import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import ModernBertConfig, ModernBertPreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging
from nanoplm.pretraining.models.modern_bert.activations import ACT2FN
from transformers.modeling_outputs import MaskedLMOutput
from nanoplm.utils import EMAModule, EMAModuleConfig
import torch.nn.functional as F
import math
from contextlib import nullcontext



def get_annealed_rate(start, end, curr_step, total_steps):
    """Compute annealed rate between start and end. Copied from Fairseq."""
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.ops.triton.rotary import apply_rotary
except ImportError:
    RotaryEmbedding = object

logger = logging.get_logger(__name__)


class ModernBertForMaskedLMWithData2Vec(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]
    _keys_to_ignore_on_save = ["model._ema"]  # Exclude EMA module from safetensors save (it's a dict, not tensors)

    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.model = ModernBertModelWithData2Vec(
            config)
        self.loss_type = "ForMaskedLM"
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=config.decoder_bias)

        self.sparse_prediction = self.config.sparse_prediction
        self.sparse_pred_ignore_index = self.config.sparse_pred_ignore_index

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLMWithData2Vec):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def get_output_embeddings(self):
        return self.decoder

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.decoder = new_embeddings
    
    def set_num_updates(self, num_updates):
        """Update EMA decay rate and step EMA teacher for Data2Vec."""
        if hasattr(self.model, 'set_num_updates'):
            self.model.set_num_updates(num_updates)
    
    def _compute_data2vec_loss(
        self,
        input_ids,
        attention_mask,
        sliding_window_mask,
        position_ids,
        inputs_embeds,
        indices,
        cu_seqlens,
        max_seqlen,
        batch_size,
        seq_len,
        labels,
        last_hidden_state,
    ):
        """Compute Data2Vec loss using EMA teacher model."""
        if self.model.ema is None or self.model.regression_head is None:
            return None
        
        # Get mask tokens (same as MLM loss)
        if self.sparse_prediction:
            mask_tokens = labels.view(-1) != self.sparse_pred_ignore_index
        else:
            # For dense prediction, we need to find masked positions
            # Assume labels contain -100 for non-masked tokens
            mask_tokens = labels.view(-1) != -100
        
        if not mask_tokens.any():
            return None
        
        with torch.no_grad():
            self.model.ema.model.eval()
            
            # Reconstruct unmasked tokens for teacher (following fairseq: teacher gets target_tokens)
            # In fairseq, target_tokens are the original (unmasked) tokens
            # We reconstruct them from labels: labels contain original token IDs for masked positions, -100 for others
            # For unmasked positions, we use input_ids (which may be masked, random, or original)
            # But for teacher, we want the original tokens, so we use labels where available, else input_ids
            if input_ids is not None:
                # Create target_tokens: use labels for masked positions (where labels != -100), else use input_ids
                target_tokens = input_ids.clone()
                if labels is not None:
                    # Where labels are valid (not -100), use them (these are the original tokens)
                    valid_labels = labels != -100
                    target_tokens[valid_labels] = labels[valid_labels]
            else:
                # If no input_ids, use inputs_embeds (but we can't create target_tokens from embeddings)
                # In this case, we'll use the masked input_ids for teacher (not ideal, but necessary)
                target_tokens = None
            
            # Teacher forward pass (with unmasked input, following fairseq line 426-429)
            # In fairseq: encoder_out = self.ema.model(target_tokens, return_all_hiddens=True)
            teacher_outputs = self.model.ema.model(
                input_ids=target_tokens if target_tokens is not None else input_ids,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds if target_tokens is None else None,
                indices=indices,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                batch_size=batch_size,
                seq_len=seq_len,
                output_hidden_states=True,
            )
            
            # Get target from top-K layers average (following fairseq line 430-432)
            # In fairseq: y = encoder_out["fc_results"], then y = y[-self.average_top_k_layers:]
            # fc_results are FFN outputs BEFORE the last residual connection in each block
            # This is critical: we use the FFN output PRIOR to residual connection, not after!
            if hasattr(teacher_outputs, 'fc_results') and teacher_outputs.fc_results:
                # Use fc_results (FFN outputs before residual) - this matches fairseq exactly
                # Detach immediately to save memory (no gradient graph needed for teacher targets)
                top_k_layers = [tl.detach() for tl in teacher_outputs.fc_results[-self.model.average_top_k_layers:]]
                
                # Apply layer normalization BEFORE averaging (fairseq lines 435-454)
                # This is critical: normalize each layer individually, then average
                # Since ModernBERT uses pre-norm, we MUST normalize here (unlike fairseq where fc_results are already normalized)
                layer_norm_target_layer = getattr(self.model.config, 'data2vec_layer_norm_target_layer', True)  # Default to True for ModernBERT
                instance_norm_target_layer = getattr(self.model.config, 'data2vec_instance_norm_target_layer', False)
                batch_norm_target_layer = getattr(self.model.config, 'data2vec_batch_norm_target_layer', False)
                
                # Note: Fairseq uses T x B x C format, we use B x T x C
                # Fairseq transposes for instance/batch norm, we don't need to
                if batch_norm_target_layer:
                    # Batch norm on each layer (fairseq lines 439-445)
                    top_k_layers = [
                        F.batch_norm(
                            tl.float(), running_mean=None, running_var=None, training=True
                        )
                        for tl in top_k_layers
                    ]
                
                if instance_norm_target_layer:
                    # Instance norm on each layer (fairseq line 447-448)
                    # Fairseq transposes TBC->BCT, but we're already B x T x C
                    top_k_layers = [F.instance_norm(tl.float()) for tl in top_k_layers]
                
                if layer_norm_target_layer:
                    # Layer norm on each layer (fairseq line 453-454)
                    top_k_layers = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in top_k_layers]
                
                # Average over layers (fairseq line 456) - sum() is optimized in PyTorch
                target = sum(top_k_layers) / len(top_k_layers)
            elif hasattr(teacher_outputs, 'hidden_states') and teacher_outputs.hidden_states:
                # Fallback: if fc_results not available, use hidden_states (but this is less accurate)
                # hidden_states are after residual connection, not what we want for data2vec
                # Detach immediately to save memory (no gradient graph needed for teacher targets)
                top_k_layers = [tl.detach() for tl in teacher_outputs.hidden_states[-self.model.average_top_k_layers:]]
                
                # Apply layer normalization BEFORE averaging (fairseq lines 435-454)
                # This is critical: normalize each layer individually, then average
                # Since ModernBERT uses pre-norm, we MUST normalize here (unlike fairseq where fc_results are already normalized)
                layer_norm_target_layer = getattr(self.model.config, 'data2vec_layer_norm_target_layer', True)  # Default to True for ModernBERT
                instance_norm_target_layer = getattr(self.model.config, 'data2vec_instance_norm_target_layer', False)
                batch_norm_target_layer = getattr(self.model.config, 'data2vec_batch_norm_target_layer', False)
                
                # Note: Fairseq uses T x B x C format, we use B x T x C
                # Fairseq transposes for instance/batch norm, we don't need to
                if batch_norm_target_layer:
                    # Batch norm on each layer (fairseq lines 439-445)
                    top_k_layers = [
                        F.batch_norm(
                            tl.float(), running_mean=None, running_var=None, training=True
                        )
                        for tl in top_k_layers
                    ]
                
                if instance_norm_target_layer:
                    # Instance norm on each layer (fairseq line 447-448)
                    # Fairseq transposes TBC->BCT, but we're already B x T x C
                    top_k_layers = [F.instance_norm(tl.float()) for tl in top_k_layers]
                
                if layer_norm_target_layer:
                    # Layer norm on each layer (fairseq line 453-454)
                    top_k_layers = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in top_k_layers]
                
                # Average over layers (fairseq line 456) - sum() is optimized in PyTorch
                target = sum(top_k_layers) / len(top_k_layers)
            else:
                # Fallback: use last hidden state
                # Detach immediately to save memory (no gradient graph needed for teacher targets)
                target = teacher_outputs[0].detach()
            
            # Apply target normalization AFTER averaging (fairseq lines 461-465)
            layer_norm_targets = getattr(self.model.config, 'data2vec_layer_norm_targets', False)
            instance_norm_targets = getattr(self.model.config, 'data2vec_instance_norm_targets', False)
            
            if layer_norm_targets:
                # Layer norm over feature dimension (fairseq line 462)
                target = F.layer_norm(target.float(), target.shape[-1:])
            elif instance_norm_targets:
                # Instance norm: transpose to (batch, features, seq_len), norm, transpose back (fairseq line 464-465)
                if target.dim() == 3:  # (batch, seq_len, features)
                    target = F.instance_norm(target.transpose(1, 2).float()).transpose(1, 2)
                else:  # Already flattened or 2D
                    target = F.instance_norm(target.float())
            
            # Handle sparse prediction case
            if self.sparse_prediction:
                target = target.view(-1, target.size(-1))
                target_masked = target[mask_tokens]
            else:
                # For dense prediction, reshape and mask
                target = target.view(-1, target.size(-1))
                target_masked = target[mask_tokens]
        
        # Student prediction (following fairseq exactly: apply regression head AFTER masking)
        # In fairseq: x = x[masked_indices], then x = self.regression_head(x)
        # IMPORTANT: Student output must be normalized the same way as teacher targets.
        # last_hidden_state is normalized by final_norm, but teacher targets are normalized
        # by layer_norm_target_layer (per-layer) + optionally layer_norm_targets (after averaging).
        # We need to ensure consistency: if teacher uses layer_norm_target_layer, student should too.
        if self.sparse_prediction:
            # For sparse prediction, last_hidden_state is already masked
            student_hidden = last_hidden_state
        else:
            # For dense prediction, mask first
            student_hidden = last_hidden_state.view(-1, last_hidden_state.size(-1))[mask_tokens]
        
        # Apply same normalization as teacher targets (if layer_norm_target_layer was used)
        # Since ModernBERT's last_hidden_state is normalized by final_norm, we need to check
        # if we should apply additional normalization to match teacher targets.
        # For now, we use last_hidden_state as-is (it's already normalized by final_norm),
        # but we should verify this matches the teacher normalization scheme.
        # Note: If teacher uses layer_norm_target_layer=True, each layer is normalized individually.
        # Student's last_hidden_state is normalized by final_norm, which is similar but not identical.
        # To be safe, we could apply layer_norm here too, but let's keep it as-is for now and verify.
        x = self.model.regression_head(student_hidden)
        y = target_masked
        
        # Loss computation using Cosine Similarity Loss (like JEPA)
        # Compute cosine similarity between student prediction and teacher target
        # x and y have shape (num_masked_tokens, hidden_size)
        cosine_similarity = F.cosine_similarity(x.float(), y.float(), dim=-1)
        # Cosine similarity loss: 1 - mean(cosine_similarity), Range: [0, 2]
        # 0 = perfect alignment, 2 = opposite directions
        d2v_loss = 1.0 - torch.mean(cosine_similarity)
        
        return d2v_loss

    @torch.compile(dynamic=True)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head(output))

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if hasattr(self, "_maybe_set_compile"):
            self._maybe_set_compile()

        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                if batch_size is None and seq_len is None:
                    if inputs_embeds is not None:
                        batch_size, seq_len = inputs_embeds.shape[:2]
                    else:
                        batch_size, seq_len = input_ids.shape[:2]
                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones(
                        (batch_size, seq_len), device=device, dtype=torch.bool)

                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, position_ids, labels = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, labels=labels
                    )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        if self.sparse_prediction and labels is not None:
            # flatten labels and output first
            labels = labels.view(-1)
            last_hidden_state = last_hidden_state.view(labels.shape[0], -1)

            # then filter out the non-masked tokens
            mask_tokens = labels != self.sparse_pred_ignore_index
            last_hidden_state = last_hidden_state[mask_tokens]
            labels = labels[mask_tokens]

        logits = (
            self.compiled_head(last_hidden_state)
            if self.config.reference_compile
            else self.decoder(self.head(last_hidden_state))
        )

        loss = None
        if labels is not None:
            mlm_loss = self.loss_function(
                logits, labels, vocab_size=self.config.vocab_size, **kwargs)
            loss = mlm_loss
            d2v_loss = None
            
            # Data2Vec Loss (if enabled)
            if self.model.use_data2vec and self.model.ema is not None:
                d2v_loss = self._compute_data2vec_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sliding_window_mask=sliding_window_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    indices=indices,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    labels=labels,
                    last_hidden_state=last_hidden_state,
                )
                if d2v_loss is not None:
                    # Combine losses (weighted)
                    d2v_weight = getattr(self.config, 'data2vec_loss_weight', 2.0)
                    loss = loss + d2v_weight * d2v_loss

        if self.config._attn_implementation == "flash_attention_2":
            # Logits padding
            with nullcontext() if self.config.repad_logits_with_grad or labels is None else torch.no_grad():
                logits = _pad_modernbert_output(
                    inputs=logits, indices=indices, batch=batch_size, seqlen=seq_len)
            # Hidden states padding
            if getattr(outputs, "hidden_states", None) is not None:
                padded_hidden_states = []
                for hs in outputs.hidden_states:
                    if hs.dim() == 3 and hs.shape[0] == 1:
                        hs = hs.squeeze(0)
                    padded_hidden_states.append(
                        _pad_modernbert_output(
                            inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    )
                outputs.hidden_states = tuple(padded_hidden_states)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        # Create standard MaskedLMOutput
        output = MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        # Store separate losses in self for logging (only if Data2Vec is enabled and labels provided)
        # Note: We can't add them to MaskedLMOutput as it's a dataclass, so we store them in the model instance
        # Accumulate losses across batches (for gradient accumulation) - similar to RecyclingMetricsCallback
        # IMPORTANT: Separate accumulators for training and evaluation
        if self.model.use_data2vec and labels is not None:
            # Initialize accumulators if they don't exist
            if not hasattr(self, '_train_mlm_loss_accumulator'):
                self._train_mlm_loss_accumulator = []
                self._train_d2v_loss_accumulator = []
                self._eval_mlm_loss_accumulator = []
                self._eval_d2v_loss_accumulator = []
            
            # Choose accumulator based on training mode
            if self.training:
                mlm_acc = self._train_mlm_loss_accumulator
                d2v_acc = self._train_d2v_loss_accumulator
            else:
                mlm_acc = self._eval_mlm_loss_accumulator
                d2v_acc = self._eval_d2v_loss_accumulator
            
            # Accumulate MLM loss for logging (detached to avoid gradient issues)
            if mlm_loss is not None:
                mlm_loss_val = mlm_loss.detach().item() if torch.is_tensor(mlm_loss) else mlm_loss
                mlm_acc.append(mlm_loss_val)
            
            # Accumulate Data2Vec loss for logging (detached to avoid gradient issues)
            if d2v_loss is not None:
                d2v_loss_val = d2v_loss.detach().item() if torch.is_tensor(d2v_loss) else d2v_loss
                d2v_acc.append(d2v_loss_val)
        
        return output


class ModernBertModelWithData2Vec(ModernBertPreTrainedModel):
    _keys_to_ignore_on_save = ["_ema"]  # Exclude EMA module from safetensors save (it's a dict, not tensors)
    
    def __init__(self, config: ModernBertConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = nn.ModuleList(
            [ModernBertEncoderLayer(config, layer_id)
             for layer_id in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.rotary_emb = ModernBertRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
              
        # Data2Vec: EMA Teacher and Regression Head
        self.ema = None  # Will be initialized later
        self.average_top_k_layers = getattr(config, 'average_top_k_layers', 4)
        self.ema_decay = getattr(config, 'ema_decay', 0.999)
        self.ema_end_decay = getattr(config, 'ema_end_decay', 0.9999)
        self.ema_anneal_end_step = getattr(config, 'ema_anneal_end_step', 40000)
        self.use_data2vec = getattr(config, 'use_data2vec', False)
        
        # Regression Head for Data2Vec
        if self.use_data2vec:
            head_layers = getattr(config, 'data2vec_head_layers', 1)
            assert head_layers >= 1
            
            embed_dim = config.hidden_size
            curr_dim = embed_dim
            projs = []
            for i in range(head_layers - 1):
                next_dim = embed_dim * 2 if i == 0 else curr_dim
                linear = nn.Linear(curr_dim, next_dim)
                linear._is_regression_head = True  # Mark for initialization
                projs.append(linear)
                projs.append(nn.GELU())
                curr_dim = next_dim
            
            final_linear = nn.Linear(curr_dim, embed_dim)
            final_linear._is_regression_head = True  # Mark for initialization
            projs.append(final_linear)
            self.regression_head = nn.Sequential(*projs)
        else:
            self.regression_head = None
        
        self.post_init()

    def _init_weights(self, module: nn.Module):
        cutoff_factor = self.config.initializer_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        def init_weight(module: nn.Module, std: float):
            nn.init.trunc_normal_(
                module.weight,
                mean=0.0,
                std=std,
                a=-cutoff_factor * std,
                b=cutoff_factor * std,
            )

            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        stds = {
            "in": self.config.initializer_range,
            "out": self.config.initializer_range / math.sqrt(2.0 * self.config.num_hidden_layers),
            "embedding": self.config.initializer_range,
            "final_out": self.config.hidden_size**-0.5,
        }

        if isinstance(module, ModernBertEmbeddings):
            init_weight(module.tok_embeddings, stds["embedding"])
        elif isinstance(module, ModernBertMLP):
            init_weight(module.Wi, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertAttention):
            init_weight(module.Wqkv, stds["in"])
            init_weight(module.Wo, stds["out"])
        elif isinstance(module, ModernBertPredictionHead):
            init_weight(module.dense, stds["out"])
        elif isinstance(module, ModernBertForMaskedLMWithData2Vec):
            init_weight(module.decoder, stds["out"])
        elif isinstance(module, nn.Linear) and hasattr(module, '_is_regression_head') and module._is_regression_head:
            # Regression head for Data2Vec: initialize with larger std for better initial predictions
            # Use "in" std instead of "out" std to allow larger initial values
            init_weight(module, stds["in"])
        elif isinstance(module, nn.Linear) and hasattr(module, '_is_s_init') and module._is_s_init:
            # s_init for Boltz2-style recycling: initialize like input projections
            init_weight(module, stds["in"])
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.embeddings.tok_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.tok_embeddings = value
    
    def make_ema_teacher(self):
        """Create EMA teacher model for Data2Vec. Copied from Fairseq."""
        ema_config = EMAModuleConfig(
            ema_decay=self.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        # Skip embeddings if ema_transformer_layers_only is True (like Fairseq)
        # This means embeddings are directly copied instead of EMA-updated
        if getattr(self.config, 'ema_transformer_layers_only', False):
            # Skip token embeddings (like Fairseq's embed_tokens)
            for k, _ in self.embeddings.tok_embeddings.named_parameters():
                skip_keys.add(f"embeddings.tok_embeddings.{k}")
            # Skip embedding layer norm (like Fairseq's layernorm_embedding)
            for k, _ in self.embeddings.norm.named_parameters():
                skip_keys.add(f"embeddings.norm.{k}")
            # Note: We don't have embed_positions (we use RoPE instead)

        self.ema = EMAModule(
            self,
            ema_config,
            copy_model=True,  # Let EMAModule handle the copy
            skip_keys=skip_keys,
        )
    
    def set_num_updates(self, num_updates):
        """Update EMA decay rate and step EMA teacher for Data2Vec. Copied from Fairseq."""
        if self.ema is None and self.regression_head is not None:
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.ema_decay != self.ema_end_decay:
                if num_updates >= self.ema_anneal_end_step:
                    decay = self.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.ema_decay,
                        self.ema_end_decay,
                        num_updates,
                        self.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self)
    
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save state dict including EMA params. Copied from Fairseq."""
        state = super().state_dict(destination, prefix, keep_vars)
        # Note: We don't add _ema here because it's a dict and safetensors can't handle it
        # Instead, EMA params should be saved separately if needed (e.g., in a checkpoint callback)
        # The _keys_to_ignore_on_save ensures _ema is not included in the model save
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load state dict including EMA params. Copied from Fairseq."""
        if self.ema is not None:
            k = prefix + "_ema"
            if k in state_dict:
                self.ema.restore(state_dict[k], True)
                del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple[torch.Tensor, ...], BaseModelOutput]:
        r"""
        sliding_window_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding or far-away tokens. In ModernBert, only every few layers
            perform global attention, while the rest perform local attention. This mask is used to avoid attending to
            far-away tokens in the local attention layers when not using Flash Attention.
        indices (`torch.Tensor` of shape `(total_unpadded_tokens,)`, *optional*):
            Indices of the non-padding tokens in the input sequence. Used for unpadding the output.
        cu_seqlens (`torch.Tensor` of shape `(batch + 1,)`, *optional*):
            Cumulative sequence lengths of the input sequences. Used to index the unpadded tensors.
        max_seqlen (`int`, *optional*):
            Maximum sequence length in the batch excluding padding tokens. Used to unpad input_ids and pad output tensors.
        batch_size (`int`, *optional*):
            Batch size of the input sequences. Used to pad the output tensors.
        seq_len (`int`, *optional*):
            Sequence length of the input sequences including padding tokens. Used to pad the output tensors.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds")

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # Collect FFN outputs (fc_results) for data2vec - FFN output BEFORE residual connection
        # This matches fairseq's fc_results behavior
        # We need fc_results if data2vec is enabled, regardless of output_hidden_states
        fc_results = [] if self.use_data2vec else None

        if hasattr(self, "_maybe_set_compile"):
            self._maybe_set_compile()

        if input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(
                input_ids, attention_mask)

        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_len), device=device, dtype=torch.bool)

        repad = False
        if self.config._attn_implementation == "flash_attention_2":
            if indices is None and cu_seqlens is None and max_seqlen is None:
                repad = True
                if inputs_embeds is None:
                    with torch.no_grad():
                        input_ids, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                            inputs=input_ids, attention_mask=attention_mask
                        )
                else:
                    inputs_embeds, indices, cu_seqlens, max_seqlen, *_ = _unpad_modernbert_input(
                        inputs=inputs_embeds, attention_mask=attention_mask
                    )
            if position_ids is None:
                position_ids = indices.unsqueeze(0)
        else:
            if position_ids is None:
                position_ids = torch.arange(
                    seq_len, device=device).unsqueeze(0)

            attention_mask, sliding_window_mask = self._update_attention_mask(
                attention_mask, output_attentions=output_attentions
            )

        hidden_states = self.embeddings(
            input_ids=input_ids, inputs_embeds=inputs_embeds)
        position_embeddings = {}
        
        for layer_type in self.config.layer_types:
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states, position_ids, layer_type)
        
        # Return FC results (FFN outputs before residual) for data2vec
        return_fc = (fc_results is not None)
        
        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                sliding_window_mask=sliding_window_mask,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings[encoder_layer.attention_type],
                output_attentions=output_attentions,
                return_fc=return_fc,
            )
            hidden_states = layer_outputs[0]
            if return_fc and len(layer_outputs) > 1:
                fc_result = layer_outputs[1]
                fc_results.append(fc_result)
            if output_attentions:
                # Attentions are at index 2 if return_fc=True, else at index 1
                attn_idx = 2 if return_fc else 1
                if len(layer_outputs) > attn_idx:
                    all_self_attentions = all_self_attentions + (layer_outputs[attn_idx],)

        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_norm(hidden_states)

        if repad:
            hidden_states = _pad_modernbert_output(
                inputs=hidden_states, indices=indices, batch=batch_size, seqlen=seq_len
            )
            if all_hidden_states is not None:
                all_hidden_states = tuple(
                    _pad_modernbert_output(
                        inputs=hs, indices=indices, batch=batch_size, seqlen=seq_len)
                    for hs in all_hidden_states
                )
            if fc_results is not None:
                fc_results = [
                    _pad_modernbert_output(
                        inputs=fc, indices=indices, batch=batch_size, seqlen=seq_len)
                    for fc in fc_results
                ]
        # If the attention implementation is FA2 and there is no need for repadding, there might still be the batch
        # dimension missing
        elif (
            self.config._attn_implementation == "flash_attention_2"
            and all_hidden_states is not None
            and all_hidden_states[-1].dim() == 2
        ):
            hidden_states = hidden_states.unsqueeze(0)
            all_hidden_states = tuple(hs.unsqueeze(0)
                                      for hs in all_hidden_states)
            if fc_results is not None:
                fc_results = [fc.unsqueeze(0) for fc in fc_results]

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # Store fc_results in a custom output object (BaseModelOutput doesn't have this field)
        # We'll access it via hidden_states tuple or add it to the output
        output = BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        # Add fc_results as an attribute for data2vec to access
        if fc_results is not None:
            output.fc_results = fc_results
        return output

    
    def _update_attention_mask(self, attention_mask: torch.Tensor, output_attentions: bool) -> torch.Tensor:
        if output_attentions:
            if self.config._attn_implementation == "sdpa":
                logger.warning_once(
                    "Outputting attentions is only supported with the 'eager' attention implementation, "
                    'not with "sdpa". Falling back to `attn_implementation="eager"`.'
                )
                self.config._attn_implementation = "eager"
            elif self.config._attn_implementation != "eager":
                logger.warning_once(
                    "Outputting attentions is only supported with the eager attention implementation, "
                    f'not with {self.config._attn_implementation}. Consider setting `attn_implementation="eager"`.'
                    " Setting `output_attentions=False`."
                )

        global_attention_mask = _prepare_4d_attention_mask(
            attention_mask, self.dtype)

        # Create position indices
        rows = torch.arange(global_attention_mask.shape[2]).unsqueeze(0)
        # Calculate distance between positions
        distance = torch.abs(rows - rows.T)

        # Create sliding window mask (1 for positions within window, 0 outside)
        window_mask = (
            (distance <= self.config.local_attention //
             2).unsqueeze(0).unsqueeze(0).to(attention_mask.device)
        )
        # Combine with existing mask
        sliding_window_mask = global_attention_mask.masked_fill(
            window_mask.logical_not(), torch.finfo(self.dtype).min)

        return global_attention_mask, sliding_window_mask


class ModernBertPredictionHead(nn.Module):
    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(
            config.hidden_size, config.hidden_size, config.classifier_bias)
        self.act = ACT2FN[config.classifier_activation]
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.dense(hidden_states)))


class ModernBertEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.norm = nn.RMSNorm(
            config.hidden_size, eps=config.norm_eps)
        self.drop = nn.Dropout(config.embedding_dropout)

    @torch.compile(dynamic=True)
    def compiled_embeddings(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.drop(self.norm(self.tok_embeddings(input_ids)).to(torch.bfloat16))

    def forward(
        self, input_ids: Optional[torch.LongTensor] = None, inputs_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = self.drop(
                self.norm(inputs_embeds).to(torch.bfloat16))
        else:
            hidden_states = (
                self.compiled_embeddings(input_ids)
                if self.config.reference_compile
                else self.drop(self.norm(self.tok_embeddings(input_ids)).to(torch.bfloat16))
            )
        return hidden_states


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            # TODO cyril: this one without `S` can be removed after deprection cycle
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


class ModernBertEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        if layer_id == 0:
            self.attn_norm = nn.Identity()
        else:
            self.attn_norm = nn.LayerNorm(
                config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.attn = ModernBertAttention(config=config, layer_id=layer_id)
        self.mlp_norm = nn.LayerNorm(
            config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)
        self.attention_type = config.layer_types[layer_id]

    @torch.compile(dynamic=True)
    def compiled_mlp(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(hidden_states))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        return_fc: Optional[bool] = False,
    ) -> torch.Tensor:
        attn_outputs = self.attn(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attn_outputs[0]
        mlp_output = (
            self.compiled_mlp(hidden_states)
            if self.config.reference_compile
            else self.mlp(self.mlp_norm(hidden_states))
        )
        
        # Store FFN output BEFORE residual connection (like fairseq's fc_result)
        fc_result = mlp_output if return_fc else None
        
        hidden_states = hidden_states + mlp_output

        # Return: (hidden_states, fc_result, ...attentions)
        # fc_result is the FFN output BEFORE residual connection (like fairseq)
        if return_fc:
            return (hidden_states, fc_result) + attn_outputs[1:]
        return (hidden_states,) + attn_outputs[1:]


class ModernBertRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: ModernBertConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.layer_types = list(set(config.layer_types))
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn: Callable = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            curr_inv_freq, curr_attention_scaling = rope_init_fn(
                self.config, device, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq",
                                 curr_inv_freq, persistent=False)
            setattr(self, f"{layer_type}_original_inv_freq", curr_inv_freq)
            setattr(self, f"{layer_type}_attention_scaling",
                    curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[ModernBertConfig] = None,
        device: Optional["torch.device"] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim",
                      None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2,
                     dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    # @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        inv_freq_expanded = inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(
            x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @
                     position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * attention_scaling
            sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class ModernBertMLP(nn.Module):
    """Applies the GLU at the end of each ModernBERT layer.

    Compared to the default BERT architecture, this block replaces :class:`~transformers.model.bert.modeling_bert.BertIntermediate`
    and :class:`~transformers.model.bert.modeling_bert.SelfOutput` with a single module that has similar functionality.
    """

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.config = config
        self.Wi = nn.Linear(config.hidden_size, int(
            config.intermediate_size) * 2, bias=config.mlp_bias)
        self.act = ACT2FN[config.hidden_activation]
        self.drop = nn.Dropout(config.mlp_dropout)
        self.Wo = nn.Linear(config.intermediate_size,
                            config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(input) * gate))


class ModernBertAttention(nn.Module):
    """Performs multi-headed self attention on a batch of unpadded sequences.

    If Flash Attention 2 is installed, this module uses Flash Attention to improve throughput.
    If Flash Attention 2 is not installed, the implementation will use PyTorch's SDPA kernel,
    which requires padding and unpadding inputs, adding some overhead.

    See `forward` method for additional details.
    """

    def __init__(self, config: ModernBertConfig, layer_id: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.attention_dropout = config.attention_dropout
        self.deterministic_flash_attn = config.deterministic_flash_attn
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.head_dim * self.num_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 *
                              self.all_head_size, bias=config.attention_bias)
        layer_type = config.layer_types[layer_id]

        if layer_id % config.global_attn_every_n_layers != 0:
            self.local_attention = (
                config.local_attention // 2, config.local_attention // 2)
            max_position_embeddings = config.local_attention
        else:
            self.local_attention = (-1, -1)
            max_position_embeddings = config.max_position_embeddings

        if config._attn_implementation == "flash_attention_2":
            if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
                if layer_type is not None and layer_type in config.rope_parameters:
                    rope_parameters_dict = config.rope_parameters[layer_type]
                else:
                    rope_parameters_dict = config.rope_parameters
                rope_theta = rope_parameters_dict.get("rope_theta", 10000.0)
            else:
                rope_theta = 10000.0  # Fallback-Wert, wie bei BERT RoPE üblich

            self.rotary_emb = ModernBertUnpaddedRotaryEmbedding(
                dim=self.head_dim,
                max_seqlen=max_position_embeddings,
                base=rope_theta,
            )
        else:
            self.rotary_emb = None

        self.Wo = nn.Linear(config.hidden_size,
                            config.hidden_size, bias=config.attention_bias)
        self.out_drop = nn.Dropout(
            config.attention_dropout) if config.attention_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.Wqkv(hidden_states)

        bs = hidden_states.shape[0]
        if self.config._attn_implementation == "flash_attention_2":
            qkv = qkv.view(-1, 3, self.num_heads, self.head_dim)
        else:
            qkv = qkv.view(bs, -1, 3, self.num_heads, self.head_dim)

        attn_outputs = MODERNBERT_ATTENTION_FUNCTION[self.config._attn_implementation](
            self,
            qkv=qkv,
            rotary_emb=self.rotary_emb,
            local_attention=self.local_attention,
            bs=bs,
            dim=self.all_head_size,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.out_drop(self.Wo(hidden_states))

        # add attentions if outputted
        return (hidden_states,) + attn_outputs[1:]


class ModernBertUnpaddedRotaryEmbedding(RotaryEmbedding):
    """
    The rotary position embeddings applied directly to unpadded sequences.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seqlen: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        max_seqlen: if max_seqlen, device, and dtype are provided, we precompute the cos_sin_cache
            up to max_seqlen. If the max_seqlen, device, or dtype during training/inference differ,
            the cos_sin_cache will be recomputed during the forward pass.
        """
        super().__init__(dim=dim, base=base, device=device, interleaved=False)
        self.max_seqlen = max_seqlen

        if max_seqlen is not None and device is not None and dtype is not None:
            self._update_cos_sin_cache(max_seqlen, device=device, dtype=dtype)

    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: Optional[int] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply rotary embedding *inplace* to qkv.
        qkv: (total_nnz, 3, nheads, headdim)
        cu_seqlens: (batch + 1,) cumulative sequence lengths
        max_seqlen: int max seq length in the batch
        """
        if max_seqlen is not None:
            self._update_cos_sin_cache(
                max_seqlen, device=qkv.device, dtype=qkv.dtype)

        qkv = apply_rotary_unpadded(
            qkv,
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        return qkv

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, scale_base={self.scale_base}"


class ApplyRotaryEmbUnpad(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cos,
        sin,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        # (total_nnz, 3, nheads, headdim)
        qkv = qkv.contiguous()
        total_nnz, _three, _nheads, headdim = qkv.shape
        # We need qkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        # qk = rearrange(qkv[:, :2], "b_s t h d -> b_s (t h) d")
        qk = qkv[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            qk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=False,
            inplace=True,
        )

        ctx.save_for_backward(cos, sin, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        return qkv

    @staticmethod
    def backward(ctx, do):
        cos, sin, cu_seqlens = ctx.saved_tensors
        do = do.contiguous()
        total_nnz, _three, _nheads, headdim = do.shape
        # We need dqkv to be contiguous so that when we reshape to combine (3, nheads) dimensions,
        # we get the same tensor
        dqk = do[:, :2].view(total_nnz, -1, headdim)
        apply_rotary(
            dqk,
            cos,
            sin,
            seqlen_offsets=0,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=False,
            inplace=True,
            conjugate=True,
        )

        return do, None, None, None, None, None, None


def _unpad_modernbert_input(
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Remove padding from input sequences.

    Args:
        inputs: (batch, seqlen, ...) or (batch, seqlen)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        position_ids: (batch, seqlen), int, position ids
        labels: (batch, seqlen), int, labels

    Returns:
        unpadded_inputs: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        cu_seqlens: (batch + 1), the cumulative sequence lengths
        max_seqlen_in_batch: int
        unpadded_position_ids: (total_nnz) or None
        unpadded_labels: (total_nnz) or None
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(
        seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    if inputs.dim() == 2:
        unpadded_inputs = inputs.flatten()[indices]
    else:
        batch, seqlen, *rest = inputs.shape
        shape = batch * seqlen
        unpadded_inputs = inputs.view(shape, *rest)[indices]

    unpadded_position_ids = position_ids.flatten(
    )[indices] if position_ids is not None else None
    unpadded_labels = labels.flatten()[indices] if labels is not None else None

    return unpadded_inputs, indices, cu_seqlens, max_seqlen_in_batch, unpadded_position_ids, unpadded_labels


def _pad_modernbert_output(
    inputs: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int,
) -> torch.Tensor:
    """
    Add padding to sequences.

    Args:
        inputs: (total_nnz, ...) or (total_nnz,), where total_nnz = number of tokens selected in attention_mask.
        indices: (total_nnz)
        batch: int, batch size
        seqlen: int, max sequence length

    Returns:
        padded_inputs: (batch, seqlen, ...) or (batch, seqlen)
    """
    if inputs.dim() == 1:
        output = torch.zeros(
            batch * seqlen, dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen)
    else:
        _, *rest = inputs.shape
        output = torch.zeros(batch * seqlen, *rest,
                             dtype=inputs.dtype, device=inputs.device)
        output[indices] = inputs
        padded_inputs = output.view(batch, seqlen, *rest)

    return padded_inputs


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return _expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(
        bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def eager_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    position_embeddings: torch.Tensor,
    output_attentions: Optional[bool] = False,
    **_kwargs,
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor]]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = position_embeddings
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    scale = module.head_dim**-0.5
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scale

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=module.attention_dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bs, -1, dim)
    if output_attentions:
        return (attn_output, attn_weights)
    return (attn_output,)


def flash_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    rotary_emb: ModernBertUnpaddedRotaryEmbedding,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    target_dtype: torch.dtype = torch.bfloat16,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # (total_seqlen, 3, nheads, headdim)
    qkv = rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
    if convert_dtype:
        # FA2 implementation only supports fp16 and bf16. If FA2 is supported,
        # bfloat16 must be supported as of FA2 2.5.7. (Turing GPUs not supported)
        orig_dtype = qkv.dtype
        qkv = qkv.to(target_dtype)

        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
        attn = attn.to(orig_dtype)  # type: ignore
    else:
        attn = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=module.attention_dropout if module.training else 0.0,
            deterministic=module.deterministic_flash_attn,
            window_size=local_attention,
        )
    return (attn.view(bs, dim),)


def sdpa_attention_forward(
    module: "ModernBertAttention",
    qkv: torch.Tensor,
    attention_mask: torch.Tensor,
    sliding_window_mask: torch.Tensor,
    position_ids: Optional[torch.LongTensor],
    local_attention: tuple[int, int],
    bs: int,
    dim: int,
    position_embeddings: torch.Tensor,
    **_kwargs,
) -> tuple[torch.Tensor]:
    # qkv: [batch_size, seqlen, 3, nheads, headdim]
    cos, sin = position_embeddings
    query, key, value = qkv.transpose(3, 1).unbind(dim=2)
    # query, key, value: [batch_size, heads, seq_len, head_dim]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if local_attention != (-1, -1):
        attention_mask = sliding_window_mask

    attn_output = (
        F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=module.attention_dropout if module.training else 0.0,
            attn_mask=attention_mask,
        )
        .transpose(1, 2)
        .contiguous()
    )
    attn_output = attn_output.view(bs, -1, dim)
    return (attn_output,)


MODERNBERT_ATTENTION_FUNCTION = {
    "flash_attention_2": flash_attention_forward,
    "eager": eager_attention_forward,
    "sdpa": sdpa_attention_forward,
}


def apply_rotary_unpadded(
    qkv,
    cos,
    sin,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
):
    """
    Arguments:
        qkv: (total_nnz, 3, nheads, headdim) - input tensor for packed QKV.
        cos, sin: (seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        inplace: if True, apply rotary embedding in-place.
        seqlen_offsets: (batch_size,) or int. Each sequence in x is shifted by this amount.
            Most commonly used in inference when we have KV cache.
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Return:
        out: (total_nnz, dim)
    rotary_dim must be <= headdim
    Apply rotary embedding to the first rotary_dim of x.
    """
    return ApplyRotaryEmbUnpad.apply(qkv, cos, sin, cu_seqlens, max_seqlen)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

