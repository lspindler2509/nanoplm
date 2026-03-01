from dataclasses import dataclass
from typing import Union, List, Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertForMaskedLM
from nanoplm.pretraining.models.modern_bert.tokenizer import ProtModernBertTokenizer
from nanoplm.pretraining.models.triangular_model.model_withTriangularAttention import ModernBertForMaskedLMWithTriangularAttention
from nanoplm.pretraining.models.triangular_model.model_with_recycling import ModernBertForMaskedLMWithRecycling
from nanoplm.pretraining.models.triangular_model.model_with_data2vec import ModernBertForMaskedLMWithData2Vec


class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x


class ModernBertMLPSwiGLU(nn.Module):
    """Replacement MLP that applies SwiGLU to each ModernBERT layer."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias)
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = SwiGLU()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states):
        x, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x, gate)))


class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x


class ModernBertMLPSwiGLU(nn.Module):
    """Replacement MLP that applies SwiGLU to each ModernBERT layer."""

    def __init__(self, config: ModernBertConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias)
        self.drop = nn.Dropout(config.mlp_dropout)
        self.act = SwiGLU()
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def forward(self, hidden_states):
        x, gate = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(self.drop(self.act(x, gate)))

@dataclass
class ProtModernBertMLMConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int = 32
    mlp_activation: str = "swiglu"
    mlp_dropout: float = 0.0
    mlp_bias: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0
    classifier_activation: str = "gelu"
    # Triangular Attention parameters (optional, only used if use_triangular_attention=True)
    use_triangular_attention: bool = False
    triangular_layers: Optional[Union[List[int], str]] = None
    triangular_pair_dim: Optional[int] = None
    triangular_heads: Optional[int] = None
    triangular_dropout: Optional[float] = None
    # Recycling parameters (optional, only used if recycling=True)
    recycling: bool = False
    n_layers_in_prelude: Optional[int] = 4
    n_layers_in_recurrent_block: Optional[int] = 8
    n_layers_in_coda: Optional[int] = 4
    mean_recurrence: Optional[int] = 4  # Average number of recycling iterations
    backprop_depth: Optional[int] = 1  # Fixed number of iterations with gradients (not randomized)
    injection_type: Optional[str] = "add"  # Options: "add", "gate", "linear", "ffn", "none"
    sampling_scheme: Optional[str] = "uniform-0-4"  # Sampling scheme for num_steps_no_grad (uniform 0-4 recurrent steps)
    state_init: Optional[str] = "like-init"  # Options: "normal", "embed", "like-init", "zero", "unit"
    recycling_mode: Optional[str] = "recurrentgpt"  # Options: "recurrentgpt" (default) or "boltz2" (additive recycling like AlphaFold/Boltz2)
    # Data2Vec parameters (optional; only required/used when use_data2vec=True)
    use_data2vec: Optional[bool] = False  # Enable Data2Vec self-supervised learning
    average_top_k_layers: Optional[int] = 4  # Number of top layers to average for teacher target
    ema_decay: Optional[float] = 0.999  # Initial EMA decay rate
    ema_end_decay: Optional[float] = 0.9999  # Final EMA decay rate
    ema_anneal_end_step: Optional[int] = 40000  # Steps to finish EMA decay annealing
    data2vec_loss_weight: Optional[float] = 0.5  # Weight for Data2Vec loss (combined with MLM loss)
    data2vec_loss_scale: Optional[float] = -1.0  # Loss scale (-1.0 = auto = 1/sqrt(hidden_size) as in fairseq, >0 = explicit scale)
    data2vec_layer_norm_targets: Optional[bool] = False  # Apply layer norm to teacher targets (improves stability)
    data2vec_instance_norm_targets: Optional[bool] = False  # Apply instance norm to teacher targets (improves stability)
    data2vec_loss_dropout: Optional[float] = 0.5  # Per-batch probability to skip data2vec loss (and teacher forward). 0.5 = 50:50, 0 = never skip. Saves time.
    ema_transformer_layers_only: Optional[bool] = False  # If True, share embeddings & embed norm (copy from student); only transformer layers get EMA (like Fairseq)
    # Data2Vec 2.0 regression head (optional; only used when use_data2vec=True). Defaults match fairseq D2vDecoderConfig.
    data2vec_head_layers: Optional[int] = 2  # MLP layers when CNN decoder off (data2vec 2.0 style)
    data2vec_use_cnn_decoder: Optional[bool] = True  # Use 1D CNN decoder before MLP (Data2Vec 2.0 default, fairseq Decoder1d)
    data2vec_decoder_dim: Optional[int] = 384  # D2vDecoderConfig default
    data2vec_decoder_kernel: Optional[int] = 5  # D2vDecoderConfig default
    data2vec_decoder_layers: Optional[int] = 5  # D2vDecoderConfig default
    data2vec_decoder_groups: Optional[int] = 16  # D2vDecoderConfig default
    data2vec_decoder_residual: Optional[bool] = True  # D2vDecoderConfig default
    data2vec_projection_layers: Optional[int] = 1  # D2vDecoderConfig default
    data2vec_projection_ratio: Optional[float] = 2.0  # D2vDecoderConfig default
    data2vec_layer_norm_target_layer: Optional[bool] = False
    data2vec_instance_norm_target_layer: Optional[bool] = False
    data2vec_batch_norm_target_layer: Optional[bool] = False

class ProtModernBertMLM(nn.Module):
    """
    Clean implementation: either standard ModernBERT OR modular segments
    No inheritance confusion, no duplicate parameters
    """

    def __init__(
        self,
        config: ProtModernBertMLMConfig
    ):
        super().__init__()
        self._keys_to_ignore_on_save = set()
        self._keys_to_ignore_on_load_missing = set()
        self._keys_to_ignore_on_load_unexpected = set()
        
        self.tokenizer = ProtModernBertTokenizer()
        self.use_triangular_attention = config.use_triangular_attention
        self.recycling = config.recycling
        self.use_data2vec = config.use_data2vec
        
        # Validate: only one of recycling or triangular_attention can be True
        if self.recycling and self.use_triangular_attention:
            raise ValueError(
                "recycling and use_triangular_attention cannot both be True. "
                "Please choose one: either recycling=True OR use_triangular_attention=True"
            )
        if self.use_data2vec and self.use_triangular_attention:
            raise ValueError(
                "use_data2vec and use_triangular_attention cannot both be True. "
                "Please choose one: either use_data2vec=True OR use_triangular_attention=True"
            )
        
        # Validate recycling parameters (required only if recycling=True)
        if self.recycling:
            if config.n_layers_in_prelude is None or config.n_layers_in_recurrent_block is None or config.n_layers_in_coda is None:
                raise ValueError(
                    "When recycling=True, n_layers_in_prelude, n_layers_in_recurrent_block, "
                    "and n_layers_in_coda must all be specified"
                )
            if config.mean_recurrence is None:
                raise ValueError("When recycling=True, mean_recurrence must be specified")
            if config.backprop_depth is None:
                raise ValueError("When recycling=True, backprop_depth must be specified")
            if config.injection_type is None:
                raise ValueError("When recycling=True, injection_type must be specified")
            if config.sampling_scheme is None:
                raise ValueError("When recycling=True, sampling_scheme must be specified")
            if config.state_init is None:
                raise ValueError("When recycling=True, state_init must be specified")
            
            total = config.n_layers_in_prelude + config.n_layers_in_recurrent_block + config.n_layers_in_coda
            if total != config.num_hidden_layers:
                raise ValueError(
                    f"Layer split mismatch: n_layers_in_prelude ({config.n_layers_in_prelude}) + "
                    f"n_layers_in_recurrent_block ({config.n_layers_in_recurrent_block}) + "
                    f"n_layers_in_coda ({config.n_layers_in_coda}) = {total}, "
                    f"but num_hidden_layers = {config.num_hidden_layers}"
                )
        
        # Create ModernBERT config
        self.config = ModernBertConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            max_position_embeddings=1024,
            mlp_dropout=config.mlp_dropout,
            mlp_bias=config.mlp_bias,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
            classifier_activation=config.classifier_activation,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=None,
            unk_token_id=self.tokenizer.unk_token_id,
            mask_token_id=self.tokenizer.mask_token_id,
            loss_type="ForMaskedLM",
            tie_word_embeddings=False,
        )
        # Build base model for SwiGLU replacement. Do not assign _base.model to self.model,
        # otherwise the same parameters appear under both model.* and bert_model.model.* in the
        # state dict (shared tensors / duplicate memory on save).
        _base = ModernBertForMaskedLM(self.config)
        _model = _base.model

        # Apply SwiGLU activation to MLP layers if specified
        if config.mlp_activation.lower() == "swiglu":
            for layer in _model.layers:
                layer.mlp = ModernBertMLPSwiGLU(self.config)

        # Manueller Fix wenn layer_types null ist:
        if not hasattr(self.config, 'layer_types') or self.config.layer_types is None:
            print("Fixing layer_types manually...")
            self.config.layer_types = [
                "sliding_attention" if bool(i % self.config.global_attn_every_n_layers) else "full_attention"
                for i in range(self.config.num_hidden_layers)
            ]
            print("layer_types nach Fix:", self.config.layer_types)

        if not hasattr(self.config, 'rope_parameters') or self.config.rope_parameters is None:
            print("Fixing rope_parameters manually...")
            # Manueller RoPE parameter fix für ältere Versionen
            self.config.rope_parameters = {
                "full_attention": {
                    "rope_theta": 160_000.0,
                    "rope_type": "default"
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0, 
                    "rope_type": "default"
                }
            }
            print("rope_parameters nach Fix:", self.config.rope_parameters)
        
        if self.recycling:
            print("🔄 Building RECYCLING architecture")
            # Pass recycling config to ModernBertConfig
            self.config.n_layers_in_prelude = config.n_layers_in_prelude
            self.config.n_layers_in_recurrent_block = config.n_layers_in_recurrent_block
            self.config.n_layers_in_coda = config.n_layers_in_coda
            self.config.mean_recurrence = config.mean_recurrence
            self.config.backprop_depth = config.backprop_depth
            self.config.injection_type = config.injection_type
            self.config.sampling_scheme = config.sampling_scheme
            self.config.state_init = config.state_init
            self.config.recycling_mode = config.recycling_mode
            # Pass Data2Vec config to ModernBertConfig
            self.config.use_data2vec = config.use_data2vec
            self.config.average_top_k_layers = config.average_top_k_layers
            self.config.ema_decay = config.ema_decay
            self.config.ema_end_decay = config.ema_end_decay
            self.config.ema_anneal_end_step = config.ema_anneal_end_step
            self.config.data2vec_loss_weight = config.data2vec_loss_weight
            self.config.data2vec_loss_scale = config.data2vec_loss_scale
            self.config.data2vec_layer_norm_targets = config.data2vec_layer_norm_targets
            self.config.data2vec_instance_norm_targets = config.data2vec_instance_norm_targets
            self.config.data2vec_loss_dropout = config.data2vec_loss_dropout
            self.bert_model = ModernBertForMaskedLMWithRecycling(self.config)
        elif self.use_triangular_attention:
            print("🔺 Building MODULAR architecture with triangular attention")
            self.bert_model = ModernBertForMaskedLMWithTriangularAttention(self.config, triangular_attention_layers=config.triangular_layers, triangular_pair_dim=config.triangular_pair_dim, triangular_heads=config.triangular_heads, triangular_dropout=config.triangular_dropout)
            print("=== Weight Check ===")
            for name, param in self.bert_model.named_parameters():
                if 'weight' in name:
                    print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}")
                    if param.data.std() > 10.0 or param.data.mean().abs() > 1.0:
                        print(f"⚠️  PROBLEMATIC: {name}")
        elif self.use_data2vec and not self.recycling and not self.use_triangular_attention:
            print("📊 Building ModernBERT architecture with Data2Vec (no recycling, no triangular attention)")
            # Pass Data2Vec config to ModernBertConfig
            self.config.use_data2vec = config.use_data2vec
            self.config.average_top_k_layers = config.average_top_k_layers
            self.config.ema_decay = config.ema_decay
            self.config.ema_end_decay = config.ema_end_decay
            self.config.ema_anneal_end_step = config.ema_anneal_end_step
            self.config.data2vec_loss_weight = config.data2vec_loss_weight
            self.config.data2vec_loss_scale = config.data2vec_loss_scale
            self.config.data2vec_layer_norm_targets = config.data2vec_layer_norm_targets
            self.config.data2vec_instance_norm_targets = config.data2vec_instance_norm_targets
            self.config.data2vec_loss_dropout = config.data2vec_loss_dropout
            self.config.data2vec_head_layers = config.data2vec_head_layers
            self.config.data2vec_use_cnn_decoder = config.data2vec_use_cnn_decoder
            self.config.data2vec_decoder_dim = config.data2vec_decoder_dim
            self.config.data2vec_decoder_kernel = config.data2vec_decoder_kernel
            self.config.data2vec_decoder_layers = config.data2vec_decoder_layers
            self.config.data2vec_decoder_groups = config.data2vec_decoder_groups
            self.config.data2vec_decoder_residual = config.data2vec_decoder_residual
            self.config.data2vec_projection_layers = config.data2vec_projection_layers
            self.config.data2vec_projection_ratio = config.data2vec_projection_ratio
            self.config.ema_transformer_layers_only = config.ema_transformer_layers_only
            self.config.data2vec_layer_norm_target_layer = config.data2vec_layer_norm_target_layer
            self.config.data2vec_instance_norm_target_layer = config.data2vec_instance_norm_target_layer
            self.config.data2vec_batch_norm_target_layer = config.data2vec_batch_norm_target_layer
            self.bert_model = ModernBertForMaskedLMWithData2Vec(self.config)
        else:
            print("🔧 Building STANDARD ModernBERT architecture")
            self.bert_model = _base  
        print(self.config)
        
        # Print model parameter count
        self._print_parameter_count()

    def _print_parameter_count(self):
        """Print detailed parameter count for the model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n📊 MODEL PARAMETER COUNT:")
        print(f"   Total parameters:       {total_params:,}")
        print(f"   Trainable parameters:   {trainable_params:,}")
    
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
        return self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,            
            inputs_embeds=inputs_embeds,
            labels=labels,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )