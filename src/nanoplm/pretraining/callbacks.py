"""Custom callbacks for training."""
import wandb
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Optional


class Data2VecUpdateCallback(TrainerCallback):
    """Callback to update EMA teacher for Data2Vec."""
    
    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs
    ):
        """Update EMA teacher after each training step."""
        if model is not None:
            if hasattr(model, 'set_num_updates'):
                model.set_num_updates(state.global_step)
            elif hasattr(model, 'bert_model') and hasattr(model.bert_model, 'set_num_updates'):
                # Fallback: try bert_model.set_num_updates
                model.bert_model.set_num_updates(state.global_step)


class ParameterLoggingCallback(TrainerCallback):
    """Callback to log specific model parameters to WandB."""
    
    def __init__(self, parameter_names: list[str]):
        """
        Initialize the callback.
        
        Args:
            parameter_names: List of parameter names to log (e.g., ['model.bert_model.layers.8.p_out.weight'])
        """
        self.parameter_names = parameter_names
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Log parameters when logging happens - adds to existing logs."""
        if wandb.run is not None and model is not None and logs is not None:
            param_logs = {}
            
            for param_name in self.parameter_names:
                try:
                    param = dict(model.named_parameters()).get(param_name)
                    
                    if param is not None:
                        # Log statistics
                        param_logs[f"params/{param_name}/mean"] = param.data.mean().item()
                        param_logs[f"params/{param_name}/std"] = param.data.std().item()
                        param_logs[f"params/{param_name}/abs_mean"] = param.data.abs().mean().item()
                        param_logs[f"params/{param_name}/max"] = param.data.max().item()
                        param_logs[f"params/{param_name}/min"] = param.data.min().item()
                        
                        # Log norm if it's a matrix
                        if param.dim() >= 2:
                            param_logs[f"params/{param_name}/frobenius_norm"] = torch.norm(param.data, p='fro').item()
                except Exception:
                    # Silently skip if parameter not found or error occurs
                    continue
            
            # Add parameter logs to existing logs dict and log directly to WandB
            if param_logs:
                wandb.log(param_logs)
        
        return control


class Data2VecLossLoggingCallback(TrainerCallback):
    """Callback to log separate MLM and Data2Vec losses to WandB."""
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Log separate losses when logging happens."""
        # Losses are extracted in compute_loss and stored in model._last_mlm_loss/_last_d2v_loss
        # This is a temporary solution - ideally losses would be in logs directly
        if wandb.run is not None and model is not None and logs is not None:
            # Structure: ProtModernBertMLM -> bert_model (ModernBertForMaskedLMWithRecycling)
            bert_model = None
            if hasattr(model, 'bert_model'):
                bert_model = model.bert_model
            elif hasattr(model, 'model'):  # Fallback
                bert_model = model.model
            
            if bert_model is not None:
                loss_logs = {}
                # Extract MLM loss if available
                if hasattr(bert_model, '_last_mlm_loss') and bert_model._last_mlm_loss is not None:
                    loss_logs["loss/mlm_loss"] = bert_model._last_mlm_loss
                
                # Extract Data2Vec loss if available
                if hasattr(bert_model, '_last_d2v_loss') and bert_model._last_d2v_loss is not None:
                    loss_logs["loss/data2vec_loss"] = bert_model._last_d2v_loss
                    # Get data2vec_loss_weight from config
                    d2v_weight = 0.5
                    if hasattr(bert_model, 'config'):
                        d2v_weight = getattr(bert_model.config, 'data2vec_loss_weight', 0.5)
                    loss_logs["loss/data2vec_loss_weighted"] = bert_model._last_d2v_loss * d2v_weight
                
                if loss_logs:
                    wandb.log(loss_logs)
        
        return control

