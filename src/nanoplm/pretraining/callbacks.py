"""Custom callbacks for training."""
import wandb
import torch
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Optional


class ParameterLoggingCallback(TrainerCallback):
    """Callback to log specific model parameters to WandB."""
    
    def __init__(self, parameter_names: list[str], log_every_n_steps: int = 100):
        """
        Initialize the callback.
        
        Args:
            parameter_names: List of parameter names to log (e.g., ['model.bert_model.triangular_layers.0.p_out.weight'])
            log_every_n_steps: Log every N training steps
        """
        self.parameter_names = parameter_names
        self.log_every_n_steps = log_every_n_steps
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Log parameters when logging happens."""
        if state.global_step % self.log_every_n_steps == 0:
            if wandb.run is not None and model is not None:
                param_logs = {}
                
                for param_name in self.parameter_names:
                    try:
                        # Get parameter by name
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
                        else:
                            # Try to find similar parameter names (for debugging)
                            all_params = list(dict(model.named_parameters()).keys())
                            similar = [p for p in all_params if param_name.split('.')[-1] in p]
                            if similar:
                                print(f"⚠️  Parameter '{param_name}' not found. Similar: {similar[:3]}")
                    except Exception as e:
                        print(f"⚠️  Error logging parameter '{param_name}': {e}")
                
                if param_logs:
                    wandb.log(param_logs, step=state.global_step)
        
        return control

