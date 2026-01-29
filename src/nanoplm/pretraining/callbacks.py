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
            # Determine if this is evaluation or training based on log keys
            # HuggingFace adds "eval_" prefix to evaluation metrics
            is_eval = any(k.startswith("eval_") for k in logs.keys())
            prefix = "eval/" if is_eval else "train/"
            
            # Structure: ProtModernBertMLM -> bert_model (ModernBertForMaskedLMWithRecycling)
            bert_model = None
            if hasattr(model, 'bert_model'):
                bert_model = model.bert_model
            elif hasattr(model, 'model'):  # Fallback
                bert_model = model.model
            
            if bert_model is not None:
                loss_logs = {}
                
                # Select appropriate accumulators based on training/evaluation mode
                if is_eval:
                    mlm_acc = getattr(bert_model, '_eval_mlm_loss_accumulator', [])
                    d2v_acc = getattr(bert_model, '_eval_d2v_loss_accumulator', [])
                else:
                    mlm_acc = getattr(bert_model, '_train_mlm_loss_accumulator', [])
                    d2v_acc = getattr(bert_model, '_train_d2v_loss_accumulator', [])
                    # For training, we need to normalize by gradient_accumulation_steps
                    gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
                
                # Calculate number of steps since last log
                # For training: Each batch corresponds to one accumulation step, so num_batches / gradient_accumulation_steps = num_steps
                # For evaluation: Each batch is one step (no gradient accumulation)
                if is_eval:
                    steps_since_last_log = max(len(mlm_acc), len(d2v_acc), 1)
                else:
                    num_batches = max(len(mlm_acc), len(d2v_acc), 0)
                    steps_since_last_log = max(num_batches // gradient_accumulation_steps, 1) if num_batches > 0 else 1
                
                # Process MLM loss
                if mlm_acc:
                    if is_eval:
                        # For evaluation, just average (no gradient accumulation normalization)
                        avg_mlm_loss = sum(mlm_acc) / len(mlm_acc)
                    else:
                        # For training, normalize by gradient_accumulation_steps then divide by steps
                        normalized_losses = [loss / gradient_accumulation_steps for loss in mlm_acc]
                        sum_mlm_loss = sum(normalized_losses)
                        avg_mlm_loss = sum_mlm_loss / steps_since_last_log if steps_since_last_log > 0 else sum_mlm_loss
                    loss_logs[f"{prefix}mlm_loss"] = avg_mlm_loss
                
                # Process Data2Vec loss
                if d2v_acc:
                    if is_eval:
                        # For evaluation, just average (no gradient accumulation normalization)
                        avg_d2v_loss = sum(d2v_acc) / len(d2v_acc)
                    else:
                        # For training, normalize by gradient_accumulation_steps then divide by steps
                        normalized_losses = [loss / gradient_accumulation_steps for loss in d2v_acc]
                        sum_d2v_loss = sum(normalized_losses)
                        avg_d2v_loss = sum_d2v_loss / steps_since_last_log if steps_since_last_log > 0 else sum_d2v_loss
                    loss_logs[f"{prefix}data2vec_loss"] = avg_d2v_loss
                    
                    # Also log weighted Data2Vec loss
                    d2v_weight = 2.0
                    if hasattr(bert_model, 'config'):
                        d2v_weight = getattr(bert_model.config, 'data2vec_loss_weight', 2.0)
                    loss_logs[f"{prefix}data2vec_loss_weighted"] = avg_d2v_loss * d2v_weight

                if loss_logs:
                    wandb.log(loss_logs)
                
                # Reset accumulators after logging (for next logging interval)
                if is_eval:
                    if hasattr(bert_model, '_eval_mlm_loss_accumulator'):
                        bert_model._eval_mlm_loss_accumulator = []
                    if hasattr(bert_model, '_eval_d2v_loss_accumulator'):
                        bert_model._eval_d2v_loss_accumulator = []
                else:
                    if hasattr(bert_model, '_train_mlm_loss_accumulator'):
                        bert_model._train_mlm_loss_accumulator = []
                    if hasattr(bert_model, '_train_d2v_loss_accumulator'):
                        bert_model._train_d2v_loss_accumulator = []
        
        return control


class MultiStepRecyclingEvalCallback(TrainerCallback):
    """Callback to evaluate with different recycling step counts (1 to mean_recurrence).
    
    After the normal evaluation, this callback runs additional evaluations with
    fixed recycling steps (1, 2, ..., mean_recurrence) and logs results to WandB
    with suffixes like eval_loss_steps_1, eval_loss_steps_2, etc.
    """
    
    def __init__(self, enabled: bool = True, trainer=None):
        """
        Args:
            enabled: Whether to enable multi-step evaluation (default: True)
            trainer: Optional Trainer instance. If None, will try to get from kwargs or state.
        """
        self.enabled = enabled
        self.trainer = trainer
        self._in_nested_eval = False
    
    def on_init_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Store trainer reference when available."""
        # Try to get trainer from kwargs (some versions pass it)
        if self.trainer is None:
            self.trainer = kwargs.get('trainer')
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run additional evaluations with different recycling step counts."""
        if self._in_nested_eval or not self.enabled or model is None:
            return
        
        # Try to get trainer from various sources
        trainer = self.trainer
        if trainer is None:
            # Try kwargs (some HuggingFace versions might pass it)
            trainer = kwargs.get('trainer')
        if trainer is None:
            # Try to get from state if available (not standard, but worth trying)
            if hasattr(state, 'trainer'):
                trainer = state.trainer
        
        if trainer is None:
            print("MultiStepRecyclingEvalCallback: trainer not found. Please pass trainer in __init__ or ensure it's available.")
            return
        
        # Get mean_recurrence from model config
        mean_recurrence = None
        if hasattr(model, 'bert_model') and hasattr(model.bert_model, 'model'):
            mean_recurrence = getattr(model.bert_model.model.config, 'mean_recurrence', None)
                
        # Get eval dataset from trainer
        # Note: HuggingFace Trainer stores eval_dataset, but it might be a dict for multiple eval datasets
        eval_dataset = trainer.eval_dataset

        self._in_nested_eval = True
        
        # Hardcoded step counts to evaluate: 1, 4, 8, and mean_recurrence (typically 12)
        step_counts = [1, 4, 8, mean_recurrence]
        
        # Run evaluation for each step count
        for num_steps in step_counts:
            # Run evaluation
            model.bert_model.config.mean_recurrence = num_steps
            model.bert_model.model.eval()
            with torch.no_grad():
                eval_results = trainer.evaluate(eval_dataset=eval_dataset)

            # Store results for this step count
            if wandb.run is not None:
                for key, value in eval_results.items():
                    if 'loss' in key:
                        wandb.log({f"eval_recycling_steps/{key}_{num_steps}": value})
            if num_steps == mean_recurrence:
                self._in_nested_eval = False
        
        return control


class RecyclingMetricsCallback(TrainerCallback):
    """Callback to log recycling metrics from model.monitor_module.
    
    Similar to RecurrentGPT's monitoring system, this callback collects
    metrics computed by monitor_module and logs them to WandB.
    """
    
    def __init__(self, enabled: bool = True, log_interval: int = 1):
        """
        Args:
            enabled: Whether to enable recycling metrics logging (default: True)
            log_interval: Log metrics every N steps (default: 1, log every step)
        """
        self.enabled = enabled
        self.log_interval = log_interval
    
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        logs: Optional[dict] = None,
        **kwargs
    ):
        """Collect and log recycling metrics from the model."""
        if not self.enabled or model is None:
            return
        
        # Check if we should log this step
        if state.global_step % self.log_interval != 0:
            return
        
        mean_recurrence = None
        if hasattr(model, 'bert_model') and hasattr(model.bert_model, 'model'):
            mean_recurrence = getattr(model.bert_model.model.config, 'mean_recurrence', None)
        
        # Determine if this is evaluation or training based on log keys
        # HuggingFace adds "eval_" prefix to evaluation metrics
        is_eval = logs is not None and any(k.startswith("eval_") for k in logs.keys())
        prefix = "eval_recycling/" if is_eval else "train_recycling/"
        
        # Get the underlying model (might be wrapped)
        base_model = model
        if hasattr(model, 'bert_model'):
            base_model = model.bert_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model
        
        # Check if model has latest_metrics
        if not hasattr(base_model, 'latest_metrics'):
            return
        
        metrics = base_model.latest_metrics
        if not metrics:
            return
        
        # Log metrics to WandB with prefix (train/ or eval/)
        if wandb.run is not None:
            recycling_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if is_eval:
                        recycling_metrics[f"{prefix}{key}_{mean_recurrence}"] = value
                    else:
                        recycling_metrics[f"{prefix}{key}"] = value

            if recycling_metrics:
                wandb.log(recycling_metrics)
        
        # Reset accumulator after logging (for next step)
        if hasattr(base_model, '_metrics_accumulator'):
            base_model._metrics_accumulator = {}
            base_model._metrics_count = 0
        
        return control

