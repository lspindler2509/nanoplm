"""
Custom Trainer for pretraining that handles optimizer state loading
for optimizers like stable_adamw that may have additional state keys.
"""

import torch
from transformers import Trainer
from typing import Optional
from pathlib import Path
import logging
import warnings
import os
import glob
import importlib.metadata as importlib_metadata
import importlib.util
from transformers.utils import is_torch_xla_available, is_sagemaker_mp_enabled, is_accelerate_available

OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"

def is_deepspeed_available():
    package_exists = importlib.util.find_spec("deepspeed") is not None

    # Check we're not importing a "deepspeed" directory somewhere but the actual library by trying to grab the version
    # AND checking it has an author field in the metadata that is HuggingFace.
    if package_exists:
        try:
            _ = importlib_metadata.metadata("deepspeed")
            return True
        except importlib_metadata.PackageNotFoundError:
            return False
if is_accelerate_available():
    from accelerate.utils import (
        load_fsdp_optimizer,
        save_fsdp_model,
    )
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper
import inspect

logger = logging.getLogger(__name__)


class PretrainingTrainer(Trainer):
    """
    Custom Trainer that handles missing optimizer state keys when resuming.
    
    This is needed for optimizers like stable_adamw that may have additional
    state keys (e.g., 'mean_square') that are not saved by the standard
    HuggingFace Trainer checkpointing.
    """
    
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(
                            checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        checkpoint_file_exists = (
            glob.glob(os.path.join(
                checkpoint, f"rank*-of-{self.args.world_size}-{OPTIMIZER_NAME}"))
            if self.is_fsdp_xla_v1_enabled
            else checkpoint_file_exists
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_xla_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                if self.is_fsdp_xla_v1_enabled:
                    optimizer_state = torch.load(
                        os.path.join(
                            checkpoint, f"rank{self.args.process_index}-of-{self.args.world_size}-{OPTIMIZER_NAME}"
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    # We only need `optimizer` when resuming from checkpoint
                    optimizer_state = optimizer_state["optimizer"]
                else:
                    optimizer_state = torch.load(
                        os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu", weights_only=True
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(
                        os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)
                import torch_xla.core.xla_model as xm
                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(
                    lr_scheduler_state, self.args.device)

                # Try loading optimizer state - handle missing keys (e.g., 'mean_square' for stable_adamw)
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                except (KeyError, RuntimeError) as e:
                    logger.warning(
                        f"Error loading optimizer state (likely missing state keys like "
                        f"'mean_square' for stable_adamw): {type(e).__name__}: {e}. "
                        "Starting with fresh optimizer state."
                    )
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    import smdistributed.modelparallel.torch as smp
                    def opt_load_hook(mod, opt):
                        try:
                            opt.load_state_dict(smp.load(os.path.join(
                                checkpoint, OPTIMIZER_NAME), partial=True))
                        except (KeyError, RuntimeError) as e:
                            logger.warning(
                                f"Error loading optimizer state (likely missing state keys like "
                                f"'mean_square' for stable_adamw): {type(e).__name__}: {e}. "
                                "Starting with fresh optimizer state."
                            )

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        try:
                            load_fsdp_optimizer(
                                self.accelerator.state.fsdp_plugin,
                                self.accelerator,
                                self.optimizer,
                                self.model,
                                checkpoint,
                                **_get_fsdp_ckpt_kwargs(),
                            )
                        except (KeyError, RuntimeError) as e:
                            logger.warning(
                                f"Error loading FSDP optimizer state (likely missing state keys like "
                                f"'mean_square' for stable_adamw): {type(e).__name__}: {e}. "
                                "Starting with fresh optimizer state."
                            )
                    else:
                        # Try loading optimizer state - handle missing keys (e.g., 'mean_square' for stable_adamw)
                        try:
                            saved_state = torch.load(
                                os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True
                            )
                            
                            # Get the actual optimizer (unwrap AcceleratedOptimizer if needed)
                            actual_optimizer = self.optimizer
                            if hasattr(self.optimizer, 'optimizer'):
                                actual_optimizer = self.optimizer.optimizer
                            
                            # First, try to load the state to see what happens
                            # If it fails, we'll catch the error and fix missing keys
                            try:
                                actual_optimizer.load_state_dict(saved_state)
                                logger.info("Successfully loaded optimizer state without missing keys.")
                            except (KeyError, RuntimeError) as load_error:
                                # If loading fails, we need to merge states manually
                                logger.info(
                                    f"Optimizer state has missing keys. Merging with current state: {load_error}"
                                )
                                
                                # Get current optimizer state to see what keys are expected
                                current_state = actual_optimizer.state_dict()
                                
                                # Check for missing keys and initialize them
                                missing_keys = []
                                saved_state_dict = saved_state.get('state', {})
                                current_state_dict = current_state.get('state', {})
                                
                                # Ensure 'state' key exists in saved_state
                                if 'state' not in saved_state:
                                    saved_state['state'] = {}
                                
                                # Iterate through all parameters in saved state
                                for param_id_str in saved_state_dict.keys():
                                    saved_param_state = saved_state_dict.get(param_id_str, {})
                                    
                                    # If this param exists in current state, check for missing keys
                                    if param_id_str in current_state_dict:
                                        current_param_state = current_state_dict[param_id_str]
                                        
                                        # Check for missing keys in this parameter's state
                                        for key in current_param_state.keys():
                                            if key not in saved_param_state:
                                                missing_keys.append((param_id_str, key))
                                                
                                                # Initialize missing key with appropriate dtype
                                                if key == 'mean_square':
                                                    # mean_square must be float32 for stable_adamw Triton kernel
                                                    # Get device from existing state
                                                    ref_tensor = None
                                                    if 'exp_avg_sq' in saved_param_state:
                                                        ref_tensor = saved_param_state['exp_avg_sq']
                                                    elif 'exp_avg' in saved_param_state:
                                                        ref_tensor = saved_param_state['exp_avg']
                                                    
                                                    if ref_tensor is not None:
                                                        # mean_square is a scalar (0D tensor) for stable_adamw
                                                        saved_state['state'][param_id_str][key] = torch.tensor(
                                                            0.0, dtype=torch.float32, device=ref_tensor.device
                                                        )
                                                    else:
                                                        # Fallback: create scalar tensor on CPU, will be moved by optimizer
                                                        saved_state['state'][param_id_str][key] = torch.tensor(
                                                            0.0, dtype=torch.float32
                                                        )
                                                    logger.info(
                                                        f"Initializing missing 'mean_square' state for param {param_id_str} "
                                                        f"in float32 (required for stable_adamw Triton kernel)"
                                                    )
                                                else:
                                                    # For other missing keys, use the current state's value
                                                    saved_state['state'][param_id_str][key] = current_param_state[key].clone()
                                                    logger.info(
                                                        f"Initializing missing '{key}' state for param {param_id_str}"
                                                    )
                                
                                if missing_keys:
                                    logger.info(
                                        f"Found {len(missing_keys)} missing state keys. Initialized them and loading merged state."
                                    )
                                
                                # Now try loading the merged state
                                actual_optimizer.load_state_dict(saved_state)
                                
                                # After loading, ensure mean_square is still float32 (in case optimizer changed it)
                                for param_id_str, param_state in actual_optimizer.state.items():
                                    if 'mean_square' in param_state:
                                        if param_state['mean_square'].dtype != torch.float32:
                                            logger.warning(
                                                f"Converting 'mean_square' to float32 for param {param_id_str} "
                                                f"(was {param_state['mean_square'].dtype})"
                                            )
                                            param_state['mean_square'] = param_state['mean_square'].to(dtype=torch.float32)
                            
                        except (KeyError, RuntimeError) as e:
                            logger.warning(
                                f"Error loading optimizer state (likely missing state keys like "
                                f"'mean_square' for stable_adamw): {type(e).__name__}: {e}. "
                                "Starting with fresh optimizer state."
                            )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(
                            checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
                

def reissue_pt_warnings(caught_warnings):
    # Reissue warnings
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category is not UserWarning:
                warnings.warn(w.message, w.category)
                

def _get_fsdp_ckpt_kwargs():
    if "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}
