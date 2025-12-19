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
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to fix mean_square dtype after first optimizer step."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # After first step, ensure mean_square is float32 for stable_adamw
        # This fixes the issue where mean_square is initialized in bfloat16
        # but Triton kernel requires float32
        if not hasattr(self, '_mean_square_fixed'):
            self._mean_square_fixed = True
            actual_optimizer = self.optimizer
            if hasattr(self.optimizer, 'optimizer'):
                actual_optimizer = self.optimizer.optimizer
            
            if any(group.get("triton", False) for group in actual_optimizer.param_groups):
                fixed = 0
                for group in actual_optimizer.param_groups:
                    if group.get("triton", False):
                        for p in group["params"]:
                            if p in actual_optimizer.state and "mean_square" in actual_optimizer.state[p]:
                                ms = actual_optimizer.state[p]["mean_square"]
                                if ms.dtype != torch.float32:
                                    actual_optimizer.state[p]["mean_square"] = ms.to(torch.float32)
                                    fixed += 1
                if fixed > 0:
                    logger.info(f"Fixed {fixed} 'mean_square' states to float32")
        
        return loss
    
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
        scheduler_file_exists = os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME))
        
        if checkpoint_file_exists and scheduler_file_exists:
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

                # ✅ NOW SAFE to load
                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    import smdistributed.modelparallel.torch as smp
                    def opt_load_hook(mod, opt):
                        optimizer_state = smp.load(os.path.join(
                            checkpoint, OPTIMIZER_NAME), partial=True)
                        try:
                            opt.load_state_dict(optimizer_state)
                        except (KeyError, RuntimeError) as e:
                            if "mean_square" in str(e):
                                logger.warning(
                                    f"Failed to load stable_adamw optimizer state: {e}. "
                                    "Starting with fresh optimizer state."
                                )
                            else:
                                raise

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
                        # Load optimizer state
                        optimizer_state = torch.load(
                            os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True
                        )
                        
                        # Try to load, but if it fails (e.g., mean_square KeyError with stable_adamw), skip it
                        try:
                            self.optimizer.load_state_dict(optimizer_state)
                            logger.info("Successfully loaded optimizer state")
                        except (KeyError, RuntimeError) as e:
                            if "mean_square" in str(e):
                                logger.warning(
                                    f"Failed to load stable_adamw optimizer state: {e}. "
                                    "Starting with fresh optimizer state."
                                )
                            else:
                                raise
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
