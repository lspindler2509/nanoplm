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
from transformers.utils import is_torch_xla_available, is_sagemaker_mp_enabled, is_accelerate_available, is_deepspeed_available
from transformers.training_args import SCHEDULER_NAME, OPTIMIZER_NAME, OPTIMIZER_NAME_BIN
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

                # Try loading optimizer state with strict=False to handle missing keys
                # (e.g., 'mean_square' for stable_adamw)
                try:
                    self.optimizer.load_state_dict(optimizer_state, strict=False)
                except (KeyError, RuntimeError) as e:
                    logger.warning(
                        f"Error loading optimizer state (likely missing state keys like "
                        f"'mean_square' for stable_adamw): {type(e).__name__}: {e}. "
                        "Starting with fresh optimizer state."
                    )
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                import smdistributed.modelparallel.torch as smp
                if is_sagemaker_mp_enabled():

                    def opt_load_hook(mod, opt):
                        try:
                            opt.load_state_dict(smp.load(os.path.join(
                                checkpoint, OPTIMIZER_NAME), partial=True), strict=False)
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
                        # Try loading optimizer state with strict=False to handle missing keys
                        # (e.g., 'mean_square' for stable_adamw)
                        try:
                            self.optimizer.load_state_dict(
                                torch.load(
                                    os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True
                                ),
                                strict=False
                            )
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
