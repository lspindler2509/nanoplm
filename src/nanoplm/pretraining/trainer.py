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

logger = logging.getLogger(__name__)


class PretrainingTrainer(Trainer):
    """
    Custom Trainer that handles missing optimizer state keys when resuming.
    
    This is needed for optimizers like stable_adamw that may have additional
    state keys (e.g., 'mean_square') that are not saved by the standard
    HuggingFace Trainer checkpointing.
    
    Also fixes loss normalization: Our model has **kwargs in forward() which
    makes the Trainer think it handles num_items_in_batch normalization, but
    it doesn't. We set model_accepts_loss_kwargs=False so the Trainer handles
    normalization correctly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Force loss normalization in Trainer since our model doesn't handle num_items_in_batch
        # The model has **kwargs in forward(), which makes Trainer think it accepts loss kwargs,
        # but it doesn't actually use num_items_in_batch for normalization.
        self.model_accepts_loss_kwargs = False