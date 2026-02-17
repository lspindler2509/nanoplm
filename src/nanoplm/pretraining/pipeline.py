import os
import time
import shutil
import torch
import torch.distributed as dist
import wandb
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union
from pathlib import Path

from safetensors.torch import load_file
from transformers import (
    TrainingArguments,
)
from nanoplm.pretraining.trainer import PretrainingTrainer

from nanoplm.pretraining.models.modern_bert import ProtModernBertMLM
from nanoplm.pretraining.dataset import ShardedDataset
from nanoplm.pretraining.collator import ProtDataCollatorForLM
from dion import Muon as DionMuon, NorMuon as DionNorMuon
from nanoplm.pretraining.optim import build_optimizer
from nanoplm.pretraining.utils import (
    compute_batch_setup,
    get_num_workers,
    prepare_run_and_steps,
)
from nanoplm.data.validation import validate_pretrain_dataset
from nanoplm.pretraining.callbacks import ParameterLoggingCallback
from nanoplm.utils.logger import logger
from nanoplm.utils.common import get_device, create_dirs, resolve_world_size


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _is_embedding_or_unembedding_param(name: str) -> bool:
    lname = name.lower()

    # HF ModernBERT naming:
    # - token embedding matrix: model.embeddings.tok_embeddings.weight
    # - MLM output head: decoder.weight / decoder.bias
    #   (decoder.weight is tied to token embeddings by default and may not appear
    #   as a distinct named parameter).
    if "embeddings.tok_embeddings" in lname:
        return True
    if lname.endswith("decoder.weight") or lname.endswith("decoder.bias"):
        return True

    # Fallbacks for other architectures.
    return (
        "embedding" in lname
        or "lm_head" in lname
        or "unembedding" in lname
    )


def _build_muon_optimizer(
    model: torch.nn.Module,
    pretrain_config: "PretrainingConfig",
):
    raw_model = _unwrap_model(model)

    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    seen: set[int] = set()

    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        if param.ndim == 1:
            adamw_params.append(param)
            continue
        if _is_embedding_or_unembedding_param(name):
            adamw_params.append(param)
            continue
        if param.ndim == 2:
            muon_params.append(param)
            continue

        # Muon is intended for hidden-layer matrices; route everything else to AdamW.
        adamw_params.append(param)

    if not muon_params:
        raise ValueError(
            "No eligible matrix parameters found for Muon (expected 2D hidden-layer weights)."
        )

    logger.info(
        "Muon grouping: "
        f"muon_params={len(muon_params)} tensors, "
        f"adamw_params={len(adamw_params)} tensors"
    )

    return build_optimizer(
        muon_params=muon_params,
        adamw_params=adamw_params,
        muon_learning_rate=pretrain_config.muon_learning_rate,
        muon_weight_decay=pretrain_config.muon_weight_decay,
        muon_cautious_weight_decay=pretrain_config.muon_cautious_weight_decay,
        muon_use_polar_express=pretrain_config.muon_use_polar_express,
        muon_momentum=pretrain_config.muon_momentum,
        muon_nesterov=pretrain_config.muon_nesterov,
        muon_eps=pretrain_config.muon_eps,
        use_normuon=str(pretrain_config.optimizer).lower() == "normuon",
        adamw_learning_rate=pretrain_config.adam_learning_rate,
        adamw_weight_decay=pretrain_config.adam_weight_decay,
        adamw_betas=(pretrain_config.adam_beta1, pretrain_config.adam_beta2),
        adamw_epsilon=pretrain_config.adam_epsilon,
    )


class TokenTrackingTrainer(Trainer):
    """Trainer subclass that injects tokens/sec into wandb logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step_tok_count = 0
        self._step_raw_tok_count = 0
        self._step_t0 = time.perf_counter()
        self._last_tokens_per_sec = 0.0
        self._last_raw_tokens_per_sec = 0.0
        self.model_accepts_loss_kwargs = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        if "attention_mask" in inputs:
            self._step_tok_count += int(inputs["attention_mask"].sum().item())
            self._step_raw_tok_count += int(inputs["attention_mask"].numel())
        elif "input_ids" in inputs:
            self._step_raw_tok_count += int(inputs["input_ids"].numel())

        loss = super().training_step(model, inputs, num_items_in_batch)

        t1 = time.perf_counter()
        elapsed = t1 - self._step_t0
        tok_count = float(self._step_tok_count)
        raw_tok_count = float(self._step_raw_tok_count)
        tok_elapsed = float(elapsed)
        if dist.is_available() and dist.is_initialized():
            if "attention_mask" in inputs:
                device = inputs["attention_mask"].device
            elif "input_ids" in inputs:
                device = inputs["input_ids"].device
            else:
                device = loss.device
            tok_tensor = torch.tensor(tok_count, device=device)
            raw_tok_tensor = torch.tensor(raw_tok_count, device=device)
            time_tensor = torch.tensor(tok_elapsed, device=device)
            dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(raw_tok_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            tok_count = tok_tensor.item()
            raw_tok_count = raw_tok_tensor.item()
            tok_elapsed = time_tensor.item()
        self._last_tokens_per_sec = tok_count / max(tok_elapsed, 1e-9)
        self._last_raw_tokens_per_sec = raw_tok_count / max(tok_elapsed, 1e-9)

        self._step_tok_count = 0
        self._step_raw_tok_count = 0
        self._step_t0 = t1
        return loss

    def log(self, logs, start_time=None, **kwargs):
        if logs is None:
            logs = {}

        optimizer = self.optimizer
        seen: set[int] = set()
        while optimizer is not None and not isinstance(optimizer, (DionMuon, DionNorMuon)):
            opt_id = id(optimizer)
            if opt_id in seen:
                break
            seen.add(opt_id)
            inner = getattr(optimizer, "optimizer", None)
            if inner is None or inner is optimizer:
                break
            optimizer = inner

        if isinstance(optimizer, (DionMuon, DionNorMuon)):
            # param_groups[0] = muon, param_groups[1] = adamw
            muon_lr = optimizer.param_groups[0]["lr"]
            adamw_lr = optimizer.param_groups[1]["lr"]
            logs["learning_rate"] = adamw_lr
            logs["adamw_lr"] = adamw_lr
            logs["muon_lr"] = muon_lr
        logs["tokens_per_sec"] = self._last_tokens_per_sec
        logs["raw_tokens_per_sec"] = self._last_raw_tokens_per_sec
        super().log(logs, start_time=start_time, **kwargs)


@dataclass
class PretrainingConfig:
    # Dataset directory (contains .data_manifest from nanoplm data from-yaml)
    dataset_dir: Union[str, Path]

    # Checkpoint and output
    ckp_dir: str = "output/pretraining"

    # Training hyperparameters
    micro_batch_size: int = 32
    num_epochs: int = 10
    warmup_ratio: float = 0.05
    warmup_steps: Optional[int] = None  # If set, takes priority over warmup_ratio
    lr_scheduler_type: str = "linear"  # Options: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adam_learning_rate: float = 1e-3
    adam_weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    # Muon-specific hyperparameters (used only when optimizer == "muon" or "normuon").
    # adam_* fields are used for the AdamW sub-optimizer.
    muon_learning_rate: float = 2e-2
    muon_weight_decay: float = 0.1
    muon_cautious_weight_decay: bool = True
    muon_use_polar_express: bool = False
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_eps: float = 1e-7
    # Target effective batch size in tokens per optimizer step.
    # gradient_accumulation_steps is inferred from this value at runtime.
    global_batch_size: int = 2 ** 20

    # Mixed precision
    bf16: bool = True
    tf32: bool = True

    # MLM settings
    mlm_probability: float = 0.3
    mask_replace_prob: float = 0.8
    random_token_prob: float = 0.1
    keep_probability: float = 0.1

    # Logging/checkpointing
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    seed: int = 42

    # Data loading
    num_workers: Union[int, str] = "auto"
    prefetch_factor: int = 2

    # Sequence packing (packs multiple sequences per row to eliminate padding waste).
    # Requires flash attention (varlen path).  Falls back to padding if disabled.
    use_packing: bool = False
    # When set, enables static-shape compilation (dynamic=False).
    # The collator pre-flattens packed batches to a fixed size
    # (target_packed_rows × max_seq_len) so torch.compile sees no shape
    # changes.  Set to ceil(micro_batch_size × avg_len / max_seq_len) + margin.
    # If unset (None), uses dynamic=True compilation.
    target_packed_rows: Optional[int] = None

    # Distributed training
    multi_gpu: bool = False
    world_size: Union[int, str] = 1
    project_name: str = "nanoplm-pretraining"
    bf16: bool = False
    save_safetensors:  bool = True


@dataclass
class ResumeConfig:
    is_resume: bool
    checkpoint_dir: str
    extra_epochs: Optional[int] = None
    normal_resume: Optional[bool] = True


def _archive_future_checkpoints(run_dir: Path, resume_step: int) -> None:
    """Archive checkpoints with steps greater than resume_step.

    When resuming from a checkpoint, any checkpoints with higher step numbers
    are moved to an archived subdirectory to prevent conflicts while preserving
    the data for potential future analysis.

    Args:
        run_dir: The run directory containing checkpoints
        resume_step: The step number being resumed from
    """
    checkpoints_to_archive = []

    for ckpt_dir in run_dir.glob("checkpoint-*"):
        try:
            step = int(ckpt_dir.name.split("-")[1])
            if step > resume_step:
                checkpoints_to_archive.append((step, ckpt_dir))
        except (IndexError, ValueError):
            continue

    if checkpoints_to_archive:
        checkpoints_to_archive.sort()

        # Create archive directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = run_dir / f"archived_{timestamp}"
        archive_dir.mkdir(exist_ok=True)

        logger.warning(
            f"Found {len(checkpoints_to_archive)} checkpoint(s) with steps > {resume_step}. "
            f"Moving to archive: {[s for s, _ in checkpoints_to_archive]}"
        )

        for step, ckpt_path in checkpoints_to_archive:
            dest = archive_dir / ckpt_path.name
            logger.info(f"Archiving checkpoint-{step} to {archive_dir.name}/")
            shutil.move(str(ckpt_path), str(dest))

        logger.info(f"Archived checkpoints moved to: {archive_dir}")




def run_pretraining(
    model: ProtModernBertMLM,
    pretrain_config: PretrainingConfig,
    resume_config: Optional[ResumeConfig] = None,
) -> None:

    device = get_device()

    tokenizer = model.tokenizer
    model.to(device)
    model = model.to(torch.bfloat16) if pretrain_config.bf16 else model

    # Validate dataset: manifest + shard files
    dataset_dir = Path(pretrain_config.dataset_dir)
    validation_result = validate_pretrain_dataset(dataset_dir)
    manifest = validation_result['manifest']

    # Get data from typed manifest
    train_shard_dir = dataset_dir / manifest.train_dir
    val_shard_dir = dataset_dir / manifest.val_dir
    train_sequences = manifest.train_sequences
    val_sequences = manifest.val_sequences

    # Load pre-tokenized binary shards
    logger.info("Using ShardedDataset for pre-tokenized binary shards")

    try:
        train_ds = ShardedDataset(data_dir=str(train_shard_dir))
        val_ds = ShardedDataset(data_dir=str(val_shard_dir))
    except FileNotFoundError as e:
        logger.error(
            f"Binary shards not found! You need to create them first.\n"
            f"Run: nanoplm data from-yaml with pipeline_mode: 'pretrain'\n"
            f"Error: {e}"
        )
        raise

    collator = ProtDataCollatorForLM(
        tokenizer=tokenizer,
        mlm_probability=pretrain_config.mlm_probability,
        mask_token_probability=pretrain_config.mask_replace_prob,
        random_token_probability=pretrain_config.random_token_prob,
        keep_probability=pretrain_config.keep_probability,
    )

    create_dirs(pretrain_config.ckp_dir)

    effective_world_size = resolve_world_size(pretrain_config.multi_gpu, pretrain_config.world_size)

    batch = compute_batch_setup(pretrain_config, manifest.max_seq_len, effective_world_size)

    inferred_grad_accum_steps = batch.grad_accum_steps
    global_batch_size_samples = batch.global_batch_size_samples

    # Prepare run info and step intervals in a single place
    (
        run_name,
        wandb_run_name,
        output_dir,
        num_epochs,
        logging_steps,
        eval_steps,
        save_steps,
        resume_step,
    ) = prepare_run_and_steps(
        pretrain_config=pretrain_config,
        resume_config=resume_config,
        train_samples=train_sequences,
        global_batch_size_samples=global_batch_size_samples,
    )

    # Configure Weights & Biases via environment variables so HF Trainer attaches correctly
    os.environ["WANDB_PROJECT"] = pretrain_config.project_name
    os.environ["WANDB_NAME"] = wandb_run_name

    num_workers = get_num_workers(
        pretrain_config.num_workers, effective_world_size)

    # Determine warmup: warmup_steps takes priority over warmup_ratio
    warmup_config = {}
    if pretrain_config.warmup_steps is not None:
        warmup_config["warmup_steps"] = pretrain_config.warmup_steps
    else:
        warmup_config["warmup_ratio"] = pretrain_config.warmup_ratio

    # Determine warmup: warmup_steps takes priority over warmup_ratio
    warmup_config = {}
    if pretrain_config.warmup_steps is not None:
        warmup_config["warmup_steps"] = pretrain_config.warmup_steps
    else:
        warmup_config["warmup_ratio"] = pretrain_config.warmup_ratio

    training_dict = {
        "output_dir": output_dir,
        "per_device_train_batch_size": pretrain_config.micro_batch_size,
        "per_device_eval_batch_size": pretrain_config.micro_batch_size,
        "gradient_accumulation_steps": inferred_grad_accum_steps,
        "num_train_epochs": num_epochs,
        "learning_rate": pretrain_config.adam_learning_rate,
        "weight_decay": pretrain_config.adam_weight_decay,
        "max_grad_norm": pretrain_config.max_grad_norm,
        "lr_scheduler_type": pretrain_config.lr_scheduler_type,
        "logging_strategy": "steps",
        **warmup_config,  # Add warmup_steps or warmup_ratio
        "logging_steps": logging_steps,
        "logging_dir": Path(output_dir) / "logs",
        "eval_strategy": "steps",
        "eval_steps": eval_steps,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "seed": pretrain_config.seed,
        "bf16": pretrain_config.bf16 and device == "cuda" and torch.cuda.is_bf16_supported(),
        "fp16": pretrain_config.bf16 and ((device == "cuda" and not torch.cuda.is_bf16_supported()) or device == "mps"),
        "tf32": pretrain_config.tf32 and device == "cuda",
        "report_to": "wandb",
        "run_name": wandb_run_name,
        "include_num_input_tokens_seen": "non_padding",  # all, no
        "dataloader_pin_memory": True if device == "cuda" else False,
        "dataloader_num_workers": num_workers,
        "dataloader_persistent_workers": False,
        "bf16": pretrain_config.bf16,
        "save_safetensors": pretrain_config.save_safetensors,
    }

    if num_workers > 0:
        training_dict["dataloader_prefetch_factor"] = pretrain_config.prefetch_factor
        training_dict["dataloader_persistent_workers"] = True

    # Configure optimizer through TrainingArguments
    optimizer_name = pretrain_config.optimizer.lower()
    custom_optimizer = None
    if optimizer_name == "adamw":
        training_dict["optim"] = "adamw_torch"
    elif optimizer_name == "stable_adamw":
        training_dict["optim"] = "stable_adamw"
    elif optimizer_name in {"muon", "normuon"}:
        custom_optimizer = _build_muon_optimizer(model, pretrain_config)
    else:
        raise ValueError(
            f"Invalid optimizer: {pretrain_config.optimizer}. "
            f"Currently supported: [adamw, stable_adamw, muon, normuon]"
        )

    if pretrain_config.multi_gpu:
        training_dict["ddp_backend"] = "nccl" if torch.cuda.is_available() else "gloo"
        training_dict["ddp_find_unused_parameters"] = True

    args = TrainingArguments(**training_dict)
    if resume_config and resume_config.is_resume:
        # print(f"🔄 Resuming training from checkpoint: {resume_config.checkpoint_dir}")
        # state = torch.load(str(resume_config.checkpoint_dir)+"/pytorch_model.bin", map_location="cpu")
        # model.load_state_dict(state, strict=False)
        ckpt_dir = Path(resume_config.checkpoint_dir)
        ckpt_path = ckpt_dir / "model.safetensors"

        logger.info(f"🔄 Resuming training from checkpoint: {ckpt_path}")

        # If the user requested a normal Trainer resume (preserve optimizer/scheduler etc.),
        # we do not manually load weights here — the Trainer will handle restoring state when
        # invoked with resume_from_checkpoint. Otherwise, keep the previous behavior of
        # loading model weights from safetensors into the model before training.
        normal = getattr(resume_config, "normal_resume", True)
        logger.info(f"Resume mode: {'normal' if normal else 'weights-only'}")
        if not normal:
            state = load_file(ckpt_path)
            model.load_state_dict(state, strict=False)

    # Find triangular attention layer parameters to log
    param_names_to_log = []
    for name, param in model.named_parameters():
        # Log key parameters from triangular attention layers:
        # - p_out.weight: Output projection (controls update size, starts at 0)
        # - proj_o.weight: Attention output projection (controls update size, starts at 0)
        # - g_in.weight, g_out.weight: Gating parameters (controls gating strength, starts at 0)
        #   Now we have separate parameters for incoming and outgoing: tri_mult_incoming.* and tri_mult_outgoing.*
        # - proj_z: Pair-to-bias projection (controls Pair→Residue influence)
        if any(key in name for key in [
            'tri_mult_incoming.p_out.weight', 'tri_mult_incoming.g_in.weight', 'tri_mult_incoming.g_out.weight',
            'tri_mult_outgoing.p_out.weight', 'tri_mult_outgoing.g_in.weight', 'tri_mult_outgoing.g_out.weight',
            'proj_o.weight', 'proj_z.1.weight'
        ]):
            param_names_to_log.append(name)
    
    # Create callbacks
    callbacks = []
    multi_step_callback = None  # Initialize to None
    
    # Recycling callbacks (only if recycling is enabled)
    use_recycling = getattr(model, 'recycling', False)
    if use_recycling:
        from nanoplm.pretraining.callbacks import MultiStepRecyclingEvalCallback, RecyclingMetricsCallback
        multi_step_callback = MultiStepRecyclingEvalCallback()
        callbacks.append(multi_step_callback)
        callbacks.append(RecyclingMetricsCallback())
        logger.info("Recycling enabled: Multi-step evaluation and recycling metrics will be logged")
    
    # Data2Vec callbacks (if Data2Vec is enabled)
    # Structure: ProtModernBertMLM -> bert_model (ModernBertForMaskedLMWithRecycling) -> model (ModernBertModelWithRecycling)
    use_data2vec = False
    if hasattr(model, 'bert_model') and hasattr(model.bert_model, 'model'):
        if hasattr(model.bert_model.model, 'use_data2vec'):
            use_data2vec = model.bert_model.model.use_data2vec
    
    if use_data2vec:
        from nanoplm.pretraining.callbacks import Data2VecUpdateCallback, Data2VecLossLoggingCallback
        callbacks.append(Data2VecUpdateCallback())
        callbacks.append(Data2VecLossLoggingCallback())
        logger.info("Data2Vec enabled: EMA teacher will be updated during training, separate losses will be logged")
    
    # Parameter logging callback
    if param_names_to_log:
        logger.info(f"Will log {len(param_names_to_log)} parameters to WandB: {param_names_to_log[:3]}...")
        param_callback = ParameterLoggingCallback(
            parameter_names=param_names_to_log
        )
        callbacks.append(param_callback)

    trainer = TokenTrackingTrainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks if callbacks else None,
        # When provided, custom optimizer/scheduler override TrainingArguments.optim.
        optimizers=(custom_optimizer, None),
    )
    
    # Assign trainer to callback after creation (HuggingFace doesn't pass it in kwargs)
    if multi_step_callback is not None:
        multi_step_callback.trainer = trainer

    logger.info("Starting Trainer")

    # Start training and capture W&B run ID immediately after trainer initialization
    try:
        if resume_config and getattr(resume_config, "normal_resume", True):
            logger.info(f"Resuming training (normal) from checkpoint: {resume_config.checkpoint_dir}")
            trainer.train(resume_from_checkpoint=str(resume_config.checkpoint_dir))
        else:
            if resume_config:
                logger.info(
                    f"Resuming training from checkpoint: {resume_config.checkpoint_dir}"
                )
            trainer.train()

        # Add W&B metadata for resume tracking
        if resume_config and resume_config.is_resume and wandb.run is not None:
            try:
                if resume_step is not None:
                    wandb.config.update(
                        {
                            "resumed_from_step": resume_step,
                            "resume_timestamp": datetime.now().isoformat(),
                        },
                        allow_val_change=True,
                    )
                    # Add tag to mark this as a resumed run
                    current_tags = list(wandb.run.tags) if wandb.run.tags else []
                    if f"resumed-from-{resume_step}" not in current_tags:
                        wandb.run.tags = current_tags + [f"resumed-from-{resume_step}"]
                    logger.info(
                        f"Added W&B metadata: resumed from step {resume_step}"
                    )
            except Exception as e:
                logger.warning(f"Failed to add W&B resume metadata: {e}")

        # Capture and save W&B run ID for future resumes (if W&B is active)
        if wandb.run is not None:
            actual_run_id = wandb.run.id
            run_id_path = Path(output_dir) / "wandb_run_id.txt"
            if (
                not run_id_path.exists()
                or run_id_path.read_text().strip() != actual_run_id
            ):
                run_id_path.write_text(actual_run_id, encoding="utf-8")
                logger.info(f"Saved W&B run ID: {actual_run_id}")
    except Exception as e:
        logger.warning(f"Error during training or saving W&B run ID: {e}")
        raise

    logger.info("Saving final model and tokenizer")
    trainer.save_model(output_dir)
    trainer.save_state()
