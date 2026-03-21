"""End-to-end training script for OmniVector model.

Supports:
- JSON (DeepSpeed) or YAML training configs
- Resume from checkpoint (--resume)
- LoRA fine-tuning
- CPU or GPU training

Usage:
    python scripts/training.py --dataset msmarco --config configs/deepspeed_zero2.json
    python scripts/training.py --config configs/stage2_generalist.yaml --resume checkpoints/stage1/checkpoint-final
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from transformers import TrainingArguments

from omnivector.data.loaders import get_loader
from omnivector.model.omnivector_model import OmniVectorModel
from omnivector.training.callbacks import (
    EarlyStoppingCallback,
    HardNegativeRefreshCallback,
    LoggingCallback,
)
from omnivector.training.trainer import OmniVectorTrainer

logger = logging.getLogger(__name__)


def load_training_arguments(
    config_path: str,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> TrainingArguments:
    """Load TrainingArguments from JSON (DeepSpeed) or YAML config.

    Args:
        config_path: Path to DeepSpeed JSON or YAML training config.
        output_dir: Output directory for checkpoints.
        resume_from_checkpoint: Path to checkpoint to resume from.

    Returns:
        HF TrainingArguments instance.
    """
    config_path_obj = Path(config_path)

    if config_path_obj.suffix in (".yaml", ".yml"):
        return _load_yaml_training_args(config_path, output_dir, resume_from_checkpoint)
    else:
        return _load_json_training_args(config_path, output_dir, resume_from_checkpoint)


def _load_json_training_args(
    config_path: str,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> TrainingArguments:
    """Load TrainingArguments from DeepSpeed JSON config."""
    with open(config_path) as f:
        config = json.load(f)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        learning_rate=config["optimizer"]["params"]["lr"],
        per_device_train_batch_size=config["train_micro_batch_size_per_gpu"],
        per_device_eval_batch_size=config["train_micro_batch_size_per_gpu"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=config["scheduler"]["params"]["warmup_num_steps"],
        max_steps=config.get("max_steps", -1),
        logging_steps=config.get("steps_per_print", 100),
        save_steps=config.get("save_steps", 1000),
        eval_steps=config.get("eval_steps", 1000),
        evaluation_strategy=config.get("evaluation_strategy", "steps"),
        save_strategy=config.get("save_strategy", "steps"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=config["fp16"]["enabled"],
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        deepspeed=config_path if "zero_optimization" in config else None,
        resume_from_checkpoint=resume_from_checkpoint,
    )


def _load_yaml_training_args(
    config_path: str,
    output_dir: str,
    resume_from_checkpoint: Optional[str] = None,
) -> TrainingArguments:
    """Load TrainingArguments from YAML config.

    Reads the 'training_args' section of a YAML file and maps it directly
    to HuggingFace TrainingArguments. Supports all fields from the YAML
    stage configs including bf16, gradient_checkpointing, and resume.

    Args:
        config_path: Path to YAML config file.
        output_dir: Output directory for checkpoints.
        resume_from_checkpoint: CLI override for resume checkpoint path.

    Returns:
        HF TrainingArguments instance.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    training_config = config.get("training_args", {})

    # CLI --resume overrides YAML resume_from_checkpoint
    checkpoint = resume_from_checkpoint or training_config.get("resume_from_checkpoint")

    # Resolve deepspeed config path
    ds_config = config.get("deepspeed")
    if ds_config and Path(ds_config).exists():
        ds_path = ds_config
    else:
        ds_path = None

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 32),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 2e-5),
        warmup_steps=training_config.get("warmup_steps", 500),
        max_steps=training_config.get("max_steps", -1),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 1000),
        eval_steps=training_config.get("eval_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        bf16=training_config.get("bf16", False),
        fp16=training_config.get("fp16", False),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=training_config.get("dataloader_pin_memory", True),
        remove_unused_columns=False,
        optim=training_config.get("optim", "adamw_torch"),
        deepspeed=ds_path,
        resume_from_checkpoint=checkpoint,
    )


def create_training_dataset(
    dataset_name: str,
    split: str = "train",
    max_samples: Optional[int] = None,
    teacher_model: Optional[str] = None,
):
    """Load training dataset.

    Args:
        dataset_name: Dataset identifier (msmarco, hotpotqa, etc.).
        split: Dataset split to load.
        max_samples: Limit dataset size (for testing).
        teacher_model: Teacher model path for hard negative mining.

    Returns:
        List of training examples with hard negatives.
    """
    logger.info(f"Loading dataset: {dataset_name} ({split})")
    loader = get_loader(dataset_name)
    dataset = loader.load(split=split, max_samples=max_samples)

    if teacher_model:
        logger.info(f"Mining hard negatives using teacher: {teacher_model}")
        loader.load_corpus(split="corpus")

    logger.info(f"Loaded {len(dataset)} training examples")
    return dataset


def main():
    """Train OmniVector model."""
    parser = argparse.ArgumentParser(description="Train OmniVector embedding model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="msmarco",
        help="Dataset identifier (msmarco, hotpotqa, beir-nfcorpus)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deepspeed_zero2.json",
        help="Training config file (JSON or YAML)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mistral-community/Mistral-7B-v0.1",
        help="Base model path or HF model ID",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model for hard negative mining",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size (for testing)",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Use LoRA fine-tuning",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Train on CPU (for testing)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from. "
        "Overrides resume_from_checkpoint in YAML config.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by DeepSpeed launcher)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {args.model_path}")
    model = OmniVectorModel.from_pretrained(
        args.model_path,
        lora=args.lora,
        device=device,
    )

    training_args = load_training_arguments(
        args.config,
        str(output_dir),
        resume_from_checkpoint=args.resume,
    )
    logger.info(f"Training arguments: {training_args}")
    if training_args.resume_from_checkpoint:
        logger.info(f"Resuming from: {training_args.resume_from_checkpoint}")

    dataset = create_training_dataset(
        args.dataset,
        split="train",
        max_samples=args.max_samples,
        teacher_model=args.teacher_model,
    )

    callbacks = [
        LoggingCallback(),
        EarlyStoppingCallback(patience=3, min_delta=1e-4),
        HardNegativeRefreshCallback(refresh_steps=5000),
    ]

    trainer = OmniVectorTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    logger.info(f"Training complete. Checkpoints saved to {output_dir}")
    model.save_pretrained(str(output_dir / "final_model"))
    logger.info("Model saved")


if __name__ == "__main__":
    main()
