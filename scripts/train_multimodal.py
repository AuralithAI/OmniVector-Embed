"""Multimodal training script for joint text + vision + audio embedding training.

Usage:
    # Vision-only (Stage 1/2 multimodal):
    python scripts/train_multimodal.py \
        --config configs/multimodal_vision.yaml \
        --text-data data/text_pairs.jsonl \
        --image-data data/image_text_pairs.jsonl \
        --image-dir data/images/ \
        --output-dir checkpoints/multimodal/

    # Full multimodal with audio:
    python scripts/train_multimodal.py \
        --config configs/stage3_multimodal.yaml \
        --text-data data/text_pairs.jsonl \
        --image-data data/image_text_pairs.jsonl \
        --audio-data data/audio_text_pairs.jsonl \
        --audio-dir data/audio/ \
        --resume checkpoints/stage2_55M/checkpoint-final \
        --output-dir checkpoints/stage3/
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multimodal embedding training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multimodal_vision.yaml",
        help="Training configuration YAML file.",
    )
    parser.add_argument(
        "--text-data",
        type=str,
        default=None,
        help="Path to text training data (JSONL).",
    )
    parser.add_argument(
        "--image-data",
        type=str,
        default=None,
        help="Path to image-text training data (JSONL).",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Root directory for images.",
    )
    parser.add_argument(
        "--video-data",
        type=str,
        default=None,
        help="Path to video-text training data (JSONL).",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Root directory for videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/multimodal",
        help="Output directory for checkpoints.",
    )
    parser.add_argument(
        "--text-checkpoint",
        type=str,
        default=None,
        help="Pre-trained text model checkpoint to initialize from.",
    )
    parser.add_argument(
        "--cross-modal-weight",
        type=float,
        default=1.0,
        help="Weight for cross-modal contrastive loss.",
    )
    parser.add_argument(
        "--freeze-vision-steps",
        type=int,
        default=1000,
        help="Steps to freeze vision encoder for text warmup.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit total dataset size.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of video frames to sample.",
    )
    parser.add_argument(
        "--audio-data",
        type=str,
        default=None,
        help="Path to audio-text training data (JSONL).",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Root directory for audio files.",
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
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML training config.

    Args:
        config_path: Path to YAML file.

    Returns:
        Config dictionary.
    """
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def build_samples(args, config: dict | None = None) -> list:
    """Build multimodal training samples from data sources.

    Reads data paths from CLI args first, then falls back to the YAML
    config's ``data_config`` section.  All data paths must point to files
    produced by ``build_dataset.py --stage 3``, which downloads media and
    creates JSONL files with local paths.

    Args:
        args: Parsed command-line arguments.
        config: Parsed YAML config dictionary (optional).

    Returns:
        List of MultimodalSample objects.
    """
    import json as _json

    from omnivector.data.multimodal_dataset import Modality, MultimodalSample
    from omnivector.data.schema import EmbeddingPair

    data_config = (config or {}).get("data_config", {})
    samples = []

    # ── Text pairs ──
    text_data = args.text_data or data_config.get("text_data")
    max_text = data_config.get("max_text_samples")
    if text_data:
        path = Path(text_data)
        if not path.exists():
            logger.error(f"Text data file not found: {path}")
            sys.exit(1)
        import os
        file_size_gb = os.path.getsize(path) / (1024 ** 3)
        cap_msg = f", capped at {max_text:,}" if max_text else ""
        logger.info(
            f"Loading text data from {path} ({file_size_gb:.1f} GB{cap_msg})..."
        )
        count = 0
        with open(path) as f:
            for line in f:
                if max_text and count >= max_text:
                    break
                record = _json.loads(line.strip())
                pair = EmbeddingPair.from_dict(record)
                samples.append(MultimodalSample.from_embedding_pair(pair))
                count += 1
                if count % 1_000_000 == 0:
                    logger.info(f"  Loaded {count:,} text samples...")
        logger.info(f"Loaded {count:,} text samples from {path}")

    # ── Image-text pairs ──
    image_data = args.image_data or data_config.get("image_data")
    image_dir = args.image_dir or data_config.get("image_dir")
    text_count = len(samples)
    if image_data and Path(image_data).exists():
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=image_data,
            image_dir=image_dir,
            format="jsonl",
            max_samples=args.max_samples,
        )
        for pair in loader.load():
            samples.append(
                MultimodalSample.from_image_text(
                    image_path=pair["image_path"],
                    caption=pair["caption"],
                    negatives=pair.get("negative_captions", []),
                )
            )
        logger.info(f"Loaded {len(samples) - text_count} image-text samples")

    # Load video-text pairs
    video_data = args.video_data or data_config.get("video_data")
    video_dir = args.video_dir or data_config.get("video_dir")
    img_count = len(samples)
    if video_data and Path(video_data).exists():
        from omnivector.data.loaders.multimodal import VideoTextLoader

        loader = VideoTextLoader(
            dataset_path=video_data,
            video_dir=video_dir,
            format="jsonl",
            max_samples=args.max_samples,
            num_frames=args.num_frames,
        )
        for pair in loader.load():
            samples.append(
                MultimodalSample.from_video_text(
                    video_path=pair["video_path"],
                    caption=pair["caption"],
                    negatives=pair.get("negative_captions", []),
                )
            )
        logger.info(f"Loaded {len(samples) - img_count} video-text samples")

    # Load audio-text pairs
    audio_data = args.audio_data or data_config.get("audio_data")
    audio_dir = args.audio_dir or data_config.get("audio_dir")
    audio_count = len(samples)
    if audio_data and Path(audio_data).exists():
        audio_path = Path(audio_data)
        with open(audio_path) as f:
            for line in f:
                record = _json.loads(line.strip())
                audio_file = record.get("audio_path", record.get("audio", ""))
                caption = record.get("caption", record.get("text", ""))

                if audio_dir and audio_file:
                    audio_file = str(Path(audio_dir) / audio_file)

                if caption:
                    samples.append(
                        MultimodalSample(
                            query_text=caption,
                            positive_text=caption,
                            negatives=record.get("negative_captions", []),
                            query_instruction="Describe the audio",
                            domain="audio_text",
                            modality=Modality.AUDIO,
                            audio_path=audio_file,
                        )
                    )
        logger.info(f"Loaded {len(samples) - audio_count} audio-text samples")

    if args.max_samples and len(samples) > args.max_samples:
        import random

        random.shuffle(samples)
        samples = samples[: args.max_samples]

    logger.info(f"Total training samples: {len(samples)}")
    return samples


def main() -> None:
    """Run multimodal training."""
    args = parse_args()

    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)

    from transformers import AutoTokenizer, TrainingArguments

    from omnivector.data.multimodal_dataset import MultimodalCollator, MultimodalDataset
    from omnivector.model.omnivector_model import OmniVectorModel
    from omnivector.training.multimodal_loss import MultimodalMRLLoss
    from omnivector.training.multimodal_trainer import MultimodalTrainer

    # Resolve model config
    model_config = config.get("model_config", {})
    use_lora = model_config.get("use_lora", False)
    lora_rank = model_config.get("lora_rank", 16)
    lora_alpha = model_config.get("lora_alpha", 32)

    # Resolve audio encoder config
    audio_config = config.get("audio_config", {})
    audio_model_name = audio_config.get("model_name")

    # Load model
    if args.text_checkpoint:
        logger.info(f"Loading from checkpoint: {args.text_checkpoint} (lora={use_lora})")
        model = OmniVectorModel.from_pretrained(
            args.text_checkpoint,
            audio_encoder=audio_model_name,
            lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
    else:
        logger.info(f"Initializing fresh model (lora={use_lora})")
        model = OmniVectorModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            audio_encoder=audio_model_name,
            lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    samples = build_samples(args, config=config)

    if not samples:
        logger.error(
            "No training samples found. Provide --text-data, --image-data, or --video-data."
        )
        sys.exit(1)

    dataset = MultimodalDataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=config.get("model_config", {}).get("max_length", 512),
        num_frames=args.num_frames,
    )

    collator = MultimodalCollator(
        tokenizer=tokenizer,
        max_negatives=config.get("training_config", {}).get("max_negatives", 7),
    )

    # Loss
    mrl_dims = tuple(config.get("model_config", {}).get("mrl_dims", [512, 1024, 2048, 4096]))
    multimodal_loss = MultimodalMRLLoss(
        mrl_dims=mrl_dims,
        cross_modal_weight=args.cross_modal_weight,
    )

    # Training args
    training_config = config.get("training_args", {})

    # Resolve resume checkpoint: CLI --resume overrides YAML config
    resume_checkpoint = args.resume or training_config.get("resume_from_checkpoint")

    # Resolve DeepSpeed config
    ds_config = config.get("deepspeed")
    if ds_config and Path(ds_config).exists():
        ds_path = ds_config
    else:
        ds_path = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=training_config.get("num_train_epochs", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        learning_rate=training_config.get("learning_rate", 1e-5),
        warmup_steps=training_config.get("warmup_steps", 500),
        max_steps=training_config.get("max_steps", 10000),
        lr_scheduler_type=training_config.get("lr_scheduler_type", "cosine"),
        weight_decay=training_config.get("weight_decay", 0.01),
        bf16=training_config.get("bf16", False),
        fp16=(
            training_config.get("fp16", False) if not training_config.get("bf16", False) else False
        ),
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        remove_unused_columns=False,
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
        deepspeed=ds_path,
        resume_from_checkpoint=resume_checkpoint,
    )

    if resume_checkpoint:
        logger.info(f"Will resume from checkpoint: {resume_checkpoint}")

    # Trainer
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        cross_modal_weight=args.cross_modal_weight,
        freeze_vision_steps=args.freeze_vision_steps,
        multimodal_loss=multimodal_loss,
    )

    logger.info("Starting multimodal training")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    model.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
