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


def build_samples(args) -> list:
    """Build multimodal training samples from data sources.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of MultimodalSample objects.
    """
    from omnivector.data.multimodal_dataset import Modality, MultimodalSample

    samples = []

    # Load text pairs
    if args.text_data:
        import json

        path = Path(args.text_data)
        if path.exists():
            with open(path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    from omnivector.data.schema import EmbeddingPair

                    pair = EmbeddingPair.from_dict(record)
                    samples.append(MultimodalSample.from_embedding_pair(pair))

            logger.info(f"Loaded {len(samples)} text samples from {path}")

    # Load image-text pairs
    text_count = len(samples)
    if args.image_data:
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=args.image_data,
            image_dir=args.image_dir,
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
    img_count = len(samples)
    if args.video_data:
        from omnivector.data.loaders.multimodal import VideoTextLoader

        loader = VideoTextLoader(
            dataset_path=args.video_data,
            video_dir=args.video_dir,
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
    audio_count = len(samples)
    if args.audio_data:
        import json as _json

        audio_path = Path(args.audio_data)
        if audio_path.exists():
            with open(audio_path) as f:
                for line in f:
                    record = _json.loads(line.strip())
                    audio_file = record.get("audio_path", record.get("audio", ""))
                    caption = record.get("caption", record.get("text", ""))

                    if args.audio_dir and audio_file:
                        audio_file = str(Path(args.audio_dir) / audio_file)

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

    # Resolve audio encoder config
    audio_config = config.get("audio_config", {})
    audio_model_name = audio_config.get("model_name")

    # Load model
    if args.text_checkpoint:
        logger.info(f"Loading from checkpoint: {args.text_checkpoint}")
        model = OmniVectorModel.from_pretrained(
            args.text_checkpoint, audio_encoder=audio_model_name,
        )
    else:
        logger.info("Initializing fresh model")
        model = OmniVectorModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1", audio_encoder=audio_model_name,
        )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    samples = build_samples(args)

    if not samples:
        logger.error("No training samples found. Provide --text-data, --image-data, or --video-data.")
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
        fp16=training_config.get("fp16", False) if not training_config.get("bf16", False) else False,
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        logging_steps=training_config.get("logging_steps", 100),
        save_steps=training_config.get("save_steps", 1000),
        save_total_limit=training_config.get("save_total_limit", 3),
        remove_unused_columns=False,
        dataloader_num_workers=training_config.get("dataloader_num_workers", 4),
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
