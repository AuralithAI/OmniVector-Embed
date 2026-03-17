"""Script to build training datasets with hard negative mining.

Supports text retrieval datasets (MSMARCO, HotpotQA, NQ, BEIR) and
multimodal datasets (LAION, WebVid, CodeSearchNet) for Stage 1/2 training.

Usage:
    python scripts/build_dataset.py --stage 1 --output-dir data/stage1
    python scripts/build_dataset.py --stage 1 --multimodal --target 8000000 --output-dir data/stage1_8M
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TEXT_DATASETS_STAGE1 = [
    "msmarco",
    "hotpotqa",
    "beir/nfcorpus",
    "beir/fiqa",
    "beir/scifact",
    "beir/arguana",
]

TEXT_DATASETS_STAGE2 = TEXT_DATASETS_STAGE1 + [
    "beir/fever",
    "beir/bioasq",
]


def encode_texts(
    texts: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with a teacher model.

    Args:
        texts: List of text strings to encode.
        model_name: HuggingFace model ID for the teacher encoder.
        batch_size: Encoding batch size.

    Returns:
        Numpy array of shape [num_texts, embed_dim].
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    logger.info(f"Encoding {len(texts)} texts with {model_name}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def load_text_datasets(
    dataset_names: list[str],
    max_samples: Optional[int] = None,
) -> list:
    """Load and combine text retrieval datasets.

    Args:
        dataset_names: List of dataset identifiers.
        max_samples: Per-dataset sample limit.

    Returns:
        Combined list of EmbeddingPair objects.
    """
    from omnivector.data.loaders.base import get_loader

    all_pairs = []
    for name in dataset_names:
        try:
            loader = get_loader(name, max_samples=max_samples)
            pairs = loader.load()
            all_pairs.extend(pairs)
            logger.info(f"Loaded {len(pairs)} pairs from {name}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")

    logger.info(f"Total text pairs: {len(all_pairs)}")
    return all_pairs


def load_multimodal_datasets(
    max_image_samples: int = 200_000,
    max_video_samples: int = 50_000,
    max_code_samples: Optional[int] = None,
) -> list:
    """Load multimodal datasets (image-text, video-text, code).

    Args:
        max_image_samples: Max image-text pairs from LAION.
        max_video_samples: Max video-text pairs from WebVid.
        max_code_samples: Max code-docstring pairs from CodeSearchNet.

    Returns:
        List of dicts with query, positive, domain, modality keys.
    """
    from datasets import load_dataset

    multimodal_pairs = []

    # LAION aesthetics (image-text)
    try:
        logger.info(f"Loading LAION aesthetics ({max_image_samples} samples)...")
        laion = load_dataset(
            "laion/relaion2B-en-research-safe",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in laion:
            if count >= max_image_samples:
                break
            text = sample.get("TEXT", sample.get("caption", ""))
            url = sample.get("URL", sample.get("url", ""))
            if text and url:
                multimodal_pairs.append({
                    "query": text,
                    "positive": text,
                    "domain": "image_text",
                    "modality": "image",
                    "image_url": url,
                })
                count += 1
        logger.info(f"Loaded {count} LAION image-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load LAION: {e}")

    # WebVid (video-text)
    try:
        logger.info(f"Loading WebVid ({max_video_samples} samples)...")
        webvid = load_dataset(
            "TempoFunk/webvid-10M",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in webvid:
            if count >= max_video_samples:
                break
            text = sample.get("caption", sample.get("name", ""))
            url = sample.get("contentUrl", sample.get("video", ""))
            if text and url:
                multimodal_pairs.append({
                    "query": text,
                    "positive": text,
                    "domain": "video_text",
                    "modality": "video",
                    "video_url": url,
                })
                count += 1
        logger.info(f"Loaded {count} WebVid video-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load WebVid: {e}")

    # AudioSet (audio-text)
    try:
        logger.info("Loading AudioSet (30k samples)...")
        audioset = load_dataset(
            "agkphysics/AudioSet",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in audioset:
            if count >= 30_000:
                break
            labels = sample.get("human_labels", sample.get("labels", ""))
            if isinstance(labels, list):
                labels = ", ".join(labels)
            video_id = sample.get("video_id", "")
            if labels:
                multimodal_pairs.append({
                    "query": labels,
                    "positive": labels,
                    "domain": "audio_text",
                    "modality": "audio",
                    "audio_id": video_id,
                })
                count += 1
        logger.info(f"Loaded {count} AudioSet audio-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load AudioSet: {e}")

    # CodeSearchNet (code-docstring)
    try:
        logger.info("Loading CodeSearchNet...")
        code_ds = load_dataset("code_search_net", "all", split="train")
        if max_code_samples:
            code_ds = code_ds.select(range(min(max_code_samples, len(code_ds))))
        for sample in code_ds:
            docstring = sample.get("func_documentation_string", "")
            code = sample.get("func_code_string", "")
            lang = sample.get("language", "unknown")
            if docstring and code:
                multimodal_pairs.append({
                    "query": docstring,
                    "positive": code,
                    "domain": f"code_{lang}",
                    "modality": "text",
                })
        logger.info(f"Loaded {len(code_ds)} CodeSearchNet pairs")
    except Exception as e:
        logger.warning(f"Failed to load CodeSearchNet: {e}")

    logger.info(f"Total multimodal pairs: {len(multimodal_pairs)}")
    return multimodal_pairs


def mine_hard_negatives(
    pairs: list,
    teacher_model: str,
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
    batch_size: int = 64,
) -> list:
    """Mine hard negatives for a list of pairs using a teacher model.

    Args:
        pairs: List of EmbeddingPair objects.
        teacher_model: HuggingFace model ID for teacher encoder.
        num_negatives: Number of hard negatives per query.
        threshold_ratio: Score threshold relative to positive score.
        batch_size: Encoding batch size.

    Returns:
        Updated list of pairs with negatives populated.
    """
    from omnivector.training.hard_negative_miner import HardNegativeMiner

    queries = [p.query for p in pairs]
    positives = [p.positive for p in pairs]

    logger.info("Encoding queries with teacher model...")
    query_embeddings = encode_texts(queries, teacher_model, batch_size)

    logger.info("Encoding positives with teacher model...")
    positive_embeddings = encode_texts(positives, teacher_model, batch_size)

    corpus_ids = list(range(len(positives)))
    miner = HardNegativeMiner(
        corpus_embeddings=positive_embeddings,
        corpus_ids=corpus_ids,
        num_negatives=num_negatives,
        threshold_ratio=threshold_ratio,
    )

    logger.info("Mining hard negatives...")
    all_negatives = miner.mine_batch(query_embeddings, corpus_ids)

    for pair, neg_ids in zip(pairs, all_negatives):
        pair.negatives = [positives[nid] for nid in neg_ids if nid < len(positives)]

    logger.info(f"Mined hard negatives for {len(pairs)} pairs")
    return pairs


def save_dataset(
    pairs: list,
    output_dir: Path,
    filename: str = "train.jsonl",
) -> Path:
    """Save dataset in JSONL format for training.

    Args:
        pairs: List of dicts or EmbeddingPair objects.
        output_dir: Directory to save dataset.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            if hasattr(pair, "query"):
                record = {
                    "query": pair.query,
                    "positive": pair.positive,
                    "negatives": getattr(pair, "negatives", []),
                    "domain": getattr(pair, "domain", "retrieval"),
                }
            else:
                record = pair
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(pairs)} pairs to {output_file}")
    return output_file


def build_stage_dataset(
    stage: int,
    output_dir: Path,
    multimodal: bool = False,
    target_size: Optional[int] = None,
    teacher_model: Optional[str] = None,
    max_samples_per_dataset: Optional[int] = None,
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
) -> None:
    """Build complete training dataset for a given stage.

    Args:
        stage: Training stage (1 or 2).
        output_dir: Output directory.
        multimodal: Include image/video/code data.
        target_size: Target total dataset size.
        teacher_model: Teacher model for hard negative mining.
        max_samples_per_dataset: Per-dataset sample limit.
        num_negatives: Hard negatives per query.
        threshold_ratio: Hard negative threshold.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = TEXT_DATASETS_STAGE1 if stage == 1 else TEXT_DATASETS_STAGE2

    # Load text datasets
    logger.info(f"Building Stage {stage} dataset...")
    text_pairs = load_text_datasets(dataset_names, max_samples=max_samples_per_dataset)

    # Mine hard negatives if teacher model provided
    if teacher_model:
        text_pairs = mine_hard_negatives(
            text_pairs,
            teacher_model=teacher_model,
            num_negatives=num_negatives,
            threshold_ratio=threshold_ratio,
        )

    # Save text pairs
    save_dataset(text_pairs, output_dir, "text_pairs.jsonl")

    all_records = []
    for pair in text_pairs:
        all_records.append({
            "query": pair.query,
            "positive": pair.positive,
            "negatives": getattr(pair, "negatives", []),
            "domain": getattr(pair, "domain", "retrieval"),
            "modality": "text",
        })

    # Load multimodal datasets
    if multimodal:
        mm_pairs = load_multimodal_datasets(
            max_image_samples=200_000,
            max_video_samples=50_000,
            max_code_samples=2_000_000 if stage == 1 else 500_000,
        )
        all_records.extend(mm_pairs)
        save_dataset(mm_pairs, output_dir, "multimodal_pairs.jsonl")

    # Subsample to target size if needed
    if target_size and len(all_records) > target_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_records), size=target_size, replace=False)
        all_records = [all_records[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {target_size} records")

    # Shuffle and save combined dataset
    rng = np.random.default_rng(42)
    rng.shuffle(all_records)
    save_dataset(all_records, output_dir, "train.jsonl")

    # Save stats
    modality_counts = {}
    domain_counts = {}
    for r in all_records:
        mod = r.get("modality", "text")
        dom = r.get("domain", "unknown")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    stats = {
        "stage": stage,
        "total_samples": len(all_records),
        "multimodal": multimodal,
        "modality_counts": modality_counts,
        "domain_counts": domain_counts,
    }
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stage {stage} dataset complete: {len(all_records)} total samples")
    logger.info(f"Modality breakdown: {modality_counts}")
    logger.info(f"Stats saved to {stats_file}")


def main():
    """CLI entry point for dataset building."""
    parser = argparse.ArgumentParser(
        description="Build training datasets with hard negative mining"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage (1=retrieval, 2=generalist)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save dataset files",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Include image/video/code multimodal data",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target total dataset size",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model for hard negative mining (e.g., BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Maximum samples per text dataset",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=7,
        help="Number of hard negatives per sample",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.95,
        help="Threshold ratio for hard negative selection",
    )

    args = parser.parse_args()

    build_stage_dataset(
        stage=args.stage,
        output_dir=args.output_dir,
        multimodal=args.multimodal,
        target_size=args.target,
        teacher_model=args.teacher_model,
        max_samples_per_dataset=args.max_samples_per_dataset,
        num_negatives=args.num_negatives,
        threshold_ratio=args.threshold_ratio,
    )


if __name__ == "__main__":
    main()
