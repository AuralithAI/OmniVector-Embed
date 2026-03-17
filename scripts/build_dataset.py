"""Script to build training datasets with hard negative mining."""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dataset_with_hard_negatives(
    dataset_name: str,
    output_dir: Path,
    teacher_model_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
) -> None:
    """Build dataset with hard negatives mined by teacher model.
    
    Args:
        dataset_name: Name of dataset to load (msmarco, hotpotqa, beir/*).
        output_dir: Directory to save dataset files.
        teacher_model_path: Path to teacher model for mining negatives.
        max_samples: Limit dataset size (None = all).
        num_negatives: Number of hard negatives per sample.
        threshold_ratio: Threshold for hard negative selection.
    
    Raises:
        FileNotFoundError: If teacher model not found.
        ValueError: If dataset name is invalid.
    """
    from omnivector.data.loaders.base import get_loader
    from omnivector.training.hard_negative_miner import HardNegativeMiner

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {dataset_name}")
    loader = get_loader(dataset_name, max_samples=max_samples)
    pairs = loader.load()

    logger.info(f"Loaded {len(pairs)} training pairs")

    if teacher_model_path:
        logger.info(f"Loading teacher model from {teacher_model_path}")
        corpus = loader.load_corpus()

        logger.info("Encoding corpus with teacher model...")
        corpus_embeddings = _encode_corpus(teacher_model_path, corpus)

        logger.info("Mining hard negatives...")
        query_embeddings = _encode_queries(
            teacher_model_path, [pair.query for pair in pairs]
        )

        corpus_ids = list(corpus.keys())
        miner = HardNegativeMiner(
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_ids,
            threshold_ratio=threshold_ratio,
            num_negatives=num_negatives,
        )

        hard_negatives_list = miner.mine_batch(query_embeddings, corpus_ids)

        for pair, hard_negs in zip(pairs, hard_negatives_list):
            pair.hard_negatives = hard_negs

        logger.info(f"Mined hard negatives for {len(pairs)} samples")
    else:
        logger.info("Skipping hard negative mining (no teacher model)")

    logger.info(f"Saving dataset to {output_dir}")
    _save_dataset(pairs, output_dir, dataset_name)


def _encode_corpus(
    model_path: str,
    corpus: dict[int, str],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode corpus using teacher model.
    
    Args:
        model_path: Path to teacher model.
        corpus: Mapping of corpus ID to text.
        batch_size: Batch size for encoding.
    
    Returns:
        Array of embeddings with shape [num_corpus, embedding_dim].
    """
    embeddings_list = []
    corpus_texts = [corpus[idx] for idx in sorted(corpus.keys())]

    for i in range(0, len(corpus_texts), batch_size):
        batch_texts = corpus_texts[i : i + batch_size]
        batch_embeddings = _get_embeddings(model_path, batch_texts)
        embeddings_list.append(batch_embeddings)

    return np.vstack(embeddings_list)


def _encode_queries(
    model_path: str,
    queries: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    """Encode queries using teacher model.
    
    Args:
        model_path: Path to teacher model.
        queries: List of query texts.
        batch_size: Batch size for encoding.
    
    Returns:
        Array of embeddings with shape [num_queries, embedding_dim].
    """
    embeddings_list = []

    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i : i + batch_size]
        batch_embeddings = _get_embeddings(model_path, batch_queries)
        embeddings_list.append(batch_embeddings)

    return np.vstack(embeddings_list)


def _get_embeddings(
    model_path: str,
    texts: list[str],
) -> np.ndarray:
    """Get embeddings from teacher model.
    
    This is a placeholder. In practice, use the teacher model
    (e.g., bge-large-en-v1.5) to encode texts.
    
    Args:
        model_path: Path to teacher model.
        texts: List of texts to encode.
    
    Returns:
        Array of embeddings.
    """
    raise NotImplementedError(
        "Integration with teacher model encoder needs to be implemented. "
        "Use SentenceTransformers or similar for this."
    )


def _save_dataset(
    pairs: list,
    output_dir: Path,
    dataset_name: str,
) -> None:
    """Save dataset in JSONL format for training.
    
    Args:
        pairs: List of EmbeddingPair objects.
        output_dir: Directory to save dataset.
        dataset_name: Name of dataset.
    """
    output_file = output_dir / f"{dataset_name}_pairs.jsonl"

    with open(output_file, "w") as f:
        for pair in pairs:
            record = {
                "query": pair.query,
                "positive": pair.positive,
                "hard_negatives": pair.hard_negatives,
                "metadata": pair.metadata,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(pairs)} pairs to {output_file}")

    stats_file = output_dir / f"{dataset_name}_stats.json"
    stats = {
        "dataset": dataset_name,
        "total_pairs": len(pairs),
        "avg_hard_negatives": np.mean(
            [len(pair.hard_negatives) for pair in pairs]
        ),
    }

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved statistics to {stats_file}")


def main():
    """CLI entry point for dataset building."""
    parser = argparse.ArgumentParser(
        description="Build training datasets with hard negative mining"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (msmarco, hotpotqa, beir/nfcorpus, etc)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save dataset files",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Path to teacher model for hard negative mining",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to load",
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

    build_dataset_with_hard_negatives(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        teacher_model_path=args.teacher_model,
        max_samples=args.max_samples,
        num_negatives=args.num_negatives,
        threshold_ratio=args.threshold_ratio,
    )


if __name__ == "__main__":
    main()
