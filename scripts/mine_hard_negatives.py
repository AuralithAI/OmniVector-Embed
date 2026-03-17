"""Offline hard negative mining script.

Encodes a corpus with a teacher model, then mines hard negatives
for each query-positive pair using FAISS approximate nearest neighbor search.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def encode_texts(texts: list[str], model_name: str, batch_size: int = 64) -> np.ndarray:
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


def mine_negatives(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: list[int],
    positive_ids: list[int],
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
) -> list[list[int]]:
    """Mine hard negatives using FAISS.
    
    Args:
        query_embeddings: Query embeddings [num_queries, dim].
        corpus_embeddings: Corpus embeddings [num_corpus, dim].
        corpus_ids: Corpus document IDs.
        positive_ids: Positive document ID for each query.
        num_negatives: Number of hard negatives per query.
        threshold_ratio: Score threshold relative to positive score.
    
    Returns:
        List of hard negative ID lists, one per query.
    """
    from omnivector.training.hard_negative_miner import HardNegativeMiner

    miner = HardNegativeMiner(
        corpus_embeddings=corpus_embeddings,
        corpus_ids=corpus_ids,
        num_negatives=num_negatives,
        threshold_ratio=threshold_ratio,
    )

    all_negatives = miner.mine_batch(
        queries=query_embeddings,
        positive_ids=positive_ids,
    )

    return all_negatives


def main():
    """Run offline hard negative mining."""
    parser = argparse.ArgumentParser(description="Mine hard negatives for training data")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (msmarco, hotpotqa, beir/nfcorpus)",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Teacher model for encoding",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/hard_negatives",
        help="Output directory for mined negatives",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=7,
        help="Number of hard negatives per query",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.95,
        help="Score threshold ratio relative to positive",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from omnivector.data.loaders import get_loader

    loader = get_loader(args.dataset, max_samples=args.max_samples)
    pairs = loader.load()
    corpus = loader.load_corpus()

    logger.info(f"Loaded {len(pairs)} pairs and {len(corpus)} corpus documents")

    queries = [p.query for p in pairs]
    corpus_texts = list(corpus.values())
    corpus_ids = list(corpus.keys())

    logger.info("Encoding corpus...")
    corpus_embeddings = encode_texts(corpus_texts, args.teacher_model, args.batch_size)

    logger.info("Encoding queries...")
    query_embeddings = encode_texts(queries, args.teacher_model, args.batch_size)

    positive_scores = np.array([
        float(np.dot(query_embeddings[i], corpus_embeddings[0]))
        for i in range(len(queries))
    ], dtype=np.float32)

    logger.info("Mining hard negatives...")
    positive_ids = list(range(len(queries)))
    all_negatives = mine_negatives(
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        corpus_ids=corpus_ids,
        positive_ids=positive_ids,
        num_negatives=args.num_negatives,
        threshold_ratio=args.threshold_ratio,
    )

    output_file = output_dir / f"{args.dataset.replace('/', '_')}_hard_negatives.jsonl"
    with open(output_file, "w") as f:
        for i, pair in enumerate(pairs):
            neg_ids = all_negatives[i] if i < len(all_negatives) else []
            neg_texts = [corpus.get(nid, "") for nid in neg_ids]
            record = {
                "query": pair.query,
                "positive": pair.positive,
                "negatives": neg_texts,
                "negative_ids": neg_ids,
                "domain": pair.domain,
            }
            f.write(json.dumps(record) + "\n")

    logger.info(f"Saved {len(pairs)} samples with hard negatives to {output_file}")

    stats = {
        "dataset": args.dataset,
        "num_pairs": len(pairs),
        "num_corpus": len(corpus),
        "num_negatives_per_query": args.num_negatives,
        "threshold_ratio": args.threshold_ratio,
        "teacher_model": args.teacher_model,
    }
    stats_file = output_dir / f"{args.dataset.replace('/', '_')}_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stats saved to {stats_file}")


if __name__ == "__main__":
    main()
