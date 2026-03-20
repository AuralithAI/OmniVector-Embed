"""Hard negative mining using FAISS and positive-aware thresholding."""

import logging
import os
from typing import Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class HardNegativeMiner:
    """Mine hard negatives using FAISS with positive-aware thresholding.

    Implements the NV-Embed-v2 approach: select negatives with scores
    below a threshold relative to the positive score.
    """

    def __init__(
        self,
        corpus_embeddings: np.ndarray,
        corpus_ids: list[int],
        threshold_ratio: float = 0.95,
        num_negatives: int = 7,
        num_threads: Optional[int] = None,
    ):
        """Initialize FAISS index and parameters.

        Args:
            corpus_embeddings: Embedding matrix of shape [num_corpus, embedding_dim].
            corpus_ids: List of corpus IDs corresponding to embeddings.
            threshold_ratio: Select negatives with score < positive_score * threshold_ratio.
            num_negatives: Number of hard negatives to return per query.
            num_threads: Number of CPU threads for FAISS search. Defaults to
                all available cores (``os.cpu_count()``).

        Raises:
            ValueError: If embeddings dimensions mismatch or inputs are invalid.
        """
        if len(corpus_embeddings) != len(corpus_ids):
            raise ValueError(
                f"corpus_embeddings ({len(corpus_embeddings)}) and corpus_ids "
                f"({len(corpus_ids)}) must have same length"
            )
        if num_negatives < 1:
            raise ValueError(f"num_negatives must be >= 1, got {num_negatives}")
        if not (0 < threshold_ratio < 1):
            raise ValueError(f"threshold_ratio must be in (0, 1), got {threshold_ratio}")

        self.corpus_embeddings = corpus_embeddings.astype(np.float32)
        self.corpus_ids = corpus_ids
        self.threshold_ratio = threshold_ratio
        self.num_negatives = num_negatives
        self.embedding_dim = corpus_embeddings.shape[1]

        # Set FAISS thread count for parallel search
        n_threads = num_threads or os.cpu_count() or 1
        faiss.omp_set_num_threads(n_threads)

        self.index = self._build_index()
        logger.info(
            f"Initialized HardNegativeMiner with {len(corpus_embeddings)} corpus items "
            f"({n_threads} threads, dim={self.embedding_dim})"
        )

    def _build_index(self) -> faiss.Index:
        """Build FAISS index for inner product search.

        Uses IVF (inverted file) index for corpora > 50K vectors for
        dramatically faster approximate search.  Falls back to exact
        ``IndexFlatIP`` for small corpora.
        """
        n = len(self.corpus_embeddings)
        dim = self.embedding_dim

        if n > 50_000:
            # IVF: sqrt(n) clusters, probe 10% of them — good speed/recall trade-off
            n_clusters = int(np.sqrt(n))
            n_probe = max(8, n_clusters // 10)

            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            logger.info(
                f"Training IVF index: {n_clusters} clusters, nprobe={n_probe} "
                f"on {n} vectors (dim={dim})..."
            )
            index.train(self.corpus_embeddings)
            index.nprobe = n_probe
            index.add(self.corpus_embeddings)
            logger.info("IVF index built and populated.")
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(self.corpus_embeddings)
            logger.info("Using exact IndexFlatIP (small corpus).")

        return index

    def mine(
        self,
        query_embedding: np.ndarray,
        positive_id: int,
        positive_score: Optional[float] = None,
        return_top_k: int = 200,
    ) -> list[int]:
        """Mine hard negatives for a single query-positive pair.

        Args:
            query_embedding: Query embedding of shape [embedding_dim].
            positive_id: ID of the positive corpus item.
            positive_score: Override score for the positive (for flexibility).
            return_top_k: Retrieve top-k candidates before threshold filtering.

        Returns:
            List of hard negative corpus IDs (up to num_negatives).

        Raises:
            ValueError: If query embedding dimension is invalid.
        """
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension {query_embedding.shape[0]} "
                f"does not match corpus dimension {self.embedding_dim}"
            )

        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, return_top_k)
        distances = distances[0]
        indices = indices[0]

        if positive_score is None:
            positive_score = distances[0]

        threshold = self.threshold_ratio * positive_score
        negatives = []

        for idx, distance in zip(indices, distances):
            corpus_id = self.corpus_ids[idx]
            if corpus_id == positive_id:
                continue
            if distance < threshold and len(negatives) < self.num_negatives:
                negatives.append(corpus_id)

        return negatives

    def mine_batch(
        self,
        query_embeddings: np.ndarray,
        positive_ids: list[int],
        positive_scores: Optional[np.ndarray] = None,
        return_top_k: int = 200,
    ) -> list[list[int]]:
        """Mine hard negatives for a batch of query-positive pairs.

        Args:
            query_embeddings: Embedding matrix of shape [batch_size, embedding_dim].
            positive_ids: List of positive corpus IDs.
            positive_scores: Optional array of positive scores.
            return_top_k: Retrieve top-k candidates per query.

        Returns:
            List of lists, where each inner list contains hard negative IDs.

        Raises:
            ValueError: If batch sizes don't match.
        """
        if len(query_embeddings) != len(positive_ids):
            raise ValueError(
                f"query_embeddings ({len(query_embeddings)}) and positive_ids "
                f"({len(positive_ids)}) must have same length"
            )

        if positive_scores is not None and len(positive_scores) != len(positive_ids):
            raise ValueError(
                f"positive_scores ({len(positive_scores)}) must match "
                f"positive_ids ({len(positive_ids)})"
            )

        import time

        query_embeddings = query_embeddings.astype(np.float32)

        total = len(query_embeddings)
        all_negatives: list[list[int]] = []

        # Process in chunks so we can log progress and avoid long blocking calls
        chunk_size = max(1024, 8192)
        num_chunks = (total + chunk_size - 1) // chunk_size
        start_time = time.time()

        for ci in range(num_chunks):
            s = ci * chunk_size
            e = min(total, s + chunk_size)
            q_chunk = query_embeddings[s:e]

            distances, indices = self.index.search(q_chunk, return_top_k)

            for local_idx, (idx_list, dist_list) in enumerate(zip(indices, distances)):
                i = s + local_idx
                positive_id = positive_ids[i]
                pos_score = positive_scores[i] if positive_scores is not None else dist_list[0]
                threshold = self.threshold_ratio * pos_score
                negatives: list[int] = []

                for idx, distance in zip(idx_list, dist_list):
                    corpus_id = self.corpus_ids[idx]
                    if corpus_id == positive_id:
                        continue
                    if distance < threshold and len(negatives) < self.num_negatives:
                        negatives.append(corpus_id)

                all_negatives.append(negatives)

            # Progress logging
            elapsed = time.time() - start_time
            done = e
            per_item = elapsed / max(1, done)
            remaining = total - done
            eta = remaining * per_item
            logger.info(
                f"HardNegativeMiner progress: chunk {ci + 1}/{num_chunks} "
                f"processed {done}/{total} queries — elapsed={elapsed:.1f}s ETA={eta:.1f}s"
            )

        return all_negatives
