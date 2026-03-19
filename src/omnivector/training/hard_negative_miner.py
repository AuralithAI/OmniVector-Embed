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

    def _build_index(self) -> faiss.IndexFlatIP:
        """Build FAISS index for inner product search."""
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.corpus_embeddings)
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

        query_embeddings = query_embeddings.astype(np.float32)
        distances, indices = self.index.search(query_embeddings, return_top_k)

        all_negatives = []
        for i, (idx_list, dist_list) in enumerate(zip(indices, distances)):
            positive_id = positive_ids[i]
            pos_score = (
                positive_scores[i]
                if positive_scores is not None
                else dist_list[0]
            )
            threshold = self.threshold_ratio * pos_score
            negatives = []

            for idx, distance in zip(idx_list, dist_list):
                corpus_id = self.corpus_ids[idx]
                if corpus_id == positive_id:
                    continue
                if distance < threshold and len(negatives) < self.num_negatives:
                    negatives.append(corpus_id)

            all_negatives.append(negatives)

        return all_negatives
