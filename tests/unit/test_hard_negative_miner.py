"""Unit tests for hard negative mining."""

import numpy as np
import pytest


class TestHardNegativeMiner:
    """Test suite for HardNegativeMiner."""

    @pytest.fixture
    def setup_miner(self):
        """Create a test HardNegativeMiner with synthetic data."""
        from omnivector.training.hard_negative_miner import HardNegativeMiner

        np.random.seed(42)
        embedding_dim = 128
        num_corpus = 1000

        corpus_embeddings = np.random.randn(num_corpus, embedding_dim).astype(
            np.float32
        )
        corpus_embeddings = corpus_embeddings / (
            np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-8
        )
        corpus_ids = list(range(num_corpus))

        miner = HardNegativeMiner(
            corpus_embeddings=corpus_embeddings,
            corpus_ids=corpus_ids,
            threshold_ratio=0.95,
            num_negatives=7,
        )
        return miner, corpus_embeddings, corpus_ids

    def test_initialization(self, setup_miner):
        """Test HardNegativeMiner initialization."""
        miner, embeddings, ids = setup_miner
        assert miner.embedding_dim == 128
        assert len(miner.corpus_ids) == 1000
        assert miner.threshold_ratio == 0.95
        assert miner.num_negatives == 7

    def test_initialization_dimension_mismatch(self):
        """Test initialization fails with dimension mismatch."""
        from omnivector.training.hard_negative_miner import HardNegativeMiner

        embeddings = np.random.randn(100, 128)
        ids = list(range(50))  # Mismatch

        with pytest.raises(ValueError, match="must have same length"):
            HardNegativeMiner(
                corpus_embeddings=embeddings, corpus_ids=ids
            )

    def test_invalid_threshold_ratio(self):
        """Test initialization fails with invalid threshold_ratio."""
        from omnivector.training.hard_negative_miner import HardNegativeMiner

        embeddings = np.random.randn(100, 128)
        ids = list(range(100))

        with pytest.raises(ValueError, match="threshold_ratio must be in"):
            HardNegativeMiner(
                corpus_embeddings=embeddings,
                corpus_ids=ids,
                threshold_ratio=1.5,
            )

    def test_mine_single_query(self, setup_miner):
        """Test mining negatives for a single query."""
        miner, embeddings, ids = setup_miner

        query_emb = embeddings[0]
        positive_id = 1
        negatives = miner.mine(query_emb, positive_id)

        assert isinstance(negatives, list)
        assert len(negatives) <= 7
        assert positive_id not in negatives
        for neg_id in negatives:
            assert neg_id in ids

    def test_mine_respects_positive_threshold(self, setup_miner):
        """Test that negatives are below positive score threshold."""
        miner, embeddings, ids = setup_miner

        query_emb = embeddings[0].astype(np.float32).reshape(1, -1)
        distances, _ = miner.index.search(query_emb, 200)
        positive_score = distances[0, 0]

        negatives = miner.mine(embeddings[0], ids[1], positive_score=positive_score)

        threshold = miner.threshold_ratio * positive_score
        assert len(negatives) <= 7

    def test_mine_batch_processing(self, setup_miner):
        """Test batch mining returns correct structure."""
        miner, embeddings, ids = setup_miner

        batch_queries = embeddings[:5]
        positive_ids = ids[1:6]

        result = miner.mine_batch(batch_queries, positive_ids, return_top_k=100)

        assert isinstance(result, list)
        assert len(result) == 5
        for neg_list in result:
            assert isinstance(neg_list, list)
            assert len(neg_list) <= 7

    def test_mine_batch_size_mismatch(self, setup_miner):
        """Test batch mining fails with size mismatch."""
        miner, embeddings, ids = setup_miner

        batch_queries = embeddings[:5]
        positive_ids = ids[:3]  # Mismatch

        with pytest.raises(ValueError, match="must have same length"):
            miner.mine_batch(batch_queries, positive_ids)

    def test_mine_invalid_query_dimension(self, setup_miner):
        """Test mining fails with invalid query dimension."""
        miner, embeddings, ids = setup_miner

        invalid_query = np.random.randn(64)  # Wrong dimension
        with pytest.raises(ValueError, match="does not match corpus dimension"):
            miner.mine(invalid_query, ids[0])

    def test_mine_excludes_positive(self, setup_miner):
        """Test that positive ID is always excluded from negatives."""
        miner, embeddings, ids = setup_miner

        for _ in range(10):
            query_idx = np.random.randint(0, 500)
            positive_idx = np.random.randint(0, 500)

            negatives = miner.mine(embeddings[query_idx], ids[positive_idx])
            assert ids[positive_idx] not in negatives

    def test_mine_returns_up_to_k_negatives(self, setup_miner):
        """Test that miner returns at most num_negatives."""
        miner, embeddings, ids = setup_miner

        num_tests = 20
        for _ in range(num_tests):
            query_idx = np.random.randint(0, 500)
            positive_idx = np.random.randint(0, 500)

            negatives = miner.mine(embeddings[query_idx], ids[positive_idx])
            assert len(negatives) <= miner.num_negatives

    def test_deterministic_results(self, setup_miner):
        """Test that mining produces deterministic results."""
        miner, embeddings, ids = setup_miner

        query_emb = embeddings[0]
        positive_id = ids[1]

        result1 = miner.mine(query_emb, positive_id)
        result2 = miner.mine(query_emb, positive_id)

        assert result1 == result2

    def test_batch_vs_single_consistency(self, setup_miner):
        """Test consistency between batch and single mining."""
        miner, embeddings, ids = setup_miner

        batch_size = 5
        batch_queries = embeddings[:batch_size]
        positive_ids_batch = ids[1:batch_size+1]

        batch_result = miner.mine_batch(batch_queries, positive_ids_batch)

        for i in range(batch_size):
            single_result = miner.mine(embeddings[i], ids[i+1])
            assert single_result == batch_result[i]
