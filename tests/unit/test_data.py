"""Unit tests for data schema and utilities."""

import logging

import pytest

logger = logging.getLogger(__name__)


class TestEmbeddingPair:
    """Test suite for EmbeddingPair schema."""

    def test_initialization_minimal(self):
        """Test minimal initialization."""
        from omnivector.data.schema import EmbeddingPair

        pair = EmbeddingPair(
            query="What is AI?",
            positive="AI is artificial intelligence.",
        )
        assert pair.query == "What is AI?"
        assert pair.positive == "AI is artificial intelligence."
        assert pair.negatives == []
        assert pair.domain == "general"
        logger.info("✓ Minimal EmbeddingPair created")

    def test_initialization_full(self):
        """Test full initialization."""
        from omnivector.data.schema import EmbeddingPair

        pair = EmbeddingPair(
            query="What is ML?",
            positive="ML is machine learning.",
            negatives=["Neg1", "Neg2"],
            query_instruction="Search",
            domain="code",
        )
        assert pair.query == "What is ML?"
        assert len(pair.negatives) == 2
        assert pair.query_instruction == "Search"
        assert pair.domain == "code"
        logger.info("✓ Full EmbeddingPair created")

    def test_from_dict(self):
        """Test creation from dictionary."""
        from omnivector.data.schema import EmbeddingPair

        data = {
            "query": "Test query",
            "positive": "Test positive",
            "negatives": ["Neg1"],
            "domain": "retrieval",
        }
        pair = EmbeddingPair.from_dict(data)
        assert pair.query == data["query"]
        assert pair.domain == data["domain"]
        logger.info("✓ EmbeddingPair created from dict")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from omnivector.data.schema import EmbeddingPair

        pair = EmbeddingPair(query="Q", positive="P", negatives=["N"], domain="test")
        d = pair.to_dict()
        assert d["query"] == "Q"
        assert d["positive"] == "P"
        assert d["domain"] == "test"
        logger.info("✓ EmbeddingPair converted to dict")

    def test_repr(self):
        """Test string representation."""
        from omnivector.data.schema import EmbeddingPair

        pair = EmbeddingPair(query="Long " * 20, positive="P", negatives=["N1", "N2"])
        repr_str = repr(pair)
        assert "EmbeddingPair" in repr_str
        assert "2" in repr_str  # 2 negatives
        logger.info(f"✓ Repr: {repr_str[:60]}...")


class TestDataPreprocessing:
    """Test suite for preprocessing utilities."""

    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        from omnivector.data.preprocessing import preprocess_text

        text = "  Hello   world  "
        result = preprocess_text(text, normalize=True)
        assert result == "Hello world"
        logger.info("✓ Basic preprocessing works")

    def test_preprocess_text_with_instruction(self):
        """Test preprocessing with instruction."""
        from omnivector.data.preprocessing import preprocess_text

        text = "What is AI?"
        instruction = "Search query"
        result = preprocess_text(text, instruction=instruction)
        assert "Instruct:" in result
        assert instruction in result
        assert text in result
        logger.info("✓ Instruction prefix added correctly")

    def test_preprocess_text_truncation(self):
        """Test text truncation."""
        from omnivector.data.preprocessing import preprocess_text

        long_text = "A" * 1000
        result = preprocess_text(long_text, max_length=100)
        assert len(result) <= 100
        logger.info(f"✓ Text truncated to {len(result)}")

    def test_preprocess_text_empty_raises(self):
        """Test empty text raises error."""
        from omnivector.data.preprocessing import preprocess_text

        with pytest.raises(ValueError, match="cannot be empty"):
            preprocess_text("   ")
        logger.info("✓ Empty text raises ValueError")

    def test_clean_text(self):
        """Test text cleaning."""
        from omnivector.data.preprocessing import clean_text

        text = "Hello   world\n\ntest"
        result = clean_text(text)
        assert "   " not in result
        assert "\n\n" not in result
        logger.info("✓ Text cleaning works")

    def test_truncate_text_end(self):
        """Test truncation at end."""
        from omnivector.data.preprocessing import truncate_text

        text = "A" * 1000
        result = truncate_text(text, max_length=100, truncate_at="end")
        assert len(result) <= 100
        logger.info(f"✓ End truncation: {len(result)}")

    def test_extract_code_instruction(self):
        """Test code detection."""
        from omnivector.data.preprocessing import extract_code_instruction

        code = "def hello():\n  pass"
        text, domain = extract_code_instruction(code)
        assert domain == "code"
        logger.info(f"✓ Code detected with domain: {domain}")

    def test_extract_sql_instruction(self):
        """Test SQL detection."""
        from omnivector.data.preprocessing import extract_code_instruction

        sql = "SELECT * FROM users WHERE id = 1"
        text, domain = extract_code_instruction(sql)
        assert domain == "sql"
        logger.info(f"✓ SQL detected with domain: {domain}")


class TestValidation:
    """Test suite for validation functions."""

    def test_validate_embedding_pair_valid(self):
        """Test validation accepts valid data."""
        from omnivector.data.schema import validate_embedding_pair

        data = {
            "query": "Test query",
            "positive": "Test positive",
        }
        pair = validate_embedding_pair(data)
        assert pair.query == "Test query"
        logger.info("✓ Valid pair passes validation")

    def test_validate_embedding_pair_missing_query(self):
        """Test validation rejects missing query."""
        from omnivector.data.schema import validate_embedding_pair

        data = {"positive": "Test positive"}
        with pytest.raises(ValueError, match="Missing required keys"):
            validate_embedding_pair(data)
        logger.info("✓ Missing query rejected")

    def test_validate_embedding_pair_empty_query(self):
        """Test validation rejects empty query."""
        from omnivector.data.schema import validate_embedding_pair

        data = {
            "query": "   ",
            "positive": "Test positive",
        }
        with pytest.raises(ValueError, match="non-empty"):
            validate_embedding_pair(data)
        logger.info("✓ Empty query rejected")
