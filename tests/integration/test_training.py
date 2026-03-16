"""Integration tests for OmniVector-Embed."""

import logging

import pytest
import torch

logger = logging.getLogger(__name__)


class TestTrainingDryRun:
    """Integration test: CPU dry run with 10 steps."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        torch.cuda.is_available() is False, reason="Requires GPU for practical training"
    )
    def test_10_step_training(self):
        """Test 10-step training with loss decreasing."""
        pytest.skip("Full training implementation in Week 3")

    @pytest.mark.integration
    def test_data_loading(self):
        """Test data loading from sample dataset."""
        from omnivector.data.schema import EmbeddingPair

        # Create sample pairs
        pairs = [
            EmbeddingPair(
                query="What is ML?",
                positive="ML is machine learning.",
                negatives=["Learning is fun.", "Python is great."],
                domain="retrieval",
            )
            for _ in range(10)
        ]

        assert len(pairs) == 10
        assert all(p.positive for p in pairs)
        logger.info("✓ Data loading test passed")


class TestONNXParity:
    """Integration test: ONNX export and parity checking."""

    @pytest.mark.integration
    @pytest.mark.skipif(True, reason="ONNX test in Week 4")
    def test_onnx_export(self):
        """Test ONNX export produces valid model."""
        pytest.skip("ONNX export implementation in Week 4")

    @pytest.mark.integration
    def test_onnx_parity_validation(self):
        """Test ONNX model produces similar outputs to PyTorch."""
        pytest.skip("ONNX parity in Week 4")


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_encode_decode_pipeline(self):
        """Test full encode-decode pipeline."""
        pytest.skip("Full e2e test in Week 8")

    @pytest.mark.integration
    def test_multimodal_encoding(self):
        """Test multimodal (text + image) encoding."""
        pytest.skip("Vision encoder in Week 5")
