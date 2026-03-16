"""Test that the package can be imported correctly."""

import logging

logger = logging.getLogger(__name__)


def test_import_omnivector():
    """Test main package import."""
    from omnivector import OmniVectorModel, __version__

    assert __version__ == "0.1.0"
    assert OmniVectorModel is not None
    logger.info(f"✓ OmniVector v{__version__} imported successfully")


def test_import_model_components():
    """Test model component imports."""
    from omnivector.model import (
        EagerMultiheadAttention,
        LatentAttentionPooling,
        MistralEmbeddingBackbone,
    )

    assert MistralEmbeddingBackbone is not None
    assert LatentAttentionPooling is not None
    assert EagerMultiheadAttention is not None
    logger.info("✓ Model components imported successfully")


def test_import_data_schema():
    """Test data schema imports."""
    from omnivector.data import EmbeddingDataset, EmbeddingPair

    assert EmbeddingPair is not None
    assert EmbeddingDataset is not None
    logger.info("✓ Data schema imported successfully")


def test_import_training():
    """Test training module imports."""
    from omnivector.training import MRLInfoNCELoss

    assert MRLInfoNCELoss is not None
    logger.info("✓ Training modules imported successfully")


def test_package_structure():
    """Test package structure is valid."""
    import omnivector

    assert hasattr(omnivector, "__version__")
    assert hasattr(omnivector, "OmniVectorModel")
    assert hasattr(omnivector, "MistralEmbeddingBackbone")
    logger.info("✓ Package structure is valid")
