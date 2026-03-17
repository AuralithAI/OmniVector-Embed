"""Unit tests for dataset loaders."""

import pytest


class TestDataLoaders:
    """Test suite for dataset loaders."""

    def test_get_loader_msmarco(self):
        """Test factory function returns MSMARCOLoader."""
        from omnivector.data.loaders.base import get_loader, MSMARCOLoader

        loader = get_loader("msmarco")
        assert isinstance(loader, MSMARCOLoader)
        assert loader.name == "msmarco"

    def test_get_loader_hotpotqa(self):
        """Test factory function returns HotpotQALoader."""
        from omnivector.data.loaders.base import get_loader, HotpotQALoader

        loader = get_loader("hotpotqa")
        assert isinstance(loader, HotpotQALoader)
        assert loader.name == "hotpotqa"

    def test_get_loader_beir(self):
        """Test factory function returns BEIRLoader."""
        from omnivector.data.loaders.base import get_loader, BEIRLoader

        loader = get_loader("beir/nfcorpus")
        assert isinstance(loader, BEIRLoader)
        assert loader.benchmark == "nfcorpus"

    def test_get_loader_invalid_dataset(self):
        """Test factory function raises on invalid dataset name."""
        from omnivector.data.loaders.base import get_loader

        with pytest.raises(ValueError, match="Unknown dataset"):
            get_loader("invalid_dataset")

    def test_msmarco_loader_init(self):
        """Test MSMARCOLoader initialization."""
        from omnivector.data.loaders.base import MSMARCOLoader

        loader = MSMARCOLoader(split="train", max_samples=100)
        assert loader.split == "train"
        assert loader.max_samples == 100
        assert loader.use_instruction_prefix is True

    def test_hotpotqa_loader_init(self):
        """Test HotpotQALoader initialization."""
        from omnivector.data.loaders.base import HotpotQALoader

        loader = HotpotQALoader(split="validation", max_samples=50)
        assert loader.split == "validation"
        assert loader.max_samples == 50
        assert loader.use_instruction_prefix is True

    def test_beir_loader_init(self):
        """Test BEIRLoader initialization."""
        from omnivector.data.loaders.base import BEIRLoader

        loader = BEIRLoader(benchmark="scifact", split="validation")
        assert loader.benchmark == "scifact"
        assert loader.split == "validation"

    def test_beir_loader_invalid_benchmark(self):
        """Test BEIRLoader rejects invalid benchmark."""
        from omnivector.data.loaders.base import BEIRLoader

        with pytest.raises(ValueError, match="Invalid BEIR benchmark"):
            BEIRLoader(benchmark="invalid_benchmark")

    def test_beir_valid_benchmarks(self):
        """Test that valid BEIR benchmarks are accepted."""
        from omnivector.data.loaders.base import BEIRLoader

        valid_benchmarks = [
            "nfcorpus",
            "scifact",
            "arguana",
            "fiqa",
            "trec-covid",
            "dbpedia-entity",
            "bioasq",
        ]

        for benchmark in valid_benchmarks:
            loader = BEIRLoader(benchmark=benchmark)
            assert loader.benchmark == benchmark

    def test_msmarco_loader_instruction_prefix(self):
        """Test MSMARCO loader with and without instruction prefix."""
        from omnivector.data.loaders.base import MSMARCOLoader

        with_prefix = MSMARCOLoader(use_instruction_prefix=True)
        assert with_prefix.use_instruction_prefix is True

        without_prefix = MSMARCOLoader(use_instruction_prefix=False)
        assert without_prefix.use_instruction_prefix is False

    def test_loader_names(self):
        """Test that loaders have correct names."""
        from omnivector.data.loaders.base import (
            MSMARCOLoader,
            HotpotQALoader,
            BEIRLoader,
        )

        assert MSMARCOLoader().name == "msmarco"
        assert HotpotQALoader().name == "hotpotqa"
        assert BEIRLoader().name == "beir/nfcorpus"
        assert BEIRLoader(benchmark="scifact").name == "beir/scifact"
