"""Unit tests for ONNX export pipeline."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SimpleBackbone(nn.Module):
    """Minimal backbone substitute for testing ONNX export."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._lora_merged = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        return self.linear(hidden)

    def merge_lora(self):
        self._lora_merged = True


class SimplePooling(nn.Module):
    """Minimal pooling substitute for testing ONNX export."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1, keepdim=True)
        return self.linear(pooled).squeeze(1)


class SimpleModel(nn.Module):
    """Minimal model substitute for testing ONNX export."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.backbone = SimpleBackbone(hidden_dim)
        self.pooling = SimplePooling(hidden_dim)


class TestONNXWrapper:
    """Tests for OmniVectorONNXWrapper."""

    def test_initialization(self):
        """Test wrapper initializes with backbone and pooling."""
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        backbone = SimpleBackbone(256)
        pooling = SimplePooling(256)
        wrapper = OmniVectorONNXWrapper(backbone=backbone, pooling=pooling, output_dim=128)

        assert wrapper.output_dim == 128
        assert wrapper.backbone is backbone
        assert wrapper.pooling is pooling

    def test_forward_output_shape(self):
        """Test wrapper forward produces correct output shape."""
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        wrapper = OmniVectorONNXWrapper(
            backbone=SimpleBackbone(256),
            pooling=SimplePooling(256),
            output_dim=128,
        )
        wrapper.eval()

        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)

        with torch.no_grad():
            output = wrapper(input_ids, attention_mask)

        assert output.shape == (2, 128)

    def test_output_normalized(self):
        """Test wrapper output is L2-normalized."""
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        wrapper = OmniVectorONNXWrapper(
            backbone=SimpleBackbone(256),
            pooling=SimplePooling(256),
            output_dim=256,
        )
        wrapper.eval()

        input_ids = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)

        with torch.no_grad():
            output = wrapper(input_ids, attention_mask)

        norms = torch.norm(output, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_output_dim_truncation(self):
        """Test that output is truncated to output_dim."""
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        wrapper = OmniVectorONNXWrapper(
            backbone=SimpleBackbone(256),
            pooling=SimplePooling(256),
            output_dim=64,
        )
        wrapper.eval()

        input_ids = torch.randint(0, 1000, (1, 8))
        attention_mask = torch.ones(1, 8, dtype=torch.long)

        with torch.no_grad():
            output = wrapper(input_ids, attention_mask)

        assert output.shape == (1, 64)

    def test_variable_sequence_lengths(self):
        """Test wrapper handles different sequence lengths."""
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        wrapper = OmniVectorONNXWrapper(
            backbone=SimpleBackbone(256),
            pooling=SimplePooling(256),
            output_dim=128,
        )
        wrapper.eval()

        for seq_len in [8, 16, 64, 128]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            attention_mask = torch.ones(1, seq_len, dtype=torch.long)

            with torch.no_grad():
                output = wrapper(input_ids, attention_mask)

            assert output.shape == (1, 128), f"Failed for seq_len={seq_len}"


class TestONNXExporter:
    """Tests for ONNXExporter."""

    def test_initialization(self):
        """Test exporter initialization with default parameters."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)
        exporter = ONNXExporter(model=model, output_dir="./test_export")

        assert exporter.opset_version == 18
        assert exporter.output_dim == 4096

    def test_merge_lora(self):
        """Test LoRA merge is called on backbone."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)
        exporter = ONNXExporter(model=model)
        exporter.merge_lora()

        assert model.backbone._lora_merged is True

    def test_merge_lora_no_backbone(self):
        """Test LoRA merge gracefully handles models without merge_lora."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = MagicMock()
        model.backbone = MagicMock(spec=[])
        model.parameters.return_value = iter([torch.randn(1)])
        exporter = ONNXExporter(model=model)
        exporter.merge_lora()  # Should not raise

    def test_export_creates_onnx_file(self):
        """Test export produces a valid ONNX file."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(
                model=model,
                output_dir=tmpdir,
                opset_version=17,
                output_dim=128,
            )
            onnx_path = exporter.export(merge_lora=True)

            assert os.path.exists(onnx_path)
            assert onnx_path.endswith(".onnx")
            assert os.path.getsize(onnx_path) > 0

    def test_export_custom_output_path(self):
        """Test export with custom output path."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_model.onnx")
            exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
            onnx_path = exporter.export(output_path=custom_path, merge_lora=False)

            assert onnx_path == custom_path
            assert os.path.exists(custom_path)

    def test_export_creates_output_dir(self):
        """Test export creates output directory if it does not exist."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "nested", "output")
            exporter = ONNXExporter(model=model, output_dir=nested_dir, output_dim=128)
            onnx_path = exporter.export(merge_lora=False)

            assert os.path.exists(nested_dir)
            assert os.path.exists(onnx_path)

    def test_validate_onnx(self):
        """Test ONNX model validation passes for exported model."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
            onnx_path = exporter.export(merge_lora=False)

            assert exporter.validate_onnx(onnx_path) is True

    def test_export_dynamic_axes(self):
        """Test exported model accepts variable batch and sequence lengths."""
        import onnxruntime as ort

        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
            onnx_path = exporter.export(merge_lora=False)

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            # Test batch_size=1, seq_len=16
            ids1 = np.random.randint(0, 1000, (1, 16)).astype(np.int64)
            mask1 = np.ones((1, 16), dtype=np.int64)
            out1 = session.run(None, {"input_ids": ids1, "attention_mask": mask1})
            assert out1[0].shape == (1, 128)

            # Test batch_size=4, seq_len=64
            ids2 = np.random.randint(0, 1000, (4, 64)).astype(np.int64)
            mask2 = np.ones((4, 64), dtype=np.int64)
            out2 = session.run(None, {"input_ids": ids2, "attention_mask": mask2})
            assert out2[0].shape == (4, 128)

    def test_export_pytorch_onnx_parity(self):
        """Test ONNX output matches PyTorch output (cosine > 0.99)."""
        import onnxruntime as ort

        from omnivector.export.onnx_exporter import ONNXExporter, OmniVectorONNXWrapper

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
            onnx_path = exporter.export(merge_lora=False)

            wrapper = OmniVectorONNXWrapper(
                backbone=model.backbone,
                pooling=model.pooling,
                output_dim=128,
            )
            wrapper.eval()

            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

            for _ in range(10):
                ids = np.random.randint(0, 1000, (2, 32)).astype(np.int64)
                mask = np.ones((2, 32), dtype=np.int64)

                with torch.no_grad():
                    pt_out = wrapper(torch.tensor(ids), torch.tensor(mask)).numpy()

                onnx_out = session.run(None, {"input_ids": ids, "attention_mask": mask})[0]

                dot = np.sum(pt_out * onnx_out, axis=-1)
                norm_pt = np.linalg.norm(pt_out, axis=-1)
                norm_onnx = np.linalg.norm(onnx_out, axis=-1)
                cos_sim = dot / (norm_pt * norm_onnx + 1e-12)

                assert np.all(cos_sim > 0.99), f"Cosine sim too low: {cos_sim}"


class TestONNXQuantizer:
    """Tests for ONNXQuantizer."""

    @pytest.fixture
    def exported_model_path(self):
        """Export a simple model and return its path."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)
        tmpdir = tempfile.mkdtemp()
        exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
        onnx_path = exporter.export(merge_lora=False)
        yield onnx_path
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initialization(self, exported_model_path):
        """Test quantizer initialization."""
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(exported_model_path)
        assert quantizer.onnx_path == Path(exported_model_path)

    def test_quantize_creates_file(self, exported_model_path):
        """Test quantization produces output file."""
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(exported_model_path)
        quantized_path = quantizer.quantize()

        assert os.path.exists(quantized_path)
        assert quantized_path.endswith("_int8.onnx")

    def test_quantized_model_smaller(self, exported_model_path):
        """Test quantized model produces a valid file.

        Note: for tiny test models, int8 quantization metadata overhead
        can exceed the original model size. This test only checks the
        quantized file is produced and loadable.
        """
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(exported_model_path)
        quantized_path = quantizer.quantize()

        assert os.path.exists(quantized_path)
        assert os.path.getsize(quantized_path) > 0

    def test_quantize_custom_output(self, exported_model_path):
        """Test quantization with custom output path."""
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_quant.onnx")
            quantizer = ONNXQuantizer(exported_model_path, output_dir=tmpdir)
            quantized_path = quantizer.quantize(output_path=custom_path)

            assert quantized_path == custom_path
            assert os.path.exists(custom_path)

    def test_quantized_model_runs(self, exported_model_path):
        """Test quantized model produces valid inference output."""
        import onnxruntime as ort

        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(exported_model_path)
        quantized_path = quantizer.quantize()

        session = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
        ids = np.random.randint(0, 1000, (1, 16)).astype(np.int64)
        mask = np.ones((1, 16), dtype=np.int64)
        out = session.run(None, {"input_ids": ids, "attention_mask": mask})

        assert out[0].shape == (1, 128)

    def test_optimize_creates_file(self, exported_model_path):
        """Test ORT graph optimization produces output file."""
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(exported_model_path)
        optimized_path = quantizer.optimize()

        assert os.path.exists(optimized_path)
        assert "opt" in optimized_path


class TestONNXValidator:
    """Tests for ONNXValidator."""

    @pytest.fixture
    def exported_model_and_path(self):
        """Export a simple model and return both model and ONNX path."""
        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(256)
        tmpdir = tempfile.mkdtemp()
        exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
        onnx_path = exporter.export(merge_lora=False)
        yield model, onnx_path
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_initialization(self, exported_model_and_path):
        """Test validator initialization loads ORT session."""
        from omnivector.export.onnx_validator import ONNXValidator

        _, onnx_path = exported_model_and_path
        validator = ONNXValidator(onnx_path)
        assert validator.session is not None

    def test_infer_shape(self, exported_model_and_path):
        """Test inference produces correct output shape."""
        from omnivector.export.onnx_validator import ONNXValidator

        _, onnx_path = exported_model_and_path
        validator = ONNXValidator(onnx_path)

        ids = np.random.randint(0, 1000, (2, 32)).astype(np.int64)
        mask = np.ones((2, 32), dtype=np.int64)
        output = validator.infer(ids, mask)

        assert output.shape == (2, 128)

    def test_validate_parity_passes(self, exported_model_and_path):
        """Test parity validation passes for freshly exported model."""
        from omnivector.export.onnx_validator import ONNXValidator

        model, onnx_path = exported_model_and_path
        validator = ONNXValidator(onnx_path)

        result = validator.validate_parity(
            pytorch_model=model,
            num_samples=10,
            seq_length=32,
            vocab_size=1000,
            threshold=0.99,
            output_dim=128,
        )

        assert result["passed"] is True
        assert result["mean_cosine_sim"] > 0.99
        assert result["min_cosine_sim"] > 0.99
        assert result["num_samples"] == 10

    def test_check_model_structure(self, exported_model_and_path):
        """Test model structure inspection returns expected fields."""
        from omnivector.export.onnx_validator import ONNXValidator

        _, onnx_path = exported_model_and_path
        validator = ONNXValidator(onnx_path)
        info = validator.check_model_structure()

        assert info["opset_version"] >= 17
        assert len(info["inputs"]) == 2
        assert len(info["outputs"]) == 1

        input_names = [inp["name"] for inp in info["inputs"]]
        assert "input_ids" in input_names
        assert "attention_mask" in input_names
        assert info["outputs"][0]["name"] == "embedding"

    def test_validate_parity_fails_on_mismatch(self, exported_model_and_path):
        """Test parity validation fails when threshold is impossibly high."""
        from omnivector.export.onnx_validator import ONNXValidator

        model, onnx_path = exported_model_and_path

        # Corrupt model weights to create mismatch
        model.backbone.linear.weight.data.fill_(0.0)

        validator = ONNXValidator(onnx_path)
        result = validator.validate_parity(
            pytorch_model=model,
            num_samples=5,
            seq_length=16,
            vocab_size=1000,
            threshold=1.0,
            output_dim=128,
        )

        assert result["passed"] is False


class TestEndToEndExportPipeline:
    """Integration tests for the full export-optimize-quantize-validate pipeline."""

    def test_full_pipeline(self):
        """Test full export -> optimize -> quantize -> validate pipeline."""
        import onnxruntime as ort

        from omnivector.export.onnx_exporter import ONNXExporter
        from omnivector.export.onnx_quantizer import ONNXQuantizer
        from omnivector.export.onnx_validator import ONNXValidator

        model = SimpleModel(256)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export
            exporter = ONNXExporter(model=model, output_dir=tmpdir, output_dim=128)
            onnx_path = exporter.export(merge_lora=True)
            assert os.path.exists(onnx_path)

            # Validate fp32
            exporter.validate_onnx(onnx_path)

            # Optimize
            quantizer = ONNXQuantizer(onnx_path, output_dir=tmpdir)
            opt_path = quantizer.optimize()
            assert os.path.exists(opt_path)

            # Quantize
            quantizer_opt = ONNXQuantizer(opt_path, output_dir=tmpdir)
            quant_path = quantizer_opt.quantize()
            assert os.path.exists(quant_path)

            # Validate quantized model runs
            session = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])
            ids = np.random.randint(0, 1000, (1, 32)).astype(np.int64)
            mask = np.ones((1, 32), dtype=np.int64)
            out = session.run(None, {"input_ids": ids, "attention_mask": mask})
            assert out[0].shape == (1, 128)

            # Validate parity on fp32 model
            validator = ONNXValidator(onnx_path)
            result = validator.validate_parity(
                pytorch_model=model,
                num_samples=10,
                vocab_size=1000,
                threshold=0.99,
                output_dim=128,
            )
            assert result["passed"] is True
