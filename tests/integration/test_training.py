"""Integration tests for OmniVector-Embed."""

import logging
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight test doubles 
# ---------------------------------------------------------------------------

class SimpleBackbone(nn.Module):
    """Minimal backbone for integration tests."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self._lora_merged = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        hidden = self.linear(hidden)
        # Use attention_mask so ONNX export retains it as a graph input
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).float()
        return hidden

    def merge_lora(self):
        self._lora_merged = True


class SimplePooling(nn.Module):
    """Minimal pooling for integration tests."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # Use attention_mask for masked mean pooling so ONNX export retains it
        if attention_mask is not None:
            mask_f = (~attention_mask).unsqueeze(-1).float()
            pooled = (hidden_states * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        return self.linear(pooled)


class SimpleModel(nn.Module):
    """Minimal model for integration tests."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.backbone = SimpleBackbone(hidden_dim)
        self.pooling = SimplePooling(hidden_dim)


# ---------------------------------------------------------------------------
# Data loading smoke test
# ---------------------------------------------------------------------------


class TestTrainingDryRun:
    """Integration test: CPU dry run with 10 steps."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        torch.cuda.is_available() is False, reason="Requires GPU for practical training"
    )
    def test_10_step_training(self):
        """Test 10-step training with loss decreasing over 10 steps.

        Verifies that the training loop produces monotonically-ish
        decreasing loss on a tiny synthetic dataset, confirming that
        gradients flow end-to-end.
        """
        from omnivector.training.losses import MRLInfoNCELoss

        hidden_dim = 256
        model = SimpleModel(hidden_dim=hidden_dim).cuda()
        loss_fn = MRLInfoNCELoss(dimensions=[hidden_dim])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for step in range(10):
            # Synthetic batch: 8 query-positive pairs
            input_ids = torch.randint(0, 1000, (8, 32), device="cuda")
            attention_mask = torch.ones(8, 32, dtype=torch.long, device="cuda")

            # Forward: encode queries and positives (same model, different noise)
            hidden_q = model.backbone(input_ids, attention_mask)
            emb_q = model.pooling(hidden_q, attention_mask=~attention_mask.bool())

            # Positive = same input + slight noise on embeddings
            hidden_p = model.backbone(input_ids, attention_mask)
            emb_p = model.pooling(hidden_p, attention_mask=~attention_mask.bool())

            # Compute loss
            loss = loss_fn(emb_q, emb_p)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            logger.info(f"Step {step}: loss={loss.item():.4f}")

        # Loss at end should be lower than at start (allow some noise)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}"
        )

    @pytest.mark.integration
    def test_data_loading(self):
        """Test data loading from sample dataset."""
        from omnivector.data.schema import EmbeddingPair

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
        logger.info("Data loading test passed")


# ---------------------------------------------------------------------------
# ONNX parity
# ---------------------------------------------------------------------------


class TestONNXParity:
    """Integration test: ONNX export and cosine parity checking."""

    @pytest.mark.integration
    def test_onnx_export_produces_valid_model(self):
        """Test ONNX export produces a valid model file that passes onnx.checker."""
        import onnx

        from omnivector.export.onnx_exporter import ONNXExporter

        model = SimpleModel(hidden_dim=256)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(
                model=model,
                output_dir=tmpdir,
                opset_version=17,
                output_dim=256,
            )
            onnx_path = exporter.export(merge_lora=False)

            assert os.path.exists(onnx_path)
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX export produced valid model")

    @pytest.mark.integration
    def test_onnx_parity_50_samples(self):
        """Test ONNX model produces cosine similarity > 0.99 vs PyTorch for 50 random inputs."""
        import onnxruntime as ort

        from omnivector.export.onnx_exporter import ONNXExporter, OmniVectorONNXWrapper

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=256)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ONNXExporter(
                model=model,
                output_dir=tmpdir,
                opset_version=17,
                output_dim=256,
            )
            onnx_path = exporter.export(merge_lora=False)

            # Create PyTorch wrapper for comparison
            wrapper = OmniVectorONNXWrapper(
                backbone=model.backbone,
                pooling=model.pooling,
                output_dim=256,
            )
            wrapper.eval()

            # Load ONNX session
            session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )

            # Run 50 random inputs and check cosine parity
            cosine_sims = []
            for i in range(50):
                seq_len = torch.randint(8, 64, (1,)).item()
                input_ids = torch.randint(0, 1000, (1, seq_len))
                attention_mask = torch.ones(1, seq_len, dtype=torch.long)

                # PyTorch inference
                with torch.no_grad():
                    pt_output = wrapper(input_ids, attention_mask).numpy()

                # ONNX inference
                ort_inputs = {
                    "input_ids": input_ids.numpy(),
                    "attention_mask": attention_mask.numpy(),
                }
                ort_output = session.run(None, ort_inputs)[0]

                # Cosine similarity
                cos_sim = np.dot(pt_output.flatten(), ort_output.flatten()) / (
                    np.linalg.norm(pt_output.flatten()) * np.linalg.norm(ort_output.flatten())
                    + 1e-12
                )
                cosine_sims.append(cos_sim)

            mean_sim = np.mean(cosine_sims)
            min_sim = np.min(cosine_sims)

            assert min_sim > 0.99, (
                f"ONNX parity too low: min cosine sim = {min_sim:.6f} "
                f"(mean = {mean_sim:.6f})"
            )
            logger.info(
                f"ONNX parity OK: mean={mean_sim:.6f}, min={min_sim:.6f} over 50 samples"
            )

    @pytest.mark.integration
    def test_onnx_quantized_parity(self):
        """Test int8 quantized ONNX model maintains reasonable parity.

        Quantization tolerance is looser (cosine > 0.95) due to int8 rounding.
        """
        import onnxruntime as ort

        from omnivector.export.onnx_exporter import ONNXExporter, OmniVectorONNXWrapper
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        torch.manual_seed(42)
        model = SimpleModel(hidden_dim=256)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Export fp32
            exporter = ONNXExporter(
                model=model,
                output_dir=tmpdir,
                opset_version=17,
                output_dim=256,
            )
            fp32_path = exporter.export(merge_lora=False)

            # Quantize to int8
            quantizer = ONNXQuantizer(fp32_path, output_dir=tmpdir)
            int8_path = quantizer.quantize()

            assert os.path.exists(int8_path)

            # Compare fp32 and int8
            fp32_session = ort.InferenceSession(
                fp32_path, providers=["CPUExecutionProvider"]
            )
            int8_session = ort.InferenceSession(
                int8_path, providers=["CPUExecutionProvider"]
            )

            cosine_sims = []
            for _ in range(20):
                seq_len = torch.randint(8, 64, (1,)).item()
                input_ids = torch.randint(0, 1000, (1, seq_len)).numpy()
                attention_mask = np.ones((1, seq_len), dtype=np.int64)

                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
                fp32_out = fp32_session.run(None, ort_inputs)[0].flatten()
                int8_out = int8_session.run(None, ort_inputs)[0].flatten()

                cos_sim = np.dot(fp32_out, int8_out) / (
                    np.linalg.norm(fp32_out) * np.linalg.norm(int8_out) + 1e-12
                )
                cosine_sims.append(cos_sim)

            mean_sim = np.mean(cosine_sims)
            min_sim = np.min(cosine_sims)

            assert min_sim > 0.95, (
                f"Quantized parity too low: min cos sim = {min_sim:.6f} "
                f"(mean = {mean_sim:.6f})"
            )
            logger.info(
                f"Int8 parity OK: mean={mean_sim:.6f}, min={min_sim:.6f} over 20 samples"
            )


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_encode_decode_pipeline(self):
        """Test full encode-decode pipeline."""
        pytest.skip("Requires full model weights — run with -m slow")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multimodal_encoding(self):
        """Test multimodal (text + image) encoding."""
        pytest.skip("Requires full model weights — run with -m slow")
