"""ONNX model validation utilities.

Compares ONNX inference output against PyTorch reference output
using cosine similarity to ensure export fidelity.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ONNXValidator:
    """Validates ONNX model parity with PyTorch reference."""

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[list[str]] = None,
    ):
        """Initialize validator.

        Args:
            onnx_path: Path to ONNX model file.
            providers: ORT execution providers. Defaults to CPUExecutionProvider.
        """
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)

    def infer(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Run ONNX inference.

        Args:
            input_ids: Token IDs [batch_size, seq_length] as int64.
            attention_mask: Attention mask [batch_size, seq_length] as int64.

        Returns:
            Embeddings [batch_size, output_dim] as float32.
        """
        outputs = self.session.run(
            ["embedding"],
            {
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
            },
        )
        return outputs[0]

    def validate_parity(
        self,
        pytorch_model,
        num_samples: int = 50,
        seq_length: int = 64,
        vocab_size: int = 32000,
        threshold: float = 0.99,
        output_dim: int = 4096,
    ) -> dict:
        """Compare ONNX output against PyTorch reference.

        Generates random inputs, runs both models, and checks cosine
        similarity per sample. All samples must exceed threshold.

        Args:
            pytorch_model: OmniVectorModel instance for reference output.
            num_samples: Number of random inputs to test.
            seq_length: Sequence length for random inputs.
            vocab_size: Vocabulary size for random token generation.
            threshold: Minimum cosine similarity per sample.
            output_dim: Output embedding dimension.

        Returns:
            Dict with keys: passed (bool), mean_cosine_sim (float),
            min_cosine_sim (float), num_samples (int), threshold (float).
        """
        from omnivector.export.onnx_exporter import OmniVectorONNXWrapper

        wrapper = OmniVectorONNXWrapper(
            backbone=pytorch_model.backbone,
            pooling=pytorch_model.pooling,
            output_dim=output_dim,
        )
        wrapper.eval()

        device = next(pytorch_model.parameters()).device
        cosine_sims = []

        for i in range(num_samples):
            length = max(8, np.random.randint(16, seq_length + 1))
            input_ids_np = np.random.randint(0, vocab_size, size=(1, length)).astype(np.int64)
            attention_mask_np = np.ones((1, length), dtype=np.int64)

            # Randomly mask some trailing tokens
            if length > 16:
                pad_start = np.random.randint(length // 2, length)
                attention_mask_np[0, pad_start:] = 0

            # PyTorch reference
            with torch.no_grad():
                pt_input_ids = torch.tensor(input_ids_np, device=device)
                pt_attention_mask = torch.tensor(attention_mask_np, device=device)
                pt_output = wrapper(pt_input_ids, pt_attention_mask).cpu().numpy()

            # ONNX inference
            onnx_output = self.infer(input_ids_np, attention_mask_np)

            # Cosine similarity
            dot = np.sum(pt_output * onnx_output, axis=-1)
            norm_pt = np.linalg.norm(pt_output, axis=-1)
            norm_onnx = np.linalg.norm(onnx_output, axis=-1)
            cos_sim = dot / (norm_pt * norm_onnx + 1e-12)
            cosine_sims.append(float(cos_sim[0]))

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Validated {i + 1}/{num_samples} samples, "
                    f"mean cosine: {np.mean(cosine_sims):.6f}"
                )

        cosine_sims_arr = np.array(cosine_sims)
        mean_sim = float(np.mean(cosine_sims_arr))
        min_sim = float(np.min(cosine_sims_arr))
        passed = bool(min_sim >= threshold)

        result = {
            "passed": passed,
            "mean_cosine_sim": mean_sim,
            "min_cosine_sim": min_sim,
            "num_samples": num_samples,
            "threshold": threshold,
        }

        if passed:
            logger.info(
                f"Validation PASSED: mean={mean_sim:.6f}, min={min_sim:.6f} "
                f"(threshold={threshold})"
            )
        else:
            logger.warning(
                f"Validation FAILED: mean={mean_sim:.6f}, min={min_sim:.6f} "
                f"(threshold={threshold})"
            )

        return result

    def check_model_structure(self) -> dict:
        """Inspect ONNX model metadata and graph info.

        Returns:
            Dict with keys: input_names, output_names, input_shapes,
            output_shapes, opset_version.
        """
        import onnx

        model = onnx.load(str(self.onnx_path))

        inputs = []
        for inp in model.graph.input:
            shape = []
            for dim in inp.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            inputs.append({"name": inp.name, "shape": shape})

        outputs = []
        for out in model.graph.output:
            shape = []
            for dim in out.type.tensor_type.shape.dim:
                if dim.dim_param:
                    shape.append(dim.dim_param)
                else:
                    shape.append(dim.dim_value)
            outputs.append({"name": out.name, "shape": shape})

        opset = model.opset_import[0].version if model.opset_import else None

        return {
            "inputs": inputs,
            "outputs": outputs,
            "opset_version": opset,
            "ir_version": model.ir_version,
        }
