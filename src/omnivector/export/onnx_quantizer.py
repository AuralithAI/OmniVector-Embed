"""ONNX int8 dynamic quantization utilities.

Applies dynamic quantization using onnxruntime with MatMulConstBOnly
to avoid quantizing attention score computation (Q @ K^T).
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ONNXQuantizer:
    """Handles int8 dynamic quantization of ONNX models."""

    def __init__(
        self,
        onnx_path: str,
        output_dir: Optional[str] = None,
    ):
        """Initialize quantizer.

        Args:
            onnx_path: Path to the fp32 ONNX model.
            output_dir: Output directory for quantized model. Defaults to same
                directory as input model.
        """
        self.onnx_path = Path(onnx_path)
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.onnx_path.parent

    def quantize(
        self,
        output_path: Optional[str] = None,
        per_channel: bool = True,
        reduce_range: bool = False,
    ) -> str:
        """Apply int8 dynamic quantization.

        Uses MatMulConstBOnly strategy: only quantizes weight-only MatMul
        nodes (linear layers), leaving Q @ K^T attention score computation
        in fp32 for numerical stability.

        Args:
            output_path: Custom output path. Defaults to
                <output_dir>/omnivector_embed_int8.onnx.
            per_channel: Use per-channel quantization for better accuracy.
            reduce_range: Use 7-bit range for older hardware compatibility.

        Returns:
            Path to quantized ONNX model.
        """
        from onnxruntime.quantization import QuantType, quantize_dynamic

        if output_path is None:
            output_path = str(self.output_dir / "omnivector_embed_int8.onnx")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Quantizing {self.onnx_path} -> {output_path}")
        logger.info(f"Settings: per_channel={per_channel}, reduce_range={reduce_range}")

        quantize_dynamic(
            model_input=str(self.onnx_path),
            model_output=output_path,
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=QuantType.QInt8,
            extra_options={"MatMulConstBOnly": True},
        )

        original_mb = os.path.getsize(self.onnx_path) / (1024 * 1024)
        quantized_mb = os.path.getsize(output_path) / (1024 * 1024)
        ratio = quantized_mb / original_mb * 100

        logger.info(
            f"Quantization complete: {original_mb:.1f} MB -> {quantized_mb:.1f} MB "
            f"({ratio:.1f}%)"
        )

        return output_path

    def optimize(self, onnx_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
        """Apply ORT graph optimizations (fuse LayerNorm, Attention, GELU).

        Should be run before quantization for best results.

        Args:
            onnx_path: Input ONNX model path. Defaults to self.onnx_path.
            output_path: Output path. Defaults to <output_dir>/omnivector_embed_opt.onnx.

        Returns:
            Path to optimized ONNX model.
        """
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = str(self.onnx_path)
        if output_path is None:
            output_path = str(self.output_dir / "omnivector_embed_opt.onnx")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = output_path

        logger.info(f"Optimizing ONNX graph: {onnx_path} -> {output_path}")

        # Creating session with optimization options triggers the optimization
        ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])

        opt_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Optimization complete: {output_path} ({opt_mb:.1f} MB)")

        return output_path
