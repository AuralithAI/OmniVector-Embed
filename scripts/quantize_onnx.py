"""Standalone quantization script for OmniVector ONNX models.

Usage:
    python scripts/quantize_onnx.py \
        --onnx-path exports/omnivector_embed.onnx \
        --output-dir exports/ \
        --optimize --validate
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Quantize OmniVector ONNX model to int8")
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to fp32 ONNX model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for quantized model. Defaults to same as input.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Custom output path for quantized model.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ORT graph optimizations before quantization.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Use per-channel quantization (default: True).",
    )
    parser.add_argument(
        "--no-per-channel",
        action="store_true",
        help="Disable per-channel quantization.",
    )
    parser.add_argument(
        "--reduce-range",
        action="store_true",
        help="Use 7-bit range for older hardware.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate quantized model output shape.",
    )
    return parser.parse_args()


def main() -> None:
    """Run quantization pipeline."""
    args = parse_args()

    if not os.path.exists(args.onnx_path):
        logger.error(f"ONNX model not found: {args.onnx_path}")
        sys.exit(1)

    from omnivector.export.onnx_quantizer import ONNXQuantizer

    quantizer = ONNXQuantizer(args.onnx_path, output_dir=args.output_dir)

    onnx_path = args.onnx_path

    if args.optimize:
        onnx_path = quantizer.optimize(onnx_path)
        logger.info(f"Optimized model: {onnx_path}")
        # Re-init quantizer with optimized model
        quantizer = ONNXQuantizer(onnx_path, output_dir=args.output_dir)

    per_channel = args.per_channel and not args.no_per_channel

    quantized_path = quantizer.quantize(
        output_path=args.output_path,
        per_channel=per_channel,
        reduce_range=args.reduce_range,
    )

    logger.info(f"Quantized model saved to: {quantized_path}")

    if args.validate:
        import numpy as np

        from omnivector.export.onnx_validator import ONNXValidator

        validator = ONNXValidator(quantized_path)
        info = validator.check_model_structure()
        logger.info(f"Model structure: {info}")

        # Smoke test with dummy input
        dummy_ids = np.random.randint(0, 1000, size=(1, 32)).astype(np.int64)
        dummy_mask = np.ones((1, 32), dtype=np.int64)
        output = validator.infer(dummy_ids, dummy_mask)
        logger.info(f"Smoke test output shape: {output.shape}")


if __name__ == "__main__":
    main()
