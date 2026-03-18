"""ONNX export script for OmniVector-Embed.

Usage:
    python scripts/export_onnx.py \
        --model-path checkpoints/stage2/final \
        --output-dir exports/ \
        --optimize --validate
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export OmniVector model to ONNX format")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained OmniVector model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx_export",
        help="Output directory for ONNX files.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=4096,
        help="Output embedding dimension.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for dummy input.",
    )
    parser.add_argument(
        "--no-merge-lora",
        action="store_true",
        help="Skip LoRA adapter merge before export.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ORT graph optimizations after export.",
    )
    parser.add_argument(
        "--quantize-int8",
        action="store_true",
        help="Apply int8 dynamic quantization after export.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate ONNX output parity against PyTorch.",
    )
    parser.add_argument(
        "--num-validation-samples",
        type=int,
        default=50,
        help="Number of samples for parity validation.",
    )
    parser.add_argument(
        "--validation-threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity threshold for validation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run ONNX export pipeline."""
    args = parse_args()

    from omnivector.export.onnx_exporter import ONNXExporter
    from omnivector.model.omnivector_model import OmniVectorModel

    logger.info(f"Loading model from {args.model_path}")
    model = OmniVectorModel.from_pretrained(args.model_path)
    model.eval()

    exporter = ONNXExporter(
        model=model,
        output_dir=args.output_dir,
        opset_version=args.opset_version,
        output_dim=args.output_dim,
    )

    onnx_path = exporter.export(
        merge_lora=not args.no_merge_lora,
        max_seq_length=args.max_seq_length,
    )
    exporter.validate_onnx(onnx_path)

    if args.optimize:
        onnx_path = exporter.optimize(onnx_path)
        logger.info(f"Optimized model saved to {onnx_path}")

    if args.quantize_int8:
        from omnivector.export.onnx_quantizer import ONNXQuantizer

        quantizer = ONNXQuantizer(onnx_path, output_dir=args.output_dir)
        onnx_path = quantizer.quantize()
        logger.info(f"Quantized model saved to {onnx_path}")

    if args.validate:
        from omnivector.export.onnx_validator import ONNXValidator

        validator = ONNXValidator(onnx_path)
        result = validator.validate_parity(
            pytorch_model=model,
            num_samples=args.num_validation_samples,
            threshold=args.validation_threshold,
            output_dim=args.output_dim,
        )

        if not result["passed"]:
            logger.error(
                f"Validation FAILED: min cosine sim {result['min_cosine_sim']:.6f} "
                f"< threshold {args.validation_threshold}"
            )
            sys.exit(1)

        logger.info("Validation PASSED")

    logger.info(f"Export complete: {onnx_path}")


if __name__ == "__main__":
    main()
