"""ONNX parity validation script for OmniVector-Embed.

Usage:
    python scripts/validate_onnx.py \
        --pytorch-model checkpoints/stage1_best \
        --onnx-model exports/omnivector_embed.onnx \
        --num-samples 50 \
        --threshold 0.99
"""

import argparse
import logging
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate ONNX parity against PyTorch model")
    parser.add_argument(
        "--pytorch-model",
        type=str,
        required=True,
        help="Path to PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        required=True,
        help="Path to ONNX model file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of random samples for parity check.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity threshold.",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=4096,
        help="Output embedding dimension.",
    )
    return parser.parse_args()


def main() -> None:
    """Run ONNX parity validation."""
    args = parse_args()

    from omnivector.export.onnx_validator import ONNXValidator
    from omnivector.model.omnivector_model import OmniVectorModel

    logger.info(f"Loading PyTorch model from {args.pytorch_model}")
    model = OmniVectorModel.from_pretrained(args.pytorch_model)
    model.eval()

    logger.info(f"Loading ONNX model from {args.onnx_model}")
    validator = ONNXValidator(args.onnx_model)

    logger.info(f"Running parity validation ({args.num_samples} samples, threshold={args.threshold})")
    result = validator.validate_parity(
        pytorch_model=model,
        num_samples=args.num_samples,
        threshold=args.threshold,
        output_dim=args.output_dim,
    )

    logger.info(f"Mean cosine similarity: {result['mean_cosine_sim']:.6f}")
    logger.info(f"Min cosine similarity:  {result['min_cosine_sim']:.6f}")
    logger.info(f"Max cosine similarity:  {result['max_cosine_sim']:.6f}")

    if result["passed"]:
        logger.info(f"PASSED: All {args.num_samples} samples above threshold {args.threshold}")
    else:
        logger.error(
            f"FAILED: min cosine sim {result['min_cosine_sim']:.6f} "
            f"< threshold {args.threshold}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
