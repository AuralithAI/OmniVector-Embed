"""Upload ONNX models to HuggingFace Hub.

Usage:
    python scripts/upload_to_hub.py \
        --repo-id omnivector/omnivector-embed-onnx \
        --onnx-path exports/omnivector_embed.onnx \
        --onnx-quantized-path exports/omnivector_embed_int8.onnx
"""

import argparse
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Upload ONNX models to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace Hub repo ID (e.g., omnivector/omnivector-embed-onnx).",
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Path to fp32 ONNX model.",
    )
    parser.add_argument(
        "--onnx-quantized-path",
        type=str,
        default=None,
        help="Path to int8 quantized ONNX model.",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload ONNX models",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository.",
    )
    return parser.parse_args()


def main() -> None:
    """Upload ONNX models to HuggingFace Hub."""
    args = parse_args()

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN environment variable not set")
        raise SystemExit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)
        logger.info(f"Repository ready: {args.repo_id}")
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Upload fp32 model
    if os.path.exists(args.onnx_path):
        logger.info(f"Uploading {args.onnx_path}...")
        api.upload_file(
            path_or_fileobj=args.onnx_path,
            path_in_repo=os.path.basename(args.onnx_path),
            repo_id=args.repo_id,
            commit_message=f"{args.commit_message} (fp32)",
        )
        size_mb = os.path.getsize(args.onnx_path) / (1024 * 1024)
        logger.info(f"Uploaded fp32 model ({size_mb:.1f} MB)")
    else:
        logger.warning(f"fp32 model not found: {args.onnx_path}")

    # Upload quantized model
    if args.onnx_quantized_path and os.path.exists(args.onnx_quantized_path):
        logger.info(f"Uploading {args.onnx_quantized_path}...")
        api.upload_file(
            path_or_fileobj=args.onnx_quantized_path,
            path_in_repo=os.path.basename(args.onnx_quantized_path),
            repo_id=args.repo_id,
            commit_message=f"{args.commit_message} (int8)",
        )
        size_mb = os.path.getsize(args.onnx_quantized_path) / (1024 * 1024)
        logger.info(f"Uploaded int8 model ({size_mb:.1f} MB)")
    elif args.onnx_quantized_path:
        logger.warning(f"Quantized model not found: {args.onnx_quantized_path}")

    logger.info(f"Upload complete: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
