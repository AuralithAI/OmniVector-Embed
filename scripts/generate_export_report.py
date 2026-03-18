"""Generate export report for ONNX release artifacts.

Usage:
    python scripts/generate_export_report.py \
        --release-dir release_artifacts/ \
        --output-file export_report.md
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate export report")
    parser.add_argument(
        "--release-dir",
        type=str,
        required=True,
        help="Directory containing release artifacts.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="export_report.md",
        help="Output file for the report.",
    )
    return parser.parse_args()


def file_sha256(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Generate markdown export report."""
    args = parse_args()
    release_dir = Path(args.release_dir)

    lines = [
        "# OmniVector-Embed Export Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Artifacts",
        "",
        "| File | Size | SHA-256 |",
        "|---|---|---|",
    ]

    onnx_files = sorted(release_dir.glob("*.onnx"))
    for f in onnx_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        sha = file_sha256(str(f))
        lines.append(f"| {f.name} | {size_mb:.1f} MB | `{sha[:16]}...` |")

    if not onnx_files:
        lines.append("| (no ONNX files found) | — | — |")

    lines.extend([
        "",
        "## Model Details",
        "",
        "- **Architecture**: Mistral-7B bidirectional + Latent Attention Pooling",
        "- **Embedding dimension**: 4096 (MRL: 512, 1024, 2048, 4096)",
        "- **ONNX opset**: 17",
        "- **Quantization**: Dynamic INT8 (MatMulConstBOnly)",
        "",
    ])

    # Include validation results if available
    validation_file = release_dir / "validation_results.json"
    if validation_file.exists():
        with open(validation_file) as vf:
            results = json.load(vf)
        lines.extend([
            "## Validation Results",
            "",
            f"- **Mean cosine similarity**: {results.get('mean_cosine_sim', 'N/A')}",
            f"- **Min cosine similarity**: {results.get('min_cosine_sim', 'N/A')}",
            f"- **Passed**: {results.get('passed', 'N/A')}",
            "",
        ])

    report = "\n".join(lines) + "\n"

    with open(args.output_file, "w") as f:
        f.write(report)

    logger.info(f"Export report written to {args.output_file}")


if __name__ == "__main__":
    main()
