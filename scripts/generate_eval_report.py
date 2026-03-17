"""Generate evaluation report from MTEB results.

Usage:
    python scripts/generate_eval_report.py \
        --results-dir eval_results/ \
        --output-file eval_report.md
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing MTEB result JSON files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="eval_report.md",
        help="Output markdown report file.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate markdown evaluation report from MTEB results."""
    args = parse_args()
    results_dir = Path(args.results_dir)

    lines = [
        "# OmniVector-Embed Evaluation Report",
        "",
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]

    # Collect all JSON result files
    result_files = sorted(results_dir.glob("*.json"))
    if not result_files:
        lines.extend([
            "No results found.",
            "",
            f"Searched directory: `{results_dir}`",
        ])
    else:
        lines.extend([
            "## Task Results",
            "",
            "| Task | Type | Main Metric | Score |",
            "|---|---|---|---|",
        ])

        scores = []
        for result_file in result_files:
            try:
                with open(result_file) as f:
                    data = json.load(f)

                task_name = data.get("task_name", result_file.stem)
                task_type = data.get("task_type", "unknown")
                main_score = data.get("main_score", data.get("score"))

                if main_score is not None:
                    scores.append(float(main_score))
                    lines.append(f"| {task_name} | {task_type} | {main_score:.4f} | {main_score:.4f} |")
                else:
                    lines.append(f"| {task_name} | {task_type} | — | — |")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse {result_file}: {e}")

        if scores:
            avg = sum(scores) / len(scores)
            lines.extend([
                "",
                f"**Average score**: {avg:.4f} (across {len(scores)} tasks)",
                "",
            ])

            # Targets
            lines.extend([
                "## Targets",
                "",
                "| Metric | Target | Achieved |",
                "|---|---|---|",
                f"| MTEB Average | ≥ 65.0 | {avg:.1f} {'✓' if avg >= 65.0 else '✗'} |",
                "",
            ])

    report = "\n".join(lines) + "\n"

    with open(args.output_file, "w") as f:
        f.write(report)

    logger.info(f"Evaluation report written to {args.output_file}")


if __name__ == "__main__":
    main()
