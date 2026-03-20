"""
Evaluation script for OmniVector-Embed.

Loads a trained checkpoint and runs MTEB benchmarks via
:class:`omnivector.eval.mteb_runner.MTEBRunner`.

Usage::

    # Retrieval only (fast)
    python scripts/evaluate.py \\
        --model-path checkpoints/stage2_55M/checkpoint-final \\
        --tasks retrieval

    # Full benchmark suite
    python scripts/evaluate.py \\
        --model-path checkpoints/stage2_55M/checkpoint-final \\
        --tasks retrieval,sts,clustering,pair_classification,reranking \\
        --output-dir eval_results \\
        --stage stage2

    # Specific MTEB tasks by name
    python scripts/evaluate.py \\
        --model-path checkpoints/stage1_8M/checkpoint-final \\
        --task-names MSMARCO,NFCorpus \\
        --stage stage1
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate OmniVector-Embed model on MTEB benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to a trained checkpoint directory (containing model.pt) "
        "or a HuggingFace model identifier.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="retrieval",
        help="Comma-separated task *types*: retrieval, sts, clustering, "
        "pair_classification, reranking (default: retrieval).",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        default=None,
        help="Comma-separated explicit MTEB task names (overrides --tasks). "
        "Example: MSMARCO,NFCorpus,STSBenchmark",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for per-task JSON results (default: eval_results).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (default: 128).",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=4096,
        help="Embedding dimension (Matryoshka slice) to evaluate (default: 4096).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["stage1", "stage2", "stage3"],
        help="Training stage — used to check benchmark targets. "
        "If omitted, target checking is skipped.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Load model with LoRA adapters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (default: cuda if available, else cpu).",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # ── Resolve device ───────────────────────────────────────────
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # ── Load model ───────────────────────────────────────────────
    from omnivector.model.omnivector_model import OmniVectorModel

    logger.info(f"Loading model from: {args.model_path}")
    model = OmniVectorModel.from_pretrained(
        model_name_or_path=args.model_path,
        lora=args.lora,
        device=device,
    )
    model.eval()
    logger.info("Model loaded successfully.")

    # ── Run MTEB evaluation ──────────────────────────────────────
    from omnivector.eval.mteb_runner import MTEBRunner

    runner = MTEBRunner(
        model=model,
        output_dir=args.output_dir,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
    )

    task_types = None
    task_names = None

    if args.task_names:
        task_names = [t.strip() for t in args.task_names.split(",") if t.strip()]
        logger.info(f"Running explicit tasks: {task_names}")
    else:
        task_types = [t.strip() for t in args.tasks.split(",") if t.strip()]
        logger.info(f"Running task types: {task_types}")

    results = runner.run(task_types=task_types, tasks=task_names)

    # ── Print summary ────────────────────────────────────────────
    runner.print_summary(results)

    # ── Check benchmark targets ──────────────────────────────────
    if args.stage:
        logger.info(f"Checking targets for {args.stage}...")
        outcomes = runner.check_targets(results, stage=args.stage)

        all_passed = all(outcomes.values()) if outcomes else True
        if all_passed:
            logger.info(f"All {args.stage} targets PASSED ✓")
        else:
            failed = [k for k, v in outcomes.items() if not v]
            logger.warning(f"FAILED targets: {failed}")

    # ── Save summary ─────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"
    summary = {
        "model_path": args.model_path,
        "device": device,
        "output_dim": args.output_dim,
        "stage": args.stage,
        "results": results,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
