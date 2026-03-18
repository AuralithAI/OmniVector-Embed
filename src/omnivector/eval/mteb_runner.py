"""MTEB evaluation runner for OmniVector embedding models.

Wraps the ``mteb`` library to run standard retrieval, STS, clustering,
pair-classification, re-ranking, and classification benchmarks against
an OmniVectorModel.

Usage::

    from omnivector.eval.mteb_runner import MTEBRunner

    runner = MTEBRunner(model=model, output_dir="eval_results")
    results = runner.run(task_types=["Retrieval"])
    runner.print_summary(results)
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Default task sets ────────────────────────────────────────────────

RETRIEVAL_TASKS = [
    "MSMARCO",
    "NFCorpus",
    "SciFact",
    "ArguAna",
    "FiQA2018",
]

STS_TASKS = [
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
]

CLUSTERING_TASKS = [
    "TwentyNewsgroupsClustering",
    "RedditClustering",
]

PAIR_CLASSIFICATION_TASKS = [
    "TwitterURLCorpus",
    "SprintDuplicateQuestions",
]

RERANKING_TASKS = [
    "AskUbuntuDupQuestions",
    "StackOverflowDupQuestions",
]

ALL_TASK_SETS: dict[str, list[str]] = {
    "retrieval": RETRIEVAL_TASKS,
    "sts": STS_TASKS,
    "clustering": CLUSTERING_TASKS,
    "pair_classification": PAIR_CLASSIFICATION_TASKS,
    "reranking": RERANKING_TASKS,
}

# Benchmark targets (stage → metric → threshold)
BENCHMARK_TARGETS: dict[str, dict[str, float]] = {
    "stage1": {
        "MSMARCO_ndcg_at_10": 0.52,
    },
    "stage2": {
        "mteb_average": 65.0,
    },
    "stage3": {
        "image_text_recall_at_1": 0.75,
        "audio_text_recall_at_1": 0.60,
    },
}


# ── Wrapper that presents an OmniVectorModel to mteb ─────────────
class _MTEBModelWrapper:
    """Thin adapter exposing the ``encode`` interface expected by mteb."""

    def __init__(
        self,
        model: Any,
        output_dim: int = 4096,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        self.model = model
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.max_length = max_length

    def encode(
        self,
        sentences: list[str],
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode *sentences* and return an ``[N, dim]`` numpy array."""
        bs = batch_size or self.batch_size
        all_embeddings: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(sentences), bs):
                batch = sentences[start: start + bs]

                if self.model.tokenizer is None:
                    raise RuntimeError(
                        "Tokenizer not loaded — cannot encode text for MTEB."
                    )

                tokens = self.model.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(next(self.model.parameters()).device)

                hidden = self.model.backbone(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                )
                emb = self.model.pooling(
                    hidden_states=hidden,
                    attention_mask=~tokens["attention_mask"].bool(),
                )
                emb = emb[:, : self.output_dim]
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                all_embeddings.append(emb.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


# ── Main runner ──────────────────────────────────────────────────────
class MTEBRunner:
    """Run MTEB benchmarks on an OmniVectorModel.

    Example::

        runner = MTEBRunner(model, output_dir="eval_results")
        results = runner.run(task_types=["retrieval", "sts"])
        runner.print_summary(results)

    Attributes:
        model: OmniVectorModel instance.
        output_dir: Directory for per-task JSON result files.
        output_dim: Embedding dimension to evaluate (MRL slice).
        batch_size: Encoding batch size.
    """

    def __init__(
        self,
        model: Any,
        output_dir: str = "./eval_results",
        output_dim: int = 4096,
        batch_size: int = 64,
    ) -> None:
        """Initialise the runner.

        Args:
            model: OmniVectorModel (or anything with a compatible backbone/pooling).
            output_dir: Where to write per-task result JSONs.
            output_dim: MRL dimension to evaluate.
            batch_size: Encoding batch size.
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        task_types: Optional[list[str]] = None,
        tasks: Optional[list[str]] = None,
        languages: Optional[list[str]] = None,
    ) -> dict[str, dict[str, float]]:
        """Run evaluation and return per-task results.

        Args:
            task_types: High-level groups — ``'retrieval'``, ``'sts'``,
                ``'clustering'``, ``'pair_classification'``, ``'reranking'``.
                Mutually exclusive with *tasks*.
            tasks: Explicit MTEB task names.  Overrides *task_types*.
            languages: Language filter (default ``['en']``).

        Returns:
            ``{task_name: {metric: value, ...}, ...}``
        """
        try:
            import mteb
        except ImportError as exc:
            raise ImportError(
                "The `mteb` package is required for evaluation.  "
                "Install it with:  pip install mteb>=1.12.0"
            ) from exc

        if languages is None:
            languages = ["en"]

        # Resolve task list
        task_names: list[str] = []
        if tasks is not None:
            task_names = list(tasks)
        elif task_types is not None:
            for tt in task_types:
                tt_lower = tt.lower()
                if tt_lower in ALL_TASK_SETS:
                    task_names.extend(ALL_TASK_SETS[tt_lower])
                else:
                    logger.warning(
                        f"Unknown task type '{tt}'.  "
                        f"Choose from: {list(ALL_TASK_SETS)}"
                    )
        else:
            # Default: retrieval only (fast)
            task_names = list(RETRIEVAL_TASKS)

        if not task_names:
            logger.warning("No tasks to evaluate.")
            return {}

        logger.info(f"Running MTEB evaluation on {len(task_names)} tasks: {task_names}")

        # Build wrapper
        wrapper = _MTEBModelWrapper(
            model=self.model,
            output_dim=self.output_dim,
            batch_size=self.batch_size,
        )

        all_results: dict[str, dict[str, float]] = {}

        for task_name in task_names:
            try:
                evaluation = mteb.MTEB(tasks=[task_name])
                task_results = evaluation.run(
                    wrapper,
                    output_folder=str(self.output_dir),
                    eval_splits=["test"],
                )

                # Extract numeric metrics
                metrics: dict[str, float] = {}
                for result in task_results:
                    if hasattr(result, "scores"):
                        for split_name, split_scores in result.scores.items():
                            for score_dict in split_scores:
                                for k, v in score_dict.items():
                                    if isinstance(v, (int, float)):
                                        metrics[f"{k}"] = float(v)
                    elif isinstance(result, dict):
                        for k, v in result.items():
                            if isinstance(v, (int, float)):
                                metrics[k] = float(v)

                all_results[task_name] = metrics
                logger.info(f"  {task_name}: {metrics}")

            except Exception as e:
                logger.warning(f"  {task_name}: FAILED — {e}")
                all_results[task_name] = {"error": str(e)}

        # Save combined results
        combined_path = self.output_dir / "mteb_results.json"
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {combined_path}")

        return all_results

    # ── Convenience helpers ──────────────────────────────────────────

    def run_retrieval(self) -> dict[str, dict[str, float]]:
        """Shortcut: run retrieval tasks only."""
        return self.run(task_types=["retrieval"])

    def run_sts(self) -> dict[str, dict[str, float]]:
        """Shortcut: run STS tasks only."""
        return self.run(task_types=["sts"])

    def run_full(self) -> dict[str, dict[str, float]]:
        """Run all supported task types."""
        return self.run(task_types=list(ALL_TASK_SETS))

    @staticmethod
    def print_summary(results: dict[str, dict[str, float]]) -> None:
        """Pretty-print evaluation results to stdout.

        Args:
            results: Output from :meth:`run`.
        """
        print("\n" + "=" * 72)
        print("MTEB Evaluation Summary")
        print("=" * 72)

        all_main_scores: list[float] = []

        for task_name, metrics in sorted(results.items()):
            if "error" in metrics:
                print(f"  {task_name:40s}  ERROR: {metrics['error']}")
                continue

            main_score = None
            for preferred in ("ndcg_at_10", "cos_sim_spearman", "accuracy", "ap", "f1"):
                if preferred in metrics:
                    main_score = metrics[preferred]
                    break
            if main_score is None and metrics:
                main_score = next(iter(metrics.values()))

            if main_score is not None:
                all_main_scores.append(main_score)
                print(f"  {task_name:40s}  {main_score:.4f}")

        if all_main_scores:
            avg = sum(all_main_scores) / len(all_main_scores)
            print("-" * 72)
            print(f"  {'Average':40s}  {avg:.4f}")
        print("=" * 72 + "\n")

    @staticmethod
    def check_targets(
        results: dict[str, dict[str, float]],
        stage: str = "stage1",
    ) -> dict[str, bool]:
        """Check whether results meet benchmark targets.

        Args:
            results: Output from :meth:`run`.
            stage: One of ``'stage1'``, ``'stage2'``, ``'stage3'``.

        Returns:
            ``{target_name: passed}``
        """
        targets = BENCHMARK_TARGETS.get(stage, {})
        outcomes: dict[str, bool] = {}

        for target_name, threshold in targets.items():
            found = False
            for task_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    composite_key = f"{task_name}_{metric_name}"
                    if composite_key == target_name or metric_name == target_name:
                        passed = float(value) >= threshold
                        outcomes[target_name] = passed
                        status = "PASS ✓" if passed else "FAIL ✗"
                        logger.info(
                            f"Target {target_name}: {value:.4f} "
                            f"(threshold {threshold}) — {status}"
                        )
                        found = True
                        break
                if found:
                    break

            if not found:
                if target_name == "mteb_average":
                    all_scores = []
                    for metrics in results.values():
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)) and k != "error":
                                all_scores.append(float(v))
                    if all_scores:
                        avg = sum(all_scores) / len(all_scores)
                        avg_pct = avg * 100 if avg < 1 else avg
                        passed = avg_pct >= threshold
                        outcomes[target_name] = passed
                        status = "PASS ✓" if passed else "FAIL ✗"
                        logger.info(
                            f"Target {target_name}: {avg_pct:.2f} "
                            f"(threshold {threshold}) — {status}"
                        )
                    else:
                        outcomes[target_name] = False
                        logger.warning(f"Target {target_name}: no scores available")
                else:
                    outcomes[target_name] = False
                    logger.warning(f"Target {target_name}: metric not found in results")

        return outcomes


# ── Lightweight internal evaluator (no mteb dependency) ──────────────
class InternalEvaluator:
    """Quick sanity-check evaluator that runs without ``mteb``.

    Computes cosine similarity between query–positive pairs and checks
    that it exceeds query–negative similarity.  Useful in CI where
    installing the full MTEB suite is expensive.
    """

    def __init__(self, model: Any, output_dim: int = 4096) -> None:
        self.model = model
        self.output_dim = output_dim

    @torch.no_grad()
    def evaluate_pairs(
        self,
        queries: list[str],
        positives: list[str],
        negatives: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Evaluate query-positive (and optionally negative) similarity.

        Args:
            queries: Query texts.
            positives: Positive passages (same length as *queries*).
            negatives: Negative passages (same length, optional).

        Returns:
            Dict with ``mean_positive_sim``, ``mean_negative_sim`` (if
            negatives given), and ``accuracy`` (fraction where
            pos_sim > neg_sim).
        """
        if len(queries) != len(positives):
            raise ValueError("queries and positives must have same length")

        wrapper = _MTEBModelWrapper(
            model=self.model,
            output_dim=self.output_dim,
        )

        q_emb = wrapper.encode(queries)
        p_emb = wrapper.encode(positives)

        pos_sims = (q_emb * p_emb).sum(axis=1)

        result: dict[str, float] = {
            "mean_positive_sim": float(np.mean(pos_sims)),
            "min_positive_sim": float(np.min(pos_sims)),
            "max_positive_sim": float(np.max(pos_sims)),
        }

        if negatives is not None:
            if len(negatives) != len(queries):
                raise ValueError("negatives must have same length as queries")
            n_emb = wrapper.encode(negatives)
            neg_sims = (q_emb * n_emb).sum(axis=1)
            result["mean_negative_sim"] = float(np.mean(neg_sims))
            result["accuracy"] = float(np.mean(pos_sims > neg_sims))

        return result
