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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# ── Default task sets ────────────────────────────────────────────────

RETRIEVAL_TASKS = [
    "MSMARCO",
    "NFCorpus",
    "SciFact",
    "ArguAna",
    "FiQA2018",
    "TRECCOVID",
    "Touche2020",
    "NQ",
    "DBPedia",
    "FEVER",
    "ClimateFEVER",
    "HotpotQA",
    "CQADupstackRetrieval",
    "QuoraRetrieval",
]

STS_TASKS = [
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STSBenchmark",
    "SICK-R",
    "BIOSSES",
]

CLUSTERING_TASKS = [
    "TwentyNewsgroupsClustering",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "ArXivClusteringP2P",
    "ArXivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
]

PAIR_CLASSIFICATION_TASKS = [
    "TwitterURLCorpus",
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
]

RERANKING_TASKS = [
    "AskUbuntuDupQuestions",
    "StackOverflowDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
]

CLASSIFICATION_TASKS = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

SUMMARIZATION_TASKS = [
    "SummEval",
]

ALL_TASK_SETS: dict[str, list[str]] = {
    "retrieval": RETRIEVAL_TASKS,
    "sts": STS_TASKS,
    "clustering": CLUSTERING_TASKS,
    "pair_classification": PAIR_CLASSIFICATION_TASKS,
    "reranking": RERANKING_TASKS,
    "classification": CLASSIFICATION_TASKS,
    "summarization": SUMMARIZATION_TASKS,
}

# Full MTEB English benchmark — all 56 tasks for leaderboard score
FULL_MTEB_TASKS: list[str] = []
for _task_list in ALL_TASK_SETS.values():
    FULL_MTEB_TASKS.extend(_task_list)

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


# ── Wrapper that presents an OmniVectorModel to mteb ──────
class _MTEBModelWrapper:
    """Adapter implementing the MTEB ``EncoderProtocol`` for OmniVectorModel.

    MTEB requires:
      - ``encode(inputs: DataLoader[BatchedInput], *, task_metadata, ...)``
      - ``similarity`` / ``similarity_pairwise``
      - ``mteb_model_meta`` property
    """

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

        # Lazy-initialised ModelMeta — created once on first access
        self._model_meta: Any | None = None

    # -- mteb_model_meta (required by EncoderProtocol) ----------------
    @property
    def mteb_model_meta(self) -> Any:
        if self._model_meta is None:
            from mteb.models.model_meta import ModelMeta

            self._model_meta = ModelMeta.create_empty(
                overwrites={
                    "name": "omnivector-embed",
                    "revision": "local",
                    "loader": None,  # already loaded
                }
            )
        return self._model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value: Any) -> None:
        self._model_meta = value

    # -- encode (new DataLoader-based signature) ----------------------
    def encode(
        self,
        inputs: "DataLoader[Any]",
        *,
        task_metadata: Any = None,
        hf_split: str = "",
        hf_subset: str = "",
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode text batches from an MTEB ``DataLoader[BatchedInput]``.

        Each batch is a dict with at least a ``"text"`` key containing
        ``list[str]``.  Returns ``[N, dim]`` numpy array.
        """
        all_embeddings: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for batch in inputs:
                # BatchedInput is a dict; grab the text list
                sentences: list[str] = batch["text"]
                if not sentences:
                    continue

                if self.model.tokenizer is None:
                    raise RuntimeError(
                        "Tokenizer not loaded — cannot encode text for MTEB."
                    )

                tokens = self.model.tokenizer(
                    sentences,
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

        if not all_embeddings:
            return np.empty((0, self.output_dim), dtype=np.float32)
        return np.concatenate(all_embeddings, axis=0)

    # -- similarity helpers (required by EncoderProtocol) -------------
    @staticmethod
    def similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> torch.Tensor:
        """Cosine similarity matrix ``[N, M]``."""
        e1 = torch.from_numpy(np.asarray(embeddings1, dtype=np.float32))
        e2 = torch.from_numpy(np.asarray(embeddings2, dtype=np.float32))
        e1 = torch.nn.functional.normalize(e1, p=2, dim=-1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=-1)
        return e1 @ e2.T

    @staticmethod
    def similarity_pairwise(
        embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> torch.Tensor:
        """Element-wise cosine similarity ``[N]``."""
        e1 = torch.from_numpy(np.asarray(embeddings1, dtype=np.float32))
        e2 = torch.from_numpy(np.asarray(embeddings2, dtype=np.float32))
        e1 = torch.nn.functional.normalize(e1, p=2, dim=-1)
        e2 = torch.nn.functional.normalize(e2, p=2, dim=-1)
        return (e1 * e2).sum(dim=-1)

    # -- Legacy encode (used by InternalEvaluator) --------------------
    def encode_sentences(
        self,
        sentences: list[str],
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode a plain list of strings (no DataLoader)."""
        bs = batch_size or self.batch_size
        all_embeddings: list[np.ndarray] = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(sentences), bs):
                batch_sents = sentences[start : start + bs]

                if self.model.tokenizer is None:
                    raise RuntimeError(
                        "Tokenizer not loaded — cannot encode text for MTEB."
                    )

                tokens = self.model.tokenizer(
                    batch_sents,
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
                    logger.warning(f"Unknown task type '{tt}'.  Choose from: {list(ALL_TASK_SETS)}")
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
                # MTEB 2.10+: use mteb.get_tasks + mteb.evaluate
                task_objs = mteb.get_tasks(tasks=[task_name])
                if not task_objs:
                    logger.warning(f"  {task_name}: task not found in MTEB registry")
                    all_results[task_name] = {"error": "task not found"}
                    continue

                logger.info(f"  Evaluating {task_name}...")
                model_result = mteb.evaluate(
                    wrapper,
                    tasks=task_objs,
                    raise_error=False,
                    encode_kwargs={"batch_size": self.batch_size},
                    overwrite_strategy="always",
                    cache=None,
                )

                # Extract numeric metrics from ModelResult
                metrics: dict[str, float] = {}
                for task_result in model_result.task_results:
                    if hasattr(task_result, "scores"):
                        for _split_name, split_scores in task_result.scores.items():
                            for score_dict in split_scores:
                                for k, v in score_dict.items():
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
        """Pretty-print evaluation results to stdout with MTEB-style scoring.

        Computes per-category averages and overall MTEB score (average of
        category averages), matching the NV-Embed v2 leaderboard methodology.

        Args:
            results: Output from :meth:`run`.
        """
        # Map task → category for per-category scoring
        task_to_category: dict[str, str] = {}
        for category, task_list in ALL_TASK_SETS.items():
            for task in task_list:
                task_to_category[task] = category

        print("\n" + "=" * 72)
        print("MTEB Evaluation Summary")
        print("=" * 72)

        category_scores: dict[str, list[float]] = {}

        for task_name, metrics in sorted(results.items()):
            if "error" in metrics:
                print(f"  {task_name:40s}  ERROR: {metrics['error']}")
                continue

            # Pick the main metric for each task type
            main_score = None
            for preferred in (
                "ndcg_at_10",
                "cos_sim_spearman",
                "v_measure",
                "accuracy",
                "ap",
                "map",
                "cos_sim_pearson",
                "f1",
            ):
                if preferred in metrics:
                    main_score = metrics[preferred]
                    break
            if main_score is None and metrics:
                main_score = next(iter(metrics.values()))

            if main_score is not None:
                category = task_to_category.get(task_name, "other")
                category_scores.setdefault(category, []).append(main_score)
                # Display as percentage if < 1 (MTEB convention)
                display = main_score * 100 if main_score < 1 else main_score
                print(f"  {task_name:40s}  {display:.2f}")

        # Per-category averages
        print("-" * 72)
        print("  Category Averages:")
        overall_category_avgs: list[float] = []
        for category in ALL_TASK_SETS:
            scores = category_scores.get(category, [])
            if scores:
                avg = sum(scores) / len(scores)
                avg_pct = avg * 100 if avg < 1 else avg
                overall_category_avgs.append(avg_pct)
                print(f"    {category:38s}  {avg_pct:.2f}  ({len(scores)} tasks)")

        # Overall MTEB score = average of category averages
        if overall_category_avgs:
            mteb_score = sum(overall_category_avgs) / len(overall_category_avgs)
            print("-" * 72)
            print(f"  {'MTEB Overall Score':40s}  {mteb_score:.2f}")
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
                            f"Target {target_name}: {value:.4f} (threshold {threshold}) — {status}"
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

        q_emb = wrapper.encode_sentences(queries)
        p_emb = wrapper.encode_sentences(positives)

        pos_sims = (q_emb * p_emb).sum(axis=1)

        result: dict[str, float] = {
            "mean_positive_sim": float(np.mean(pos_sims)),
            "min_positive_sim": float(np.min(pos_sims)),
            "max_positive_sim": float(np.max(pos_sims)),
        }

        if negatives is not None:
            if len(negatives) != len(queries):
                raise ValueError("negatives must have same length as queries")
            n_emb = wrapper.encode_sentences(negatives)
            neg_sims = (q_emb * n_emb).sum(axis=1)
            result["mean_negative_sim"] = float(np.mean(neg_sims))
            result["accuracy"] = float(np.mean(pos_sims > neg_sims))

        return result
