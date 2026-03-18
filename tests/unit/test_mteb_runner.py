"""Unit tests for MTEB runner and InternalEvaluator (Fix 7).

Tests the MTEBRunner, _MTEBModelWrapper, InternalEvaluator, task set
constants, BENCHMARK_TARGETS, and helper methods — all without requiring
the heavy ``mteb`` library (mocked where needed).
"""

import json
import logging
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from omnivector.eval.mteb_runner import (
    MTEBRunner,
    InternalEvaluator,
    _MTEBModelWrapper,
    RETRIEVAL_TASKS,
    STS_TASKS,
    CLUSTERING_TASKS,
    PAIR_CLASSIFICATION_TASKS,
    RERANKING_TASKS,
    ALL_TASK_SETS,
    BENCHMARK_TARGETS,
)


# ── Task-set constants ──────────────────────────────────────────────


class TestTaskSets:
    def test_retrieval_tasks_non_empty(self):
        assert len(RETRIEVAL_TASKS) >= 3

    def test_sts_tasks_non_empty(self):
        assert len(STS_TASKS) >= 3

    def test_all_task_sets_keys(self):
        assert set(ALL_TASK_SETS) == {
            "retrieval", "sts", "clustering",
            "pair_classification", "reranking",
        }

    def test_benchmark_targets_stages(self):
        assert "stage1" in BENCHMARK_TARGETS
        assert "stage2" in BENCHMARK_TARGETS
        assert "stage3" in BENCHMARK_TARGETS


# ── _MTEBModelWrapper ────────────────────────────────────────────────


def _make_mock_model(embed_dim=128, seq_len=32):
    """Build a lightweight mock model for wrapper tests.

    The mock correctly:
    - Returns a *fresh* iterator from ``parameters()`` each call
    - Adapts backbone/pooling output to match the actual batch size
    - Returns tokenizer outputs whose batch dim matches the input list
    """
    # A real tiny parameter so `next(model.parameters()).device` works
    _param = torch.zeros(1)

    model = Mock()
    model.eval = Mock()
    model.train = Mock()
    model.tokenizer = Mock()
    model.parameters = Mock(side_effect=lambda: iter([_param]))

    # The tokenizer is called with a *list* of strings.  We need the
    # returned object's __getitem__ to return tensors whose batch dim
    # equals len(sentences).  We capture that via a side_effect on the
    # tokenizer call itself.
    class _TokOutput:
        """Mimics a BatchEncoding that tracks batch size."""
        def __init__(self, batch_size, sl):
            self._bs = batch_size
            self._sl = sl
        def to(self, device):
            return self
        def __getitem__(self, key):
            return torch.ones(self._bs, self._sl, dtype=torch.long)

    def _tokenizer_side_effect(sentences, **kwargs):
        bs = len(sentences) if isinstance(sentences, list) else 1
        return _TokOutput(bs, seq_len)

    model.tokenizer.side_effect = _tokenizer_side_effect

    # backbone / pooling return tensors whose batch dim matches input_ids
    def _backbone_side_effect(**kwargs):
        ids = kwargs.get("input_ids", torch.ones(1, seq_len))
        bs = ids.shape[0]
        return torch.randn(bs, seq_len, embed_dim)

    def _pooling_side_effect(**kwargs):
        hs = kwargs.get("hidden_states", torch.ones(1, seq_len, embed_dim))
        bs = hs.shape[0]
        return torch.randn(bs, embed_dim)

    model.backbone = Mock(side_effect=_backbone_side_effect)
    model.pooling = Mock(side_effect=_pooling_side_effect)
    return model


class TestMTEBModelWrapper:
    def test_encode_returns_numpy(self):
        model = _make_mock_model(embed_dim=128)
        wrapper = _MTEBModelWrapper(model, output_dim=128, batch_size=4)
        result = wrapper.encode(["hello", "world"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 128)

    def test_encode_batching(self):
        model = _make_mock_model(embed_dim=64)
        wrapper = _MTEBModelWrapper(model, output_dim=64, batch_size=1)
        result = wrapper.encode(["a", "b", "c"])
        assert result.shape[0] == 3
        assert model.backbone.call_count == 3

    def test_encode_l2_normalized(self):
        model = _make_mock_model(embed_dim=64)
        wrapper = _MTEBModelWrapper(model, output_dim=64, batch_size=4)
        result = wrapper.encode(["hello"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_raises_without_tokenizer(self):
        model = _make_mock_model()
        model.tokenizer = None
        wrapper = _MTEBModelWrapper(model, output_dim=128)
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            wrapper.encode(["hello"])


# ── MTEBRunner ───────────────────────────────────────────────────────


def _patch_mteb():
    """Return a context manager that injects a mock ``mteb`` module.

    ``mteb`` is imported *inside* ``MTEBRunner.run()`` via
    ``import mteb``, so we patch it in ``sys.modules``.
    """
    mock_mteb = MagicMock()
    mock_eval = MagicMock()
    mock_eval.run = MagicMock(return_value=[])
    mock_mteb.MTEB = MagicMock(return_value=mock_eval)
    return patch.dict(sys.modules, {"mteb": mock_mteb}), mock_mteb


class TestMTEBRunner:
    def test_init_creates_output_dir(self, tmp_path):
        out = tmp_path / "eval_out"
        runner = MTEBRunner(model=Mock(), output_dir=str(out))
        assert out.exists()
        assert runner.output_dim == 4096

    def test_run_defaults_to_retrieval(self, tmp_path):
        """When no task_types given, defaults to retrieval tasks."""
        patcher, mock_mteb = _patch_mteb()
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patcher:
            runner.run()
        assert mock_mteb.MTEB.call_count == len(RETRIEVAL_TASKS)

    def test_run_with_explicit_tasks(self, tmp_path):
        patcher, mock_mteb = _patch_mteb()
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patcher:
            runner.run(tasks=["FakeTask1", "FakeTask2"])
        assert mock_mteb.MTEB.call_count == 2

    def test_run_empty_on_unknown_type(self, tmp_path, caplog):
        caplog.set_level(logging.WARNING)
        patcher, _ = _patch_mteb()
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patcher:
            results = runner.run(task_types=["nonexistent_type"])
        assert results == {}
        assert "Unknown task type" in caplog.text

    def test_run_saves_json(self, tmp_path):
        mock_mteb = MagicMock()
        mock_result = MagicMock()
        mock_result.scores = {"test": [{"ndcg_at_10": 0.55}]}
        mock_eval = MagicMock()
        mock_eval.run = MagicMock(return_value=[mock_result])
        mock_mteb.MTEB = MagicMock(return_value=mock_eval)

        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patch.dict(sys.modules, {"mteb": mock_mteb}):
            runner.run(tasks=["MSMARCO"])

        json_path = tmp_path / "out" / "mteb_results.json"
        assert json_path.exists()

    def test_run_retrieval_shortcut(self, tmp_path):
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patch.object(runner, "run", return_value={}) as mock_run:
            runner.run_retrieval()
            mock_run.assert_called_once_with(task_types=["retrieval"])

    def test_run_sts_shortcut(self, tmp_path):
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patch.object(runner, "run", return_value={}) as mock_run:
            runner.run_sts()
            mock_run.assert_called_once_with(task_types=["sts"])

    def test_run_full_shortcut(self, tmp_path):
        runner = MTEBRunner(model=Mock(), output_dir=str(tmp_path / "out"))
        with patch.object(runner, "run", return_value={}) as mock_run:
            runner.run_full()
            mock_run.assert_called_once_with(task_types=list(ALL_TASK_SETS))


class TestMTEBRunnerPrintSummary:
    def test_print_summary(self, capsys):
        results = {
            "MSMARCO": {"ndcg_at_10": 0.55, "recall_at_100": 0.91},
            "SciFact": {"ndcg_at_10": 0.72},
        }
        MTEBRunner.print_summary(results)
        captured = capsys.readouterr()
        assert "MSMARCO" in captured.out
        assert "SciFact" in captured.out
        assert "Average" in captured.out

    def test_print_summary_with_error(self, capsys):
        results = {"FailedTask": {"error": "dataset not found"}}
        MTEBRunner.print_summary(results)
        captured = capsys.readouterr()
        assert "ERROR" in captured.out

    def test_print_summary_empty(self, capsys):
        MTEBRunner.print_summary({})
        captured = capsys.readouterr()
        assert "MTEB Evaluation Summary" in captured.out


class TestMTEBRunnerCheckTargets:
    def test_check_targets_pass(self):
        results = {"MSMARCO": {"ndcg_at_10": 0.60}}
        outcomes = MTEBRunner.check_targets(results, stage="stage1")
        assert outcomes["MSMARCO_ndcg_at_10"] is True

    def test_check_targets_fail(self):
        results = {"MSMARCO": {"ndcg_at_10": 0.30}}
        outcomes = MTEBRunner.check_targets(results, stage="stage1")
        assert outcomes["MSMARCO_ndcg_at_10"] is False

    def test_check_targets_missing_metric(self):
        results = {"SomeOther": {"accuracy": 0.9}}
        outcomes = MTEBRunner.check_targets(results, stage="stage1")
        assert outcomes.get("MSMARCO_ndcg_at_10") is False

    def test_check_targets_mteb_average(self):
        results = {
            "T1": {"score": 0.70},
            "T2": {"score": 0.68},
        }
        outcomes = MTEBRunner.check_targets(results, stage="stage2")
        # average is 0.69 → 69% → passes 65.0 threshold
        assert outcomes["mteb_average"] is True


# ── InternalEvaluator ────────────────────────────────────────────────


class TestInternalEvaluator:
    def test_evaluate_pairs_returns_metrics(self):
        model = _make_mock_model(embed_dim=64)
        evaluator = InternalEvaluator(model, output_dim=64)
        result = evaluator.evaluate_pairs(
            queries=["q1", "q2"],
            positives=["p1", "p2"],
        )
        assert "mean_positive_sim" in result
        assert "min_positive_sim" in result
        assert "max_positive_sim" in result
        assert isinstance(result["mean_positive_sim"], float)

    def test_evaluate_pairs_with_negatives(self):
        model = _make_mock_model(embed_dim=64)
        evaluator = InternalEvaluator(model, output_dim=64)
        result = evaluator.evaluate_pairs(
            queries=["q1", "q2"],
            positives=["p1", "p2"],
            negatives=["n1", "n2"],
        )
        assert "mean_negative_sim" in result
        assert "accuracy" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_pairs_length_mismatch(self):
        model = _make_mock_model(embed_dim=64)
        evaluator = InternalEvaluator(model, output_dim=64)
        with pytest.raises(ValueError, match="same length"):
            evaluator.evaluate_pairs(queries=["q1"], positives=["p1", "p2"])

    def test_evaluate_pairs_negatives_length_mismatch(self):
        model = _make_mock_model(embed_dim=64)
        evaluator = InternalEvaluator(model, output_dim=64)
        with pytest.raises(ValueError, match="same length"):
            evaluator.evaluate_pairs(
                queries=["q1", "q2"],
                positives=["p1", "p2"],
                negatives=["n1"],
            )
