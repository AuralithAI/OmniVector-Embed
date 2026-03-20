"""Unit tests for training callbacks."""

import logging
from unittest.mock import Mock

import numpy as np
import torch

from omnivector.training.callbacks import (
    EarlyStoppingCallback,
    HardNegativeRefreshCallback,
    LoggingCallback,
)


class TestHardNegativeRefreshCallback:
    """Test HardNegativeRefreshCallback."""

    def test_initialization(self):
        """Test callback initialization with new signature."""
        callback = HardNegativeRefreshCallback(refresh_steps=1000)
        assert callback.refresh_steps == 1000
        assert callback.last_refresh_step == 0
        assert callback.miner is None
        assert callback.corpus_texts is None
        assert callback.corpus_ids is None
        assert callback.train_dataset is None
        assert callback.tokenizer is None
        assert callback.encode_batch_size == 64
        assert callback.max_seq_length == 512
        assert callback.device == "cpu"

    def test_initialization_with_all_params(self):
        """Test initialization with all new parameters."""
        miner = Mock()
        tokenizer = Mock()
        corpus = ["text a", "text b"]
        ids = [0, 1]
        dataset = [Mock()]
        callback = HardNegativeRefreshCallback(
            refresh_steps=500,
            miner=miner,
            corpus_texts=corpus,
            corpus_ids=ids,
            train_dataset=dataset,
            tokenizer=tokenizer,
            encode_batch_size=32,
            max_seq_length=256,
            device="cuda",
        )
        assert callback.miner is miner
        assert callback.refresh_steps == 500
        assert callback.corpus_texts is corpus
        assert callback.corpus_ids is ids
        assert callback.tokenizer is tokenizer
        assert callback.encode_batch_size == 32
        assert callback.max_seq_length == 256
        assert callback.device == "cuda"

    def test_initialization_with_miner(self):
        """Test initialization with miner instance."""
        miner = Mock()
        callback = HardNegativeRefreshCallback(refresh_steps=500, miner=miner)
        assert callback.miner is miner
        assert callback.refresh_steps == 500

    def test_refresh_triggered_at_interval(self):
        """Test that refresh is triggered at correct step interval."""
        miner = Mock()
        callback = HardNegativeRefreshCallback(refresh_steps=1000, miner=miner)

        state = Mock()
        state.global_step = 1000
        control = Mock()

        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 1000

    def test_no_refresh_before_interval(self):
        """Test that refresh is not triggered before interval."""
        miner = Mock()
        callback = HardNegativeRefreshCallback(refresh_steps=1000, miner=miner)

        state = Mock()
        state.global_step = 500
        control = Mock()

        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 0

    def test_no_refresh_without_miner(self):
        """Test that callback does nothing without miner."""
        callback = HardNegativeRefreshCallback(refresh_steps=1000, miner=None)

        state = Mock()
        state.global_step = 1000
        control = Mock()

        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 0

    def test_multiple_refresh_cycles(self):
        """Test multiple refresh cycles."""
        miner = Mock()
        callback = HardNegativeRefreshCallback(refresh_steps=500, miner=miner)

        state = Mock()
        control = Mock()

        state.global_step = 500
        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 500

        state.global_step = 1000
        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 1000

        state.global_step = 1200
        callback.on_step_end(Mock(), state, control)
        assert callback.last_refresh_step == 1000

    def test_refresh_skips_without_corpus_texts(self, caplog):
        """_refresh_negatives skips when corpus_texts is None."""
        caplog.set_level(logging.WARNING)
        miner = Mock()
        callback = HardNegativeRefreshCallback(
            refresh_steps=100,
            miner=miner,
            corpus_texts=None,
            train_dataset=[Mock()],
        )
        callback._refresh_negatives(model=Mock())
        assert "Skipping refresh" in caplog.text

    def test_refresh_skips_without_train_dataset(self, caplog):
        """_refresh_negatives skips when train_dataset is None."""
        caplog.set_level(logging.WARNING)
        miner = Mock()
        callback = HardNegativeRefreshCallback(
            refresh_steps=100,
            miner=miner,
            corpus_texts=["a", "b"],
            train_dataset=None,
        )
        callback._refresh_negatives(model=Mock())
        assert "Skipping refresh" in caplog.text

    def test_refresh_rebuilds_index_without_model(self, caplog):
        """Falls back to cached embeddings when model is None."""
        caplog.set_level(logging.WARNING)
        miner = Mock()
        sample = Mock()
        sample.query_embedding = None
        delattr(sample, "query_embedding")

        callback = HardNegativeRefreshCallback(
            refresh_steps=100,
            miner=miner,
            corpus_texts=["a", "b"],
            train_dataset=[sample],
            tokenizer=None,  # no tokenizer → cannot re-encode
        )
        callback._refresh_negatives(model=Mock())
        miner._build_index.assert_called_once()
        assert "Cannot re-encode" in caplog.text

    def test_encode_corpus_produces_embeddings(self):
        """_encode_corpus returns numpy array with correct shape."""
        embed_dim = 128
        corpus = ["hello", "world", "test"]

        # Mock model
        model = Mock()
        model.eval = Mock()
        model.train = Mock()
        model.backbone = Mock(return_value=torch.randn(3, 32, embed_dim))
        model.pooling = Mock(return_value=torch.randn(3, embed_dim))

        # Mock tokenizer
        tokenizer = Mock()
        tok_output = Mock()
        tok_output.to = Mock(return_value=tok_output)
        tok_output.__getitem__ = lambda self, key: torch.ones(3, 32, dtype=torch.long)
        tokenizer.return_value = tok_output

        callback = HardNegativeRefreshCallback(
            refresh_steps=100,
            miner=Mock(),
            corpus_texts=corpus,
            train_dataset=[],
            tokenizer=tokenizer,
            encode_batch_size=10,
        )

        embeddings = callback._encode_corpus(model)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 3
        assert embeddings.dtype == np.float32
        model.eval.assert_called_once()
        model.train.assert_called_once()


class TestLoggingCallback:
    """Test LoggingCallback."""

    def test_logs_loss_and_lr(self, caplog):
        """Test logging of loss and learning rate."""
        import logging

        caplog.set_level(logging.INFO)

        callback = LoggingCallback()
        state = Mock()
        state.global_step = 100

        logs = {"loss": 2.5, "learning_rate": 0.0001}
        callback.on_log(Mock(), state, Mock(), logs=logs)

        assert "loss=2.5000" in caplog.text
        assert "lr=1.00e-04" in caplog.text

    def test_handles_missing_logs(self, caplog):
        """Test that callback handles None logs gracefully."""
        import logging

        caplog.set_level(logging.INFO)

        callback = LoggingCallback()
        callback.on_log(Mock(), Mock(), Mock(), logs=None)

    def test_logs_only_when_loss_present(self, caplog):
        """Test that callback only logs when loss is present."""
        import logging

        caplog.set_level(logging.INFO)

        callback = LoggingCallback()
        state = Mock()
        state.global_step = 50

        logs = {"learning_rate": 0.0001}
        callback.on_log(Mock(), state, Mock(), logs=logs)

        assert "Step 50" not in caplog.text


class TestEarlyStoppingCallback:
    """Test EarlyStoppingCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        assert callback.patience == 3
        assert callback.min_delta == 1e-4
        assert callback.best_loss is None
        assert callback.patience_counter == 0

    def test_first_evaluation_sets_baseline(self, caplog):
        """Test that first evaluation sets baseline loss."""
        import logging

        caplog.set_level(logging.INFO)

        callback = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        state = Mock()
        state.global_step = 0
        control = Mock()

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        assert callback.best_loss == 5.0
        assert callback.patience_counter == 0
        assert "Initial best loss: 5.0000" in caplog.text

    def test_improvement_resets_patience(self, caplog):
        """Test that improvement resets patience counter."""
        import logging

        caplog.set_level(logging.INFO)

        callback = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        state = Mock()
        control = Mock()

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        metrics = {"eval_loss": 4.9}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        assert callback.best_loss == 4.9
        assert callback.patience_counter == 0

    def test_no_improvement_increments_patience(self, caplog):
        """Test that no improvement increments patience counter."""
        import logging

        caplog.set_level(logging.INFO)

        callback = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        state = Mock()
        control = Mock()
        control.should_training_stop = False

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        assert callback.patience_counter == 1
        assert control.should_training_stop is False

    def test_early_stopping_triggered(self, caplog):
        """Test that early stopping is triggered after patience exceeded."""
        import logging

        caplog.set_level(logging.INFO)

        callback = EarlyStoppingCallback(patience=2, min_delta=1e-4)
        state = Mock()
        control = Mock()
        control.should_training_stop = False

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)
        assert control.should_training_stop is False

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)
        assert control.should_training_stop is True
        assert "Stopping training" in caplog.text

    def test_min_delta_threshold(self):
        """Test that min_delta threshold is respected."""
        callback = EarlyStoppingCallback(patience=3, min_delta=0.1)
        state = Mock()
        control = Mock()

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        metrics = {"eval_loss": 4.95}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)

        assert callback.best_loss == 5.0
        assert callback.patience_counter == 1

    def test_handles_missing_metrics(self):
        """Test that callback handles missing eval_loss gracefully."""
        callback = EarlyStoppingCallback(patience=3)
        control = Mock()

        callback.on_evaluate(Mock(), Mock(), control, metrics=None)
        assert callback.best_loss is None

        callback.on_evaluate(Mock(), Mock(), control, metrics={})
        assert callback.best_loss is None

    def test_sequential_patience_tracking(self):
        """Test patience tracking across multiple evaluations."""
        callback = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        state = Mock()
        control = Mock()

        metrics = {"eval_loss": 5.0}
        callback.on_evaluate(Mock(), state, control, metrics=metrics)
        assert callback.patience_counter == 0

        for _ in range(3):
            metrics = {"eval_loss": 5.0}
            callback.on_evaluate(Mock(), state, control, metrics=metrics)

        assert callback.patience_counter == 3
        assert control.should_training_stop is True
