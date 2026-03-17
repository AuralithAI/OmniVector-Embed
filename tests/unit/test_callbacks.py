"""Unit tests for training callbacks."""

import pytest
from unittest.mock import Mock, MagicMock

from omnivector.training.callbacks import (
    HardNegativeRefreshCallback,
    LoggingCallback,
    EarlyStoppingCallback,
)


class TestHardNegativeRefreshCallback:
    """Test HardNegativeRefreshCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = HardNegativeRefreshCallback(refresh_steps=1000)
        assert callback.refresh_steps == 1000
        assert callback.last_refresh_step == 0
        assert callback.miner is None
        assert callback.corpus_embeddings is None
        assert callback.train_dataset is None

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
