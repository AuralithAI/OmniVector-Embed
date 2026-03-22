"""Unit tests for training infrastructure."""

from unittest.mock import Mock

import pytest
import torch

from omnivector.training.trainer import OmniVectorTrainer


def _make_trainer(**overrides):
    """Create OmniVectorTrainer without invoking HF Trainer.__init__.

    Uses object.__new__ to bypass the heavy __init__ so unit tests
    can exercise individual methods in isolation.
    """
    trainer = object.__new__(OmniVectorTrainer)
    trainer.model = overrides.get("model", Mock())
    trainer.args = overrides.get("args", Mock(output_dir="./checkpoints"))
    trainer.state = overrides.get("state", Mock(global_step=0))
    trainer.control = overrides.get("control", Mock(should_log=False, should_save=False))
    trainer.optimizer = overrides.get("optimizer", Mock(param_groups=[{"lr": 1e-4}]))
    trainer.train_dataset = overrides.get("train_dataset", [])
    return trainer


class TestOmniVectorTrainer:
    """Test OmniVectorTrainer."""

    @pytest.fixture
    def trainer(self):
        """Create lightweight trainer for testing."""
        return _make_trainer()

    def test_trainer_inheritance(self):
        """Test that OmniVectorTrainer inherits from HF Trainer."""
        from transformers import Trainer

        assert issubclass(OmniVectorTrainer, Trainer)

    def test_compute_loss_with_loss_attr(self, trainer):
        """Test compute_loss with model output that has .loss attribute."""
        model = Mock()
        output = Mock()
        output.loss = torch.tensor(2.5)

        model.return_value = output
        inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        loss = trainer.compute_loss(model, inputs)

        assert loss == output.loss

    def test_compute_loss_returns_outputs(self, trainer):
        """Test compute_loss with return_outputs=True."""
        model = Mock()
        output = Mock()
        output.loss = torch.tensor(2.5)
        model.return_value = output

        inputs = {"input_ids": Mock()}
        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

        assert loss == output.loss
        assert outputs is output

    def test_compute_loss_removes_labels(self, trainer):
        """Test that compute_loss strips labels before forwarding."""
        model = Mock()
        output = Mock()
        output.loss = torch.tensor(1.0)
        model.return_value = output

        inputs = {"input_ids": Mock(), "labels": Mock()}
        trainer.compute_loss(model, inputs)

        assert "labels" not in inputs

    def test_compute_loss_handles_tuple_output(self, trainer):
        """Test compute_loss when model returns a tuple."""
        loss_tensor = torch.tensor(2.5)
        result = (loss_tensor, Mock())
        model = Mock(return_value=result)

        inputs = {"input_ids": Mock()}
        loss = trainer.compute_loss(model, inputs)

        assert loss is loss_tensor

    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving creates correct path and trainer state."""
        model = Mock()
        args = Mock(output_dir=str(tmp_path), save_total_limit=None)
        trainer = _make_trainer(model=model, args=args)
        trainer.state.global_step = 1000

        trainer._save_checkpoint(model, None)
        model.save_pretrained.assert_called_once_with(f"{tmp_path}/checkpoint-1000")
        trainer.state.save_to_json.assert_called_once_with(
            f"{tmp_path}/checkpoint-1000/trainer_state.json"
        )

    def test_multiple_checkpoint_saves(self, tmp_path):
        """Test sequential checkpoint saves."""
        model = Mock()
        args = Mock(output_dir=str(tmp_path), save_total_limit=None)
        trainer = _make_trainer(model=model, args=args)

        trainer.state.global_step = 1000
        trainer._save_checkpoint(model, None)

        trainer.state.global_step = 2000
        trainer._save_checkpoint(model, None)

        assert model.save_pretrained.call_count == 2

    def test_get_learning_rate(self, trainer):
        """Test learning rate retrieval from optimizer."""
        lr = trainer._get_learning_rate()
        assert lr == 1e-4

    def test_dataset_access(self, trainer):
        """Test train_dataset is accessible."""
        trainer.train_dataset = [{"query": "test", "positive": "doc"}]

        assert len(trainer.train_dataset) == 1
        assert trainer.train_dataset[0]["query"] == "test"
