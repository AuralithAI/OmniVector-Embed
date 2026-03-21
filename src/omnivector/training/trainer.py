"""Custom trainer for OmniVector embedding model."""

import logging

from transformers import Trainer

logger = logging.getLogger(__name__)


class OmniVectorTrainer(Trainer):
    """HF Trainer subclass with custom loss computation for MRL.

    Integrates with DeepSpeed ZeRO-2 for distributed training and
    supports hard negative mining callbacks for curriculum learning.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute MRL InfoNCE loss across all dimensions.

        The model's forward pass handles MRL loss computation internally.
        This method extracts the loss value and returns it in the format
        expected by the HF Trainer.

        Args:
            model: The model being trained.
            inputs: Batch of training inputs.
            return_outputs: Whether to return model outputs alongside loss.

        Returns:
            Loss value or tuple of (loss, outputs) if return_outputs=True.
        """
        if "labels" in inputs:
            del inputs["labels"]

        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        """Log, save, and evaluate with custom loss formatting.

        Extends the base trainer to format loss output and handle
        model checkpointing.

        Args:
            tr_loss: Training loss accumulation.
            grad_norm: Gradient norm from training step.
            model: Model being trained.
            trial: Optional trial object for hyperparameter tuning.
            epoch: Current epoch.
            ignore_keys_for_eval: Keys to ignore during evaluation.
        """
        if self.control.should_log:
            logs: dict = {}
            tr_loss_scalar = (tr_loss / max(1, self.state.global_step)).item()
            logs["loss"] = round(tr_loss_scalar, 4)
            logs["learning_rate"] = self._get_learning_rate()

            self.log(logs)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=None)

    def _get_learning_rate(self) -> float:
        """Get current learning rate from optimizer.

        Retrieves the learning rate from the first parameter group
        of the optimizer.

        Returns:
            Current learning rate as float.
        """
        return self.optimizer.param_groups[0]["lr"]

    def _save_checkpoint(self, model, trial, metrics=None):
        """Save model checkpoint.

        Args:
            model: Model to save.
            trial: Optional trial for naming.
            metrics: Optional metrics to track best checkpoint.
        """
        checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"
        self.model.save_pretrained(checkpoint_folder)
        logger.info(f"Saved checkpoint to {checkpoint_folder}")
