"""Training callbacks for OmniVector model."""

import logging
from typing import Optional

from transformers import TrainerCallback, TrainerState, TrainerControl

logger = logging.getLogger(__name__)


class HardNegativeRefreshCallback(TrainerCallback):
    """Refresh hard negatives at regular intervals during training.
    
    Implements curriculum learning by periodically re-mining hard negatives
    using the current model state.
    """

    def __init__(
        self,
        refresh_steps: int = 5000,
        miner=None,
        corpus_embeddings=None,
        corpus_ids=None,
        train_dataset=None,
    ):
        """Initialize callback.
        
        Args:
            refresh_steps: Re-mine hard negatives every N steps.
            miner: Optional HardNegativeMiner instance for mining.
            corpus_embeddings: Numpy array of corpus embeddings for re-indexing.
            corpus_ids: List of corpus IDs corresponding to embeddings.
            train_dataset: Reference to training dataset for updating negatives.
        """
        self.refresh_steps = refresh_steps
        self.miner = miner
        self.corpus_embeddings = corpus_embeddings
        self.corpus_ids = corpus_ids
        self.train_dataset = train_dataset
        self.last_refresh_step = 0

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Check if hard negative refresh is due.
        
        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            **kwargs: Additional callback arguments.
        """
        if self.miner is None:
            return

        current_step = state.global_step
        steps_since_refresh = current_step - self.last_refresh_step

        if steps_since_refresh >= self.refresh_steps:
            logger.info(
                f"Refreshing hard negatives at step {current_step} "
                f"(last refresh at {self.last_refresh_step})"
            )
            self._refresh_negatives(kwargs.get("model"))
            self.last_refresh_step = current_step

    def _refresh_negatives(self, model=None):
        """Re-mine hard negatives using current model state.
        
        Args:
            model: Current model for re-encoding (optional, uses cached if None).
        """
        if self.corpus_embeddings is None or self.train_dataset is None:
            logger.warning("Skipping refresh: corpus_embeddings or train_dataset not set")
            return

        import numpy as np

        self.miner.corpus_embeddings = self.corpus_embeddings
        self.miner.corpus_ids = self.corpus_ids
        self.miner._build_index()

        refreshed = 0
        for sample in self.train_dataset:
            if not hasattr(sample, "query_embedding"):
                continue
            query_emb = np.array(sample.query_embedding, dtype=np.float32).reshape(1, -1)
            neg_ids = self.miner.mine(
                query_emb,
                positive_id=getattr(sample, "positive_id", -1),
                positive_score=getattr(sample, "positive_score", 1.0),
            )
            if neg_ids:
                sample.negatives = [
                    self.corpus_ids[nid] if self.corpus_ids else str(nid)
                    for nid in neg_ids
                ]
                refreshed += 1

        logger.info(f"Refreshed negatives for {refreshed} samples")


class LoggingCallback(TrainerCallback):
    """Log training metrics at each step."""

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        """Log metrics when trainer logs.
        
        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            logs: Dictionary of metrics to log.
            **kwargs: Additional arguments.
        """
        if logs is None:
            return

        if "loss" in logs:
            logger.info(
                f"Step {state.global_step}: "
                f"loss={logs['loss']:.4f}, "
                f"lr={logs.get('learning_rate', 0):.2e}"
            )


class EarlyStoppingCallback(TrainerCallback):
    """Stop training if loss plateaus."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4):
        """Initialize callback.
        
        Args:
            patience: Number of evaluations without improvement before stopping.
            min_delta: Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.patience_counter = 0

    def on_evaluate(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        metrics=None,
        **kwargs,
    ):
        """Check for early stopping criterion.
        
        Args:
            args: Training arguments.
            state: Current trainer state.
            control: Trainer control object.
            metrics: Dictionary of evaluation metrics.
            **kwargs: Additional arguments.
        """
        if metrics is None or "eval_loss" not in metrics:
            return

        eval_loss = metrics["eval_loss"]

        if self.best_loss is None:
            self.best_loss = eval_loss
            logger.info(f"Initial best loss: {self.best_loss:.4f}")
        elif eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.patience_counter = 0
            logger.info(f"New best loss: {self.best_loss:.4f}")
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement for {self.patience_counter}/{self.patience} evaluations"
            )

            if self.patience_counter >= self.patience:
                control.should_training_stop = True
                logger.info("Stopping training due to early stopping criterion")
