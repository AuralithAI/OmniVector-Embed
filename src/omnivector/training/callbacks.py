"""Training callbacks for OmniVector model."""

import logging
from typing import Optional

import numpy as np
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)


class HardNegativeRefreshCallback(TrainerCallback):
    """Refresh hard negatives at regular intervals during training.

    Implements curriculum learning by periodically re-mining hard negatives
    using the **current model's representations** (not stale cached embeddings).
    The model is used to re-encode the corpus, then FAISS is rebuilt and
    negatives are re-mined with the up-to-date similarity landscape.
    """

    def __init__(
        self,
        refresh_steps: int = 5000,
        miner=None,
        corpus_texts: Optional[list] = None,
        corpus_ids: Optional[list] = None,
        train_dataset=None,
        tokenizer=None,
        encode_batch_size: int = 64,
        max_seq_length: int = 512,
        device: str = "cpu",
    ):
        """Initialize callback.

        Args:
            refresh_steps: Re-mine hard negatives every N steps.
            miner: Optional HardNegativeMiner instance for mining.
            corpus_texts: List of corpus text strings for re-encoding.
            corpus_ids: List of corpus IDs corresponding to texts.
            train_dataset: Reference to training dataset for updating negatives.
            tokenizer: Tokenizer for encoding corpus with current model.
            encode_batch_size: Batch size for corpus re-encoding.
            max_seq_length: Max sequence length for tokenisation.
            device: Device for encoding ('cpu' or 'cuda').
        """
        self.refresh_steps = refresh_steps
        self.miner = miner
        self.corpus_texts = corpus_texts
        self.corpus_ids = corpus_ids
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.encode_batch_size = encode_batch_size
        self.max_seq_length = max_seq_length
        self.device = device
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
            **kwargs: Additional callback arguments (must contain ``model``).
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

    @torch.no_grad()
    def _encode_corpus(self, model) -> "np.ndarray":
        """Re-encode the full corpus with the current model weights.

        Args:
            model: Current OmniVectorModel.

        Returns:
            Numpy array of shape [num_corpus, embed_dim].
        """
        import numpy as np

        model.eval()

        all_embeddings = []
        for start in range(0, len(self.corpus_texts), self.encode_batch_size):
            batch_texts = self.corpus_texts[start : start + self.encode_batch_size]

            tokens = self.tokenizer(
                batch_texts,
                max_length=self.max_seq_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            hidden = model.backbone(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            embeddings = model.pooling(
                hidden_states=hidden,
                attention_mask=~tokens["attention_mask"].bool(),
            )

            # L2 normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
            all_embeddings.append(embeddings.cpu().numpy())

        model.train()
        return np.concatenate(all_embeddings, axis=0).astype(np.float32)

    def _refresh_negatives(self, model=None):
        """Re-mine hard negatives using current model state.

        If a model and tokenizer are available the corpus is **re-encoded**
        with the current weights so that the FAISS index reflects the
        model's evolving representation.  Falls back to cached embeddings
        only when re-encoding is not possible.

        Args:
            model: Current OmniVectorModel instance.
        """
        if self.corpus_texts is None or self.train_dataset is None:
            logger.warning("Skipping refresh: corpus_texts or train_dataset not set")
            return

        import numpy as np

        # Re-encode corpus with current model if possible
        if model is not None and self.tokenizer is not None:
            logger.info("Re-encoding corpus with current model weights...")
            new_embeddings = self._encode_corpus(model)
            self.miner.corpus_embeddings = new_embeddings
            self.miner.index = self.miner._build_index()
            logger.info(f"FAISS index rebuilt with {len(new_embeddings)} fresh embeddings")
        else:
            # Fallback: rebuild index from whatever embeddings are cached
            logger.warning(
                "Cannot re-encode corpus (model or tokenizer missing). "
                "Rebuilding FAISS index from cached embeddings."
            )
            self.miner.index = self.miner._build_index()

        refreshed = 0
        for sample in self.train_dataset:
            if not hasattr(sample, "query_embedding"):
                continue
            query_emb = np.array(sample.query_embedding, dtype=np.float32).reshape(1, -1)
            neg_ids = self.miner.mine(
                query_emb.squeeze(),
                positive_id=getattr(sample, "positive_id", -1),
                positive_score=getattr(sample, "positive_score", 1.0),
            )
            if neg_ids:
                sample.negatives = [
                    self.corpus_texts[nid] if nid < len(self.corpus_texts) else str(nid)
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
            logger.info(f"No improvement for {self.patience_counter}/{self.patience} evaluations")

            if self.patience_counter >= self.patience:
                control.should_training_stop = True
                logger.info("Stopping training due to early stopping criterion")
