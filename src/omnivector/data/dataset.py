"""
Dataset and DataCollator for embedding training.

Provides:
- EmbeddingDataset: PyTorch Dataset for embedding pairs
- EmbeddingDataCollator: Collates batches with variable negatives
"""

import logging
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

from omnivector.data.schema import EmbeddingPair

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """
    PyTorch Dataset for embedding training.

    Stores training examples and provides access by index.

    Attributes:
        pairs: List of EmbeddingPair objects
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        pairs: list[EmbeddingPair],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> None:
        """
        Initialize dataset.

        Args:
            pairs: List of EmbeddingPair objects
            tokenizer: Pre-initialized tokenizer
            max_length: Maximum token sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

        logger.info(
            f"EmbeddingDataset initialized with {len(pairs)} pairs, max_length={max_length}"
        )

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get single example.

        Args:
            idx: Example index

        Returns:
            Dictionary with tokenized query, positive, negatives, and metadata
        """
        pair = self.pairs[idx]

        # Prepare texts with instruction prefix
        query_text = pair.query
        if pair.query_instruction:
            query_text = f"Instruct: {pair.query_instruction}\nQuery: {query_text}"

        # Tokenize query and positive (no instruction for positive)
        query_tokens = self.tokenizer(
            query_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        positive_tokens = self.tokenizer(
            pair.positive,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize negatives
        negative_tokens_list = []
        for neg in pair.negatives:
            neg_tokens = self.tokenizer(
                neg,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            negative_tokens_list.append(neg_tokens)

        return {
            "query": query_tokens,
            "positive": positive_tokens,
            "negatives": negative_tokens_list,
            "domain": pair.domain,
            "query_instruction": pair.query_instruction,
        }


class EmbeddingDataCollator:
    """
    Custom collator for embedding training batches.

    Handles variable number of negatives and creates proper batch format.

    Attributes:
        tokenizer: HuggingFace tokenizer
        max_negatives: Maximum negatives per example (for padding)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_negatives: int = 7,
        pad_to_multiple_of: int = 8,
    ) -> None:
        """
        Initialize collator.

        Args:
            tokenizer: Pre-initialized tokenizer
            max_negatives: Max negatives to include per batch
            pad_to_multiple_of: Pad sequence lengths to multiple of this value
        """
        self.tokenizer = tokenizer
        self.max_negatives = max_negatives
        self.pad_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """
        Collate batch.

        Args:
            batch: List of examples from EmbeddingDataset

        Returns:
            Batch dictionary with aligned tensors

        Raises:
            ValueError: If batch is empty
        """
        if not batch:
            raise ValueError("Empty batch")

        # Collect inputs
        query_inputs = [item["query"] for item in batch]
        positive_inputs = [item["positive"] for item in batch]

        # Pad queries and positives using standard collator
        query_batch = self.pad_collator(
            [
                {"input_ids": q["input_ids"][0], "attention_mask": q["attention_mask"][0]}
                for q in query_inputs
            ]
        )
        positive_batch = self.pad_collator(
            [
                {"input_ids": p["input_ids"][0], "attention_mask": p["attention_mask"][0]}
                for p in positive_inputs
            ]
        )

        # Collect negatives (variable length)
        max_negs = min(max(len(item["negatives"]) for item in batch), self.max_negatives)

        result = {
            "query_input_ids": query_batch["input_ids"],
            "query_attention_mask": query_batch["attention_mask"],
            "positive_input_ids": positive_batch["input_ids"],
            "positive_attention_mask": positive_batch["attention_mask"],
        }

        # If no negatives in any example, the model will use in-batch negatives
        if max_negs == 0:
            return result

        negative_input_ids_list = []
        negative_attention_mask_list = []

        for item in batch:
            negs = item["negatives"][:max_negs]
            # Pad with empty if not enough negatives — use [1, seq_len] shape
            # to match the tokenizer output format so [0] indexing works below
            while len(negs) < max_negs:
                negs.append(
                    {
                        "input_ids": torch.zeros_like(
                            query_batch["input_ids"][0]
                        ).unsqueeze(0),
                        "attention_mask": torch.zeros_like(
                            query_batch["attention_mask"][0]
                        ).unsqueeze(0),
                    }
                )

            neg_input_ids = torch.stack(
                [neg["input_ids"][0] for neg in negs]
            )  # [max_negs, seq_len]
            neg_attention_mask = torch.stack(
                [neg["attention_mask"][0] for neg in negs]
            )  # [max_negs, seq_len]

            negative_input_ids_list.append(neg_input_ids)
            negative_attention_mask_list.append(neg_attention_mask)

        # Stack negatives
        result["negative_input_ids"] = torch.stack(
            negative_input_ids_list
        )  # [batch_size, max_negs, seq_len]
        result["negative_attention_mask"] = torch.stack(
            negative_attention_mask_list
        )  # [batch_size, max_negs, seq_len]

        return result
