"""
Mistral embedding backbone with bidirectional attention and LoRA support.

This module provides MistralEmbeddingBackbone which:
1. Loads Mistral-7B-v0.1 with eager attention (no SDPA/Flash)
2. Overrides causal masking to allow bidirectional attention
3. Supports LoRA adapters via peft
4. Maintains ONNX export compatibility
"""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    MistralModel,
)

logger = logging.getLogger(__name__)


class MistralEmbeddingBackbone(nn.Module):
    """
    Mistral-7B backbone configured for embeddings with bidirectional attention.

    Architecture:
    - Base model: mistralai/Mistral-7B-v0.1
    - Attention: Bidirectional (override _update_causal_mask)
    - Implementation: eager (explicitly setting no SDPA/Flash Attention)
    - LoRA: Optional adapters for parameter-efficient fine-tuning

    Attributes:
        model: MistralModel instance
        config: MistralConfig
        use_lora: Whether LoRA adapters are applied
        lora_config: LoraConfig if use_lora=True
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        """
        Initialize the Mistral embedding backbone.

        Args:
            model_name: HuggingFace model identifier
            use_lora: Whether to apply LoRA adapters
            lora_rank: LoRA rank (r)
            lora_alpha: LoRA alpha scaling factor
            lora_dropout: LoRA dropout probability

        Raises:
            ValueError: If model loading fails or config is incompatible
        """
        super().__init__()

        # Load config and model with eager attention (critical for ONNX)
        try:
            config = AutoConfig.from_pretrained(model_name)
            self.model = MistralModel.from_pretrained(
                model_name,
                config=config,
                attn_implementation="eager",  # NEVER change this
            )
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {e}")

        self.config = self.model.config
        self.use_lora = use_lora
        self.lora_config: Optional[LoraConfig] = None

        # Apply LoRA if requested
        if use_lora:
            self.lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.model = get_peft_model(self.model, self.lora_config)
            logger.info(
                f"Applied LoRA: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}"
            )

        # Override causal masking for bidirectional attention
        self._enable_bidirectional_attention()

    def _enable_bidirectional_attention(self) -> None:
        """
        Override the causal mask to allow bidirectional attention.

        In transformers >= 4.40, the attention masking logic is in _update_causal_mask.
        By returning None, all tokens can attend to all other tokens (bidirectional).

        Test: Verify that hidden_state[0] changes when a future token is modified.
        """
        original_method = self.model._update_causal_mask

        def no_causal_mask(*args: tuple, **kwargs: dict) -> None:
            """Return None to disable causal masking."""
            return None

        self.model._update_causal_mask = no_causal_mask  # type: ignore
        logger.info("Bidirectional attention enabled (_update_causal_mask overridden)")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass through the Mistral backbone.

        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length], optional
            token_type_ids: Token type IDs, optional (Mistral doesn't use)
            position_ids: Position IDs, optional
            output_hidden_states: Whether to return all hidden states

        Returns:
            If output_hidden_states=False: last_hidden_state [batch_size, seq_length, hidden_size]
            If output_hidden_states=True: (last_hidden_state, hidden_states_tuple)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if output_hidden_states:
            return outputs.last_hidden_state, outputs.hidden_states
        return outputs.last_hidden_state

    def merge_lora(self) -> None:
        """
        Merge LoRA adapters into base model weights.

        Critical before ONNX export to eliminate branching operations.
        After merging, the model can be used normally without peft.

        Raises:
            RuntimeError: If LoRA is not applied or already merged
        """
        if not self.use_lora:
            raise RuntimeError("LoRA not applied. Call with use_lora=True at init.")

        if hasattr(self.model, "merge_and_unload"):
            self.model = self.model.merge_and_unload()
            self.use_lora = False
            logger.info("LoRA adapters merged into base model")
        else:
            raise RuntimeError("Model does not support merge_and_unload()")

    def unmerge_lora(self) -> None:
        """
        Unmerge LoRA adapters (restore original weights).

        Allows continued training after merging.

        Raises:
            RuntimeError: If LoRA not applied or not currently merged
        """
        if not self.use_lora:
            raise RuntimeError("LoRA not applied.")

        if hasattr(self.model, "unmerge"):
            self.model.unmerge()
            logger.info("LoRA adapters unmerged from base model")
        else:
            raise RuntimeError("Model does not support unmerge()")

    def get_hidden_size(self) -> int:
        """Get the hidden dimension size."""
        return self.config.hidden_size

    def get_num_layers(self) -> int:
        """Get the number of transformer layers."""
        return self.config.num_hidden_layers

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())


def create_bidirectional_mistral(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    use_lora: bool = False,
) -> MistralEmbeddingBackbone:
    """
    Convenience function to create a bidirectional Mistral backbone.

    Args:
        model_name: HuggingFace model identifier
        use_lora: Whether to apply LoRA

    Returns:
        Initialized MistralEmbeddingBackbone
    """
    return MistralEmbeddingBackbone(model_name=model_name, use_lora=use_lora)
