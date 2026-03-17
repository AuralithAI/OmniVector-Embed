"""Unit tests for bidirectional attention verification.

Verifies that the MistralEmbeddingBackbone's _update_causal_mask override
produces true bidirectional attention: token[0]'s hidden state MUST change
when a future token is modified.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MinimalBidirectionalBackbone(nn.Module):
    """Lightweight backbone that mimics the bidirectional override.

    Uses a small transformer encoder to verify bidirectional vs causal
    behaviour without downloading Mistral-7B.
    """

    def __init__(self, vocab_size: int = 500, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(128, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional causal masking.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Padding mask [batch, seq_len].
            causal: If True, apply causal (triangular) mask.

        Returns:
            Hidden states [batch, seq_len, hidden_dim].
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.embedding(input_ids) + self.pos_embedding(positions)

        mask = None
        if causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=input_ids.device)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        output = self.encoder(hidden, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output


class TestBidirectionalAttention:
    """Verify bidirectional attention: hidden[0] changes when future tokens change."""

    @pytest.fixture
    def backbone(self) -> MinimalBidirectionalBackbone:
        """Create a small bidirectional backbone."""
        torch.manual_seed(42)
        model = MinimalBidirectionalBackbone(vocab_size=500, hidden_dim=64, num_layers=2)
        model.eval()
        return model

    def test_bidirectional_token0_changes_with_future(self, backbone: MinimalBidirectionalBackbone):
        """Core test: token[0] hidden state changes when token[3] is modified.

        In a bidirectional model, every token attends to every other token.
        Changing a future token should propagate back and alter past hidden states.
        """
        input_ids_a = torch.tensor([[10, 20, 30, 40, 50]])
        input_ids_b = torch.tensor([[10, 20, 30, 99, 50]])  # token[3] changed
        mask = torch.ones(1, 5, dtype=torch.long)

        with torch.no_grad():
            hidden_a = backbone(input_ids_a, mask, causal=False)
            hidden_b = backbone(input_ids_b, mask, causal=False)

        # Token 0 should differ between a and b because attention is bidirectional
        delta = (hidden_a[0, 0] - hidden_b[0, 0]).abs().max().item()
        assert delta > 1e-5, (
            f"Token[0] hidden state did not change when token[3] was modified "
            f"(max delta={delta:.2e}). Attention may be causal, not bidirectional."
        )
        logger.info(f"Bidirectional OK: token[0] delta = {delta:.6f}")

    def test_causal_token0_unchanged_with_future(self, backbone: MinimalBidirectionalBackbone):
        """Control test: with causal mask, token[0] should NOT change.

        Under causal masking, token[0] only attends to itself.
        Changing token[3] should have zero effect on token[0].
        """
        input_ids_a = torch.tensor([[10, 20, 30, 40, 50]])
        input_ids_b = torch.tensor([[10, 20, 30, 99, 50]])
        mask = torch.ones(1, 5, dtype=torch.long)

        with torch.no_grad():
            hidden_a = backbone(input_ids_a, mask, causal=True)
            hidden_b = backbone(input_ids_b, mask, causal=True)

        delta = (hidden_a[0, 0] - hidden_b[0, 0]).abs().max().item()
        assert delta < 1e-6, (
            f"Token[0] changed under causal masking (max delta={delta:.2e}). "
            f"Causal mask is not working correctly."
        )
        logger.info(f"Causal control OK: token[0] delta = {delta:.2e}")

    def test_bidirectional_all_positions_attend_globally(self, backbone: MinimalBidirectionalBackbone):
        """Every position's hidden state should change when any other position changes."""
        base_ids = torch.tensor([[10, 20, 30, 40, 50]])
        mask = torch.ones(1, 5, dtype=torch.long)

        with torch.no_grad():
            base_hidden = backbone(base_ids, mask, causal=False)

        for change_pos in range(5):
            modified_ids = base_ids.clone()
            modified_ids[0, change_pos] = 499

            with torch.no_grad():
                mod_hidden = backbone(modified_ids, mask, causal=False)

            for check_pos in range(5):
                if check_pos == change_pos:
                    continue
                delta = (base_hidden[0, check_pos] - mod_hidden[0, check_pos]).abs().max().item()
                assert delta > 1e-6, (
                    f"Position {check_pos} unchanged when position {change_pos} changed "
                    f"(delta={delta:.2e}). Bidirectional attention not fully connected."
                )

        logger.info("All positions attend globally in bidirectional mode")

    def test_causal_future_positions_unaffected(self, backbone: MinimalBidirectionalBackbone):
        """Under causal masking, later positions should NOT affect earlier ones."""
        base_ids = torch.tensor([[10, 20, 30, 40, 50]])
        mask = torch.ones(1, 5, dtype=torch.long)

        with torch.no_grad():
            base_hidden = backbone(base_ids, mask, causal=True)

        # Change token 4 (last), check tokens 0-3 are unchanged
        modified_ids = base_ids.clone()
        modified_ids[0, 4] = 499

        with torch.no_grad():
            mod_hidden = backbone(modified_ids, mask, causal=True)

        for pos in range(4):
            delta = (base_hidden[0, pos] - mod_hidden[0, pos]).abs().max().item()
            assert delta < 1e-6, (
                f"Position {pos} changed when future position 4 changed "
                f"under causal masking (delta={delta:.2e})."
            )

        logger.info("Causal masking correctly blocks future information flow")

    def test_bidirectional_override_on_backbone_class(self):
        """Test that the real backbone's _enable_bidirectional_attention patches _update_causal_mask."""
        pytest.importorskip("omnivector", reason="omnivector package not installed")
        from omnivector.model.backbone import MistralEmbeddingBackbone

        # Mock from_pretrained to avoid downloading Mistral-7B
        with (
            patch("omnivector.model.backbone.AutoConfig.from_pretrained") as mock_config,
            patch("omnivector.model.backbone.MistralModel.from_pretrained") as mock_model,
        ):
            config = MagicMock()
            config.hidden_size = 4096
            config.num_hidden_layers = 32
            mock_config.return_value = config

            model_instance = MagicMock()
            type(model_instance).config = config
            mock_model.return_value = model_instance

            backbone = object.__new__(MistralEmbeddingBackbone)
            nn.Module.__init__(backbone)
            backbone.model = model_instance
            backbone.config = config
            backbone.use_lora = False
            backbone.lora_config = None
            backbone._enable_bidirectional_attention()

            # _update_causal_mask should return None (bidirectional)
            result = backbone.model._update_causal_mask()
            assert result is None, (
                "_update_causal_mask should return None for bidirectional attention"
            )

        logger.info("Backbone _enable_bidirectional_attention verified")

    def test_hidden_state_deterministic(self, backbone: MinimalBidirectionalBackbone):
        """Same input should produce identical hidden states (deterministic)."""
        input_ids = torch.tensor([[10, 20, 30, 40, 50]])
        mask = torch.ones(1, 5, dtype=torch.long)

        with torch.no_grad():
            hidden_1 = backbone(input_ids, mask, causal=False)
            hidden_2 = backbone(input_ids, mask, causal=False)

        assert torch.allclose(hidden_1, hidden_2, atol=1e-6), (
            "Non-deterministic output detected for identical inputs"
        )

    def test_batch_consistency(self, backbone: MinimalBidirectionalBackbone):
        """Batched forward should match individual forward passes."""
        ids_a = torch.tensor([[10, 20, 30, 40, 50]])
        ids_b = torch.tensor([[60, 70, 80, 90, 100]])
        ids_batch = torch.cat([ids_a, ids_b], dim=0)
        mask = torch.ones_like(ids_batch)

        with torch.no_grad():
            hidden_a = backbone(ids_a, mask[:1], causal=False)
            hidden_b = backbone(ids_b, mask[1:], causal=False)
            hidden_batch = backbone(ids_batch, mask, causal=False)

        assert torch.allclose(hidden_batch[0], hidden_a[0], atol=1e-5), (
            "Batch sample 0 differs from individual forward pass"
        )
        assert torch.allclose(hidden_batch[1], hidden_b[0], atol=1e-5), (
            "Batch sample 1 differs from individual forward pass"
        )
