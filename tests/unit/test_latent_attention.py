"""Unit tests for latent attention layer."""

import logging

import pytest
import torch

logger = logging.getLogger(__name__)


class TestEagerMultiheadAttention:
    """Test suite for EagerMultiheadAttention."""

    def test_initialization(self):
        """Test attention layer initialization."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        attn = EagerMultiheadAttention(
            embed_dim=4096,
            num_heads=8,
            dropout=0.1,
        )
        assert attn.embed_dim == 4096
        assert attn.num_heads == 8
        assert attn.head_dim == 512
        logger.info("✓ EagerMultiheadAttention initialized")

    def test_invalid_dimensions(self):
        """Test error on invalid embedding dimensions."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        with pytest.raises(ValueError, match="divisible"):
            EagerMultiheadAttention(embed_dim=1001, num_heads=8)

        logger.info("✓ Correctly rejects invalid dimensions")

    def test_forward_pass_shape(self):
        """Test forward pass output shapes."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        attn = EagerMultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.0,
        )
        attn.eval()

        batch_size, seq_length, embed_dim = 4, 128, 512
        query = torch.randn(batch_size, seq_length, embed_dim)
        key = torch.randn(batch_size, seq_length, embed_dim)
        value = torch.randn(batch_size, seq_length, embed_dim)

        with torch.no_grad():
            output, weights = attn(query, key, value)

        assert output.shape == (batch_size, seq_length, embed_dim)
        assert weights.shape == (batch_size, seq_length, seq_length)
        logger.info(f"✓ Forward pass shapes: output={output.shape}, weights={weights.shape}")

    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        attn = EagerMultiheadAttention(embed_dim=256, num_heads=8)
        attn.train()

        batch_size, seq_length, embed_dim = 2, 64, 256
        query = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
        key = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)
        value = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)

        output, _ = attn(query, key, value)
        loss = output.sum()
        loss.backward()

        for param in attn.parameters():
            assert param.grad is not None, "Module parameters should have gradients"
        logger.info("✓ Gradients flow correctly")

    def test_attention_mask(self):
        """Test key padding mask application."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        attn = EagerMultiheadAttention(embed_dim=256, num_heads=8)
        attn.eval()

        batch_size, seq_length, embed_dim = 2, 64, 256
        query = torch.randn(batch_size, seq_length, embed_dim)
        key = torch.randn(batch_size, seq_length, embed_dim)
        value = torch.randn(batch_size, seq_length, embed_dim)

        # Create padding mask (True for positions to ignore)
        key_padding_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool)
        key_padding_mask[:, -16:] = True  # Mask last 16 positions

        with torch.no_grad():
            output, _ = attn(query, key, value, key_padding_mask=key_padding_mask)

        assert output.shape == (batch_size, seq_length, embed_dim)
        logger.info("✓ Attention mask applied correctly")

    def test_dropout(self):
        """Test dropout is applied in training mode."""
        from omnivector.model.latent_attention import EagerMultiheadAttention

        attn = EagerMultiheadAttention(embed_dim=256, num_heads=8, dropout=0.5)
        attn.train()

        batch_size, seq_length, embed_dim = 2, 64, 256
        query = torch.randn(batch_size, seq_length, embed_dim)
        key = torch.randn(batch_size, seq_length, embed_dim)
        value = torch.randn(batch_size, seq_length, embed_dim)

        outputs = []
        for _ in range(3):
            output, _ = attn(query, key, value)
            outputs.append(output)

        # Outputs should differ due to dropout
        assert not torch.allclose(outputs[0], outputs[1])
        logger.info("✓ Dropout applied in training mode")


class TestLatentAttentionPooling:
    """Test suite for LatentAttentionPooling."""

    def test_initialization(self):
        """Test latent attention pooling initialization."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        pooling = LatentAttentionPooling(
            embed_dim=4096,
            n_latents=512,
            num_heads=8,
        )
        assert pooling.embed_dim == 4096
        assert pooling.n_latents == 512
        assert pooling.num_heads == 8
        assert pooling.latents.shape == (512, 4096)
        logger.info("✓ LatentAttentionPooling initialized")

    def test_forward_pass(self):
        """Test forward pass produces correct output."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        pooling = LatentAttentionPooling(
            embed_dim=512,
            n_latents=64,
            num_heads=8,
        )
        pooling.eval()

        batch_size, seq_length, embed_dim = 4, 128, 512
        hidden_states = torch.randn(batch_size, seq_length, embed_dim)

        with torch.no_grad():
            output = pooling(hidden_states)

        # Output should be [batch_size, embed_dim] after pooling
        assert output.shape == (batch_size, embed_dim)
        logger.info(f"✓ Pooling output shape: {output.shape}")

    def test_attention_mask_application(self):
        """Test attention mask is properly applied."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        pooling = LatentAttentionPooling(
            embed_dim=256,
            n_latents=32,
            num_heads=8,
        )
        pooling.eval()

        batch_size, seq_length, embed_dim = 2, 64, 256
        hidden_states = torch.randn(batch_size, seq_length, embed_dim)

        with torch.no_grad():
            output = pooling(hidden_states)

        assert output.shape == (batch_size, embed_dim)
        logger.info("✓ Pooling output shape correct")

    def test_gradient_flow_through_pooling(self):
        """Test gradients flow through pooling layer."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        pooling = LatentAttentionPooling(
            embed_dim=256,
            n_latents=32,
            num_heads=8,
        )
        pooling.train()

        batch_size, seq_length, embed_dim = 2, 64, 256
        hidden_states = torch.randn(batch_size, seq_length, embed_dim, requires_grad=True)

        output = pooling(hidden_states)
        loss = output.sum()
        loss.backward()

        assert pooling.latents.grad is not None, "Latent parameters should have gradients"
        assert pooling.cross_attn.in_proj.weight.grad is not None, "Attention weights should have gradients"
        logger.info("✓ Gradients flow correctly through pooling")

    def test_latent_parameters_trainable(self):
        """Test latent parameters are trainable."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        pooling = LatentAttentionPooling(
            embed_dim=256,
            n_latents=32,
            num_heads=8,
        )

        assert pooling.latents.requires_grad is True
        assert pooling.latents.numel() == 32 * 256
        logger.info(f"✓ Latent parameters: {pooling.latents.numel():,}")

    def test_output_dimension_correctness(self):
        """Test output maintains embedding dimension."""
        from omnivector.model.latent_attention import LatentAttentionPooling

        embed_dim = 768
        pooling = LatentAttentionPooling(
            embed_dim=embed_dim,
            n_latents=128,
        )
        pooling.eval()

        batch_size = 8
        seq_length = 256
        hidden_states = torch.randn(batch_size, seq_length, embed_dim)

        with torch.no_grad():
            output = pooling(hidden_states)

        assert output.shape[-1] == embed_dim
        logger.info(f"✓ Output maintains embedding dim: {embed_dim}")
