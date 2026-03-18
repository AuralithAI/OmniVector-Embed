"""Unit tests for EagerCrossAttention (Fix 1).

Validates the cross-attention module with separate Q / KV projections:
correct shapes, gradient flow, and true cross-attention behaviour.
"""

import math

import pytest
import torch

from omnivector.model.latent_attention import EagerCrossAttention


class TestEagerCrossAttentionInit:
    """Initialization and validation."""

    def test_initialization(self):
        attn = EagerCrossAttention(embed_dim=256, num_heads=8)
        assert attn.embed_dim == 256
        assert attn.num_heads == 8
        assert attn.head_dim == 32
        assert attn.scale == pytest.approx(1.0 / math.sqrt(32))

    def test_has_separate_projections(self):
        attn = EagerCrossAttention(embed_dim=256, num_heads=8)
        assert hasattr(attn, "q_proj")
        assert hasattr(attn, "k_proj")
        assert hasattr(attn, "v_proj")
        assert hasattr(attn, "out_proj")
        assert attn.q_proj is not attn.k_proj
        assert attn.k_proj is not attn.v_proj

    def test_projection_shapes(self):
        attn = EagerCrossAttention(embed_dim=512, num_heads=8)
        assert attn.q_proj.weight.shape == (512, 512)
        assert attn.k_proj.weight.shape == (512, 512)
        assert attn.v_proj.weight.shape == (512, 512)
        assert attn.out_proj.weight.shape == (512, 512)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="divisible"):
            EagerCrossAttention(embed_dim=100, num_heads=8)

    def test_no_bias_option(self):
        attn = EagerCrossAttention(embed_dim=256, num_heads=8, bias=False)
        assert attn.q_proj.bias is None
        assert attn.k_proj.bias is None


class TestEagerCrossAttentionForward:
    """Forward pass shape and behaviour."""

    def test_basic_forward_shape(self):
        attn = EagerCrossAttention(embed_dim=256, num_heads=8, dropout=0.0)
        attn.eval()
        B, Lq, Lkv, E = 4, 32, 128, 256
        q = torch.randn(B, Lq, E)
        k = torch.randn(B, Lkv, E)
        with torch.no_grad():
            out, wt = attn(q, k, k)
        assert out.shape == (B, Lq, E)
        assert wt.shape == (B, Lq, Lkv)

    def test_different_sequence_lengths(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.0)
        attn.eval()
        B, Lq, Lkv, E = 2, 16, 64, 128
        with torch.no_grad():
            out, wt = attn(torch.randn(B, Lq, E), torch.randn(B, Lkv, E), torch.randn(B, Lkv, E))
        assert out.shape == (B, Lq, E)
        assert wt.shape == (B, Lq, Lkv)

    def test_query_and_kv_truly_separate(self):
        """Changing KV source must change output."""
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.0)
        attn.eval()
        B, Lq, Lkv, E = 2, 8, 32, 128
        q = torch.randn(B, Lq, E)
        ka = torch.randn(B, Lkv, E)
        kb = torch.randn(B, Lkv, E)
        with torch.no_grad():
            oa, _ = attn(q, ka, ka)
            ob, _ = attn(q, kb, kb)
        assert not torch.allclose(oa, ob, atol=1e-5)

    def test_key_padding_mask_zeros_attention(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.0)
        attn.eval()
        B, Lq, Lkv, E = 2, 8, 32, 128
        mask = torch.zeros(B, Lkv, dtype=torch.bool)
        mask[:, -16:] = True
        with torch.no_grad():
            _, wt = attn(torch.randn(B, Lq, E), torch.randn(B, Lkv, E),
                         torch.randn(B, Lkv, E), key_padding_mask=mask)
        assert wt[:, :, -16:].abs().max() < 1e-5

    def test_attention_weights_sum_to_one(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.0)
        attn.eval()
        B, Lq, Lkv, E = 2, 8, 32, 128
        with torch.no_grad():
            _, wt = attn(torch.randn(B, Lq, E), torch.randn(B, Lkv, E), torch.randn(B, Lkv, E))
        torch.testing.assert_close(wt.sum(dim=-1), torch.ones(B, Lq), atol=1e-5, rtol=0)


class TestEagerCrossAttentionGradients:
    """Gradient flow."""

    def test_gradients_flow_to_all_projections(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4)
        attn.train()
        B, Lq, Lkv, E = 2, 8, 32, 128
        q = torch.randn(B, Lq, E, requires_grad=True)
        k = torch.randn(B, Lkv, E, requires_grad=True)
        v = torch.randn(B, Lkv, E, requires_grad=True)
        out, _ = attn(q, k, v)
        out.sum().backward()
        assert attn.q_proj.weight.grad is not None
        assert attn.k_proj.weight.grad is not None
        assert attn.v_proj.weight.grad is not None
        assert attn.out_proj.weight.grad is not None
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_kv_gradients_nonzero(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.0)
        attn.train()
        q = torch.randn(1, 4, 128, requires_grad=True)
        k = torch.randn(1, 16, 128, requires_grad=True)
        out, _ = attn(q, k, k)
        out.sum().backward()
        assert attn.k_proj.weight.grad.norm().item() > 0

    def test_dropout_in_training_mode(self):
        attn = EagerCrossAttention(embed_dim=128, num_heads=4, dropout=0.5)
        attn.train()
        q = torch.randn(2, 8, 128)
        k = torch.randn(2, 32, 128)
        o1, _ = attn(q, k, k)
        o2, _ = attn(q, k, k)
        assert not torch.allclose(o1, o2)


class TestLatentPoolingUsesCrossAttn:
    """LatentAttentionPooling must use EagerCrossAttention."""

    def test_pooling_cross_attn_type(self):
        from omnivector.model.latent_attention import LatentAttentionPooling
        pooling = LatentAttentionPooling(embed_dim=256, n_latents=16, num_heads=8)
        assert isinstance(pooling.cross_attn, EagerCrossAttention)

    def test_pooling_output_changes_with_hidden_states(self):
        from omnivector.model.latent_attention import LatentAttentionPooling
        pooling = LatentAttentionPooling(embed_dim=128, n_latents=8, num_heads=4)
        pooling.eval()
        with torch.no_grad():
            oa = pooling(torch.randn(2, 32, 128))
            ob = pooling(torch.randn(2, 32, 128))
        assert not torch.allclose(oa, ob, atol=1e-5)
