"""
Latent attention pooling layer with ONNX-compatible attention implementation.

This module implements:
1. EagerMultiheadAttention: Multihead attention using explicit matmuls (self-attention)
2. EagerCrossAttention: Cross-attention with separate Q / KV projections
3. LatentAttentionPooling: Learns n_latents latent vectors that aggregate context

Critical: Do NOT use nn.MultiheadAttention because PyTorch 2.x routes it through
SDPA internally, which breaks ONNX export. Use explicit matmul-based implementation.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EagerMultiheadAttention(nn.Module):
    """
    Self-attention using explicit matmuls (ONNX-safe).

    Uses a unified in_proj for Q, K, V — suitable for **self-attention**
    where query, key, and value come from the same source.

    For cross-attention (query ≠ key/value), use ``EagerCrossAttention``.

    Attributes:
        embed_dim: Input/output dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        scale: Scaling factor (1/sqrt(d_k))
        in_proj: Linear layer for Q, K, V projection (unified)
        out_proj: Linear layer for output projection
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """
        Initialize EagerMultiheadAttention (self-attention).

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability after softmax
            bias: Whether to include bias in projections

        Raises:
            ValueError: If embed_dim not divisible by num_heads
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        # Unified projection for Q, K, V (3 * embed_dim output)
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of self-attention.

        Note: ``key`` and ``value`` are ignored — all three projections
        are derived from ``query`` via the unified ``in_proj``.  This is
        intentional for self-attention.  Use ``EagerCrossAttention`` when
        the key/value source differs from the query.

        Args:
            query: Input tensor [batch, L, E]
            key: Ignored (kept for API compat with cross-attention swap)
            value: Ignored
            key_padding_mask: Boolean mask [batch, L], True for padding
            attn_mask: Additive attention mask [L, L]

        Returns:
            Tuple of (output [batch, L, E], attn_weights [batch, L, L])
        """
        batch_size, tgt_len, embed_dim = query.shape

        # Project Q, K, V from the **same** source (self-attention)
        qkv = self.in_proj(query)  # [batch, tgt_len, 3*embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each [batch, tgt_len, embed_dim]

        # Reshape for multihead
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, tgt_len, head_dim]

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask (additive)
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        # Softmax + NaN guard for fully-masked rows
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )

        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, L, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)

        output = self.out_proj(attn_output)
        attn_weights_avg = attn_weights.mean(dim=1)  # [batch, L, L]

        return output, attn_weights_avg


class EagerCrossAttention(nn.Module):
    """
    Cross-attention with **separate** Q and KV projections (ONNX-safe).

    Query comes from one source (e.g. latent vectors) while key/value
    come from another (e.g. encoder hidden states).  Using separate
    projections is critical — a shared ``in_proj`` would project key/value
    from the query source, producing incorrect cross-attention.

    All operations use explicit matmuls so the graph exports cleanly to
    ONNX opset ≥ 18.

    Attributes:
        embed_dim: Input/output dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head
        scale: Scaling factor (1/sqrt(d_k))
        q_proj: Linear projection for queries
        k_proj: Linear projection for keys
        v_proj: Linear projection for values
        out_proj: Linear projection for output
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        """
        Initialize EagerCrossAttention.

        Args:
            embed_dim: Total embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability after softmax
            bias: Whether to include bias in projections

        Raises:
            ValueError: If embed_dim not divisible by num_heads
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dropout_p = dropout

        # Separate projections for cross-attention
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.

        Args:
            query: Query tensor [batch, L_q, E] (e.g. latent vectors)
            key:   Key tensor   [batch, L_kv, E] (e.g. encoder hidden states)
            value: Value tensor [batch, L_kv, E] (same source as key)
            key_padding_mask: Boolean mask [batch, L_kv], True = ignore
            attn_mask: Additive attention mask [L_q, L_kv]

        Returns:
            Tuple of (output [batch, L_q, E], attn_weights [batch, L_q, L_kv])
        """
        batch_size, tgt_len, _ = query.shape
        src_len = key.size(1)

        # Separate projections from different sources
        q = self.q_proj(query)  # [batch, L_q, E]   — from latents
        k = self.k_proj(key)  # [batch, L_kv, E]  — from encoder
        v = self.v_proj(value)  # [batch, L_kv, E]  — from encoder

        # Reshape for multihead
        q = q.reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, seq_len, head_dim]

        # Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Additive attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            scores = scores + attn_mask

        # Key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )

        # Softmax + NaN guard
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )

        attn_weights = self.dropout(attn_weights)

        # Attention-weighted values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, tgt_len, self.embed_dim)

        output = self.out_proj(attn_output)
        attn_weights_avg = attn_weights.mean(dim=1)

        return output, attn_weights_avg


class LatentAttentionPooling(nn.Module):
    """
    Latent attention pooling layer for embeddings.

    Learns n_latents trainable latent vectors and uses cross-attention
    to aggregate information from the input sequence. This is analogous
    to the pooling mechanism in NV-Embed-v2.

    Architecture:
    - Latents: n_latents x embed_dim (learnable)
    - Cross-attention: latents attend to input sequence
    - MLP: Optional projection layer
    - Final pooling: Mean over latents

    Attributes:
        n_latents: Number of latent vectors
        embed_dim: Input embedding dimension
        num_heads: Number of attention heads
        latents: Learnable latent vectors [n_latents, embed_dim]
    """

    def __init__(
        self,
        embed_dim: int,
        n_latents: int = 512,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize latent attention pooling.

        Args:
            embed_dim: Input and output embedding dimension (4096)
            n_latents: Number of learnable latent vectors (512)
            num_heads: Number of attention heads (8)
            ffn_dim: Dimension of feed-forward network (2048)
            dropout: Dropout probability

        Raises:
            ValueError: If embed_dim not divisible by num_heads
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_latents = n_latents
        self.num_heads = num_heads

        # Learnable latents
        self.latents = nn.Parameter(torch.randn(n_latents, embed_dim) / math.sqrt(embed_dim))

        # Cross-attention: latents (query) attend to encoder hidden states (key/value).
        # Must use EagerCrossAttention with separate Q / KV projections —
        # a shared in_proj would project K,V from the latent source, not the encoder.
        self.cross_attn = EagerCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feed-forward network (MLPs after attention)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of latent attention pooling.

        Args:
            hidden_states: Input sequence [batch_size, seq_length, embed_dim]
            attention_mask: Attention mask [batch_size, seq_length], optional
                           True for positions to ignore

        Returns:
            Pooled embeddings [batch_size, embed_dim]
        """
        batch_size = hidden_states.size(0)

        # Expand latents for batch: [n_latents, embed_dim] -> [batch, n_latents, embed_dim]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Layer norm input
        hidden_states = self.ln_pre(hidden_states)

        # Cross-attention: latents (query) attend to hidden_states (key/value)
        attn_output, _ = self.cross_attn(
            query=latents,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=attention_mask,  # Inverse of normal mask (True for padding)
        )

        # Residual connection + FFN
        latent_repr = attn_output + latents
        latent_repr = latent_repr + self.ffn(self.ln_post(latent_repr))

        # Mean pool over latents: [batch, n_latents, embed_dim] -> [batch, embed_dim]
        output = latent_repr.mean(dim=1)

        return output


def create_pooling_layer(
    embed_dim: int = 4096,
    n_latents: int = 512,
    num_heads: int = 8,
) -> LatentAttentionPooling:
    """
    Convenience function to create a latent attention pooling layer.

    Args:
        embed_dim: Embedding dimension (default 4096)
        n_latents: Number of latents (default 512)
        num_heads: Number of heads (default 8)

    Returns:
        LatentAttentionPooling instance
    """
    return LatentAttentionPooling(
        embed_dim=embed_dim,
        n_latents=n_latents,
        num_heads=num_heads,
    )
