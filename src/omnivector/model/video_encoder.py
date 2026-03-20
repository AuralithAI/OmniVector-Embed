"""
Video encoder using frame sampling and temporal pooling.

Extracts frames from videos and encodes them to 4096-dim embeddings
using the vision encoder with temporal aggregation.
"""

import logging

import torch
import torch.nn as nn

from omnivector.model.latent_attention import EagerMultiheadAttention

logger = logging.getLogger(__name__)


class VideoEncoder(nn.Module):
    """
    Video encoder using frame sampling and temporal pooling.

    Architecture:
    - Frame sampling: Extract n_frames from video
    - Vision encoding: Encode each frame with SigLIP
    - Temporal pooling: Aggregate frame embeddings (mean/attention)

    Attributes:
        vision_encoder: SigLIP vision encoder
        n_frames: Number of frames to sample
        pooling_method: 'mean' or 'attention'
        embed_dim: Embedding dimension
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        n_frames: int = 8,
        pooling_method: str = "mean",
        embed_dim: int = 4096,
    ) -> None:
        """
        Initialize video encoder.

        Args:
            vision_encoder: Pre-initialized vision encoder
            n_frames: Number of frames to sample (default 8)
            pooling_method: 'mean' or 'attention' for temporal aggregation
            embed_dim: Embedding dimension
        """
        super().__init__()

        self.vision_encoder = vision_encoder
        self.n_frames = n_frames
        self.pooling_method = pooling_method
        self.embed_dim = embed_dim

        if pooling_method == "attention":
            self.temporal_attention = EagerMultiheadAttention(embed_dim, num_heads=8, dropout=0.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode video frames to embedding.

        Args:
            frames: Video frames [batch_size, n_frames, 3, height, width]

        Returns:
            Video embedding [batch_size, embed_dim]
        """
        batch_size, n_frames, *frame_shape = frames.shape

        # Flatten batch and frames for encoding
        frames_flat = frames.reshape(-1, *frame_shape)

        # Encode all frames
        frame_embeddings = self.vision_encoder(frames_flat)
        frame_embeddings = frame_embeddings.reshape(batch_size, n_frames, -1)

        # Temporal pooling
        if self.pooling_method == "mean":
            video_embedding = frame_embeddings.mean(dim=1)
        elif self.pooling_method == "attention":
            attn_output, _ = self.temporal_attention(
                frame_embeddings, frame_embeddings, frame_embeddings
            )
            video_embedding = attn_output.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        return video_embedding
