"""
Vision encoder using SigLIP for multimodal embeddings.

Encodes images to 4096-dim embeddings aligned with text embeddings.
Uses SigLIP-SO400M as the base vision model with projection to target dimension.
"""


import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP-based vision encoder for image embeddings.

    Architecture:
    - Base: Open-source SigLIP model (1152-dim output)
    - Projection: Linear layer to target embedding dimension (4096)

    Attributes:
        model_name: SigLIP model identifier
        embed_dim: Target embedding dimension
        vision_model: Base SigLIP vision model
        projection: Linear projection layer
    """

    def __init__(
        self,
        model_name: str = "SigLIP-SO400M",
        embed_dim: int = 4096,
    ) -> None:
        """
        Initialize SigLIP vision encoder.

        Args:
            model_name: SigLIP model to use
            embed_dim: Target embedding dimension (default 4096)
        """
        super().__init__()

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.vision_model_dim = 1152  # SigLIP-SO400M output dimension

        try:
            # Import here to avoid hard dependency
            from open_clip import create_model_from_pretrained

            self.vision_model, self.preprocess = create_model_from_pretrained(
                "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
            )
        except ImportError:
            logger.warning("open_clip not available, using placeholder")
            self.vision_model = None
            self.preprocess = None

        # Projection to target dimension
        self.projection = nn.Linear(self.vision_model_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: Image tensors [batch_size, 3, height, width]

        Returns:
            Image embeddings [batch_size, embed_dim]

        Raises:
            RuntimeError: If vision model not initialized
        """
        if self.vision_model is None:
            raise RuntimeError("Vision model not initialized. Install open_clip.")

        # Extract vision features
        with torch.no_grad():
            image_features = self.vision_model.encode_image(images)
            # Normalize
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

        # Project to target dimension
        projected = self.projection(image_features)

        # Normalize output
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-6)

        return projected

    def get_preprocess(self):
        """Get image preprocessing function."""
        if self.preprocess is None:
            raise RuntimeError("Preprocessing not available. Install open_clip.")
        return self.preprocess

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters (only projection)."""
        return self.projection.weight.numel() + self.projection.bias.numel()

    @property
    def total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
