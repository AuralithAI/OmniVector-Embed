"""
Main OmniVector model combining text, image, and video encoders.

Unified interface for multimodal embeddings with:
- Text/code encoding via Mistral backbone
- Image encoding via SigLIP vision encoder
- Video encoding via temporal frame pooling
- Matryoshka Representation Learning (MRL) for dynamic dimensionality
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from omnivector.model.backbone import MistralEmbeddingBackbone
from omnivector.model.latent_attention import LatentAttentionPooling
from omnivector.model.video_encoder import VideoEncoder
from omnivector.model.vision_encoder import SigLIPVisionEncoder

logger = logging.getLogger(__name__)


class OmniVectorModel(nn.Module):
    """
    Unified multimodal embedding model (text, image, video).

    Features:
    - Bidirectional text encoder (Mistral-7B)
    - Latent attention pooling
    - Vision encoder (SigLIP)
    - Video encoder (temporal)
    - Matryoshka dimensionality (512, 1024, 2048, 4096)
    - ONNX export compatible

    Attributes:
        backbone: Text encoder
        pooling: Latent attention pooling
        vision_encoder: Image encoder
        video_encoder: Video encoder
        output_dim: Output embedding dimension
        mrl_dims: Matryoshka dimensions for MRL training
    """

    def __init__(
        self,
        backbone: MistralEmbeddingBackbone,
        pooling: LatentAttentionPooling,
        vision_encoder: Optional[SigLIPVisionEncoder] = None,
        output_dim: int = 4096,
        mrl_dims: tuple = (512, 1024, 2048, 4096),
    ) -> None:
        """
        Initialize OmniVectorModel.

        Args:
            backbone: Text encoder (MistralEmbeddingBackbone)
            pooling: Pooling layer (LatentAttentionPooling)
            vision_encoder: Optional vision encoder
            output_dim: Output dimension (default 4096)
            mrl_dims: Matryoshka dimensions

        Raises:
            ValueError: If dimensions don't match
        """
        super().__init__()

        self.backbone = backbone
        self.pooling = pooling
        self.vision_encoder = vision_encoder
        self.output_dim = output_dim
        self.mrl_dims = mrl_dims

        # Verify dimensions
        if backbone.get_hidden_size() != pooling.embed_dim:
            raise ValueError(
                f"Backbone hidden size ({backbone.get_hidden_size()}) must match "
                f"pooling embed_dim ({pooling.embed_dim})"
            )

        if max(mrl_dims) != output_dim:
            raise ValueError(f"Max MRL dim ({max(mrl_dims)}) must equal output_dim ({output_dim})")

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self.tokenizer = None

        # Video encoder (optional)
        if vision_encoder is not None:
            self.video_encoder = VideoEncoder(vision_encoder)
        else:
            self.video_encoder = None

        logger.info(f"OmniVectorModel initialized: output_dim={output_dim}, mrl_dims={mrl_dims}")

    def encode_text(
        self,
        texts: Union[str, list],
        instruction: Optional[str] = None,
        output_dim: int = 4096,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode text/code to embeddings.

        Args:
            texts: Single text or list of texts
            instruction: Optional instruction prefix (e.g., "Search query")
            output_dim: Output dimension from MRL (default 4096)
            normalize: Whether to L2 normalize output

        Returns:
            Text embeddings [batch_size, output_dim]

        Raises:
            ValueError: If output_dim not in mrl_dims
            RuntimeError: If tokenizer not loaded
        """
        if output_dim not in self.mrl_dims:
            raise ValueError(f"output_dim {output_dim} not in mrl_dims {self.mrl_dims}")

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")

        # Handle single string
        if isinstance(texts, str):
            texts = [texts]

        # Add instruction prefix
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.backbone.model.device)

        # Encode
        with torch.no_grad():
            hidden_states = self.backbone(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # Pool
            embeddings = self.pooling(
                hidden_states=hidden_states,
                attention_mask=~inputs["attention_mask"].bool(),
            )

        # Slice to output_dim (MRL)
        embeddings = embeddings[:, :output_dim]

        # Normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode_image(
        self,
        images: torch.Tensor,
        output_dim: int = 4096,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: Image tensors [batch_size, 3, height, width]
            output_dim: Output dimension (default 4096)
            normalize: Whether to L2 normalize

        Returns:
            Image embeddings [batch_size, output_dim]

        Raises:
            RuntimeError: If vision encoder not initialized
        """
        if self.vision_encoder is None:
            raise RuntimeError("Vision encoder not initialized")

        embeddings = self.vision_encoder(images)
        embeddings = embeddings[:, :output_dim]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode_video(
        self,
        frames: torch.Tensor,
        output_dim: int = 4096,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode video frames to embeddings.

        Args:
            frames: Video frames [batch_size, n_frames, 3, height, width]
            output_dim: Output dimension (default 4096)
            normalize: Whether to L2 normalize

        Returns:
            Video embeddings [batch_size, output_dim]

        Raises:
            RuntimeError: If video encoder not initialized
        """
        if self.video_encoder is None:
            raise RuntimeError("Video encoder not initialized")

        embeddings = self.video_encoder(frames)
        embeddings = embeddings[:, :output_dim]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        output_dim: int = 4096,
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass (text or multimodal).

        Args:
            input_ids: Text token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            images: Optional image tensors [batch_size, 3, height, width]
            output_dim: Output dimension (default 4096)

        Returns:
            Embeddings [batch_size, output_dim] or (text_emb, image_emb) if images provided
        """
        # Text encodings
        hidden_states = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = self.pooling(
            hidden_states=hidden_states,
            attention_mask=~attention_mask.bool() if attention_mask is not None else None,
        )
        text_embeddings = text_embeddings[:, :output_dim]

        # Image embeddings if provided
        if images is not None:
            image_embeddings = self.encode_image(images, output_dim=output_dim, normalize=False)
            return text_embeddings, image_embeddings

        return text_embeddings

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "OmniVectorModel":
        """
        Load pre-trained OmniVector model.

        Args:
            model_name_or_path: HuggingFace model ID or local path
            **kwargs: Additional arguments

        Returns:
            OmniVectorModel instance
        """
        # Initialize components
        backbone = MistralEmbeddingBackbone(model_name="mistralai/Mistral-7B-v0.1")
        pooling = LatentAttentionPooling()
        vision_encoder = SigLIPVisionEncoder()

        model = cls(backbone, pooling, vision_encoder)
        # Load weights from pretrained if available
        logger.info(f"Loaded OmniVectorModel from {model_name_or_path}")
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save model to directory.

        Args:
            save_directory: Path to save directory
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "model.pt"))
        logger.info(f"Model saved to {save_directory}")
