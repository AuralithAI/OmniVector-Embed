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

from omnivector.model.audio_encoder import WhisperAudioEncoder
from omnivector.model.backbone import MistralEmbeddingBackbone
from omnivector.model.latent_attention import LatentAttentionPooling
from omnivector.model.video_encoder import VideoEncoder
from omnivector.model.vision_encoder import SigLIPVisionEncoder

logger = logging.getLogger(__name__)


class OmniVectorModel(nn.Module):
    """
    Unified multimodal embedding model (text, image, video, audio).

    Features:
    - Bidirectional text encoder (Mistral-7B)
    - Latent attention pooling
    - Vision encoder (SigLIP)
    - Video encoder (temporal)
    - Audio encoder (Whisper)
    - Matryoshka dimensionality (512, 1024, 2048, 4096)
    - ONNX export compatible

    Attributes:
        backbone: Text encoder
        pooling: Latent attention pooling
        vision_encoder: Image encoder
        video_encoder: Video encoder
        audio_encoder: Audio encoder
        output_dim: Output embedding dimension
        mrl_dims: Matryoshka dimensions for MRL training
    """

    def __init__(
        self,
        backbone: MistralEmbeddingBackbone,
        pooling: LatentAttentionPooling,
        vision_encoder: Optional[SigLIPVisionEncoder] = None,
        audio_encoder: Optional[WhisperAudioEncoder] = None,
        output_dim: int = 4096,
        mrl_dims: tuple = (512, 1024, 2048, 4096),
    ) -> None:
        """
        Initialize OmniVectorModel.

        Args:
            backbone: Text encoder (MistralEmbeddingBackbone)
            pooling: Pooling layer (LatentAttentionPooling)
            vision_encoder: Optional vision encoder
            audio_encoder: Optional audio encoder (Whisper)
            output_dim: Output dimension (default 4096)
            mrl_dims: Matryoshka dimensions

        Raises:
            ValueError: If dimensions don't match
        """
        super().__init__()

        self.backbone = backbone
        self.pooling = pooling
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
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

    def encode_audio(
        self,
        audio_features: torch.Tensor,
        output_dim: int = 4096,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Encode audio spectrograms to embeddings.

        Args:
            audio_features: Log-mel spectrogram [batch_size, n_mels, seq_len].
            output_dim: Output dimension (default 4096).
            normalize: Whether to L2 normalize.

        Returns:
            Audio embeddings [batch_size, output_dim].

        Raises:
            RuntimeError: If audio encoder not initialized.
        """
        if self.audio_encoder is None:
            raise RuntimeError("Audio encoder not initialized")

        embeddings = self.audio_encoder(audio_features)
        embeddings = embeddings[:, :output_dim]

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

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
    def from_pretrained(
        cls,
        model_name_or_path: str,
        lora: bool = False,
        device: str = "cpu",
        **kwargs,
    ) -> "OmniVectorModel":
        """
        Load a pre-trained OmniVector model from a local checkpoint or construct
        a fresh model from Mistral-7B weights.

        If *model_name_or_path* contains a ``model.pt`` state-dict saved by
        :meth:`save_pretrained`, the weights are loaded on top of the newly
        constructed model.  Otherwise the Mistral-7B backbone is loaded from
        the HuggingFace Hub and all other components are randomly initialised
        (suitable for starting Stage-1 training).

        Args:
            model_name_or_path: Local directory (with ``model.pt``) or HF model ID
                for the Mistral backbone.
            lora: Whether to apply LoRA adapters on the backbone.
            device: Target device (``'cpu'``, ``'cuda'``, ``'cuda:0'``, …).
            **kwargs: Extra options forwarded to component constructors:
                - ``audio_encoder`` (str): Whisper variant name, e.g. ``'whisper-tiny'``.
                  If *None*, no audio encoder is created.
                - ``vision_encoder`` (bool): Whether to create the SigLIP vision encoder.
                  Defaults to *True*.
                - ``freeze_vision_backbone`` (bool): Freeze vision backbone (default True).

        Returns:
            Fully initialised OmniVectorModel on *device*.

        Raises:
            FileNotFoundError: If *model_name_or_path* looks like a local path
                but does not exist.
        """
        import os

        local_checkpoint = None
        backbone_name = "mistralai/Mistral-7B-v0.1"

        # Detect whether path points to a saved checkpoint directory
        if os.path.isdir(model_name_or_path):
            candidate = os.path.join(model_name_or_path, "model.pt")
            if os.path.isfile(candidate):
                local_checkpoint = candidate
                logger.info(f"Found local checkpoint: {candidate}")
            else:
                logger.info(
                    f"Directory {model_name_or_path} has no model.pt — "
                    f"will construct a fresh model."
                )

        # Build backbone
        backbone = MistralEmbeddingBackbone(
            model_name=backbone_name,
            use_lora=lora,
        )

        # Build pooling
        pooling = LatentAttentionPooling(embed_dim=backbone.get_hidden_size())

        # Build optional vision encoder
        vision_encoder = None
        if kwargs.pop("vision_encoder", True):
            freeze_vision = kwargs.pop("freeze_vision_backbone", True)
            vision_encoder = SigLIPVisionEncoder(freeze_backbone=freeze_vision)

        # Build optional audio encoder
        audio_model_name = kwargs.pop("audio_encoder", None)
        audio_encoder = None
        if audio_model_name:
            audio_encoder = WhisperAudioEncoder(model_name=audio_model_name)

        model = cls(
            backbone=backbone,
            pooling=pooling,
            vision_encoder=vision_encoder,
            audio_encoder=audio_encoder,
        )

        # Load saved weights if available
        if local_checkpoint is not None:
            state_dict = torch.load(local_checkpoint, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {local_checkpoint}")

        model = model.to(device)
        logger.info(
            f"OmniVectorModel ready on {device} "
            f"(lora={lora}, vision={'yes' if vision_encoder else 'no'}, "
            f"audio={'yes' if audio_encoder else 'no'})"
        )
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
