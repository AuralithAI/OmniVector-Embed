"""Audio encoder using Whisper for multimodal embeddings.

Encodes audio spectrograms to 4096-dim embeddings aligned with
text/image/video embeddings. Uses Whisper-tiny encoder (384-dim hidden)
as the base audio model with a learned 2-layer MLP projection
(384 → 2240 → 4096) to the target dimension.
"""

import contextlib
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WhisperAudioEncoder(nn.Module):
    """Whisper-based audio encoder for audio embeddings.

    Architecture:
        - Base: Whisper-tiny encoder (384-dim hidden states)
        - Projection: 2-layer MLP (384 → 2240 → 4096) with LayerNorm + GELU
        - Mean-pool over encoder time steps before projection

    The two-layer MLP with non-linearity bridges the dimension gap
    (384 → 4096 ≈ 10.7×) with a mid-dimension of (384 + 4096) // 2 = 2240,
    providing sufficient capacity for the cross-modal projection.

    Attributes:
        model_name: Whisper model identifier.
        embed_dim: Target embedding dimension.
        encoder_dim: Whisper encoder hidden size (e.g. 384 for tiny).
        whisper_encoder: Base Whisper encoder model.
        projection: MLP projection to target dimension.
    """

    WHISPER_DIM = 384  # whisper-tiny hidden size
    WHISPER_MODELS = {
        "whisper-tiny": ("openai/whisper-tiny", 384),
        "whisper-base": ("openai/whisper-base", 512),
        "whisper-small": ("openai/whisper-small", 768),
    }

    def __init__(
        self,
        model_name: str = "whisper-tiny",
        embed_dim: int = 4096,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        """Initialize Whisper audio encoder.

        Args:
            model_name: Whisper model variant ('whisper-tiny', 'whisper-base',
                'whisper-small'). Defaults to whisper-tiny.
            embed_dim: Target embedding dimension (default 4096).
            dropout: Dropout rate in projection MLP.
            freeze_encoder: Whether to freeze the Whisper encoder weights.
                Only the projection head is trained by default.

        Raises:
            ValueError: If model_name is not a supported Whisper variant.
        """
        super().__init__()

        if model_name not in self.WHISPER_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from: {list(self.WHISPER_MODELS.keys())}"
            )

        self.model_name = model_name
        self.embed_dim = embed_dim
        hf_name, self.encoder_dim = self.WHISPER_MODELS[model_name]

        # Load Whisper encoder
        self.whisper_encoder: Optional[nn.Module] = None
        self.feature_extractor = None
        self._freeze_encoder = freeze_encoder

        try:
            from transformers import WhisperFeatureExtractor, WhisperModel

            whisper_model = WhisperModel.from_pretrained(hf_name)
            self.whisper_encoder = whisper_model.encoder
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(hf_name)

            if freeze_encoder:
                for param in self.whisper_encoder.parameters():
                    param.requires_grad = False
                logger.info(f"Whisper encoder frozen ({model_name})")

        except (ImportError, OSError) as e:
            logger.warning(
                f"Failed to load Whisper model '{hf_name}': {e}. "
                f"Using placeholder — audio encoding will raise RuntimeError."
            )

        # Two-layer MLP projection: encoder_dim → mid_dim → embed_dim
        mid_dim = (self.encoder_dim + embed_dim) // 2
        self.projection = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, embed_dim),
        )

        self._init_projection()
        logger.info(
            f"WhisperAudioEncoder initialized: {model_name} " f"({self.encoder_dim} → {embed_dim})"
        )

    def _init_projection(self) -> None:
        """Initialize projection weights with Xavier uniform."""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Encode audio to embeddings.

        Args:
            audio_features: Log-mel spectrogram features from
                WhisperFeatureExtractor. Shape [batch_size, n_mels, seq_len]
                (typically [B, 80, 3000] for 30s audio at 16kHz).

        Returns:
            Audio embeddings [batch_size, embed_dim], L2-normalized.

        Raises:
            RuntimeError: If Whisper encoder was not loaded successfully.
        """
        if self.whisper_encoder is None:
            raise RuntimeError(
                "Whisper encoder not initialized. "
                "Install transformers and ensure model weights are accessible."
            )

        # Whisper encoder expects [batch, n_mels, seq_len]
        ctx = torch.no_grad() if self._freeze_encoder else contextlib.nullcontext()
        with ctx:
            encoder_output = self.whisper_encoder(audio_features)
            hidden_states = encoder_output.last_hidden_state  # [B, T, encoder_dim]

        # Mean pool over time dimension
        pooled = hidden_states.mean(dim=1)  # [B, encoder_dim]

        # Project to target dimension
        projected = self.projection(pooled)  # [B, embed_dim]

        # L2 normalize
        projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-6)

        return projected

    def preprocess_audio(
        self,
        audio_arrays: list,
        sampling_rate: int = 16_000,
    ) -> torch.Tensor:
        """Preprocess raw audio waveforms to log-mel spectrograms.

        Args:
            audio_arrays: List of 1D numpy arrays (raw waveform at 16kHz).
            sampling_rate: Audio sampling rate (default 16kHz for Whisper).

        Returns:
            Batched log-mel spectrogram tensor [batch, n_mels, seq_len].

        Raises:
            RuntimeError: If feature extractor was not loaded.
        """
        if self.feature_extractor is None:
            raise RuntimeError(
                "Whisper feature extractor not initialized. "
                "Install transformers and ensure model weights are accessible."
            )

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_features

    def unfreeze_encoder(self) -> None:
        """Unfreeze the Whisper encoder for fine-tuning."""
        if self.whisper_encoder is not None:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = True
            self._freeze_encoder = False
            logger.info("Whisper encoder unfrozen for fine-tuning")

    @property
    def trainable_parameters(self) -> int:
        """Count trainable parameters (projection only when frozen)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
