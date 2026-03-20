"""Tests for WhisperAudioEncoder and audio integration across the stack.

Covers:
- WhisperAudioEncoder init, forward, freeze/unfreeze
- OmniVectorModel.encode_audio integration
- MultimodalMRLLoss with audio embeddings
- MultimodalCollator audio batching
- Modality.AUDIO enum and MultimodalSample.from_audio_text
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ── WhisperAudioEncoder Unit Tests ──


class TestWhisperAudioEncoder:
    """Tests for WhisperAudioEncoder module."""

    def test_import(self):
        """Audio encoder module is importable."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        assert WhisperAudioEncoder is not None

    def test_init_supported_models(self):
        """All supported model names are accepted."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        for name in ("whisper-tiny", "whisper-base", "whisper-small"):
            assert name in WhisperAudioEncoder.WHISPER_MODELS

    def test_init_unsupported_model(self):
        """Unsupported model name raises ValueError."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with pytest.raises(ValueError, match="Unsupported model"):
            WhisperAudioEncoder(model_name="whisper-mega")

    def test_projection_dimensions(self):
        """Projection MLP has correct input/output dimensions."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with patch(
            "omnivector.model.audio_encoder.WhisperAudioEncoder.__init__", return_value=None
        ):
            encoder = WhisperAudioEncoder.__new__(WhisperAudioEncoder)

        nn.Module.__init__(encoder)
        encoder.model_name = "whisper-tiny"
        encoder.embed_dim = 4096
        encoder.encoder_dim = 384
        encoder.whisper_encoder = None
        encoder.feature_extractor = None
        encoder._freeze_encoder = True

        mid_dim = (384 + 4096) // 2
        encoder.projection = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, mid_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, 4096),
        )

        x = torch.randn(2, 384)
        out = encoder.projection(x)
        assert out.shape == (2, 4096)

    def test_forward_with_mock_encoder(self):
        """Forward pass produces correct output shape with mocked Whisper."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with patch(
            "omnivector.model.audio_encoder.WhisperAudioEncoder.__init__", return_value=None
        ):
            encoder = WhisperAudioEncoder.__new__(WhisperAudioEncoder)

        nn.Module.__init__(encoder)
        encoder.model_name = "whisper-tiny"
        encoder.embed_dim = 4096
        encoder.encoder_dim = 384
        encoder._freeze_encoder = True
        encoder.feature_extractor = None

        mock_whisper = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 1500, 384)
        mock_whisper.return_value = mock_output
        encoder.whisper_encoder = mock_whisper

        mid_dim = (384 + 4096) // 2
        encoder.projection = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, mid_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mid_dim, 4096),
        )

        audio_input = torch.randn(2, 80, 3000)
        output = encoder(audio_input)

        assert output.shape == (2, 4096)
        norms = output.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-5, rtol=1e-5)

    def test_forward_without_encoder_raises(self):
        """Forward without Whisper encoder raises RuntimeError."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with patch(
            "omnivector.model.audio_encoder.WhisperAudioEncoder.__init__", return_value=None
        ):
            encoder = WhisperAudioEncoder.__new__(WhisperAudioEncoder)

        nn.Module.__init__(encoder)
        encoder.whisper_encoder = None

        with pytest.raises(RuntimeError, match="Whisper encoder not initialized"):
            encoder(torch.randn(1, 80, 3000))

    def test_trainable_parameters_frozen(self):
        """When encoder frozen, only projection params are trainable."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with patch(
            "omnivector.model.audio_encoder.WhisperAudioEncoder.__init__", return_value=None
        ):
            encoder = WhisperAudioEncoder.__new__(WhisperAudioEncoder)

        nn.Module.__init__(encoder)
        fake_encoder = nn.Linear(10, 10)
        for p in fake_encoder.parameters():
            p.requires_grad = False
        encoder.whisper_encoder = fake_encoder
        encoder.projection = nn.Linear(10, 20)
        encoder._freeze_encoder = True

        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_proj = encoder.projection.weight.numel() + encoder.projection.bias.numel()
        assert trainable == total_proj

    def test_unfreeze_encoder(self):
        """unfreeze_encoder() makes all encoder params trainable."""
        from omnivector.model.audio_encoder import WhisperAudioEncoder

        with patch(
            "omnivector.model.audio_encoder.WhisperAudioEncoder.__init__", return_value=None
        ):
            encoder = WhisperAudioEncoder.__new__(WhisperAudioEncoder)

        nn.Module.__init__(encoder)
        fake_encoder = nn.Linear(10, 10)
        for p in fake_encoder.parameters():
            p.requires_grad = False
        encoder.whisper_encoder = fake_encoder
        encoder.projection = nn.Linear(10, 20)
        encoder._freeze_encoder = True

        encoder.unfreeze_encoder()
        assert encoder._freeze_encoder is False
        for p in encoder.whisper_encoder.parameters():
            assert p.requires_grad


# ── OmniVectorModel Audio Integration ──


class TestOmniVectorModelAudio:
    """Tests for OmniVectorModel with audio encoder."""

    def test_model_has_audio_encoder_attribute(self):
        """OmniVectorModel accepts audio_encoder param."""
        from omnivector.model.omnivector_model import OmniVectorModel

        sig = OmniVectorModel.__init__.__code__.co_varnames
        assert "audio_encoder" in sig

    def test_encode_audio_method_exists(self):
        """OmniVectorModel has encode_audio method."""
        from omnivector.model.omnivector_model import OmniVectorModel

        assert hasattr(OmniVectorModel, "encode_audio")

    def test_encode_audio_without_encoder_raises(self):
        """encode_audio raises RuntimeError when no audio encoder."""
        from omnivector.model.backbone import MistralEmbeddingBackbone
        from omnivector.model.latent_attention import LatentAttentionPooling
        from omnivector.model.omnivector_model import OmniVectorModel

        with patch.object(MistralEmbeddingBackbone, "__init__", return_value=None):
            backbone = MistralEmbeddingBackbone.__new__(MistralEmbeddingBackbone)
        nn.Module.__init__(backbone)
        backbone.model = MagicMock()
        backbone.model.config.hidden_size = 4096

        with patch.object(LatentAttentionPooling, "__init__", return_value=None):
            pooling = LatentAttentionPooling.__new__(LatentAttentionPooling)
        nn.Module.__init__(pooling)
        pooling.embed_dim = 4096
        pooling.output_dim = 4096

        with patch.object(backbone, "get_hidden_size", return_value=4096):
            model = OmniVectorModel(
                backbone=backbone,
                pooling=pooling,
                vision_encoder=None,
                audio_encoder=None,
            )

        with pytest.raises(RuntimeError, match="Audio encoder not initialized"):
            model.encode_audio(torch.randn(1, 80, 3000))


# ── MultimodalMRLLoss with Audio ──


class TestMultimodalLossAudio:
    """Tests for audio support in MultimodalMRLLoss."""

    def test_loss_with_audio_embeddings(self):
        """Loss computation works with audio embeddings."""
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(
            mrl_dims=(512, 1024, 2048, 4096),
            cross_modal_weight=0.2,
        )

        batch_size = 4
        dim = 4096
        query = torch.randn(batch_size, dim)
        positive = torch.randn(batch_size, dim)
        audio_emb = torch.randn(batch_size, dim)
        audio_mask = torch.ones(batch_size, dtype=torch.bool)

        result = loss_fn(
            query_embeddings=query,
            positive_embeddings=positive,
            audio_embeddings=audio_emb,
            text_for_audio=query,
            audio_mask=audio_mask,
        )

        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)
        assert result["loss"].item() > 0
        assert "audio_cross_modal_loss_weighted" in result

    def test_loss_with_vision_and_audio(self):
        """Loss supports simultaneous vision and audio."""
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(
            mrl_dims=(512, 1024, 2048, 4096),
            cross_modal_weight=0.2,
        )

        batch_size = 4
        dim = 4096
        query = torch.randn(batch_size, dim)
        positive = torch.randn(batch_size, dim)
        visual_emb = torch.randn(batch_size, dim)
        audio_emb = torch.randn(batch_size, dim)
        vis_mask = torch.ones(batch_size, dtype=torch.bool)
        aud_mask = torch.ones(batch_size, dtype=torch.bool)

        result = loss_fn(
            query_embeddings=query,
            positive_embeddings=positive,
            visual_embeddings=visual_emb,
            text_for_visual=query,
            visual_mask=vis_mask,
            audio_embeddings=audio_emb,
            text_for_audio=query,
            audio_mask=aud_mask,
        )

        assert "loss" in result
        assert "cross_modal_loss_weighted" in result
        assert "audio_cross_modal_loss_weighted" in result

        text_only = loss_fn(
            query_embeddings=query,
            positive_embeddings=positive,
        )
        assert result["loss"].item() > text_only["loss"].item()

    def test_loss_without_audio_unchanged(self):
        """Backward compat: loss without audio produces same results."""
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(
            mrl_dims=(512, 1024, 2048, 4096),
            cross_modal_weight=0.2,
        )

        torch.manual_seed(42)
        query = torch.randn(4, 4096)
        positive = torch.randn(4, 4096)

        result = loss_fn(
            query_embeddings=query,
            positive_embeddings=positive,
        )

        assert "audio_cross_modal_loss_weighted" not in result
        assert "loss" in result


# ── MultimodalDataset + Collator Audio ──


class TestMultimodalDatasetAudio:
    """Tests for audio modality in MultimodalDataset/Collator."""

    def test_audio_modality_enum(self):
        """AUDIO is a valid Modality."""
        from omnivector.data.multimodal_dataset import Modality

        assert Modality.AUDIO == "audio"
        assert Modality.AUDIO.value == "audio"

    def test_multimodal_sample_from_audio(self):
        """MultimodalSample.from_audio_text creates correct sample."""
        from omnivector.data.multimodal_dataset import Modality, MultimodalSample

        sample = MultimodalSample.from_audio_text(
            audio_path="/data/audio.wav",
            caption="A dog barking in the park",
        )

        assert sample.modality == Modality.AUDIO
        assert sample.audio_path == "/data/audio.wav"
        assert sample.query_text == "A dog barking in the park"
        assert sample.domain == "audio_text"

    def test_collator_handles_audio(self):
        """MultimodalCollator produces audio_features and has_audio."""
        from omnivector.data.multimodal_dataset import MultimodalCollator

        tokenizer = MagicMock()
        collator = MultimodalCollator(tokenizer=tokenizer)

        seq_len = 32
        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "audio",
                "domain": "audio_text",
                "image": None,
                "video": None,
                "audio": torch.randn(80, 3000),
            },
            {
                "query_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "text",
                "domain": "retrieval",
                "image": None,
                "video": None,
                "audio": None,
            },
        ]

        collated = collator(batch)

        assert collated["has_audio"] is True
        assert collated["audio_features"] is not None
        assert collated["audio_features"].shape == (2, 80, 3000)
        assert collated["audio_mask"] is not None
        assert collated["audio_mask"].tolist() == [True, False]

    def test_collator_no_audio(self):
        """Collator with no audio samples produces None audio fields."""
        from omnivector.data.multimodal_dataset import MultimodalCollator

        tokenizer = MagicMock()
        collator = MultimodalCollator(tokenizer=tokenizer)

        seq_len = 32
        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.ones(1, seq_len, dtype=torch.long),
                    "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "text",
                "domain": "retrieval",
                "image": None,
                "video": None,
                "audio": None,
            },
        ]

        collated = collator(batch)

        assert collated["has_audio"] is False
        assert collated["audio_features"] is None
        assert collated["audio_mask"] is None
