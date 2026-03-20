"""Unit tests for multimodal training pipeline.

Tests vision data loaders, multimodal dataset/collator, cross-modal
contrastive loss, and multimodal trainer with mock vision components.
"""

import json
import logging
import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class MockVisionEncoder(nn.Module):
    """Minimal vision encoder for testing."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.projection = nn.Linear(3 * 8 * 8, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        flat = images.reshape(batch, -1)[:, : 3 * 8 * 8]
        if flat.shape[1] < 3 * 8 * 8:
            flat = torch.nn.functional.pad(flat, (0, 3 * 8 * 8 - flat.shape[1]))
        return self.projection(flat)


class MockBackbone(nn.Module):
    """Minimal text backbone for testing."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(1000, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        return self.linear(self.embedding(input_ids))

    def get_hidden_size(self):
        return self.linear.out_features

    def merge_lora(self):
        pass


class MockPooling(nn.Module):
    """Minimal pooling for testing."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_states, attention_mask=None):
        return self.linear(hidden_states.mean(dim=1))


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = 0

    def tokenize_fn(text, **kwargs):
        max_len = kwargs.get("max_length", 32)
        ids = torch.randint(1, 100, (1, max_len))
        mask = torch.ones(1, max_len, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    tokenizer.side_effect = tokenize_fn
    tokenizer.__call__ = tokenize_fn
    return tokenizer


@pytest.fixture
def sample_jsonl_dir():
    """Create temp dir with sample JSONL files for testing loaders."""
    tmpdir = tempfile.mkdtemp()

    # Image-text JSONL
    image_text_file = os.path.join(tmpdir, "image_text.jsonl")
    with open(image_text_file, "w") as f:
        for i in range(5):
            record = {
                "image_path": f"img_{i}.jpg",
                "caption": f"A photo of object number {i}",
                "negative_captions": [f"Wrong caption {j}" for j in range(2)],
            }
            f.write(json.dumps(record) + "\n")

    # Video-text JSONL
    video_text_file = os.path.join(tmpdir, "video_text.jsonl")
    with open(video_text_file, "w") as f:
        for i in range(3):
            record = {
                "video_path": f"vid_{i}.mp4",
                "caption": f"A video showing action number {i}",
            }
            f.write(json.dumps(record) + "\n")

    # COCO-format JSON
    coco_file = os.path.join(tmpdir, "coco_captions.json")
    coco_data = {
        "images": [{"id": i, "file_name": f"COCO_{i:06d}.jpg"} for i in range(4)],
        "annotations": [
            {"image_id": i, "caption": f"COCO caption for image {i}"} for i in range(4)
        ],
    }
    with open(coco_file, "w") as f:
        json.dump(coco_data, f)

    yield tmpdir

    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# ImageTextLoader tests
# ---------------------------------------------------------------------------


class TestImageTextLoader:
    """Tests for ImageTextLoader."""

    def test_load_jsonl(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "image_text.jsonl"),
            format="jsonl",
        )
        pairs = loader.load()
        assert len(pairs) == 5
        assert pairs[0]["image_path"] == "img_0.jpg"
        assert "caption" in pairs[0]

    def test_load_jsonl_with_max_samples(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "image_text.jsonl"),
            format="jsonl",
            max_samples=3,
        )
        pairs = loader.load()
        assert len(pairs) == 3

    def test_load_jsonl_with_image_dir(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "image_text.jsonl"),
            image_dir="/data/images",
            format="jsonl",
        )
        pairs = loader.load()
        path = pairs[0]["image_path"].replace("\\", "/")
        assert path.startswith("/data/images")

    def test_load_coco(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "coco_captions.json"),
            format="coco",
        )
        pairs = loader.load()
        assert len(pairs) == 4
        assert "COCO" in pairs[0]["image_path"]

    def test_load_jsonl_not_found(self):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(dataset_path="/nonexistent/path.jsonl", format="jsonl")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_unknown_format(self):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(dataset_path="test.jsonl", format="parquet")
        with pytest.raises(ValueError, match="Unknown format"):
            loader.load()

    def test_to_embedding_pairs(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import ImageTextLoader

        loader = ImageTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "image_text.jsonl"),
            format="jsonl",
        )
        pairs = loader.to_embedding_pairs()
        assert len(pairs) == 5
        assert pairs[0].domain.startswith("image_text:")
        assert pairs[0].query_instruction == "Describe the image"


class TestVideoTextLoader:
    """Tests for VideoTextLoader."""

    def test_load_jsonl(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import VideoTextLoader

        loader = VideoTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "video_text.jsonl"),
            format="jsonl",
        )
        pairs = loader.load()
        assert len(pairs) == 3
        assert pairs[0]["video_path"] == "vid_0.mp4"

    def test_load_with_video_dir(self, sample_jsonl_dir):
        from omnivector.data.loaders.multimodal import VideoTextLoader

        loader = VideoTextLoader(
            dataset_path=os.path.join(sample_jsonl_dir, "video_text.jsonl"),
            video_dir="/data/videos",
            format="jsonl",
        )
        pairs = loader.load()
        path = pairs[0]["video_path"].replace("\\", "/")
        assert path.startswith("/data/videos")


# ---------------------------------------------------------------------------
# MultimodalSample tests
# ---------------------------------------------------------------------------


class TestMultimodalSample:
    """Tests for MultimodalSample."""

    def test_from_embedding_pair(self):
        from omnivector.data.multimodal_dataset import Modality, MultimodalSample
        from omnivector.data.schema import EmbeddingPair

        pair = EmbeddingPair(
            query="test query",
            positive="test positive",
            negatives=["neg1"],
            domain="retrieval",
        )
        sample = MultimodalSample.from_embedding_pair(pair)
        assert sample.modality == Modality.TEXT
        assert sample.query_text == "test query"
        assert sample.image_path is None

    def test_from_image_text(self):
        from omnivector.data.multimodal_dataset import Modality, MultimodalSample

        sample = MultimodalSample.from_image_text(
            image_path="/img/test.jpg",
            caption="A cat sitting on a mat",
        )
        assert sample.modality == Modality.IMAGE
        assert sample.image_path == "/img/test.jpg"
        assert sample.query_text == "A cat sitting on a mat"

    def test_from_video_text(self):
        from omnivector.data.multimodal_dataset import Modality, MultimodalSample

        sample = MultimodalSample.from_video_text(
            video_path="/vid/test.mp4",
            caption="A person walking",
            negatives=["A car driving"],
        )
        assert sample.modality == Modality.VIDEO
        assert sample.video_path == "/vid/test.mp4"
        assert len(sample.negatives) == 1


# ---------------------------------------------------------------------------
# MultimodalDataset tests
# ---------------------------------------------------------------------------


class TestMultimodalDataset:
    """Tests for MultimodalDataset."""

    def test_dataset_length(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalDataset, MultimodalSample

        samples = [
            MultimodalSample(query_text="q1", positive_text="p1"),
            MultimodalSample(query_text="q2", positive_text="p2"),
        ]
        ds = MultimodalDataset(samples=samples, tokenizer=mock_tokenizer)
        assert len(ds) == 2

    def test_text_sample_structure(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalDataset, MultimodalSample

        samples = [MultimodalSample(query_text="test query", positive_text="test positive")]
        ds = MultimodalDataset(samples=samples, tokenizer=mock_tokenizer)
        item = ds[0]

        assert "query_tokens" in item
        assert "positive_tokens" in item
        assert item["modality"] == "text"
        assert item["image"] is None
        assert item["video"] is None

    def test_modality_counts_logged(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import (
            Modality,
            MultimodalDataset,
            MultimodalSample,
        )

        samples = [
            MultimodalSample(query_text="q1", positive_text="p1", modality=Modality.TEXT),
            MultimodalSample(
                query_text="q2", positive_text="p2", modality=Modality.IMAGE, image_path="x.jpg"
            ),
            MultimodalSample(
                query_text="q3", positive_text="p3", modality=Modality.IMAGE, image_path="y.jpg"
            ),
        ]
        ds = MultimodalDataset(samples=samples, tokenizer=mock_tokenizer)
        assert len(ds) == 3


# ---------------------------------------------------------------------------
# MultimodalCollator tests
# ---------------------------------------------------------------------------


class TestMultimodalCollator:
    """Tests for MultimodalCollator."""

    def test_collate_text_batch(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalCollator

        collator = MultimodalCollator(tokenizer=mock_tokenizer, max_negatives=3)

        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "text",
                "domain": "general",
                "image": None,
                "video": None,
            }
            for _ in range(4)
        ]

        result = collator(batch)
        assert result["query_input_ids"].shape[0] == 4
        assert result["has_images"] is False
        assert result["has_videos"] is False

    def test_collate_image_batch(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalCollator

        collator = MultimodalCollator(tokenizer=mock_tokenizer)

        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "image",
                "domain": "image_text",
                "image": torch.randn(3, 32, 32),
                "video": None,
            }
            for _ in range(3)
        ]

        result = collator(batch)
        assert result["has_images"] is True
        assert result["images"].shape == (3, 3, 32, 32)
        assert result["image_mask"].all()

    def test_collate_mixed_batch(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalCollator

        collator = MultimodalCollator(tokenizer=mock_tokenizer)

        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "text",
                "domain": "general",
                "image": None,
                "video": None,
            },
            {
                "query_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "negative_tokens": [],
                "modality": "image",
                "domain": "image_text",
                "image": torch.randn(3, 32, 32),
                "video": None,
            },
        ]

        result = collator(batch)
        assert result["has_images"] is True
        assert result["images"].shape[0] == 2
        # First sample has no image, second has image
        assert result["image_mask"][0].item() is False
        assert result["image_mask"][1].item() is True

    def test_collate_with_negatives(self, mock_tokenizer):
        from omnivector.data.multimodal_dataset import MultimodalCollator

        collator = MultimodalCollator(tokenizer=mock_tokenizer, max_negatives=2)

        neg = {
            "input_ids": torch.randint(0, 100, (1, 16)),
            "attention_mask": torch.ones(1, 16, dtype=torch.long),
        }
        batch = [
            {
                "query_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "positive_tokens": {
                    "input_ids": torch.randint(0, 100, (1, 16)),
                    "attention_mask": torch.ones(1, 16, dtype=torch.long),
                },
                "negative_tokens": [neg, neg],
                "modality": "text",
                "domain": "general",
                "image": None,
                "video": None,
            }
            for _ in range(2)
        ]

        result = collator(batch)
        assert result["negative_input_ids"].shape == (2, 2, 16)


# ---------------------------------------------------------------------------
# CrossModalContrastiveLoss tests
# ---------------------------------------------------------------------------


class TestCrossModalContrastiveLoss:
    """Tests for CrossModalContrastiveLoss."""

    def test_initialization(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss()
        assert loss_fn.mrl_dims == (512, 1024, 2048, 4096)
        assert loss_fn.temperature.item() > 0

    def test_forward_shape(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(64, 128))
        visual = torch.randn(4, 128)
        text = torch.randn(4, 128)
        result = loss_fn(visual, text)

        assert "loss" in result
        assert result["loss"].dim() == 0
        assert result["loss"].item() > 0

    def test_loss_decreases_with_alignment(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(32,), learnable_temperature=False)

        # Misaligned embeddings
        visual_random = torch.randn(8, 32)
        text_random = torch.randn(8, 32)
        loss_random = loss_fn(visual_random, text_random)["loss"].item()

        # Aligned embeddings (identical)
        aligned = torch.randn(8, 32)
        loss_aligned = loss_fn(aligned, aligned)["loss"].item()

        assert loss_aligned < loss_random

    def test_visual_mask(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(32,), learnable_temperature=False)
        visual = torch.randn(4, 32)
        text = torch.randn(4, 32)
        mask = torch.tensor([True, True, False, False])

        result = loss_fn(visual, text, visual_mask=mask)
        assert result["loss"].item() > 0

    def test_empty_mask_returns_zero(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(32,))
        visual = torch.randn(4, 32)
        text = torch.randn(4, 32)
        mask = torch.tensor([False, False, False, False])

        result = loss_fn(visual, text, visual_mask=mask)
        assert result["loss"].item() == 0.0

    def test_temperature_is_learnable(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(32,), learnable_temperature=True)
        assert loss_fn.log_temperature.requires_grad is True

    def test_symmetric_loss(self):
        from omnivector.training.multimodal_loss import CrossModalContrastiveLoss

        loss_fn = CrossModalContrastiveLoss(mrl_dims=(32,), learnable_temperature=False)
        a = torch.randn(4, 32)
        b = torch.randn(4, 32)

        loss_ab = loss_fn(a, b)["loss"].item()
        loss_ba = loss_fn(b, a)["loss"].item()

        # Symmetric loss should be approximately equal
        assert abs(loss_ab - loss_ba) < 0.1


# ---------------------------------------------------------------------------
# MultimodalMRLLoss tests
# ---------------------------------------------------------------------------


class TestMultimodalMRLLoss:
    """Tests for combined MultimodalMRLLoss."""

    def test_text_only(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))
        q = torch.randn(4, 128)
        p = torch.randn(4, 128)
        result = loss_fn(query_embeddings=q, positive_embeddings=p)

        assert "loss" in result
        assert "text_loss" in result
        assert result["loss"].item() > 0

    def test_with_visual(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128), cross_modal_weight=1.0)
        q = torch.randn(4, 128)
        p = torch.randn(4, 128)
        vis = torch.randn(4, 128)
        result = loss_fn(
            query_embeddings=q,
            positive_embeddings=p,
            visual_embeddings=vis,
            text_for_visual=q,
        )

        assert "loss" in result
        assert "text_loss" in result
        assert "cross_modal_loss_weighted" in result

    def test_cross_modal_weight_zero(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss

        loss_fn = MultimodalMRLLoss(mrl_dims=(64,), cross_modal_weight=0.0)
        q = torch.randn(4, 64)
        p = torch.randn(4, 64)
        vis = torch.randn(4, 64)

        result = loss_fn(
            query_embeddings=q,
            positive_embeddings=p,
            visual_embeddings=vis,
            text_for_visual=q,
        )

        # With weight=0, cross-modal loss shouldn't contribute
        assert result["cross_modal_loss_weighted"] == 0.0


# ---------------------------------------------------------------------------
# MultimodalTrainer tests
# ---------------------------------------------------------------------------


class TestMultimodalTrainer:
    """Tests for MultimodalTrainer.compute_loss."""

    def _make_model(self, dim=128):
        """Create a minimal model for testing."""
        model = MagicMock()
        backbone = MockBackbone(dim)
        pooling = MockPooling(dim)
        vision = MockVisionEncoder(dim)

        model.backbone = backbone
        model.pooling = pooling
        model.vision_encoder = vision
        model.video_encoder = None
        model.parameters = lambda: backbone.parameters()

        return model

    def test_compute_loss_text_only(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss
        from omnivector.training.multimodal_trainer import MultimodalTrainer

        model = self._make_model(128)
        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))

        trainer = object.__new__(MultimodalTrainer)
        trainer.multimodal_loss_fn = loss_fn
        trainer.freeze_vision_steps = 0
        trainer._vision_frozen = False

        inputs = {
            "query_input_ids": torch.randint(0, 100, (2, 16)),
            "query_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "positive_input_ids": torch.randint(0, 100, (2, 16)),
            "positive_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "negative_input_ids": torch.zeros(2, 0, 16, dtype=torch.long),
            "negative_attention_mask": torch.zeros(2, 0, 16, dtype=torch.long),
            "has_images": False,
            "has_videos": False,
        }

        loss = trainer.compute_loss(model, inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_compute_loss_with_images(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss
        from omnivector.training.multimodal_trainer import MultimodalTrainer

        model = self._make_model(128)
        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))

        trainer = object.__new__(MultimodalTrainer)
        trainer.multimodal_loss_fn = loss_fn
        trainer.freeze_vision_steps = 0
        trainer._vision_frozen = False

        inputs = {
            "query_input_ids": torch.randint(0, 100, (2, 16)),
            "query_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "positive_input_ids": torch.randint(0, 100, (2, 16)),
            "positive_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "negative_input_ids": torch.zeros(2, 0, 16, dtype=torch.long),
            "negative_attention_mask": torch.zeros(2, 0, 16, dtype=torch.long),
            "images": torch.randn(2, 3, 32, 32),
            "image_mask": torch.tensor([True, True]),
            "has_images": True,
            "has_videos": False,
        }

        loss = trainer.compute_loss(model, inputs)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_compute_loss_removes_labels(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss
        from omnivector.training.multimodal_trainer import MultimodalTrainer

        model = self._make_model(128)
        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))

        trainer = object.__new__(MultimodalTrainer)
        trainer.multimodal_loss_fn = loss_fn
        trainer.freeze_vision_steps = 0
        trainer._vision_frozen = False

        inputs = {
            "labels": torch.zeros(2),
            "query_input_ids": torch.randint(0, 100, (2, 16)),
            "query_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "positive_input_ids": torch.randint(0, 100, (2, 16)),
            "positive_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "negative_input_ids": torch.zeros(2, 0, 16, dtype=torch.long),
            "negative_attention_mask": torch.zeros(2, 0, 16, dtype=torch.long),
            "has_images": False,
            "has_videos": False,
        }

        trainer.compute_loss(model, inputs)
        assert "labels" not in inputs

    def test_compute_loss_with_negatives(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss
        from omnivector.training.multimodal_trainer import MultimodalTrainer

        model = self._make_model(128)
        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))

        trainer = object.__new__(MultimodalTrainer)
        trainer.multimodal_loss_fn = loss_fn
        trainer.freeze_vision_steps = 0
        trainer._vision_frozen = False

        inputs = {
            "query_input_ids": torch.randint(0, 100, (2, 16)),
            "query_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "positive_input_ids": torch.randint(0, 100, (2, 16)),
            "positive_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "negative_input_ids": torch.randint(0, 100, (2, 3, 16)),
            "negative_attention_mask": torch.ones(2, 3, 16, dtype=torch.long),
            "has_images": False,
            "has_videos": False,
        }

        loss = trainer.compute_loss(model, inputs)
        assert loss.item() > 0

    def test_return_outputs(self):
        from omnivector.training.multimodal_loss import MultimodalMRLLoss
        from omnivector.training.multimodal_trainer import MultimodalTrainer

        model = self._make_model(128)
        loss_fn = MultimodalMRLLoss(mrl_dims=(64, 128))

        trainer = object.__new__(MultimodalTrainer)
        trainer.multimodal_loss_fn = loss_fn
        trainer.freeze_vision_steps = 0
        trainer._vision_frozen = False

        inputs = {
            "query_input_ids": torch.randint(0, 100, (2, 16)),
            "query_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "positive_input_ids": torch.randint(0, 100, (2, 16)),
            "positive_attention_mask": torch.ones(2, 16, dtype=torch.long),
            "negative_input_ids": torch.zeros(2, 0, 16, dtype=torch.long),
            "negative_attention_mask": torch.zeros(2, 0, 16, dtype=torch.long),
            "has_images": False,
            "has_videos": False,
        }

        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)
        assert isinstance(loss, torch.Tensor)
        assert "query_embeddings" in outputs
        assert "positive_embeddings" in outputs
