import logging
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

# Ensure project root is on sys.path so `scripts/` is importable in CI
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="session")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}")
        return None


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks.",
    ]


@pytest.fixture
def sample_embedding_pair():
    from omnivector.data.schema import EmbeddingPair

    return EmbeddingPair(
        query="What is machine learning?",
        positive="Machine learning is a subset of AI focused on data-driven learning.",
        negatives=["Python is a programming language.", "The weather is sunny today."],
        query_instruction="Search query",
        domain="retrieval",
    )


@pytest.fixture
def sample_batch(sample_embedding_pair):
    return [sample_embedding_pair] * 4


@pytest.fixture
def sample_audio_path(tmp_path):
    import numpy as np

    audio_file = tmp_path / "test_audio.wav"
    sample_rate = 16000
    duration = 1
    num_samples = sample_rate * duration
    audio_data = np.random.randn(num_samples)

    try:
        import soundfile as sf

        sf.write(str(audio_file), audio_data, sample_rate)
        return str(audio_file)
    except ImportError:
        pytest.skip("soundfile not installed")


@pytest.fixture
def sample_video_path(tmp_path):
    import numpy as np

    video_file = tmp_path / "test_video.mp4"

    try:
        import cv2

        frame_width, frame_height = 640, 480
        fps = 30
        duration = 1
        num_frames = fps * duration

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_file), fourcc, fps, (frame_width, frame_height))

        for _ in range(num_frames):
            frame = np.random.randint(0, 255, (frame_height, frame_width, 3), dtype=np.uint8)
            out.write(frame)

        out.release()
        return str(video_file)
    except ImportError:
        pytest.skip("opencv-python not installed")


@pytest.fixture
def mock_backbone(device):
    from omnivector.model.backbone import MistralEmbeddingBackbone

    try:
        backbone = MistralEmbeddingBackbone(
            model_name="mistralai/Mistral-7B-v0.1",
            use_lora=False,
        )
        backbone = backbone.to(device)
        backbone.eval()
        return backbone
    except Exception as e:
        logger.warning(f"Failed to create backbone: {e}")
        return None


@pytest.fixture
def mock_pooling():
    from omnivector.model.latent_attention import LatentAttentionPooling

    pooling = LatentAttentionPooling(
        embed_dim=4096,
        n_latents=512,
        num_heads=8,
    )
    pooling.eval()
    return pooling
