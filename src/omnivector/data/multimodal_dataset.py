"""Multimodal dataset and collator for joint text-image-video training.

Extends the text-only EmbeddingDataset to handle mixed-modality batches
where each sample can be text-text, image-text, or video-text.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from omnivector.data.schema import EmbeddingPair

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported input modalities."""
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class MultimodalSample:
    """Single training sample that may contain text, image, video, or audio.

    Attributes:
        query_text: Text query (always present for contrastive learning).
        positive_text: Positive text passage.
        negatives: Hard negative texts.
        query_instruction: Instruction prefix.
        domain: Domain tag.
        modality: Primary modality (text, image, video, audio).
        image_path: Path to image file (if modality == IMAGE).
        video_path: Path to video file (if modality == VIDEO).
        audio_path: Path to audio file (if modality == AUDIO).
    """

    def __init__(
        self,
        query_text: str,
        positive_text: str,
        negatives: Optional[list[str]] = None,
        query_instruction: Optional[str] = None,
        domain: str = "general",
        modality: Modality = Modality.TEXT,
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ):
        self.query_text = query_text
        self.positive_text = positive_text
        self.negatives = negatives or []
        self.query_instruction = query_instruction
        self.domain = domain
        self.modality = modality
        self.image_path = image_path
        self.video_path = video_path
        self.audio_path = audio_path

    @classmethod
    def from_embedding_pair(cls, pair: EmbeddingPair) -> "MultimodalSample":
        """Create from text EmbeddingPair."""
        return cls(
            query_text=pair.query,
            positive_text=pair.positive,
            negatives=pair.negatives,
            query_instruction=pair.query_instruction,
            domain=pair.domain,
            modality=Modality.TEXT,
        )

    @classmethod
    def from_image_text(cls, image_path: str, caption: str, negatives: Optional[list[str]] = None) -> "MultimodalSample":
        """Create from image-text pair."""
        return cls(
            query_text=caption,
            positive_text=caption,
            negatives=negatives or [],
            query_instruction="Describe the image",
            domain="image_text",
            modality=Modality.IMAGE,
            image_path=image_path,
        )

    @classmethod
    def from_video_text(cls, video_path: str, caption: str, negatives: Optional[list[str]] = None) -> "MultimodalSample":
        """Create from video-text pair."""
        return cls(
            query_text=caption,
            positive_text=caption,
            negatives=negatives or [],
            query_instruction="Describe the video",
            domain="video_text",
            modality=Modality.VIDEO,
            video_path=video_path,
        )

    @classmethod
    def from_audio_text(cls, audio_path: str, caption: str, negatives: Optional[list[str]] = None) -> "MultimodalSample":
        """Create from audio-text pair."""
        return cls(
            query_text=caption,
            positive_text=caption,
            negatives=negatives or [],
            query_instruction="Describe the audio",
            domain="audio_text",
            modality=Modality.AUDIO,
            audio_path=audio_path,
        )


class MultimodalDataset(Dataset):
    """Dataset for mixed-modality training.

    Each sample contains text tokens and optionally image/video tensors.
    The collator handles padding and batching across modalities.

    Attributes:
        samples: List of MultimodalSample objects.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum token sequence length.
        image_transform: Callable to transform image path -> tensor.
        video_transform: Callable to transform video path -> tensor.
    """

    def __init__(
        self,
        samples: list[MultimodalSample],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        image_transform: Optional[Any] = None,
        video_transform: Optional[Any] = None,
        audio_transform: Optional[Any] = None,
        image_size: int = 384,
        num_frames: int = 8,
        audio_sampling_rate: int = 16_000,
    ):
        """Initialize multimodal dataset.

        Args:
            samples: List of training samples.
            tokenizer: Pre-initialized tokenizer.
            max_length: Maximum token sequence length.
            image_transform: Image preprocessing transform (PIL -> tensor).
                If None, uses default resize+normalize.
            video_transform: Video preprocessing transform.
            audio_transform: Audio preprocessing callable (path -> tensor).
            image_size: Target image size for default transform.
            num_frames: Number of video frames to sample.
            audio_sampling_rate: Audio sampling rate for Whisper (16kHz).
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_transform = image_transform
        self.video_transform = video_transform
        self.audio_transform = audio_transform
        self.image_size = image_size
        self.num_frames = num_frames
        self.audio_sampling_rate = audio_sampling_rate

        # Count modalities
        counts = {m: 0 for m in Modality}
        for s in samples:
            counts[s.modality] += 1

        logger.info(
            f"MultimodalDataset: {len(samples)} samples "
            f"(text={counts[Modality.TEXT]}, image={counts[Modality.IMAGE]}, "
            f"video={counts[Modality.VIDEO]}, audio={counts[Modality.AUDIO]})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a single sample.

        Returns:
            Dict with keys:
            - query_tokens: tokenized query text
            - positive_tokens: tokenized positive text
            - negative_tokens: list of tokenized negatives
            - modality: str modality type
            - image: tensor [3, H, W] or None
            - video: tensor [n_frames, 3, H, W] or None
            - audio: tensor [n_mels, seq_len] or None
            - domain: str domain tag
        """
        sample = self.samples[idx]

        # Prepare query text with instruction
        query_text = sample.query_text
        if sample.query_instruction:
            query_text = f"Instruct: {sample.query_instruction}\nQuery: {query_text}"

        # Tokenize text
        query_tokens = self.tokenizer(
            query_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        positive_tokens = self.tokenizer(
            sample.positive_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        negative_tokens = []
        for neg in sample.negatives:
            neg_tokens = self.tokenizer(
                neg,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            negative_tokens.append(neg_tokens)

        result = {
            "query_tokens": query_tokens,
            "positive_tokens": positive_tokens,
            "negative_tokens": negative_tokens,
            "modality": sample.modality.value,
            "domain": sample.domain,
            "image": None,
            "video": None,
            "audio": None,
        }

        # Load image if applicable
        if sample.modality == Modality.IMAGE and sample.image_path:
            result["image"] = self._load_image(sample.image_path)

        # Load video if applicable
        if sample.modality == Modality.VIDEO and sample.video_path:
            result["video"] = self._load_video(sample.video_path)

        # Load audio if applicable
        if sample.modality == Modality.AUDIO and sample.audio_path:
            result["audio"] = self._load_audio(sample.audio_path)

        return result

    def _load_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess image.

        Args:
            image_path: Path to image file.

        Returns:
            Image tensor [3, H, W] or None on failure.
        """
        try:
            if self.image_transform:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                return self.image_transform(img)

            # Default transform: resize, to tensor, normalize
            from PIL import Image
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            img = Image.open(image_path).convert("RGB")
            return transform(img)
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None

    def _load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess video frames.

        Args:
            video_path: Path to video file.

        Returns:
            Video tensor [n_frames, 3, H, W] or None on failure.
        """
        try:
            if self.video_transform:
                return self.video_transform(video_path)

            from omnivector.data.preprocessing import preprocess_video

            result = preprocess_video(
                video_path,
                num_frames=self.num_frames,
                target_size=(self.image_size, self.image_size),
            )

            frames = np.array(result["frames"], dtype=np.float32)
            # [N, H, W, 3] -> [N, 3, H, W], normalize to [0, 1]
            frames = frames.transpose(0, 3, 1, 2) / 255.0
            return torch.from_numpy(frames)
        except Exception as e:
            logger.warning(f"Failed to load video {video_path}: {e}")
            return None

    def _load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess audio to log-mel spectrogram.

        Args:
            audio_path: Path to audio file (.wav, .mp3, .flac).

        Returns:
            Log-mel spectrogram tensor [n_mels, seq_len] or None on failure.
        """
        try:
            if self.audio_transform:
                return self.audio_transform(audio_path)

            # Default: load audio with librosa and convert to Whisper features
            import librosa

            waveform, sr = librosa.load(
                audio_path,
                sr=self.audio_sampling_rate,
                mono=True,
            )

            # Use Whisper feature extractor if available
            try:
                from transformers import WhisperFeatureExtractor

                extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
                features = extractor(
                    waveform,
                    sampling_rate=self.audio_sampling_rate,
                    return_tensors="pt",
                )
                return features.input_features[0]  # [n_mels, seq_len]
            except (ImportError, OSError):
                # Fallback: compute log-mel manually
                mel = librosa.feature.melspectrogram(
                    y=waveform, sr=sr, n_mels=80, n_fft=400, hop_length=160,
                )
                log_mel = librosa.power_to_db(mel, ref=np.max)
                return torch.from_numpy(log_mel.astype(np.float32))

        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            return None


class MultimodalCollator:
    """Collates mixed-modality batches for training.

    Groups samples by modality within a batch, pads text sequences,
    and stacks image/video tensors where available.

    Attributes:
        tokenizer: HuggingFace tokenizer.
        max_negatives: Maximum negatives per sample.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_negatives: int = 7,
        image_size: int = 384,
    ):
        """Initialize collator.

        Args:
            tokenizer: Pre-initialized tokenizer.
            max_negatives: Max negatives to include per batch.
            image_size: Image tensor size for creating zero placeholders.
        """
        self.tokenizer = tokenizer
        self.max_negatives = max_negatives
        self.image_size = image_size

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of multimodal samples.

        Args:
            batch: List of sample dicts from MultimodalDataset.

        Returns:
            Collated batch dict with keys:
            - query_input_ids, query_attention_mask
            - positive_input_ids, positive_attention_mask
            - negative_input_ids, negative_attention_mask
            - images: [batch_size, 3, H, W] or None
            - videos: [batch_size, n_frames, 3, H, W] or None
            - audio_features: [batch_size, n_mels, seq_len] or None
            - modalities: list[str] per sample
            - has_images, has_videos, has_audio: bool
        """
        if not batch:
            raise ValueError("Empty batch")

        batch_size = len(batch)

        # Collate text tokens
        query_input_ids = torch.stack([b["query_tokens"]["input_ids"][0] for b in batch])
        query_attention_mask = torch.stack([b["query_tokens"]["attention_mask"][0] for b in batch])
        positive_input_ids = torch.stack([b["positive_tokens"]["input_ids"][0] for b in batch])
        positive_attention_mask = torch.stack([b["positive_tokens"]["attention_mask"][0] for b in batch])

        # Collate negatives (variable length)
        max_negs = min(
            max((len(b["negative_tokens"]) for b in batch), default=0),
            self.max_negatives,
        )

        seq_len = query_input_ids.shape[1]

        if max_negs > 0:
            neg_input_ids_list = []
            neg_attention_mask_list = []

            for item in batch:
                negs = item["negative_tokens"][:max_negs]
                while len(negs) < max_negs:
                    negs.append({
                        "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
                        "attention_mask": torch.zeros(1, seq_len, dtype=torch.long),
                    })

                neg_input_ids_list.append(torch.stack([n["input_ids"][0] for n in negs]))
                neg_attention_mask_list.append(torch.stack([n["attention_mask"][0] for n in negs]))

            negative_input_ids = torch.stack(neg_input_ids_list)
            negative_attention_mask = torch.stack(neg_attention_mask_list)
        else:
            negative_input_ids = torch.zeros(batch_size, 0, seq_len, dtype=torch.long)
            negative_attention_mask = torch.zeros(batch_size, 0, seq_len, dtype=torch.long)

        # Collate images
        images = [b["image"] for b in batch]
        has_images = any(img is not None for img in images)

        if has_images:
            # Replace None images with zero tensors
            ref_img = next(img for img in images if img is not None)
            images_tensor = torch.stack([
                img if img is not None else torch.zeros_like(ref_img)
                for img in images
            ])
            image_mask = torch.tensor([img is not None for img in images], dtype=torch.bool)
        else:
            images_tensor = None
            image_mask = None

        # Collate videos
        videos = [b["video"] for b in batch]
        has_videos = any(vid is not None for vid in videos)

        if has_videos:
            ref_vid = next(vid for vid in videos if vid is not None)
            videos_tensor = torch.stack([
                vid if vid is not None else torch.zeros_like(ref_vid)
                for vid in videos
            ])
            video_mask = torch.tensor([vid is not None for vid in videos], dtype=torch.bool)
        else:
            videos_tensor = None
            video_mask = None

        # Collate audio
        audios = [b.get("audio") for b in batch]
        has_audio = any(aud is not None for aud in audios)

        if has_audio:
            ref_aud = next(aud for aud in audios if aud is not None)
            audios_tensor = torch.stack([
                aud if aud is not None else torch.zeros_like(ref_aud)
                for aud in audios
            ])
            audio_mask = torch.tensor([aud is not None for aud in audios], dtype=torch.bool)
        else:
            audios_tensor = None
            audio_mask = None

        modalities = [b["modality"] for b in batch]

        return {
            "query_input_ids": query_input_ids,
            "query_attention_mask": query_attention_mask,
            "positive_input_ids": positive_input_ids,
            "positive_attention_mask": positive_attention_mask,
            "negative_input_ids": negative_input_ids,
            "negative_attention_mask": negative_attention_mask,
            "images": images_tensor,
            "image_mask": image_mask,
            "videos": videos_tensor,
            "video_mask": video_mask,
            "audio_features": audios_tensor,
            "audio_mask": audio_mask,
            "modalities": modalities,
            "has_images": has_images,
            "has_videos": has_videos,
            "has_audio": has_audio,
        }
