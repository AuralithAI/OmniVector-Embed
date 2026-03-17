"""Image-text pair data loader for multimodal training.

Loads image-text datasets from common formats (COCO captions, Flickr30k,
Conceptual Captions, custom JSONL) into EmbeddingPair-compatible format.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from omnivector.data.schema import EmbeddingPair

logger = logging.getLogger(__name__)


class ImageTextLoader:
    """Loads image-text pairs from various dataset formats.

    Supports:
    - COCO Captions format (annotations JSON with image paths)
    - Flickr30k format
    - Simple JSONL ({"image_path": ..., "caption": ...})
    - HuggingFace datasets
    """

    def __init__(
        self,
        dataset_path: str,
        image_dir: Optional[str] = None,
        format: str = "jsonl",
        max_samples: Optional[int] = None,
        split: str = "train",
    ):
        """Initialize image-text loader.

        Args:
            dataset_path: Path to dataset file or HF dataset name.
            image_dir: Root directory for images if paths are relative.
            format: Dataset format ('jsonl', 'coco', 'flickr30k', 'hf').
            max_samples: Maximum number of samples to load.
            split: Dataset split to use.
        """
        self.dataset_path = dataset_path
        self.image_dir = Path(image_dir) if image_dir else None
        self.format = format
        self.max_samples = max_samples
        self.split = split

    def load(self) -> list[dict]:
        """Load image-text pairs.

        Returns:
            List of dicts with keys: image_path (str), caption (str),
            negative_captions (list[str]), domain (str).
        """
        if self.format == "jsonl":
            return self._load_jsonl()
        elif self.format == "coco":
            return self._load_coco()
        elif self.format == "hf":
            return self._load_hf()
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _load_jsonl(self) -> list[dict]:
        """Load from JSONL format."""
        pairs = []
        path = Path(self.dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path) as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break

                record = json.loads(line.strip())
                image_path = record.get("image_path", record.get("image", ""))

                if self.image_dir:
                    image_path = str(self.image_dir / image_path)

                pairs.append({
                    "image_path": image_path,
                    "caption": record.get("caption", record.get("text", "")),
                    "negative_captions": record.get("negative_captions", []),
                    "domain": record.get("domain", "image_text"),
                })

        logger.info(f"Loaded {len(pairs)} image-text pairs from {path}")
        return pairs

    def _load_coco(self) -> list[dict]:
        """Load from COCO Captions JSON format."""
        path = Path(self.dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"COCO annotations not found: {path}")

        with open(path) as f:
            data = json.load(f)

        # Build image ID -> filename mapping
        image_map = {}
        for img in data.get("images", []):
            image_map[img["id"]] = img["file_name"]

        pairs = []
        for ann in data.get("annotations", []):
            if self.max_samples and len(pairs) >= self.max_samples:
                break

            image_id = ann["image_id"]
            filename = image_map.get(image_id, "")
            image_path = filename

            if self.image_dir:
                image_path = str(self.image_dir / filename)

            pairs.append({
                "image_path": image_path,
                "caption": ann["caption"],
                "negative_captions": [],
                "domain": "image_text",
            })

        logger.info(f"Loaded {len(pairs)} COCO caption pairs from {path}")
        return pairs

    def _load_hf(self) -> list[dict]:
        """Load from HuggingFace datasets."""
        from datasets import load_dataset

        ds = load_dataset(self.dataset_path, split=self.split)

        if self.max_samples:
            ds = ds.select(range(min(self.max_samples, len(ds))))

        pairs = []
        for sample in ds:
            # Detect common column names
            image_col = None
            text_col = None

            for col in ["image", "image_path", "file_name"]:
                if col in sample:
                    image_col = col
                    break

            for col in ["caption", "text", "sentence"]:
                if col in sample:
                    text_col = col
                    break

            if image_col and text_col:
                pairs.append({
                    "image_path": str(sample[image_col]),
                    "caption": sample[text_col],
                    "negative_captions": [],
                    "domain": "image_text",
                })

        logger.info(f"Loaded {len(pairs)} pairs from HF dataset {self.dataset_path}")
        return pairs

    def to_embedding_pairs(self, pairs: Optional[list[dict]] = None) -> list[EmbeddingPair]:
        """Convert image-text pairs to EmbeddingPair format for text-side training.

        This converts captions into text-format EmbeddingPairs for joint
        training with text data. The image paths are stored in the domain field.

        Args:
            pairs: Pre-loaded pairs, or None to load fresh.

        Returns:
            List of EmbeddingPair objects.
        """
        if pairs is None:
            pairs = self.load()

        return [
            EmbeddingPair(
                query=p["caption"],
                positive=p["caption"],
                negatives=p.get("negative_captions", []),
                query_instruction="Describe the image",
                domain=f"image_text:{p['image_path']}",
            )
            for p in pairs
        ]


class VideoTextLoader:
    """Loads video-text pairs from dataset files.

    Supports:
    - Simple JSONL ({"video_path": ..., "caption": ...})
    - WebVid/MSR-VTT style annotations
    """

    def __init__(
        self,
        dataset_path: str,
        video_dir: Optional[str] = None,
        format: str = "jsonl",
        max_samples: Optional[int] = None,
        num_frames: int = 8,
    ):
        """Initialize video-text loader.

        Args:
            dataset_path: Path to dataset file.
            video_dir: Root directory for videos if paths are relative.
            format: Dataset format ('jsonl').
            max_samples: Maximum number of samples to load.
            num_frames: Number of frames to sample per video.
        """
        self.dataset_path = dataset_path
        self.video_dir = Path(video_dir) if video_dir else None
        self.format = format
        self.max_samples = max_samples
        self.num_frames = num_frames

    def load(self) -> list[dict]:
        """Load video-text pairs.

        Returns:
            List of dicts with keys: video_path (str), caption (str),
            negative_captions (list[str]), domain (str), num_frames (int).
        """
        if self.format == "jsonl":
            return self._load_jsonl()
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _load_jsonl(self) -> list[dict]:
        """Load from JSONL format."""
        pairs = []
        path = Path(self.dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        with open(path) as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break

                record = json.loads(line.strip())
                video_path = record.get("video_path", record.get("video", ""))

                if self.video_dir:
                    video_path = str(self.video_dir / video_path)

                pairs.append({
                    "video_path": video_path,
                    "caption": record.get("caption", record.get("text", "")),
                    "negative_captions": record.get("negative_captions", []),
                    "domain": record.get("domain", "video_text"),
                    "num_frames": self.num_frames,
                })

        logger.info(f"Loaded {len(pairs)} video-text pairs from {path}")
        return pairs
