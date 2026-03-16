import logging
import re
from typing import Optional, Union

logger = logging.getLogger(__name__)


def preprocess_text(
    text: str,
    instruction: Optional[str] = None,
    normalize: bool = True,
    max_length: int = 512,
) -> str:
    if not isinstance(text, str):
        raise ValueError("Text must be string")

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty after preprocessing")

    if normalize:
        text = re.sub(r"\s+", " ", text)

    if instruction:
        text = f"Instruct: {instruction}\nQuery: {text}"

    if len(text) > max_length:
        text = text[:max_length].rstrip()

    return text


def preprocess_code(
    code: str,
    language: Optional[str] = None,
    max_length: int = 2048,
) -> str:
    if not isinstance(code, str):
        raise ValueError("Code must be string")

    code = code.strip()

    if not code:
        raise ValueError("Code cannot be empty")

    if len(code) > max_length:
        code = code[:max_length].rstrip()

    if language:
        code = f"Language: {language}\nCode:\n{code}"

    return code


def preprocess_audio(
    audio_path: str,
    sample_rate: int = 16000,
    duration: Optional[float] = None,
) -> dict:
    try:
        import librosa
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "librosa required for audio processing. Install with: pip install librosa"
        ) from e

    if not isinstance(audio_path, str):
        raise ValueError("Audio path must be string")

    try:
        audio_data, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        raise ValueError(f"Failed to load audio: {e}") from e

    if duration is not None:
        max_samples = int(duration * sample_rate)
        audio_data = audio_data[:max_samples]

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return {
        "audio": audio_data,
        "sample_rate": sample_rate,
        "mfcc": mfcc,
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "duration": len(audio_data) / sample_rate,
    }


def preprocess_video(
    video_path: str,
    num_frames: int = 8,
    target_size: tuple[int, int] = (224, 224),
) -> dict:
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "opencv-python required for video processing. Install with: pip install opencv-python"
        ) from e

    if not isinstance(video_path, str):
        raise ValueError("Video path must be string")

    try:
        cap = cv2.VideoCapture(video_path)
    except Exception as e:
        raise ValueError(f"Failed to open video: {e}") from e

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("No frames extracted from video")

    return {
        "frames": frames,
        "num_frames": len(frames),
        "fps": fps,
        "total_frames": total_frames,
        "duration": total_frames / fps if fps > 0 else 0,
    }


def detect_modality(data: Union[str, dict]) -> str:
    if isinstance(data, str):
        if data.endswith((".mp4", ".avi", ".mov", ".mkv")):
            return "video"
        elif data.endswith((".wav", ".mp3", ".flac", ".ogg")):
            return "audio"
        elif "```" in data or "def " in data or "class " in data:
            return "code"
        else:
            return "text"
    elif isinstance(data, dict):
        if "frames" in data:
            return "video"
        elif "audio" in data or "mfcc" in data:
            return "audio"
        else:
            return "text"
    return "text"


def clean_text(text: str) -> str:
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def truncate_text(text: str, max_length: int = 512, truncate_at: str = "end") -> str:
    if len(text) <= max_length:
        return text

    if truncate_at == "end":
        return text[:max_length].rstrip()
    elif truncate_at == "middle":
        half = max_length // 2
        return text[:half] + " ... " + text[-(half - 5) :]
    else:
        raise ValueError(f"Unknown truncate_at: {truncate_at}")


def extract_code_instruction(text: str) -> tuple[str, str]:
    if "```" in text or "def " in text or "class " in text:
        return text, "code"
    elif any(
        keyword in text.upper() for keyword in ["SELECT", "INSERT", "UPDATE", "DELETE", "FROM"]
    ):
        return text, "sql"
    else:
        return text, "general"
