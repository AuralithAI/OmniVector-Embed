"""Download audio datasets for Stage 3 multimodal training.

Uses HuggingFace `datasets` library to stream audio directly from parquet
files — no corrupted zip archives, no YouTube downloads, no external CDNs.

Sources:
  1. ESC-50: 2,000 environmental sounds (dog, rain, chainsaw, etc.)
     - Audio embedded in parquet, category labels → generate captions
  2. People's Speech: 1.5M English speech clips with transcripts
     - Audio embedded in parquet, transcripts serve as captions
  3. WavCaps metadata + any successfully extracted FLAC files
     - Uses whatever audio already exists on disk from prior runs

Usage:
    python scripts/download_wavcaps_audio.py \\
        --output-dir data/stage3_multimodal \\
        --max-samples 20000

Output:
    data/stage3_multimodal/audio/           — WAV/FLAC audio files
    data/stage3_multimodal/audio_text_pairs.jsonl  — {audio_path, caption, domain}
"""

import argparse
import json
import logging
import os
import numpy as np
from pathlib import Path

# Force HF datasets to use soundfile for audio decoding (not torchcodec)
os.environ["HF_AUDIO_DECODER"] = "soundfile"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── ESC-50 category → natural language caption templates ──────────────

ESC50_CAPTIONS = {
    "dog": "A dog barking",
    "rooster": "A rooster crowing",
    "pig": "A pig oinking",
    "cow": "A cow mooing",
    "frog": "A frog croaking",
    "cat": "A cat meowing",
    "hen": "A hen clucking",
    "insects": "Insects buzzing and chirping",
    "sheep": "A sheep bleating",
    "crow": "A crow cawing",
    "rain": "The sound of rain falling",
    "sea_waves": "Ocean waves crashing on shore",
    "crackling_fire": "A fire crackling",
    "crickets": "Crickets chirping at night",
    "chirping_birds": "Birds chirping in the distance",
    "water_drops": "Water drops dripping",
    "wind": "The wind blowing strongly",
    "pouring_water": "Water being poured",
    "toilet_flush": "A toilet flushing",
    "thunderstorm": "A thunderstorm with lightning",
    "crying_baby": "A baby crying",
    "sneezing": "Someone sneezing",
    "clapping": "People clapping their hands",
    "breathing": "Heavy breathing sounds",
    "coughing": "Someone coughing",
    "footsteps": "Footsteps walking on a surface",
    "laughing": "Someone laughing",
    "brushing_teeth": "The sound of brushing teeth",
    "snoring": "A person snoring loudly",
    "drinking_sipping": "Someone drinking or sipping a beverage",
    "door_wood_knock": "A knock on a wooden door",
    "mouse_click": "A mouse clicking",
    "keyboard_typing": "Typing on a keyboard",
    "door_wood_creaks": "A wooden door creaking open",
    "can_opening": "Opening a can",
    "washing_machine": "A washing machine running",
    "vacuum_cleaner": "A vacuum cleaner in operation",
    "clock_alarm": "An alarm clock ringing",
    "clock_tick": "A clock ticking steadily",
    "glass_breaking": "Glass shattering and breaking",
    "helicopter": "A helicopter flying overhead",
    "chainsaw": "A chainsaw cutting wood",
    "siren": "An emergency siren wailing",
    "car_horn": "A car horn honking",
    "engine": "An engine running",
    "train": "A train passing by on tracks",
    "church_bells": "Church bells ringing",
    "airplane": "An airplane flying overhead",
    "fireworks": "Fireworks exploding in the sky",
    "hand_saw": "A hand saw cutting back and forth",
}


def download_esc50(output_audio_dir: Path, max_samples: int) -> list[dict]:
    """Download ESC-50 environmental sounds via HF datasets.

    Audio is embedded directly in parquet — no external downloads needed.

    Args:
        output_audio_dir: Directory to save audio files.
        max_samples: Max samples to download.

    Returns:
        List of {audio_path, caption, domain} dicts.
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    logger.info("Loading ESC-50 dataset from HuggingFace (audio in parquet)...")
    # Load WITHOUT decoding audio — avoids torchcodec/FFmpeg dependency entirely
    ds = load_dataset("ashraq/esc50", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info(f"  ESC-50 loaded: {len(ds)} samples")

    pairs = []
    count = 0
    for idx, sample in enumerate(ds):
        if count >= max_samples:
            break

        category = sample.get("category", "")
        audio_data = sample.get("audio", {})
        if not audio_data or not category:
            continue

        caption = ESC50_CAPTIONS.get(
            category, f"The sound of {category.replace('_', ' ')}"
        )

        # audio_data is {"bytes": b"...", "path": "..."} when decode=False
        raw_bytes = audio_data.get("bytes")
        if not raw_bytes:
            continue

        # Save as WAV using soundfile to decode + re-encode
        filename = f"esc50_{idx:05d}.wav"
        filepath = output_audio_dir / filename
        if not filepath.exists():
            try:
                array, sr = sf.read(io.BytesIO(raw_bytes))
                sf.write(str(filepath), array, sr)
            except Exception as e:
                logger.warning(f"  Failed to decode ESC-50 sample {idx}: {e}")
                continue

        pairs.append({
            "audio_path": filename,
            "caption": caption,
            "domain": "audio_text",
        })
        count += 1

        if count % 500 == 0:
            logger.info(
                f"  [ESC-50] Saved {count}/{min(max_samples, len(ds))} audio files"
            )

    logger.info(f"  [ESC-50] Complete: {count} audio-text pairs")
    return pairs


def download_peoples_speech(
    output_audio_dir: Path, max_samples: int
) -> list[dict]:
    """Download People's Speech clips via HF datasets (streaming).

    Streams audio directly from parquet — no external downloads needed.

    Args:
        output_audio_dir: Directory to save audio files.
        max_samples: Max samples to download.

    Returns:
        List of {audio_path, caption, domain} dicts.
    """
    import io

    import soundfile as sf
    from datasets import Audio, load_dataset

    logger.info("Loading People's Speech dataset (streaming mode)...")
    ds = load_dataset(
        "MLCommons/peoples_speech",
        name="clean",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    # Disable audio decoding — we decode with soundfile ourselves
    ds = ds.cast_column("audio", Audio(decode=False))
    logger.info("  People's Speech stream opened, downloading samples...")

    pairs = []
    count = 0
    skipped = 0
    scanned = 0
    for sample in ds:
        if count >= max_samples:
            break

        scanned += 1
        if scanned <= 20 and scanned % 5 == 0:
            logger.info(
                f"  [People's Speech] Scanned {scanned} records so far "
                f"(saved {count}, skipped {skipped})"
            )
        elif scanned <= 100 and scanned % 25 == 0:
            logger.info(
                f"  [People's Speech] Scanned {scanned} records so far "
                f"(saved {count}, skipped {skipped})"
            )

        text = sample.get("text", "").strip()
        audio_data = sample.get("audio", {})

        if not text or len(text) < 20:
            skipped += 1
            continue

        # audio_data is {"bytes": b"...", "path": "..."} when decode=False
        raw_bytes = audio_data.get("bytes")
        if not raw_bytes:
            skipped += 1
            continue

        # Decode with soundfile
        try:
            array, sr = sf.read(io.BytesIO(raw_bytes))
        except Exception:
            skipped += 1
            continue

        # Skip very short clips (< 2 seconds)
        duration = len(array) / sr
        if duration < 2.0:
            skipped += 1
            continue

        # Truncate very long clips to 30 seconds
        if duration > 30.0:
            array = array[: int(30.0 * sr)]

        # Save as WAV
        filename = f"speech_{count:06d}.wav"
        filepath = output_audio_dir / filename
        if not filepath.exists():
            sf.write(str(filepath), array, sr)

        # Use transcript as caption with a prefix for variety
        caption = f"A person saying: {text}"

        pairs.append({
            "audio_path": filename,
            "caption": caption,
            "domain": "audio_text",
        })
        count += 1

        if count == 1:
            logger.info(
                f"  [People's Speech] First sample saved successfully "
                f"(scanned {scanned}, skipped {skipped}). Streaming..."
            )
        elif count % 500 == 0:
            logger.info(
                f"  [People's Speech] Saved {count}/{max_samples} "
                f"(scanned {scanned}, skipped {skipped})"
            )

    logger.info(
        f"  [People's Speech] Complete: {count} audio-text pairs "
        f"(skipped {skipped})"
    )
    return pairs


def collect_existing_wavcaps(
    output_audio_dir: Path, output_dir: Path, max_samples: int
) -> list[dict]:
    """Collect any WavCaps FLAC files already on disk from prior runs.

    Reads WavCaps JSON metadata from HF cache and matches against existing
    audio files in audio_dir.

    Args:
        output_audio_dir: Directory containing audio files.
        output_dir: Base output directory (has .cache/).
        max_samples: Max samples to collect.

    Returns:
        List of {audio_path, caption, domain} dicts.
    """
    existing_flacs = {
        f.name
        for f in output_audio_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".flac", ".wav", ".mp3")
        and not f.name.startswith(("esc50_", "speech_"))
    }

    if not existing_flacs:
        logger.info("  No existing WavCaps FLAC files found on disk")
        return []

    logger.info(
        f"  Found {len(existing_flacs)} existing WavCaps audio files on disk"
    )

    # Try to load metadata to get captions
    cache_dir = output_dir / ".cache"
    pairs = []

    source_configs = {
        "freesound": {
            "json_dir": "json_files/FreeSound",
            "json_file": "fsd_final.json",
        },
        "bbc_sound_effects": {
            "json_dir": "json_files/BBC_Sound_Effects",
            "json_file": "bbc_final.json",
        },
        "soundbible": {
            "json_dir": "json_files/SoundBible",
            "json_file": "sb_final.json",
        },
    }

    for source_name, src in source_configs.items():
        if len(pairs) >= max_samples:
            break

        try:
            from huggingface_hub import hf_hub_download

            logger.info(f"  Downloading {source_name} metadata...")
            json_path = hf_hub_download(
                repo_id="cvssp/WavCaps",
                filename=f"{src['json_dir']}/{src['json_file']}",
                repo_type="dataset",
                cache_dir=str(cache_dir / "hf_cache"),
            )

            logger.info(f"  Parsing {source_name} JSON...")
            with open(json_path) as f:
                data = json.load(f)

            records = (
                data.get("data", data) if isinstance(data, dict) else data
            )
            logger.info(
                f"  Loaded {len(records)} metadata from {source_name}, "
                f"matching against {len(existing_flacs)} files on disk..."
            )

            matched = 0
            scanned = 0
            for rec in records:
                if len(pairs) >= max_samples:
                    break

                scanned += 1
                if scanned % 50000 == 0:
                    logger.info(
                        f"    [{source_name}] Scanned {scanned}/{len(records)} "
                        f"records, {matched} matches so far"
                    )

                caption = rec.get("caption", "") or rec.get(
                    "raw_description", ""
                )
                if not caption:
                    continue

                audio_file = ""
                for key in ("id", "file_name", "fname", "audio"):
                    val = rec.get(key, "")
                    if val:
                        audio_file = Path(str(val)).name
                        break
                if not audio_file:
                    continue

                if not any(
                    audio_file.endswith(ext)
                    for ext in (".flac", ".wav", ".mp3")
                ):
                    audio_file += ".flac"

                if audio_file in existing_flacs:
                    pairs.append({
                        "audio_path": audio_file,
                        "caption": caption,
                        "domain": "audio_text",
                    })
                    matched += 1

            logger.info(
                f"  [{source_name}] Scanned {scanned} records, "
                f"matched {matched} captions to files on disk"
            )

        except Exception as e:
            logger.warning(
                f"  Could not load {source_name} metadata: {e}"
            )
            continue

    logger.info(
        f"  [WavCaps existing] Total: {len(pairs)} audio-caption pairs"
    )
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Download audio datasets for Stage 3 multimodal training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/stage3_multimodal",
        help="Output directory (audio/ subdir and audio_text_pairs.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20_000,
        help="Maximum audio-text pairs to produce.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["esc50", "peoples_speech", "wavcaps_existing"],
        choices=["esc50", "peoples_speech", "wavcaps_existing"],
        help="Audio sources to use.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_pairs: list[dict] = []
    remaining = args.max_samples

    # ── Source 1: ESC-50 (2,000 environmental sounds) ─────────────
    if "esc50" in args.sources and remaining > 0:
        logger.info(
            f"\n{'=' * 60}\n"
            f"Source 1: ESC-50 (2,000 environmental sounds)\n"
            f"{'=' * 60}"
        )
        esc50_limit = min(2000, remaining)
        pairs = download_esc50(audio_dir, esc50_limit)
        all_pairs.extend(pairs)
        remaining -= len(pairs)
        logger.info(
            f"  ESC-50: {len(pairs)} pairs, remaining quota: {remaining}"
        )

    # ── Source 2: WavCaps existing files on disk ──────────────────
    if "wavcaps_existing" in args.sources and remaining > 0:
        logger.info(
            f"\n{'=' * 60}\n"
            f"Source 2: WavCaps (existing files on disk)\n"
            f"{'=' * 60}"
        )
        pairs = collect_existing_wavcaps(audio_dir, output_dir, remaining)
        all_pairs.extend(pairs)
        remaining -= len(pairs)
        logger.info(
            f"  WavCaps existing: {len(pairs)} pairs, "
            f"remaining quota: {remaining}"
        )

    # ── Source 3: People's Speech (streaming, fills remaining) ────
    if "peoples_speech" in args.sources and remaining > 0:
        logger.info(
            f"\n{'=' * 60}\n"
            f"Source 3: People's Speech "
            f"({remaining} clips to fill quota)\n"
            f"{'=' * 60}"
        )
        pairs = download_peoples_speech(audio_dir, remaining)
        all_pairs.extend(pairs)
        remaining -= len(pairs)
        logger.info(
            f"  People's Speech: {len(pairs)} pairs, "
            f"remaining quota: {remaining}"
        )

    # ── Build final JSONL ─────────────────────────────────────────
    jsonl_path = output_dir / "audio_text_pairs.jsonl"
    with open(jsonl_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {len(all_pairs)} audio-text pairs to {jsonl_path}")

    # Summary
    source_counts: dict[str, int] = {}
    for pair in all_pairs:
        name = pair["audio_path"]
        if name.startswith("esc50_"):
            key = "esc50"
        elif name.startswith("speech_"):
            key = "peoples_speech"
        else:
            key = "wavcaps"
        source_counts[key] = source_counts.get(key, 0) + 1

    stats = {
        "total_pairs": len(all_pairs),
        "source_breakdown": source_counts,
        "audio_dir": str(audio_dir),
        "jsonl_path": str(jsonl_path),
    }
    logger.info(f"Audio download complete:\n{json.dumps(stats, indent=2)}")

    # Audio files on disk
    audio_count = sum(
        1
        for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".flac", ".wav", ".mp3")
    )
    logger.info(f"Total audio files on disk: {audio_count}")


if __name__ == "__main__":
    main()
