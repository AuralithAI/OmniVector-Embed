"""Download WavCaps audio dataset for Stage 3 multimodal training.

Downloads FLAC audio files + caption metadata from the WavCaps dataset
(cvssp/WavCaps on HuggingFace) which provides pre-extracted audio with
natural language captions from FreeSound, BBC Sound Effects, and SoundBible.

This replaces the broken AudioSet/YouTube approach — WavCaps ships actual
audio files as zip archives on HuggingFace.

Usage:
    python scripts/download_wavcaps_audio.py \
        --output-dir data/stage3_multimodal \
        --max-samples 20000 \
        --sources freesound bbc_sound_effects soundbible

Output:
    data/stage3_multimodal/audio/           — extracted FLAC files
    data/stage3_multimodal/audio_text_pairs.jsonl  — {audio_path, caption, domain}
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# WavCaps HuggingFace repo structure
WAVCAPS_REPO = "cvssp/WavCaps"

# Source configs: json metadata path + zip directory on HF
SOURCES = {
    "freesound": {
        "json_dir": "json_files/FreeSound",
        "json_file": "fsd_final.json",
        "zip_dir": "Zip_files/FreeSound",
        "total_clips": 262_300,
        "avg_duration_s": 86.0,
    },
    "bbc_sound_effects": {
        "json_dir": "json_files/BBC_Sound_Effects",
        "json_file": "bbc_final.json",
        "zip_dir": "Zip_files/BBC_Sound_Effects",
        "total_clips": 31_201,
        "avg_duration_s": 115.0,
    },
    "soundbible": {
        "json_dir": "json_files/SoundBible",
        "json_file": "sb_final.json",
        "zip_dir": "Zip_files/SoundBible",
        "total_clips": 1_232,
        "avg_duration_s": 13.0,
    },
}


def download_json_metadata(source_name: str, cache_dir: Path) -> list[dict]:
    """Download and parse WavCaps JSON metadata for a source.

    Args:
        source_name: One of 'freesound', 'bbc_sound_effects', 'soundbible'.
        cache_dir: Local cache directory.

    Returns:
        List of dicts with 'id', 'caption', 'audio_filename' keys.
    """
    from huggingface_hub import hf_hub_download

    source = SOURCES[source_name]
    json_path = hf_hub_download(
        repo_id=WAVCAPS_REPO,
        filename=f"{source['json_dir']}/{source['json_file']}",
        repo_type="dataset",
        cache_dir=str(cache_dir / "hf_cache"),
    )

    with open(json_path) as f:
        data = json.load(f)

    # WavCaps JSON structure: {"data": [{"id": ..., "caption": ..., ...}, ...]}
    records = data.get("data", data) if isinstance(data, dict) else data
    logger.info(f"Loaded {len(records)} metadata records from {source_name}")
    return records


def download_and_extract_zips(
    source_name: str,
    output_audio_dir: Path,
    cache_dir: Path,
    needed_files: set[str] | None = None,
) -> set[str]:
    """Download zip archives from WavCaps and extract FLAC files.

    Args:
        source_name: Source name.
        output_audio_dir: Directory to extract audio files into.
        cache_dir: Cache directory for zip downloads.
        needed_files: If provided, only extract these filenames.

    Returns:
        Set of extracted filenames.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    source = SOURCES[source_name]
    zip_dir = source["zip_dir"]

    # List all zip files in the source directory
    files = api.list_repo_tree(
        repo_id=WAVCAPS_REPO,
        path_in_repo=zip_dir,
        repo_type="dataset",
    )
    zip_files = [f.rfilename for f in files if f.rfilename.endswith(".zip")]
    logger.info(f"Found {len(zip_files)} zip files for {source_name}")

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    extracted = set()

    # Check what's already extracted — skip download if we have enough
    existing = {
        f.name for f in output_audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".flac", ".wav", ".mp3")
    }
    if needed_files:
        already_have = existing & needed_files
        if len(already_have) >= len(needed_files):
            logger.info(
                f"  Already have {len(already_have)}/{len(needed_files)} "
                f"audio files for {source_name} — skipping download"
            )
            return already_have
        logger.info(
            f"  Found {len(already_have)}/{len(needed_files)} already on disk, "
            f"need {len(needed_files) - len(already_have)} more"
        )
    elif existing:
        logger.info(f"  Found {len(existing)} audio files already on disk")
    extracted = set(existing)

    for i, zip_filename in enumerate(zip_files):
        logger.info(
            f"Downloading zip {i + 1}/{len(zip_files)}: {zip_filename}..."
        )

        from huggingface_hub import hf_hub_download

        zip_path = hf_hub_download(
            repo_id=WAVCAPS_REPO,
            filename=zip_filename,
            repo_type="dataset",
            cache_dir=str(cache_dir / "hf_cache"),
        )

        # Extract FLAC files
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                audio_names = [
                    n for n in zf.namelist()
                    if Path(n).name and Path(n).name.endswith((".flac", ".wav", ".mp3"))
                    and (not needed_files or Path(n).name in needed_files)
                ]
                total = len(audio_names)
                logger.info(f"Extracting {total} audio files from {zip_filename}...")
                for idx, name in enumerate(audio_names):
                    basename = Path(name).name

                    dest = output_audio_dir / basename
                    if dest.exists():
                        extracted.add(basename)
                    else:
                        with zf.open(name) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                        extracted.add(basename)

                    if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                        logger.info(
                            f"  [{source_name}] Extracted {idx + 1}/{total} "
                            f"({len(extracted)} total so far)"
                        )

        except zipfile.BadZipFile:
            logger.warning(
                f"Bad zip file: {zip_filename}, extracting with unzip (more lenient)..."
            )
            import subprocess

            # Use unzip directly — it handles corrupted central directories
            # much better than zip -F (which prompts for split archives)
            try:
                extract_tmp = output_audio_dir / "_unzip_tmp"
                extract_tmp.mkdir(exist_ok=True)
                logger.info(f"  Running: unzip -o {zip_filename} -> {extract_tmp}")
                proc = subprocess.run(
                    ["unzip", "-o", zip_path, "-d", str(extract_tmp)],
                    timeout=1800,  # 30 min max for large archives
                )
                if proc.returncode not in (0, 1):
                    # unzip returns 1 for warnings (still extracts), 2+ for errors
                    logger.warning(
                        f"  unzip returned code {proc.returncode}, "
                        f"checking for partial extraction..."
                    )

                # Collect extracted audio files
                audio_files = [
                    f for f in extract_tmp.rglob("*")
                    if f.is_file() and f.suffix.lower() in (".flac", ".wav", ".mp3")
                ]
                total = len(audio_files)
                logger.info(f"  unzip found {total} audio files, moving to {output_audio_dir}...")

                for idx, f in enumerate(audio_files):
                    basename = f.name
                    if needed_files and basename not in needed_files:
                        continue
                    dest = output_audio_dir / basename
                    if not dest.exists():
                        f.rename(dest)
                    extracted.add(basename)

                    if (idx + 1) % 1000 == 0 or (idx + 1) == total:
                        logger.info(
                            f"  [{source_name}] Moving {idx + 1}/{total} "
                            f"({len(extracted)} kept so far)"
                        )

                # Cleanup tmp
                import shutil
                shutil.rmtree(extract_tmp, ignore_errors=True)
                logger.info(
                    f"  Extraction complete: {len(extracted)} audio files "
                    f"from {source_name}"
                )
            except FileNotFoundError:
                logger.error(
                    "'unzip' command not found — install with: "
                    "apt-get install -y unzip"
                )
            except subprocess.TimeoutExpired:
                logger.warning("unzip timed out after 30 minutes")
            except Exception as e3:
                logger.warning(f"unzip extraction failed: {e3}")

        if needed_files and len(extracted) >= len(needed_files):
            logger.info(f"All needed files extracted ({len(extracted)})")
            break

        logger.info(
            f"Extracted so far: {len(extracted)} audio files from {source_name}"
        )

    return extracted


def build_audio_jsonl(
    records: list[dict],
    extracted_files: set[str],
    output_dir: Path,
    max_samples: int,
) -> int:
    """Build audio_text_pairs.jsonl from metadata + extracted files.

    Args:
        records: WavCaps metadata records.
        extracted_files: Set of filenames that were successfully extracted.
        output_dir: Output directory.
        max_samples: Maximum samples to include.

    Returns:
        Number of pairs written.
    """
    jsonl_path = output_dir / "audio_text_pairs.jsonl"
    count = 0

    with open(jsonl_path, "w") as f:
        for record in records:
            if count >= max_samples:
                break

            # WavCaps metadata fields vary by source
            caption = record.get("caption", "")
            if not caption:
                caption = record.get("raw_description", "")
            if not caption:
                continue

            # Audio filename — different field names per source
            audio_file = ""
            for key in ("id", "file_name", "fname", "audio"):
                val = record.get(key, "")
                if val:
                    audio_file = str(val)
                    break

            # Normalize to just the filename
            audio_file = Path(audio_file).name
            if not audio_file:
                continue

            # Add extension if missing
            if not any(audio_file.endswith(ext) for ext in (".flac", ".wav", ".mp3")):
                audio_file += ".flac"

            if audio_file not in extracted_files:
                continue

            pair = {
                "audio_path": audio_file,
                "caption": caption,
                "domain": "audio_text",
            }
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1

    logger.info(f"Wrote {count} audio-text pairs to {jsonl_path}")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Download WavCaps audio for Stage 3 training"
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
        default=["freesound", "bbc_sound_effects", "soundbible"],
        choices=list(SOURCES.keys()),
        help="WavCaps sources to download from.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HF downloads. Defaults to output-dir/.cache",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    audio_dir = output_dir / "audio"
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_extracted = set()
    remaining = args.max_samples

    for source_name in args.sources:
        if remaining <= 0:
            break

        source_info = SOURCES[source_name]
        logger.info(
            f"\n{'=' * 60}\n"
            f"Processing {source_name} "
            f"({source_info['total_clips']} clips available)\n"
            f"{'=' * 60}"
        )

        # 1. Download JSON metadata
        records = download_json_metadata(source_name, cache_dir)

        # Limit to what we need
        records = records[:remaining]

        # 2. Build set of needed audio filenames
        needed_files = set()
        for rec in records:
            for key in ("id", "file_name", "fname", "audio"):
                val = rec.get(key, "")
                if val:
                    fname = Path(str(val)).name
                    if not any(fname.endswith(ext) for ext in (".flac", ".wav", ".mp3")):
                        fname += ".flac"
                    needed_files.add(fname)
                    break

        logger.info(f"Need {len(needed_files)} audio files from {source_name}")

        # 3. Download and extract zips
        extracted = download_and_extract_zips(
            source_name=source_name,
            output_audio_dir=audio_dir,
            cache_dir=cache_dir,
            needed_files=needed_files,
        )

        all_records.extend(records)
        all_extracted.update(extracted)
        remaining -= len(extracted & needed_files)

        logger.info(
            f"{source_name}: extracted {len(extracted)} files, "
            f"remaining quota: {max(remaining, 0)}"
        )

    # 4. Build final JSONL
    count = build_audio_jsonl(
        records=all_records,
        extracted_files=all_extracted,
        output_dir=output_dir,
        max_samples=args.max_samples,
    )

    # 5. Summary
    stats = {
        "sources": args.sources,
        "total_extracted": len(all_extracted),
        "audio_text_pairs": count,
        "audio_dir": str(audio_dir),
    }
    logger.info(f"\nAudio download complete: {json.dumps(stats, indent=2)}")

    # Update dataset_stats.json if it exists
    stats_file = output_dir / "dataset_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            existing_stats = json.load(f)
        existing_stats["audio_text_pairs"] = count
        existing_stats["audio_source"] = "WavCaps (FreeSound + BBC + SoundBible)"
        existing_stats["audio_download_failed"] = len(all_extracted) - count
        with open(stats_file, "w") as f:
            json.dump(existing_stats, f, indent=2)
        logger.info(f"Updated {stats_file}")


if __name__ == "__main__":
    main()
