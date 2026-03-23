#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# OmniVector-Embed — Full Training Pipeline
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   - NVIDIA GPU(s) with CUDA drivers
#   - Git clone of this repository
#
# Usage:
#   chmod +x run.sh && ./run.sh
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  OmniVector-Embed Training Pipeline"
echo "  Started: $(date)"
echo "============================================"


# ───────────────────────────────────────────────────────────────────
# Environment Setup
# ───────────────────────────────────────────────────────────────────

echo ""
echo "[1/8] Checking Python installation..."

# Detect OS
OS="$(uname -s)"
case "$OS" in
    Linux*)   PLATFORM="linux" ;;
    Darwin*)  PLATFORM="mac" ;;
    MINGW*|MSYS*|CYGWIN*)  PLATFORM="windows" ;;
    *)        echo "Unsupported OS: $OS"; exit 1 ;;
esac
echo "  Detected platform: $PLATFORM"

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "  Python not found. Installing..."
    if [ "$PLATFORM" = "linux" ]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y python3 python3-pip
        elif command -v yum &>/dev/null; then
            sudo yum install -y python3 python3-pip
        else
            echo "  ERROR: No supported package manager found. Install Python 3.10+ manually."
            exit 1
        fi
    elif [ "$PLATFORM" = "mac" ]; then
        if command -v brew &>/dev/null; then
            brew install python@3.11
        else
            echo "  ERROR: Homebrew not found. Install Python from https://python.org"
            exit 1
        fi
    else
        echo "  ERROR: Install Python 3.10+ from https://python.org"
        exit 1
    fi
    PYTHON=python3
fi

PYTHON_VERSION=$($PYTHON --version 2>&1)
echo "  Using: $PYTHON_VERSION"

# Upgrade pip
echo ""
echo "[2/8] Upgrading pip..."
$PYTHON -m pip install --upgrade pip 2>/dev/null || $PYTHON -m ensurepip --upgrade

# Create and activate venv
echo ""
echo "[3/8] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists, reusing"
fi

if [ "$PLATFORM" = "windows" ]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi
echo "  Activated venv: $(which python)"

# Install dependencies
echo ""
echo "[4/8] Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Detect GPU compute capability and install matching PyTorch
echo "  Checking CUDA / GPU compatibility..."
NEED_NIGHTLY=false
if command -v nvidia-smi &>/dev/null; then
    # Extract compute capability (e.g. 12.0 for Blackwell sm_120)
    SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')
    SM_MAJOR=$(echo "$SM_VERSION" | cut -d. -f1)
    echo "  Detected GPU compute capability: sm_${SM_VERSION} (major=${SM_MAJOR})"

    if [ "${SM_MAJOR:-0}" -ge 12 ]; then
        echo "  Blackwell (sm_12x) detected — installing PyTorch nightly with CUDA 12.8..."
        NEED_NIGHTLY=true
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    elif [ "${SM_MAJOR:-0}" -ge 10 ]; then
        echo "  Hopper/Ada (sm_${SM_MAJOR}x) detected — installing PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
fi

# Install project deps — use --no-deps first to avoid overwriting the PyTorch we just installed,
# then install remaining dependencies separately
if [ "$NEED_NIGHTLY" = true ]; then
    echo "  Installing project (preserving PyTorch nightly)..."
    pip install --no-deps -e "."
    pip install --no-deps -e ".[dev,test,vision,multimodal]" 2>/dev/null || true
    # Install all non-torch dependencies from pyproject.toml
    pip install "transformers==4.44.2" "peft==0.12.0" "accelerate>=0.27.0" \
        "numpy>=1.24.0" "pydantic>=2.0.0" "pillow>=9.0.0" "einops>=0.7.0" \
        "tensorboard>=2.14.0" "onnx>=1.16.0" "onnxruntime>=1.18.0" \
        "onnxscript>=0.1.0" "optimum[onnxruntime]>=1.21.0" "deepspeed>=0.14.0" \
        "faiss-cpu>=1.8.0" "mteb>=1.12.0" "datasets>=2.16.0" "pyarrow>=14.0.0" \
        "tqdm>=4.66.0" "pyyaml>=6.0" "requests>=2.31.0" \
        "sentence-transformers>=3.0.0" \
        "ruff>=0.3.0" "mypy>=1.8.0" "pre-commit>=3.6.0" \
        "pytest>=7.4.0" "pytest-cov>=4.1.0" "pytest-xdist>=3.5.0" \
        "pytest-timeout>=2.1.0"
else
    pip install -e ".[dev,test,vision,multimodal]"
fi
pip install deepspeed

# Stage 3 needs yt-dlp (audio download) and ffmpeg (audio conversion)
pip install yt-dlp
if command -v ffmpeg &>/dev/null; then
    echo "  ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
else
    echo "  Installing ffmpeg (required for audio extraction)..."
    if [ "$PLATFORM" = "linux" ]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y ffmpeg 2>/dev/null || echo "  WARNING: ffmpeg install failed — audio download will be skipped"
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y ffmpeg 2>/dev/null || echo "  WARNING: ffmpeg install failed"
        fi
    elif [ "$PLATFORM" = "mac" ]; then
        brew install ffmpeg 2>/dev/null || echo "  WARNING: ffmpeg install failed"
    fi
fi

# Verify GPU
echo ""
echo "  Verifying GPU setup..."
python -c "
import torch
n = torch.cuda.device_count()
if n == 0:
    print('  WARNING: No CUDA GPUs detected. Training will be slow on CPU.')
else:
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1e9
        print(f'  GPU {i}: {props.name} ({mem_gb:.1f} GB, sm_{props.major}{props.minor})')
    print(f'  Total: {n} GPU(s) ready')
"

echo ""
echo "  Environment setup complete."
sleep 5


# ───────────────────────────────────────────────────────────────────
# Data Pipeline
# ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "[5/8] Building Stage 1 dataset (8M target)..."
echo "============================================"
python scripts/build_dataset.py \
    --stage 1 \
    --multimodal \
    --target 8000000 \
    --teacher-model BAAI/bge-large-en-v1.5 \
    --output-dir data/stage1_8M

echo "  Stage 1 data build complete."
sleep 30

echo ""
echo "============================================"
echo "[6/8] Building Stage 2 dataset (55M target)..."
echo "============================================"
python scripts/build_dataset.py \
    --stage 2 \
    --multimodal \
    --target 55000000 \
    --add-synthetic 200000 \
    --output-dir data/stage2_55M

echo "  Stage 2 data build complete."
sleep 30

echo ""
echo "  Mining hard negatives..."
python scripts/mine_hard_negatives.py \
    --dataset msmarco \
    --teacher-model BAAI/bge-large-en-v1.5 \
    --output-dir data/hard_negatives

echo "  Hard negative mining complete."
sleep 30

# Stage 3 data: download multimodal media from LAION URLs collected in Stage 2
echo ""
echo "============================================"
echo "  Building Stage 3 multimodal dataset..."
echo "============================================"
python scripts/build_dataset.py \
    --stage 3 \
    --source-dir data/stage2_55M \
    --output-dir data/stage3_multimodal \
    --max-images 100000 \
    --max-video 10000 \
    --max-audio 20000 \
    --download-workers 16

echo "  Stage 3 data build complete."
sleep 30

# Stage 3 audio: download from WavCaps (replaces broken AudioSet/YouTube approach)
echo ""
echo "  ── Downloading WavCaps audio (FreeSound + BBC + SoundBible) ──"
python scripts/download_wavcaps_audio.py \
    --output-dir data/stage3_multimodal \
    --max-samples 20000 \
    --sources freesound bbc_sound_effects soundbible

echo "  WavCaps audio download complete."
sleep 10


# ───────────────────────────────────────────────────────────────────
# Multi-Stage Training (DeepSpeed ZeRO-2 + LoRA + bf16)
# ───────────────────────────────────────────────────────────────────

NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
echo ""
echo "============================================"
echo "[7/8] Training on ${NUM_GPUS} GPU(s)..."
echo "============================================"

# Stage 1: Retrieval — 20k steps
echo ""
echo "  ── Stage 1: Retrieval (20k steps) ──"
deepspeed --num_gpus="$NUM_GPUS" scripts/training.py \
    --config configs/stage1_retrieval.yaml \
    --dataset msmarco \
    --output-dir checkpoints/stage1_8M \
    --lora

echo "  Stage 1 training complete."
sleep 60

# Stage 2: Generalist — 18k steps, loads Stage 1 weights
echo ""
echo "  ── Stage 2: Generalist (18k steps) ──"
deepspeed --num_gpus="$NUM_GPUS" scripts/training.py \
    --config configs/stage2_generalist.yaml \
    --dataset msmarco \
    --output-dir checkpoints/stage2_55M \
    --model-path checkpoints/stage1_8M/final_model \
    --lora

echo "  Stage 2 training complete."
sleep 60

# Stage 3: Multimodal alignment — 12k steps
echo ""
echo "  ── Stage 3: Multimodal (12k steps) ──"
deepspeed --num_gpus="$NUM_GPUS" scripts/train_multimodal.py \
    --config configs/stage3_multimodal.yaml \
    --output-dir checkpoints/stage3 \
    --text-checkpoint checkpoints/stage2_55M/final_model

echo "  Stage 3 training complete."
sleep 30


# ───────────────────────────────────────────────────────────────────
# Evaluation & ONNX Export
# ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "[8/8] Evaluation & Export..."
echo "============================================"

# Evaluate Stage 1 — retrieval baseline
echo ""
echo "  ── Evaluating Stage 1 (retrieval baseline) ──"
python scripts/evaluate.py \
    --model-path checkpoints/stage1_8M/final_model \
    --tasks retrieval \
    --output-dir eval_results/stage1 \
    --stage stage1 \
    --lora

# Evaluate Stage 2 — generalist (text benchmark)
echo ""
echo "  ── Evaluating Stage 2 (generalist) ──"
python scripts/evaluate.py \
    --model-path checkpoints/stage2_55M/final_model \
    --tasks retrieval,sts,clustering,pair_classification,reranking,classification,summarization \
    --output-dir eval_results/stage2 \
    --stage stage2 \
    --lora

# Evaluate Stage 3 — full MTEB benchmark (56 tasks, NV-Embed v2 leaderboard)
echo ""
echo "  ── Evaluating Stage 3 (full MTEB — 56 tasks) ──"
python scripts/evaluate.py \
    --model-path checkpoints/stage3/final_model \
    --full-mteb \
    --output-dir eval_results/stage3 \
    --stage stage3 \
    --lora

# ONNX export of final model
echo ""
echo "  ── Exporting final model to ONNX ──"
python scripts/export_onnx.py \
    --model-path checkpoints/stage3/final_model \
    --output-dir onnx_export \
    --optimize \
    --quantize-int8 \
    --validate

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Finished: $(date)"
echo "  Log: $LOG_FILE"
echo "============================================"
