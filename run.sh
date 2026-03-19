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
pip install --upgrade pip

# Detect GPU compute capability and install matching PyTorch
echo "  Checking CUDA / GPU compatibility..."
NEED_NIGHTLY=false
if command -v nvidia-smi &>/dev/null; then
    # Extract compute capability (e.g. 12.0 for Blackwell sm_120)
    SM_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '[:space:]')
    SM_MAJOR=$(echo "$SM_VERSION" | cut -d. -f1)
    echo "  Detected GPU compute capability: sm_${SM_VERSION} (major=${SM_MAJOR})"

    if [ "${SM_MAJOR:-0}" -ge 12 ]; then
        echo "  Blackwell (sm_12x) detected — installing PyTorch nightly with CUDA 12.8 support..."
        NEED_NIGHTLY=true
        pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    elif [ "${SM_MAJOR:-0}" -ge 10 ]; then
        echo "  Hopper/Ada (sm_${SM_MAJOR}x) detected — installing PyTorch with CUDA 12.4..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    fi
fi

pip install -e ".[dev,test,vision]"
pip install deepspeed

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

# Stage 2: Generalist — 18k steps, resumes from Stage 1
echo ""
echo "  ── Stage 2: Generalist (18k steps) ──"
deepspeed --num_gpus="$NUM_GPUS" scripts/training.py \
    --config configs/stage2_generalist.yaml \
    --dataset msmarco \
    --output-dir checkpoints/stage2_55M \
    --lora \
    --resume checkpoints/stage1_8M/checkpoint-final

echo "  Stage 2 training complete."
sleep 60

# Stage 3: Multimodal alignment — 12k steps
echo ""
echo "  ── Stage 3: Multimodal (12k steps) ──"
deepspeed --num_gpus="$NUM_GPUS" scripts/train_multimodal.py \
    --config configs/stage3_multimodal.yaml

echo "  Stage 3 training complete."
sleep 30


# ───────────────────────────────────────────────────────────────────
# Evaluation & ONNX Export
# ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo "[8/8] Evaluation & Export..."
echo "============================================"

python scripts/evaluate.py \
    --model-path checkpoints/stage2_55M/checkpoint-final \
    --tasks retrieval,sts,clustering \
    --output-dir eval_results \
    --stage stage2 \
    --lora

python scripts/export_onnx.py \
    --model-path checkpoints/stage2_55M/checkpoint-final \
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
