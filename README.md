# OmniVector-Embed

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PyTorch 2.2–2.3](https://img.shields.io/badge/PyTorch-2.2–2.3-red)](https://pytorch.org)
[![Transformers 4.44.2](https://img.shields.io/badge/Transformers-4.44.2-yellow)](https://huggingface.co/transformers)
[![ONNX Opset 18](https://img.shields.io/badge/ONNX-Opset%2018-purple)](https://onnx.ai)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

| Stack | Version |
|---|---|
| **Python** | ≥ 3.9 |
| **PyTorch** | ≥ 2.2.0, < 2.4.0 |
| **Transformers** | 4.44.2 (pinned) |
| **PEFT** | 0.12.0 (pinned) |
| **ONNX Runtime** | ≥ 1.18.0 |
| **ONNX Opset** | 18 |
| **DeepSpeed** | ≥ 0.14.0 (ZeRO-2) |
| **FAISS** | CPU (IndexFlatIP) |
| **Base Model** | Mistral-7B-v0.1 |
| **Vision Encoder** | SigLIP-SO400M |
| **Quantization** | Dynamic INT8 |

Production-grade multimodal embedding model that replicates and extends [NV-Embed-v2](https://arxiv.org/abs/2405.17428). Unified 4096-dimensional embeddings for text, code, image, and video — with ONNX export and int8 quantization for deployment.

## Key Differentiators

| Feature | NV-Embed-v2 | OmniVector-Embed |
|---|---|---|
| ONNX export | Not possible (SDPA ops) | Opset 18 + int8 quantization |
| Modalities | Text only | Text + Code + Image + Video |
| Attention | SDPA/Flash | Eager bidirectional (export-safe) |
| Deployment | GPU-only inference | CPU/GPU via ORT |
| Fine-tuning | Full fine-tune | LoRA (rank 16, 0.1% params) |

## Architecture

```
Input → Mistral-7B (bidirectional, eager attention, LoRA)
      → Latent Attention Pooling (512 latents × 4096, 8 heads)
      → Matryoshka dimensions [512, 1024, 2048, 4096]
      → L2 normalize
```

- **Backbone**: `mistralai/Mistral-7B-v0.1` with `_update_causal_mask → None` for bidirectional attention
- **Pooling**: Cross-attention with learned latent queries, followed by mean pooling
- **Vision**: SigLIP-SO400M (1152 → 4096 projection) for images, temporal attention for video
- **Loss**: InfoNCE + MRL with in-batch negatives and FAISS hard negative mining
- **Training**: 2-stage (Stage 1: retrieval 20k steps, Stage 2: generalist 18k steps) with DeepSpeed ZeRO-2

See [docs/architecture.md](docs/architecture.md) for a detailed component guide.

## Installation

```bash
# From source
git clone https://github.com/AuralithAI/OmniVector-Embed.git
cd OmniVector-Embed
pip install -e ".[dev,test]"

# CPU-only PyTorch (CI / no GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

**Requirements**: Python ≥ 3.9, PyTorch ≥ 2.2 < 2.4, transformers == 4.44.2, peft == 0.12.0

## Quick Start

### Encode text

```python
from omnivector.model.omnivector_model import OmniVectorModel

model = OmniVectorModel.from_pretrained("path/to/checkpoint")
model.eval()

# Full 4096-d embeddings
embeddings = model.encode_text(["What is machine learning?", "Deep learning overview"])

# Matryoshka: use only first 1024 dims for faster retrieval
embeddings_1024 = model.encode_text(["search query"], output_dim=1024)
```

### Encode images

```python
from PIL import Image

image = Image.open("photo.jpg")
embedding = model.encode_image(image, output_dim=4096)
```

### Encode video

```python
embedding = model.encode_video("clip.mp4", output_dim=4096)
```

## Training

### 1. Build dataset

```bash
# Text-only (MSMARCO, HotpotQA, BEIR)
python scripts/build_dataset.py --stage 1 --output-dir data/stage1

# With multimodal data (LAION, WebVid, CodeSearchNet) targeting 8M pairs
python scripts/build_dataset.py --stage 1 --multimodal --target 8000000 \
    --teacher-model BAAI/bge-large-en-v1.5 --output-dir data/stage1_8M
```

### 2. Mine hard negatives (optional, offline)

```bash
python scripts/mine_hard_negatives.py \
    --dataset msmarco \
    --teacher-model BAAI/bge-large-en-v1.5 \
    --output-dir data/hard_negatives
```

### 3. Train

```bash
# Stage 1: Retrieval (20k steps, 2× A100 80GB)
python scripts/training.py --config configs/stage1_retrieval.yaml

# Stage 2: Generalist (18k steps)
python scripts/training.py --config configs/stage2_generalist.yaml

# Multimodal (image + video)
python scripts/train_multimodal.py --config configs/multimodal_vision.yaml
```

### 4. Evaluate

```bash
python scripts/evaluate.py --model-path checkpoints/stage1_best --tasks retrieval
```

## ONNX Export

```bash
# Full pipeline: export → optimize → quantize → validate
python scripts/export_onnx.py \
    --model-path checkpoints/stage1_best \
    --output-dir onnx_export \
    --optimize \
    --quantize-int8 \
    --validate
```

This produces:

| File | Size (est.) | Use case |
|---|---|---|
| `omnivector_embed.onnx` | ~14 GB | Full precision (fp32) |
| `omnivector_embed_opt.onnx` | ~14 GB | ORT graph-optimized (fused attention/LN) |
| `omnivector_embed_int8.onnx` | ~3.5 GB | Dynamic int8 quantization |

### Inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("onnx_export/omnivector_embed_int8.onnx")
output = session.run(None, {
    "input_ids": input_ids_numpy,
    "attention_mask": attention_mask_numpy,
})[0]
```

## Benchmark Targets

| Benchmark | Target | Notes |
|---|---|---|
| MSMARCO NDCG@10 | ≥ 0.52 | Stage 1 retrieval |
| MTEB (preview) | ≥ 65 | Averaged across tasks |
| ONNX cosine parity | > 0.99 | fp32 PyTorch vs ONNX |
| Int8 cosine parity | > 0.95 | fp32 vs int8 ONNX |

## Project Structure

```
OmniVector-Embed/
├── configs/                        # Training & model configs
│   ├── stage1_retrieval.yaml       # Stage 1: 20k steps, LR 2e-5, 8M pairs
│   ├── stage2_generalist.yaml      # Stage 2: 18k steps, LR 1.5e-5
│   ├── multimodal_vision.yaml      # Image + video training
│   ├── lora.yaml                   # LoRA hyperparameters
│   └── deepspeed_zero2.json        # ZeRO-2 config
├── scripts/                        # CLI entry points
│   ├── build_dataset.py            # Data pipeline (text + multimodal)
│   ├── mine_hard_negatives.py      # Offline FAISS hard negative mining
│   ├── training.py                 # Text training
│   ├── train_multimodal.py         # Multimodal training
│   ├── export_onnx.py              # ONNX export + optimize + quantize
│   ├── quantize_onnx.py            # Standalone quantization
│   └── evaluate.py                 # MTEB evaluation
├── src/omnivector/
│   ├── model/
│   │   ├── backbone.py             # Bidirectional Mistral-7B + LoRA
│   │   ├── latent_attention.py     # Latent attention pooling
│   │   ├── omnivector_model.py     # Main model (encode_text/image/video)
│   │   ├── vision_encoder.py       # SigLIP-SO400M encoder
│   │   └── video_encoder.py        # Temporal video encoder
│   ├── data/
│   │   ├── schema.py               # EmbeddingPair Pydantic schema
│   │   ├── preprocessing.py        # Tokenization + collation
│   │   ├── loaders/                # MSMARCO, HotpotQA, BEIR, COCO, video
│   │   └── multimodal_dataset.py   # Unified multimodal dataset + collator
│   ├── training/
│   │   ├── trainer.py              # OmniVectorTrainer (HF Trainer subclass)
│   │   ├── multimodal_trainer.py   # MultimodalTrainer
│   │   ├── losses.py               # MRL InfoNCE loss
│   │   ├── multimodal_loss.py      # Cross-modal contrastive + MRL loss
│   │   ├── callbacks.py            # Hard neg refresh, logging, early stop
│   │   └── hard_negative_miner.py  # FAISS IndexFlatIP miner
│   ├── export/
│   │   ├── onnx_exporter.py        # ONNX export + ORT optimizer
│   │   ├── onnx_quantizer.py       # Int8 dynamic quantization
│   │   └── onnx_validator.py       # Cosine parity validation
│   └── eval/                       # MTEB evaluation utilities
├── tests/
│   ├── unit/                       # ~100 unit tests
│   └── integration/                # ONNX parity, data loading
├── docs/
│   └── architecture.md             # Detailed architecture guide
└── pyproject.toml                  # Dependencies + tool config
```

## Testing

```bash
# All unit tests (no GPU required, skips slow tests)
pytest

# Include integration tests
pytest -m "not slow"

# Full test suite including model download tests
pytest -m ""

# With coverage
pytest --cov=src/omnivector --cov-report=html
```

## Version Pins

These versions are **load-bearing** — do not upgrade without testing:

- `transformers==4.44.2` — `_update_causal_mask` API for bidirectional attention + RoPE real-valued ops for ONNX
- `peft==0.12.0` — `merge_and_unload()` + DeepSpeed compatibility
- `torch>=2.2.0,<2.4.0` — ONNX dynamo export requires opset ≥ 18

## License

Apache 2.0 — see [LICENSE](LICENSE).