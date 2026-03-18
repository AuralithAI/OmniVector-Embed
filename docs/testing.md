# Testing Guide

Comprehensive guide for running OmniVector-Embed's test suite.

## Overview

| Category | Count | GPU Required | Description |
|---|---|---|---|
| **Unit** | ~280 | No | Component-level tests (models, data, losses, export) |
| **Integration** | ~15 | Partial | ONNX parity, data loading, training loop |
| **Slow** | ~19 | Yes | Full Mistral-7B backbone, end-to-end pipelines |
| **Total** | ~314 | — | All tests across `tests/unit/` and `tests/integration/` |

## Quick Reference

```bash
# Default: all tests except slow (no GPU needed)
pytest

# Everything including slow tests (GPU recommended)
pytest -m ""

# Only unit tests
pytest tests/unit/

# Only integration tests
pytest tests/integration/

# A single test file
pytest tests/unit/test_latent_attention.py

# A single test by name
pytest -k "test_cross_attention_output_shape"
```

## Test Markers

Tests are tagged with [pytest markers](https://docs.pytest.org/en/stable/example/markers.html)
defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `unit` | Pure CPU tests — no GPU, no large model downloads |
| `integration` | End-to-end flows (ONNX export, data loading, training dry-run) |
| `slow` | Downloads Mistral-7B or runs full-model inference — needs GPU + time |

The default `addopts` in `pyproject.toml` includes `-m 'not slow'`, so bare
`pytest` automatically skips the 19 slow tests.

```bash
# Override the default marker filter
pytest -m ""                   # run everything
pytest -m "slow"               # only slow tests
pytest -m "integration"        # only integration tests
pytest -m "not slow and not integration"  # unit only
```

## Running on GPU

### Single GPU

```bash
# Set visible device
CUDA_VISIBLE_DEVICES=0 pytest -m "" -v

# Only the slow tests that actually need a GPU
CUDA_VISIBLE_DEVICES=0 pytest -m "slow" -v
```

### Multi-GPU (2+ GPUs)

pytest itself is single-process by default. To use multiple GPUs:

**Option A — pytest-xdist (parallel workers, one GPU each)**

`pytest-xdist` is already in `pyproject.toml` dev dependencies.

```bash
# 2 workers, each pinned to a separate GPU
pytest -m "" -n 2 --dist loadscope -v
```

> **Note:** `--dist loadscope` keeps all tests in the same file/class on the
> same worker, which avoids GPU memory conflicts from test fixtures.

To explicitly pin GPU per worker, create or update `conftest.py` with a
worker-aware device fixture (already supported — the `device` fixture in
`tests/conftest.py` returns CPU; on GPU machines slow tests use
`torch.cuda.is_available()` guards and select CUDA automatically).

For manual control:

```bash
# Worker 0 → GPU 0, Worker 1 → GPU 1
CUDA_VISIBLE_DEVICES=0,1 pytest -m "" -n 2 --dist loadscope -v
```

**Option B — run subsets in parallel shells**

```bash
# Terminal 1: unit + integration on GPU 0
CUDA_VISIBLE_DEVICES=0 pytest tests/unit/ tests/integration/ -m "not slow" -v &

# Terminal 2: slow tests on GPU 1
CUDA_VISIBLE_DEVICES=1 pytest -m "slow" -v &

wait
```

**Option C — DeepSpeed multi-GPU training test**

The 10-step integration training test (`test_10_step_training`) runs on a
single GPU. For multi-GPU training validation, use the training scripts
directly:

```bash
# Quick 10-step sanity check on 2 GPUs via DeepSpeed
deepspeed --num_gpus=2 scripts/training.py \
    --config configs/stage1_retrieval.yaml \
    --dataset msmarco \
    --output-dir /tmp/test_run \
    --lora \
    --max-steps 10
```

## Coverage

```bash
# HTML coverage report
pytest --cov=src/omnivector --cov-report=html -m "not slow"
open htmlcov/index.html

# Terminal summary
pytest --cov=src/omnivector --cov-report=term-missing -m "not slow"

# With branch coverage (configured in pyproject.toml)
pytest --cov=src/omnivector --cov-report=term-missing --cov-branch
```

Target: **80%+ line coverage** on `src/omnivector/`.

## Test Structure

```
tests/
├── conftest.py                     # Shared fixtures (device, tokenizer, samples)
├── unit/
│   ├── test_audio_encoder.py       # WhisperAudioEncoder
│   ├── test_backbone.py            # MistralEmbeddingBackbone (slow — loads 7B)
│   ├── test_bidirectionality.py    # Causal mask override verification
│   ├── test_build_dataset.py       # build_dataset.py domain balancing
│   ├── test_callbacks.py           # HardNegativeRefreshCallback
│   ├── test_cross_attention.py     # EagerCrossAttention module
│   ├── test_data.py                # EmbeddingPair schema + preprocessing
│   ├── test_hard_negative_miner.py # FAISS miner
│   ├── test_latent_attention.py    # LatentAttentionPooling
│   ├── test_loaders.py             # MSMARCO, HotpotQA, BEIR data loaders
│   ├── test_model.py               # OmniVectorModel (slow — loads 7B)
│   ├── test_mteb_runner.py         # MTEB evaluation runner
│   ├── test_multimodal.py          # Multimodal dataset + collator
│   ├── test_multimodal_training.py # Multimodal trainer + loss
│   ├── test_onnx_export.py         # ONNX export, optimize, quantize
│   ├── test_trainer.py             # OmniVectorTrainer
│   ├── test_training_configs.py    # YAML config validation
│   └── test_vision_encoder.py      # SigLIPVisionEncoder freeze/unfreeze
└── integration/
    ├── test_imports.py             # Package import smoke test
    └── test_training.py            # 10-step training, ONNX parity, quantization
```

## Fixtures

Key fixtures from `tests/conftest.py`:

| Fixture | Scope | Description |
|---|---|---|
| `device` | session | `torch.device("cpu")` — override with GPU in slow tests |
| `tokenizer` | session | Mistral-7B tokenizer (skips if download fails) |
| `sample_texts` | function | 3 short English sentences |
| `sample_embedding_pair` | function | Single `EmbeddingPair` with query/positive/negatives |
| `sample_batch` | function | 4× `EmbeddingPair` batch |
| `sample_audio_path` | function | Temp WAV file (requires `soundfile`) |
| `sample_video_path` | function | Temp MP4 file (requires `opencv-python`) |
| `mock_backbone` | function | Full `MistralEmbeddingBackbone` (skips if unavailable) |
| `mock_pooling` | function | `LatentAttentionPooling(4096, 512, 8)` |

## CI Configuration

The default pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--strict-markers -v --tb=short --timeout=3600 --timeout-method=thread -m 'not slow'"
markers = [
    "unit: Unit tests that don't require GPU",
    "integration: Integration tests that may require GPU",
    "slow: Tests that take a long time to run or download large models",
]
```

**CI pipeline recommendation:**

| Stage | Command | Environment |
|---|---|---|
| Lint | `ruff check src/ tests/` | Any |
| Type check | `mypy src/omnivector/` | Any |
| Unit tests | `pytest tests/unit/ -m "not slow"` | CPU, no GPU |
| Integration | `pytest tests/integration/ -m "not slow"` | CPU, no GPU |
| Full suite | `pytest -m "" --timeout=7200` | GPU instance |
| Coverage | `pytest --cov=src/omnivector --cov-report=xml` | CPU |

## Troubleshooting

### Common issues

**Tests skipped with "Requires GPU"**
```
SKIPPED [1] tests/integration/test_training.py: Requires GPU for practical training
```
Expected on CPU-only machines. Run on a GPU machine to execute these.

**Tokenizer download fails**
```
WARNING - Failed to load tokenizer
```
Ensure `HF_TOKEN` is set or you have accepted the Mistral model license on HuggingFace.

**ONNX tests fail with opset error**
We use opset 17 (max supported by PyTorch 2.3.x). If you see opset errors,
verify `torch` version: `python -c "import torch; print(torch.__version__)"`.

**Timeout on slow tests**
Default timeout is 3600s (1 hour). For very slow machines:
```bash
pytest -m "slow" --timeout=7200
```

**Out of GPU memory**
Slow tests load the full Mistral-7B model. Requires ≥24 GB VRAM per GPU.
Run with a single test at a time if memory is tight:
```bash
CUDA_VISIBLE_DEVICES=0 pytest -m "slow" -k "test_backbone" --timeout=7200
```
