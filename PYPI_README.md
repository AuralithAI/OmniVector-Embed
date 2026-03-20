# OmniVector-Embed

[![PyPI](https://img.shields.io/pypi/v/omnivector-embed)](https://pypi.org/project/omnivector-embed/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://github.com/AuralithAI/OmniVector-Embed/blob/main/LICENSE)

**Production-grade multimodal embedding model** — unified 4096-dimensional embeddings for text, code, image, video, and audio. Built on Mistral-7B with ONNX export and int8 quantization for deployment.

Replicates and extends [NV-Embed-v2](https://arxiv.org/abs/2405.17428) with multimodal support and CPU-friendly inference.

## Installation

```bash
pip install omnivector-embed
```

With vision support:
```bash
pip install omnivector-embed[vision]
```

## Quick Start

```python
from omnivector.model import OmniVectorModel

# Load a trained model
model = OmniVectorModel.from_pretrained("AuralithAI/omnivector-embed-v1")

# Encode text
embeddings = model.encode(["What is machine learning?", "ML is a subset of AI."])

# Encode with Matryoshka dimensionality
embeddings_512 = model.encode(["query"], output_dim=512)
embeddings_4096 = model.encode(["query"], output_dim=4096)
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal** | Text, code, image, video, and audio in one embedding space |
| **Matryoshka** | Flexible output dimensions: 512, 1024, 2048, 4096 |
| **ONNX Export** | Opset 17 with dynamic int8 quantization |
| **CPU Inference** | Full ONNX Runtime support — no GPU required |
| **LoRA Training** | Fine-tune with 0.1% of parameters (rank 16) |
| **3-Stage Pipeline** | Retrieval → Generalist → Multimodal training |

## Architecture

```
Input → Mistral-7B (bidirectional, eager attention, LoRA)
      → Latent Attention Pooling (512 latents × 4096, 8 heads)
      → Matryoshka dimensions [512, 1024, 2048, 4096]
      → L2 normalize
```

- **Backbone**: Mistral-7B-v0.1 with bidirectional attention
- **Pooling**: Cross-attention with learned latent queries
- **Vision**: SigLIP-SO400M (1152 → 4096 projection)
- **Audio**: Whisper-tiny (384 → 4096 MLP projection)
- **Loss**: InfoNCE + Matryoshka Representation Learning + cross-modal contrastive

## ONNX Export

```python
from omnivector.export import OnnxExporter

exporter = OnnxExporter(model_path="path/to/model", opset_version=17)
exporter.export("model.onnx")

# Quantize to int8
from omnivector.export import OnnxQuantizer
OnnxQuantizer.quantize_dynamic("model.onnx", "model_int8.onnx")
```

## Evaluation

Built-in MTEB evaluation:

```bash
python scripts/evaluate.py --model-path path/to/model --tasks retrieval
```

## Training

3-stage training pipeline with DeepSpeed ZeRO-2:

```bash
# Stage 1: Retrieval (text pairs with hard negatives)
python scripts/training.py --config configs/stage1_retrieval.yaml

# Stage 2: Generalist (55M+ pairs)
python scripts/training.py --config configs/stage2_generalist.yaml

# Stage 3: Multimodal (image/video/audio + text)
python scripts/train_multimodal.py --config configs/stage3_multimodal.yaml
```

## Stack

| Component | Version |
|-----------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.2.0 |
| Transformers | 4.44.2 |
| PEFT | 0.12.0 |
| ONNX Runtime | ≥ 1.18.0 |
| DeepSpeed | ≥ 0.14.0 |

## Links

- **GitHub**: [AuralithAI/OmniVector-Embed](https://github.com/AuralithAI/OmniVector-Embed)
- **Paper**: [NV-Embed-v2 (arXiv:2405.17428)](https://arxiv.org/abs/2405.17428)
- **Documentation**: [docs/architecture.md](https://github.com/AuralithAI/OmniVector-Embed/blob/main/docs/architecture.md)

## License

Apache 2.0 — see [LICENSE](https://github.com/AuralithAI/OmniVector-Embed/blob/main/LICENSE).
