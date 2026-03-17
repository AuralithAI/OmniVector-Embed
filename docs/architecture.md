# OmniVector-Embed Architecture Guide

Detailed technical reference for every component in the OmniVector-Embed system.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Backbone: Bidirectional Mistral-7B](#backbone)
3. [Latent Attention Pooling](#latent-attention-pooling)
4. [Matryoshka Representation Learning](#matryoshka-representation-learning)
5. [Vision Encoder](#vision-encoder)
6. [Video Encoder](#video-encoder)
7. [Loss Functions](#loss-functions)
8. [Hard Negative Mining](#hard-negative-mining)
9. [Training Pipeline](#training-pipeline)
10. [ONNX Export Pipeline](#onnx-export-pipeline)
11. [Data Pipeline](#data-pipeline)
12. [Design Decisions](#design-decisions)

---

## System Overview

```
                    ┌──────────────┐
                    │   Input      │
                    │ (text/image/ │
                    │  video/code) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
        ┌─────▼─────┐ ┌───▼────┐ ┌────▼─────┐
        │  Mistral   │ │ SigLIP │ │  Video   │
        │ Backbone   │ │ Vision │ │ Encoder  │
        │ (bidir.)   │ │ Encoder│ │ (temp.)  │
        └─────┬──────┘ └───┬────┘ └────┬─────┘
              │            │            │
              │     ┌──────▼──────┐     │
              │     │ Linear Proj │     │
              │     │ 1152 → 4096 │     │
              │     └──────┬──────┘     │
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼───────┐
                    │   Latent     │
                    │  Attention   │
                    │   Pooling    │
                    │ (512 × 4096) │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  MRL Slice   │
                    │ [512..4096]  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ L2 Normalize │
                    └──────────────┘
```

All modalities produce 4096-dimensional, L2-normalized embeddings in a shared vector space.

---

## Backbone

**File**: `src/omnivector/model/backbone.py`

### MistralEmbeddingBackbone

The text encoder is Mistral-7B-v0.1, modified for embedding generation:

| Parameter | Value | Rationale |
|---|---|---|
| Hidden size | 4096 | Native Mistral dimension, no projection needed |
| Layers | 32 | Full Mistral depth |
| Attention | Bidirectional | Override `_update_causal_mask → None` |
| Implementation | `eager` | Required for ONNX export (no SDPA/Flash) |
| LoRA targets | q_proj, k_proj, v_proj, o_proj | Adapter fine-tuning (0.1% params) |

### Bidirectional Attention

Standard Mistral uses causal masking (each token only attends to past tokens). For embeddings, we need bidirectional attention so every token's representation incorporates the full context.

```python
def _enable_bidirectional_attention(self):
    def no_causal_mask(*args, **kwargs):
        return None
    self.model._update_causal_mask = no_causal_mask
```

This relies on `transformers==4.44.2` where `_update_causal_mask` is the single gateway for causal masking. Upgrading transformers requires verifying this API still works.

**Verification**: `test_bidirectionality.py` confirms:
- Bidirectional: changing token[3] alters token[0]'s hidden state
- Control: under causal mask, changing token[3] does NOT alter token[0]

### LoRA Configuration

```yaml
r: 16              # Rank
alpha: 32           # Scaling factor (effective scale = alpha/r = 2.0)
dropout: 0.1        # During training only
targets: [q_proj, k_proj, v_proj, o_proj]
bias: none
task_type: FEATURE_EXTRACTION
```

Before ONNX export, LoRA adapters are merged via `merge_and_unload()` to eliminate conditional branching in the computation graph.

---

## Latent Attention Pooling

**File**: `src/omnivector/model/latent_attention.py`

Converts variable-length hidden states into a fixed-size embedding.

### Architecture

1. **Learned latent queries**: 512 trainable vectors of dimension 4096
2. **Cross-attention**: Latent queries attend to encoder hidden states (8 heads)
3. **FFN**: Two-layer MLP with GELU activation
4. **Mean pooling**: Average across the 512 latent outputs → single 4096-d vector

### EagerMultiheadAttention

Custom attention implementation using explicit matmul operations instead of `F.multi_head_attention` or SDPA:

```python
# Explicit Q/K/V projections and matmuls
Q = self.q_proj(query)
K = self.k_proj(key)
V = self.v_proj(value)
scores = torch.bmm(Q, K.transpose(-2, -1)) / sqrt(d_k)
attn = softmax(scores)
output = torch.bmm(attn, V)
```

This avoids:
- **SDPA**: Uses `aten::scaled_dot_product_attention` which has no ONNX opset mapping
- **Flash Attention**: Requires CUDA and uses custom kernels
- **NaN propagation**: `torch.where` guards on softmax output prevent NaN from padding

### Why 512 Latents?

NV-Embed-v2 uses 512 latents. More latents increase capacity but also increase memory and compute. 512 provides a good balance — each latent can specialize for different semantic aspects of the input.

---

## Matryoshka Representation Learning

**File**: `src/omnivector/training/losses.py`

MRL trains the model so that prefix slices of the embedding are independently useful:

| Dimension | Weight | Use Case |
|---|---|---|
| 512 | 0.5 | Fast approximate retrieval |
| 1024 | 0.75 | Balanced speed/quality |
| 2048 | 1.0 | High-quality retrieval |
| 4096 | 1.0 | Maximum quality |

The loss is:

$$\mathcal{L}_{MRL} = \sum_{d \in \{512, 1024, 2048, 4096\}} w_d \cdot \mathcal{L}_{InfoNCE}(\text{embed}[:d])$$

Weights are **fixed buffers** (not learnable). Lower dimensions get lower weight because their representational capacity is inherently limited.

### At inference time

Users can choose any prefix dimension:

```python
embedding_fast = model.encode_text(texts, output_dim=512)   # 8× less storage
embedding_full = model.encode_text(texts, output_dim=4096)  # maximum quality
```

---

## Vision Encoder

**File**: `src/omnivector/model/vision_encoder.py`

### SigLIPVisionEncoder

| Component | Detail |
|---|---|
| Base model | `google/siglip-so400m-patch14-384` |
| Output dim | 1152 (SigLIP native) |
| Projection | Linear(1152, 4096) |
| Normalization | L2 after projection |

SigLIP is chosen over CLIP because:
- Sigmoid loss (no softmax) scales better with batch size
- SO400M variant has better image understanding than ViT-L
- 384px input captures more detail than 224px CLIP

---

## Video Encoder

**File**: `src/omnivector/model/video_encoder.py`

### VideoEncoder

Processes video as a sequence of frame embeddings:

1. **Frame sampling**: Uniform sampling of N frames from video
2. **Per-frame encoding**: Each frame through SigLIP vision encoder → 4096-d
3. **Temporal attention**: EagerMultiheadAttention across frame sequence
4. **Mean pooling**: Average across frames → single 4096-d embedding

The temporal attention module uses the same `EagerMultiheadAttention` as latent attention to maintain ONNX compatibility.

---

## Loss Functions

### MRLInfoNCELoss (`src/omnivector/training/losses.py`)

Standard InfoNCE with MRL and hard negative support:

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\exp(\text{sim}(q, p^+) / \tau) + \sum_{p^-} \exp(\text{sim}(q, p^-) / \tau)}$$

- **Temperature** ($\tau$): 0.07 (trainable)
- **In-batch negatives**: All other positives in batch serve as negatives (Stage 1 only)
- **Hard negatives**: 7 per query from FAISS mining
- **Negative similarity**: Computed via `torch.bmm` for correct `[B, N]` shape

### CrossModalContrastiveLoss (`src/omnivector/training/multimodal_loss.py`)

Symmetric cross-modal alignment:

$$\mathcal{L}_{cross} = \frac{1}{2}(\mathcal{L}_{t \rightarrow v} + \mathcal{L}_{v \rightarrow t})$$

Where $\mathcal{L}_{t \rightarrow v}$ is InfoNCE from text to vision and vice versa.

### MultimodalMRLLoss

Combines text MRL loss and cross-modal loss with configurable weighting:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{MRL} + \beta \cdot \mathcal{L}_{cross}$$

---

## Hard Negative Mining

**File**: `src/omnivector/training/hard_negative_miner.py`

### FAISS IndexFlatIP

Uses exact inner-product search (FAISS `IndexFlatIP`) to find hard negatives:

1. Encode all passages with teacher model (`BAAI/bge-large-en-v1.5`)
2. For each query, retrieve top-K most similar passages
3. Filter: score < positive_score × threshold_ratio (0.95)
4. Keep top 7 negatives per query

### Positive-Aware Threshold

Negatives must score below 95% of the positive's score. This prevents false negatives (passages that are actually relevant but not in the ground truth) from being used as negatives.

### Online Refresh

`HardNegativeRefreshCallback` re-mines negatives periodically during training (every N steps) as the model's representations evolve.

---

## Training Pipeline

### Two-Stage Training

**Stage 1: Retrieval** (`configs/stage1_retrieval.yaml`)
- 20,000 steps, LR 2e-5, cosine schedule
- ~8M pairs (text + multimodal with 40% multimodal ratio)
- In-batch negatives ON
- Hard negatives: 7 per query
- Target: MSMARCO NDCG@10 ≥ 0.52

**Stage 2: Generalist** (`configs/stage2_generalist.yaml`)
- 18,000 steps, LR 1.5e-5
- Broader task mix
- In-batch negatives OFF (prevents task interference)
- Target: MTEB average ≥ 65

### DeepSpeed ZeRO-2

```json
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "reduce_scatter": true,
        "overlap_comm": true
    },
    "fp16": {"enabled": true},
    "gradient_clipping": 1.0
}
```

ZeRO-2 partitions gradients and optimizer states across GPUs. Stage 2 (not stage 3) because:
- Stage 3 partitions parameters too, which adds communication overhead
- 7B model with LoRA fits in 2× A100 80GB with ZeRO-2

### OmniVectorTrainer

Subclass of HuggingFace `Trainer` that:
- Overrides `compute_loss` for MRL InfoNCE
- Integrates hard negative miner
- Supports gradient checkpointing
- Custom callbacks for logging, early stopping, and negative refresh

---

## ONNX Export Pipeline

**Files**: `src/omnivector/export/`

### Export Flow

```
PyTorch Model
    │
    ▼ merge LoRA (W + BA → W')
    │
    ▼ OmniVectorONNXWrapper (clean forward: input_ids, mask → embedding)
    │
    ▼ torch.onnx.export (opset 18, dynamic axes)
    │
    ▼ ORT Transformer Optimizer (fuse LayerNorm, Attention, GELU, SkipLN)
    │
    ▼ Dynamic INT8 Quantization (MatMulConstBOnly)
    │
    ▼ Cosine Parity Validation (threshold > 0.99 fp32, > 0.95 int8)
```

### Why Opset 18?

PyTorch's dynamo-based ONNX export enforces a minimum opset of 18. Opset 18 also provides native `LayerNormalization` which simplifies the graph.

### ORT Transformer Optimizer

```python
from onnxruntime.transformers import optimizer
opt_model = optimizer.optimize_model(
    "model.onnx",
    model_type="bert",
    num_heads=32,
    hidden_size=4096,
)
```

This fuses multi-head attention patterns (Q/K/V projections + matmul + softmax + output projection) into optimized ORT kernels, providing ~20-30% inference speedup.

### INT8 Quantization

Dynamic quantization with `MatMulConstBOnly=True`:
- Only quantizes weight matrices (not activations)
- No calibration data needed
- ~4× model size reduction
- Minimal accuracy loss (cosine similarity > 0.95 vs fp32)

---

## Data Pipeline

**Files**: `src/omnivector/data/`, `scripts/build_dataset.py`

### Text Datasets

| Dataset | Type | Size (est.) |
|---|---|---|
| MSMARCO | Passage retrieval | ~500k |
| HotpotQA | Multi-hop QA | ~100k |
| BEIR (6 tasks) | Diverse retrieval | ~200k |
| CodeSearchNet | Code-docstring | ~2M |

### Multimodal Datasets

| Dataset | Type | Size (target) |
|---|---|---|
| LAION (aesthetic) | Image-text | 200k |
| WebVid | Video-text | 50k |

### Data Format

All datasets are normalized to JSONL with the schema:

```json
{
    "query": "What is gradient descent?",
    "positive": "Gradient descent is an optimization algorithm...",
    "negatives": ["Support vector machines...", "Random forest..."],
    "domain": "retrieval",
    "modality": "text"
}
```

---

## Design Decisions

### Why eager attention instead of SDPA?

SDPA (`F.scaled_dot_product_attention`) uses `aten::scaled_dot_product_attention` which has no ONNX opset mapping. Flash Attention requires CUDA and custom kernels. By using explicit matmul operations, the model exports cleanly to ONNX.

**Trade-off**: ~30% slower training on GPU vs SDPA/Flash. Acceptable because we use LoRA (only 0.1% of params are trainable) and total training is ~3 days.

### Why not increase embedding dimension beyond 4096?

- 4096 is Mistral-7B's native hidden size. Going higher requires a projection head that adds parameters without adding representational capacity
- NV-Embed-v2 uses 4096, OpenAI text-embedding-3-large uses 3072, E5-Mistral uses 4096
- Matryoshka lets users trade dimension for speed at inference time

### Why fixed MRL weights instead of learnable?

Learnable weights collapsed to near-zero for lower dimensions during training. Fixed weights `[0.5, 0.75, 1.0, 1.0]` ensure all dimensions receive meaningful gradient signal.

### Why transformers==4.44.2?

- `_update_causal_mask` API is stable and accessible for monkey-patching
- RoPE uses real-valued operations (not complex) which export to ONNX
- Later versions changed the causal mask API, breaking bidirectional attention
- `peft==0.12.0` tested with this version for `merge_and_unload()`

### Why FAISS IndexFlatIP instead of approximate search?

For hard negative mining, we need exact scores to apply the positive-aware threshold. Approximate methods (IVF, HNSW) could miss critical false negatives. The corpus size (~8M) is manageable for exact search with FAISS on a single GPU.
