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
7. [Audio Encoder](#audio-encoder)
8. [Loss Functions](#loss-functions)
9. [Hard Negative Mining](#hard-negative-mining)
10. [Training Pipeline](#training-pipeline)
11. [ONNX Export Pipeline](#onnx-export-pipeline)
12. [Data Pipeline](#data-pipeline)
13. [Evaluation (MTEB)](#evaluation-mteb)
14. [Design Decisions](#design-decisions)

---

## System Overview

```
                    ┌──────────────┐
                    │   Input      │
                    │ (text/image/ │
                    │ video/audio/ │
                    │    code)     │
                    └──────┬───────┘
                           │
         ┌─────────────┬───┴────┬─────────────┐
         │             │        │             │
   ┌─────▼──────┐ ┌────▼───┐ ┌──▼──────┐ ┌────▼──────┐
   │  Mistral   │ │ SigLIP │ │  Video  │ │  Whisper  │
   │ Backbone   │ │ Vision │ │ Encoder │ │  Audio    │
   │ (bidir.)   │ │ Encoder│ │ (temp.) │ │  Encoder  │
   └─────┬──────┘ └────┬───┘ └──┬──────┘ └───┬───────┘
         │             │        │             │
         │      ┌──────▼──────┐ │      ┌──────▼──────┐
         │      │ Linear Proj │ │      │  MLP Proj   │
         │      │ 1152 → 4096 │ │      │ 384 → 4096  │
         │      └──────┬──────┘ │      └──────┬──────┘
         │             │        │             │
         └─────────────┴────┬───┴─────────────┘
                            │
              ┌─────────────▼───────────────┐
              │ dtype cast (mixed precision)│
              └─────────────┬───────────────┘
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

All modalities produce 4096-dimensional, L2-normalized embeddings in a shared vector space. During mixed-precision training, encoder outputs are explicitly cast to match the backbone's dtype before entering the pooling layer.

---

## Backbone

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

### Model Loading (`from_pretrained`)

`OmniVectorModel.from_pretrained` handles both HuggingFace Hub and local checkpoints:

1. **Local checkpoint**: If the path contains `model.pt`, loads the state dict directly with `strict=False` to tolerate partial checkpoints
2. **HuggingFace Hub**: Downloads the model via `AutoModelForCausalLM.from_pretrained` with `attn_implementation="eager"` and trust_remote_code
3. **LoRA**: Optionally applies LoRA adapters at load time via the `lora` parameter (dict with LoRA config)
4. **Multimodal encoders**: Optionally initializes vision and audio encoders via `vision_encoder` and `audio_encoder` kwargs, with `freeze_vision_backbone` control

```python
# From local checkpoint (fine-tuned)
model = OmniVectorModel.from_pretrained(
    "checkpoints/stage2_55M",
    device="cuda",
)

# From HuggingFace with LoRA
model = OmniVectorModel.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    lora={"r": 16, "alpha": 32, "dropout": 0.1},
)
```

---

## Latent Attention Pooling

Converts variable-length hidden states into a fixed-size embedding.

### Architecture

1. **Learned latent queries**: 512 trainable vectors of dimension 4096
2. **Cross-attention**: Latent queries attend to encoder hidden states (8 heads) via `EagerCrossAttention`
3. **FFN**: Two-layer MLP with GELU activation
4. **Mean pooling**: Average across the 512 latent outputs → single 4096-d vector

### Two Attention Variants

The module provides two ONNX-safe attention implementations. Both avoid `F.scaled_dot_product_attention` and `nn.MultiheadAttention` (which PyTorch 2.x routes through SDPA internally), using only explicit `torch.matmul` operations that export cleanly to ONNX opset ≥ 18.

#### EagerCrossAttention (cross-attention)

Used by `LatentAttentionPooling` where query and key/value come from **different** sources (latent vectors attend to encoder hidden states). Separate `q_proj`, `k_proj`, `v_proj` projections ensure keys and values are derived from the correct source:

```python
# Query from latent vectors, Key/Value from encoder hidden states
Q = self.q_proj(query)     # query source (latent vectors)
K = self.k_proj(key)       # key source (encoder hidden states)
V = self.v_proj(value)     # value source (encoder hidden states)
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
attn = softmax(scores)     # + NaN guard for fully-masked rows
output = torch.matmul(attn, V)
```

This distinction is critical. A shared `in_proj` (as used in self-attention) would incorrectly project keys and values from the query source, making cross-attention degenerate into self-attention and significantly degrading retrieval quality.

#### EagerMultiheadAttention (self-attention)

Used by `VideoEncoder` for temporal attention across video frames where query, key, and value originate from the **same** source. A single unified `in_proj` layer produces Q, K, V:

```python
# All from the same source (frame embeddings)
qkv = self.in_proj(query)       # [batch, L, 3*embed_dim]
Q, K, V = qkv.chunk(3, dim=-1)
scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
attn = softmax(scores)
output = torch.matmul(attn, V)
```

Both variants include:
- **NaN guard**: `torch.where(isnan(attn), zeros, attn)` prevents NaN propagation from fully-masked padding rows
- **Additive attention mask** and **key padding mask** support
- **Dropout** after softmax (training only)

### Why 512 Latents?

NV-Embed-v2 uses 512 latents. More latents increase capacity but also increase memory and compute. 512 provides a good balance — each latent can specialize for different semantic aspects of the input.

---

## Matryoshka Representation Learning

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

### SigLIPVisionEncoder

| Component | Detail |
|---|---|
| Base model | `google/siglip-so400m-patch14-384` |
| Output dim | 1152 (SigLIP native) |
| Projection | Linear(1152, 4096) |
| Normalization | L2 after projection |
| Backbone control | `freeze_backbone` parameter (default: frozen) |

SigLIP is chosen over CLIP because:
- Sigmoid loss (no softmax) scales better with batch size
- SO400M variant has better image understanding than ViT-L
- 384px input captures more detail than 224px CLIP

### Backbone Freeze / Unfreeze

By default the SigLIP backbone is frozen and only the projection layer trains. This is correct for Stage 1 and Stage 2 where the vision encoder provides fixed feature extraction:

```python
encoder = SigLIPVisionEncoder(freeze_backbone=True)   # default
encoder.trainable_parameters  # only projection weights
```

For Stage 3 multimodal alignment the backbone can be unfrozen for end-to-end fine-tuning:

```python
encoder.unfreeze_backbone()   # all SigLIP params become trainable
encoder.freeze_backbone()     # re-freeze if needed
```

When the backbone is frozen, `torch.no_grad()` is used in forward to skip gradient computation entirely. When unfrozen, `contextlib.nullcontext()` is used so gradients flow through.

---

## Video Encoder

### VideoEncoder

Processes video as a sequence of frame embeddings:

1. **Frame sampling**: Uniform sampling of N frames from video
2. **Per-frame encoding**: Each frame through SigLIP vision encoder → 4096-d
3. **Temporal attention**: EagerMultiheadAttention across frame sequence
4. **Mean pooling**: Average across frames → single 4096-d embedding

The temporal attention module uses the same `EagerMultiheadAttention` as the video encoder's self-attention to maintain ONNX compatibility. (Note: Latent Attention Pooling uses `EagerCrossAttention` — a different variant with separate Q/KV projections.)

---

## Audio Encoder

### WhisperAudioEncoder

| Component | Detail |
|---|---|
| Base model | `openai/whisper-tiny` |
| Encoder hidden dim | 384 |
| Projection | 2-layer MLP: 384 → 2240 → 4096 |
| MLP activation | GELU with LayerNorm |
| Pooling | Mean over encoder time steps |
| Normalization | L2 after projection |
| Encoder control | `freeze_encoder` parameter (default: frozen) |

The two-layer MLP bridges the large dimension gap (384 → 4096, ≈ 10.7×) via a mid-dimension of `(384 + 4096) // 2 = 2240`, providing sufficient capacity for the cross-modal projection without a single bottleneck layer.

### Supported Variants

| Variant | HuggingFace ID | Hidden Dim |
|---|---|---|
| whisper-tiny | `openai/whisper-tiny` | 384 |
| whisper-base | `openai/whisper-base` | 512 |
| whisper-small | `openai/whisper-small` | 768 |

Whisper-tiny is the default for training efficiency. The encoder is frozen by default — only the projection MLP trains. When `freeze_encoder=False`, gradients flow through the Whisper encoder using `contextlib.nullcontext()` instead of `torch.no_grad()`.

---

## Loss Functions

### MRLInfoNCELoss

Standard InfoNCE with MRL and hard negative support:

$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(q, p^+) / \tau)}{\exp(\text{sim}(q, p^+) / \tau) + \sum_{p^-} \exp(\text{sim}(q, p^-) / \tau)}$$

- **Temperature** ($\tau$): 0.07 (trainable)
- **In-batch negatives**: All other positives in batch serve as negatives (Stage 1 only)
- **Hard negatives**: 7 per query from FAISS mining
- **Negative similarity**: Computed via `torch.bmm` for correct `[B, N]` shape

### CrossModalContrastiveLoss

Symmetric cross-modal alignment:

$$\mathcal{L}_{cross} = \frac{1}{2}(\mathcal{L}_{t \rightarrow v} + \mathcal{L}_{v \rightarrow t})$$

Where $\mathcal{L}_{t \rightarrow v}$ is InfoNCE from text to vision and vice versa.

### MultimodalMRLLoss

Combines text MRL loss and cross-modal loss with configurable weighting:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{MRL} + \beta \cdot \mathcal{L}_{cross}$$

---

## Hard Negative Mining

### FAISS IndexFlatIP

Uses exact inner-product search (FAISS `IndexFlatIP`) to find hard negatives:

1. Encode all passages with teacher model (`BAAI/bge-large-en-v1.5`)
2. For each query, retrieve top-K most similar passages
3. Filter: score < positive_score × threshold_ratio (0.95)
4. Keep top 7 negatives per query

### Positive-Aware Threshold

Negatives must score below 95% of the positive's score. This prevents false negatives (passages that are actually relevant but not in the ground truth) from being used as negatives.

### Online Refresh with Model Re-Encoding

`HardNegativeRefreshCallback` re-mines negatives periodically during training (every N steps) as the model's representations evolve.

Unlike a naive approach that rebuilds the FAISS index from stale cached embeddings, the callback **re-encodes the entire corpus with the current model's weights** before rebuilding the index. This ensures the similarity landscape used for mining reflects the model's evolving representations.

The re-encoding pipeline:

1. Put model in `eval()` mode with `torch.no_grad()`
2. Tokenize corpus texts in batches using the provided tokenizer
3. Run each batch through `model.backbone` → `model.pooling` → L2-normalize
4. Concatenate all embeddings and replace `miner.corpus_embeddings`
5. Rebuild the FAISS `IndexFlatIP` from the fresh embeddings
6. Re-mine negatives for all training samples
7. Return model to `train()` mode

Required initialization:

```python
callback = HardNegativeRefreshCallback(
    refresh_steps=5000,
    miner=hard_negative_miner,
    corpus_texts=corpus_text_list,
    tokenizer=tokenizer,
    encode_batch_size=64,
    max_seq_length=512,
    device="cuda",
    train_dataset=dataset,
)
```

If the model or tokenizer is unavailable, the callback falls back to rebuilding the FAISS index from whatever embeddings are already cached.

---

## Training Pipeline

### Three-Stage Training

**Stage 1: Retrieval** (`configs/stage1_retrieval.yaml`)
- 20,000 steps, LR 2e-5, cosine schedule
- ~8M pairs (text + multimodal with 40% multimodal ratio)
- In-batch negatives ON
- Hard negatives: 7 per query
- Target: MSMARCO NDCG@10 ≥ 0.52

**Stage 2: Generalist** (`configs/stage2_generalist.yaml`)
- 18,000 steps, LR 1.5e-5
- ~55M pairs with domain-balanced upsampling (see [Data Pipeline](#data-pipeline))
- In-batch negatives OFF (prevents task interference)
- Target: MTEB average ≥ 65

**Stage 3: Multimodal Alignment** (`configs/stage3_multimodal.yaml`)
- 12,000 steps, LR 5e-6
- Cross-modal contrastive loss with configurable weight (default 0.2)
- Vision backbone optionally unfrozen for end-to-end fine-tuning
- Target: image-text recall@1 ≥ 0.75, audio-text recall@1 ≥ 0.60

### Mixed-Precision Dtype Casting

Under FP16/BF16 mixed-precision training with DeepSpeed, different model components may produce different dtypes. The `MultimodalTrainer.compute_loss` method explicitly casts all encoder outputs to match the text backbone's dtype before computing similarity:

```python
target_dtype = query_embeddings.dtype  # backbone output dtype

# Cast vision, video, audio embeddings to match
visual_embeddings = model.vision_encoder(images).to(dtype=target_dtype)
audio_embeddings = model.audio_encoder(audio_feats).to(dtype=target_dtype)
```

Without this cast, `torch.matmul` between fp16 and fp32 tensors raises a RuntimeError under autocast.

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

### MultimodalTrainer

Extended trainer for Stage 3 multimodal alignment that:
- Handles mixed text/image/video/audio batches in a single forward pass
- Applies dtype casting for mixed-precision safety across encoder outputs
- Computes weighted combination of MRL text loss and cross-modal contrastive loss
- Supports vision backbone unfreezing during training

---

## ONNX Export Pipeline

### Export Flow

```
PyTorch Model
    │
    ▼ merge LoRA (W + BA → W')
    │
    ▼ OmniVectorONNXWrapper (clean forward: input_ids, mask → embedding)
    │
    ▼ torch.onnx.export (opset 17, dynamic axes)
    │
    ▼ ORT Transformer Optimizer (fuse LayerNorm, Attention, GELU, SkipLN)
    │
    ▼ Dynamic INT8 Quantization (MatMulConstBOnly)
    │
    ▼ Cosine Parity Validation (threshold > 0.99 fp32, > 0.95 int8)
```

### Why Opset 17?

Opset 17 provides native `LayerNormalization` which simplifies the exported graph, and is the highest opset fully supported by PyTorch's legacy `torch.onnx.export()` in PyTorch 2.3.x. It avoids the opset 18 `ReduceMean` attribute-to-input migration issue that causes checker failures with the legacy exporter.

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

### Domain-Balanced Upsampling

Stage 2 training targets ~55M pairs. When upsampling from a smaller base pool, naive uniform random sampling would replicate the existing domain distribution (e.g. 90% retrieval, 5% QA, 5% code), amplifying the imbalance.

Instead, `build_dataset.py` applies a **domain-balanced upsampling strategy**:

1. **Compute per-domain counts** from the existing pool
2. **Apply inverse-frequency weighting**: `weight = sqrt(1 / domain_fraction)` — this boosts underrepresented domains without fully flattening the distribution (square-root dampening preserves some natural skew)
3. **Cap maximum amplification** at 10× to prevent extreme oversampling of very rare domains
4. **Normalize weights** to produce a proper probability distribution
5. **Allocate samples** proportional to the balanced weights
6. **Fill remaining slots** with uniform sampling from underrepresented domains

This ensures that smaller but critical domains (code search, QA) receive meaningful representation in the upsampled dataset while large domains (retrieval) remain the majority.

Domain distribution is logged before and after upsampling for auditability.

---

## Evaluation (MTEB)

### MTEBRunner

The `MTEBRunner` provides a structured interface for running MTEB benchmarks against an OmniVectorModel:

```python
from omnivector.eval.mteb_runner import MTEBRunner

runner = MTEBRunner(model=model, output_dir="eval_results", output_dim=4096)
results = runner.run(task_types=["retrieval", "sts"])
runner.print_summary(results)
```

Supported task types with curated default task sets:

| Task Type | Default Tasks | Key Metric |
|---|---|---|
| **Retrieval** | MSMARCO, NFCorpus, SciFact, ArguAna, FiQA2018 | NDCG@10 |
| **STS** | STS12–16, STSBenchmark, SICK-R | Spearman correlation |
| **Clustering** | TwentyNewsgroups, RedditClustering | V-measure |
| **Pair Classification** | TwitterURLCorpus, SprintDuplicateQuestions | AP |
| **Re-ranking** | AskUbuntuDupQuestions, StackOverflowDupQuestions | MAP |

### Model Wrapper

The `_MTEBModelWrapper` adapts an OmniVectorModel to the `encode(sentences)` interface expected by the `mteb` library. It handles:
- Batch tokenization with configurable `max_length` and `batch_size`
- Forward pass through `model.backbone` → `model.pooling`
- MRL dimension slicing (`output_dim` parameter)
- L2 normalization of final embeddings
- Automatic device placement

### Benchmark Targets

Each training stage has predefined quality gates:

| Stage | Target | Threshold |
|---|---|---|
| Stage 1 (Retrieval) | MSMARCO NDCG@10 | ≥ 0.52 |
| Stage 2 (Generalist) | MTEB average | ≥ 65.0 |
| Stage 3 (Multimodal) | Image-text recall@1 | ≥ 0.75 |
| Stage 3 (Multimodal) | Audio-text recall@1 | ≥ 0.60 |

Check targets programmatically:

```python
outcomes = runner.check_targets(results, stage="stage1")
# {'MSMARCO_ndcg_at_10': True}  — PASS ✓
```

### InternalEvaluator

A lightweight evaluator that runs without the `mteb` dependency. Computes cosine similarity between query–positive pairs and optionally checks that positive similarity exceeds negative similarity. Useful for CI pipelines:

```python
from omnivector.eval.mteb_runner import InternalEvaluator

evaluator = InternalEvaluator(model=model, output_dim=4096)
results = evaluator.evaluate_pairs(
    queries=["What is ML?"],
    positives=["Machine learning is..."],
    negatives=["Cooking recipes..."],
)
# {'mean_positive_sim': 0.87, 'mean_negative_sim': 0.12, 'accuracy': 1.0}
```

### Result Storage

Per-task results are saved as JSON in the output directory. A combined `mteb_results.json` file contains all task metrics for easy comparison across runs.

---

## Design Decisions

### Why eager attention instead of SDPA?

SDPA (`F.scaled_dot_product_attention`) uses `aten::scaled_dot_product_attention` which has no ONNX opset mapping. Flash Attention requires CUDA and custom kernels. By using explicit matmul operations, the model exports cleanly to ONNX.

**Trade-off**: ~30% slower training on GPU vs SDPA/Flash. Acceptable because we use LoRA (only 0.1% of params are trainable) and total training is ~3 days.

### Why separate cross-attention and self-attention classes?

Cross-attention (latent queries attending to encoder hidden states) requires separate `q_proj`, `k_proj`, `v_proj` because query and key/value come from different sources. A unified `in_proj` would project key and value from the query source — which is mathematically incorrect for cross-attention and causes the pooling layer to degenerate into self-attention over the latent queries alone. Self-attention (video temporal frames) can use a shared `in_proj` since Q, K, V all derive from the same frame embeddings.

### Why freeze the vision/audio backbone by default?

In Stage 1 and 2, the vision and audio encoders serve as fixed feature extractors. Freezing them: (1) saves GPU memory, (2) prevents catastrophic forgetting of pre-trained visual/audio features, and (3) speeds up training since no gradients are computed for the backbone. The `freeze_backbone` / `freeze_encoder` parameter and corresponding `unfreeze_*` / `freeze_*` methods allow controlled unfreezing for Stage 3 end-to-end fine-tuning.

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

### Why re-encode the corpus for hard negative refresh?

As the model trains, its embedding space shifts. Hard negatives mined at step 0 become stale by step 5000 — what was once a "hard" negative may now be trivially distinguishable, or a previously easy negative may have moved close to the positive in the new embedding space. Re-encoding the corpus with the current model weights before rebuilding the FAISS index ensures that the mined negatives are genuinely challenging under the model's current representations, providing a stronger training signal throughout training.

### Why domain-balanced upsampling instead of uniform sampling?

Stage 2 upsamples to ~55M pairs from a smaller base pool. Uniform random upsampling preserves the original distribution (e.g. 90% retrieval), which starves smaller but critical domains (QA, code search) of representation. Square-root inverse-frequency weighting (`weight = sqrt(1/frac)`) boosts rare domains without fully flattening the distribution — retrieval still dominates, but QA and code get meaningfully more samples. The 10× cap on amplification prevents pathological oversampling of extremely rare domains.

### Why explicit dtype casting in multimodal training?

Under FP16/BF16 mixed-precision training with DeepSpeed, the backbone may run in fp16 while vision/audio encoders (which may be frozen and not wrapped by autocast) produce fp32 outputs. Computing `torch.matmul` between tensors of different dtypes raises a RuntimeError. Explicitly casting all encoder outputs to match `query_embeddings.dtype` is the minimal, safe fix that works regardless of the autocast configuration.
