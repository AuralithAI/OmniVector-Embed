# OmniVector-Embed: Production Implementation Plan

## Context
Build a production-grade multimodal embedding model that replicates and improves upon NVIDIA NV-Embed-v2 (paper: arxiv 2405.17428). Key differentiators vs NV-Embed-v2:
1. **ONNX export** (opset 17 + int8 quantization) — NV-Embed-v2 cannot export to ONNX due to SDPA ops
2. **Multimodal** — text + code + image + video in a single unified 4096-dim space
3. **Fine-tuned on your data** — Siemens monoliths, TensorRT docs, PR diffs
4. **Free & local** — no API dependency, runs on your Linux GPU cluster

**Starting state**: Empty repo (README, .gitignore, LICENSE, `docs/2405.17428v3.pdf`)

---

## Architecture (Locked)

| Component | Spec |
|---|---|
| Base model | `mistralai/Mistral-7B-v0.1` |
| Attention (training) | Bidirectional — override `_update_causal_mask` to return `None` |
| Attention impl | **Always** `attn_implementation="eager"` — no SDPA, no Flash Attention |
| Pooling | Latent Attention Layer: 512 trainable latents × 4096, 8 heads, cross-attn + MLP + mean pool |
| Output dim | 4096 with Matryoshka (MRL) at [512, 1024, 2048, 4096] |
| LoRA | rank=16, alpha=32, dropout=0.1, targets: q/k/v/o_proj |
| Vision encoder | SigLIP-SO400M (1152-dim → linear proj → 4096) |
| Version pins | `transformers==4.44.2`, `peft==0.12.0` (load-bearing for causal mask API + LoRA export) |

---

## Repository Structure

```
OmniVector-Embed/
├── .github/workflows/
│   ├── ci.yml              # lint (ruff) + mypy + unit tests — required for PR merge
│   ├── evaluate.yml        # MTEB + internal eval — self-hosted GPU runner
│   └── export.yml          # ONNX export → int8 quantize → validate — triggered by evaluate
├── configs/
│   ├── stage1_retrieval.yaml
│   ├── stage2_generalist.yaml
│   ├── lora.yaml
│   └── deepspeed_zero2.json
├── scripts/
│   ├── train.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── evaluate.py
│   ├── mine_hard_negatives.py
│   └── build_dataset.py
├── src/omnivector/
│   ├── model/
│   │   ├── backbone.py         # MistralEmbeddingBackbone — bidirectional + LoRA
│   │   ├── latent_attention.py # LatentAttentionPooling + EagerMultiheadAttention
│   │   ├── vision_encoder.py   # SigLIPVisionEncoder (SigLIP → proj → 4096)
│   │   ├── video_encoder.py    # VideoEncoder (frames → SigLIP → temporal pool)
│   │   └── omnivector_model.py # OmniVectorModel — unified routing + MRL slicing
│   ├── training/
│   │   ├── losses.py           # MRLInfoNCELoss — InfoNCE at [512,1024,2048,4096]
│   │   ├── trainer.py          # HF Trainer subclass + DeepSpeed ZeRO-2
│   │   ├── hard_negative_miner.py  # FAISS-based positive-aware mining
│   │   └── callbacks.py
│   ├── data/
│   │   ├── schema.py           # Pydantic v2 EmbeddingPair dataclass
│   │   ├── dataset.py          # EmbeddingDataset + EmbeddingDataCollator
│   │   ├── preprocessing.py    # Tokenization + instruction prefix injection
│   │   └── loaders/            # msmarco.py, hotpotqa.py, nq.py, beir.py, synthetic.py, …
│   ├── export/
│   │   ├── onnx_exporter.py    # OmniVectorONNXWrapper + torch.onnx.export (opset 17)
│   │   ├── onnx_quantizer.py   # int8 dynamic quant (MatMulConstBOnly=True)
│   │   └── onnx_validator.py   # cosine sim > 0.99 vs PyTorch output
│   └── eval/
│       ├── mteb_runner.py
│       └── internal_eval.py
├── tests/
│   ├── unit/                   # test_backbone, test_latent_attention, test_losses, …
│   └── integration/            # test_onnx_parity, test_forward_pass, test_mrl_dims
├── pyproject.toml              # hatchling build, all deps pinned
├── .ruff.toml
└── mypy.ini
```

---

## Critical Implementation Details

### 1. Bidirectional Attention Conversion (`backbone.py`)
```python
class MistralEmbeddingBackbone(nn.Module):
    def __init__(self, config):
        self.model = MistralModel.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            attn_implementation="eager"  # NEVER change this
        )
        # Override causal mask (transformers >= 4.40 API)
        self.model._update_causal_mask = lambda *args, **kwargs: None
```
**Test**: Verify token at position 0 can see information from position T-1 by checking hidden states change when a future informative token is modified.

### 2. ONNX-Safe Latent Attention (`latent_attention.py`)
Do NOT use `nn.MultiheadAttention` — PyTorch 2.x routes it through SDPA internally.
Implement `EagerMultiheadAttention` as explicit matmuls:
```python
# Q @ K.T / sqrt(d) → softmax → @ V  (all standard ONNX ops)
attn = (Q @ K.transpose(-2, -1)) * scale
attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
attn = attn.softmax(dim=-1)
out = attn @ V
```
This maps entirely to ONNX `MatMul`, `Softmax`, `Add`, `Mul` — no custom ops.

### 3. MRL Loss (`losses.py`)
```python
# Apply InfoNCE at each dim independently, weighted sum
for dim, weight in zip([512, 1024, 2048, 4096], [0.5, 0.75, 1.0, 1.0]):
    q_slice = F.normalize(query_emb[:, :dim], dim=-1)
    p_slice = F.normalize(pos_emb[:, :dim], dim=-1)
    # sim matrix [B, B], hard negs appended
    total_loss += weight * F.cross_entropy(sim_matrix / temperature, labels)
```

### 4. ONNX Export Strategy (`onnx_exporter.py`)
```python
# Step 1: Merge LoRA into base weights (eliminates branching)
peft_model.merge_adapter()

# Step 2: Export with dynamic shapes
torch.onnx.export(
    wrapper,
    args=(dummy_input_ids, dummy_mask),
    f="omnivector_embed.onnx",
    opset_version=17,  # LayerNorm as native op
    dynamic_axes={
        "input_ids":      {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embedding":      {0: "batch_size"},
    },
)

# Step 3: ORT optimizer (fuses LayerNorm, Attention, GELU)
# python -m onnxruntime.transformers.optimizer --model_type bert --num_heads 32 --hidden_size 4096

# Step 4: int8 dynamic quantization
quantize_dynamic(..., weight_type=QuantType.QInt8,
                 extra_options={"MatMulConstBOnly": True})  # don't quantize attention scores
```

### 5. RoPE ONNX Compatibility
Mistral uses RoPE. `transformers==4.44.2` implements `apply_rotary_pos_emb` with explicit `cos`/`sin` real-valued ops → maps cleanly to ONNX `Mul`/`Add`. **Do not upgrade transformers** — newer versions may break this.

### 6. Hard Negative Mining
```python
threshold = 0.95 * positive_score   # positive-aware (NV-Embed-v2 method)
negatives = [c for c in top200 if score(c) < threshold and c != positive][:7]
```
Run offline before training, then refresh every 5k steps via `HardNegativeRefreshCallback`.

### 7. Instruction Prefix Format
```
Query side:   "Instruct: {task_description}\nQuery: {query_text}"
Passage side: no prefix (asymmetric — same as NV-Embed-v2)
```

---

## Training Configuration

### Stage 1 — Retrieval (20k steps)
- LR: 2e-5, cosine, 1k warmup
- Batch: 16/GPU × 4 grad accum × 2 GPUs = 128 effective
- Loss: InfoNCE + MRL, **in-batch negatives ON**, 7 hard negs
- Data: MSMARCO, HotpotQA, NQ, PAQ, StackExchange, NLI, BEIR suite, MIRACL, Mr.TyDi (~1.5M pairs)
- Hardware: 2× A100 80GB (or 4× RTX 3090 24GB), fp16, DeepSpeed ZeRO-2, gradient checkpointing

### Stage 2 — Generalist (18k steps)
- LR: 1.5e-5, resume from Stage 1
- Loss: **in-batch negatives OFF** (prevents signal corruption on classification/STS tasks)
- Data: All Stage 1 + synthetic (120k) + custom Siemens/TensorRT/PR-diffs (50k)

### DeepSpeed ZeRO-2 Config (for 2× A100)
- micro-batch 16 × grad-accum 4 × 2 GPUs = 128 effective batch
- fp16 mixed precision, gradient clipping 1.0

---

## Data Pipeline

| Source | Volume | Type |
|---|---|---|
| MSMARCO, HotpotQA, NQ, PAQ | ~800k | Retrieval |
| StackExchange, NLI, SQuAD | ~400k | Diverse |
| BEIR (ArguAna, BioASQ, FiQA, FEVER, HoVer, SciFact, NFCorpus) | ~200k | Domain retrieval |
| MIRACL, Mr.TyDi | ~100k | Multilingual |
| Synthetic (LLM-generated) | 120k | Short-long, code-comment |
| Custom (Siemens, TensorRT, PRs) | 50k | Domain-specific |

Hard negatives mined using `bge-large-en-v1.5` as teacher model (CPU-feasible).

---

## GitHub Workflows

**`ci.yml`** — On every PR to `main`/`develop`:
- `ruff check` + `ruff format --check`
- `mypy src/omnivector/ --strict`
- `pytest tests/unit/ --cov` (no GPU needed)
- Branch protection: both checks required before merge

**`evaluate.yml`** — Manual dispatch with `checkpoint_path` input:
- Runs on self-hosted Linux GPU runner
- Executes MTEB-Code + BEIR + internal eval
- On success → auto-triggers `export.yml`

**`export.yml`** — Auto-triggered or manual dispatch:
- CPU runner (5 min budget)
- Export → optimize → int8 quantize → validate (cosine > 0.99)
- Upload `omnivector_embed_int8.onnx` as release artifact

---

## pyproject.toml Key Pins
```toml
"transformers==4.44.2",   # load-bearing: _update_causal_mask API + RoPE real-valued ops
"peft==0.12.0",           # load-bearing: merge_adapter + DeepSpeed compat
"torch>=2.2.0,<2.4.0",
"onnx>=1.16.0",
"onnxruntime>=1.18.0",
"optimum[onnxruntime]>=1.21.0",
"faiss-cpu>=1.8.0",
"deepspeed>=0.14.0",
"mteb>=1.12.0",
```

---

## 8-Week Milestones

| Week | Goal | Milestone |
|---|---|---|
| 1 | Repo skeleton + core model | Backbone + LatentAttn unit tests green, CI pipeline running |
| 2 | Data pipeline + hard neg miner | `build_dataset.py` produces batches for MSMARCO |
| 3 | Training infrastructure | 100-step CPU dry run with loss decreasing |
| 4 | ONNX export pipeline | int8 ONNX validated (cosine > 0.99), export.yml green |
| 5 | Vision encoder + multimodal | Image embedding path working, vision unit tests pass |
| 6 | Stage 1 training (GPU) | Stage 1 checkpoint, NDCG@10 > 0.50 on MSMARCO dev |
| 7 | Stage 2 + full eval + export | MTEB results logged, final ONNX artifact ready |
| 8 | Polish + release | v0.1.0 tag, README with results, model on HuggingFace Hub |

---

## Hardware Requirements

| Use case | Minimum | Recommended |
|---|---|---|
| Training | 4× RTX 3090 24GB | 2× A100 80GB |
| ONNX export | CPU only | CPU only |
| Inference | CPU (int8 ONNX) | Any |
| Hard neg mining | CPU (FAISS) | CPU |

Update cycle (every 2-3 months): curate 10k new pairs → re-run Stage 2 only (4-8 hrs) → export new ONNX → hot-swap via NFS/S3.

---

## Top 5 Pitfalls to Avoid

1. **SDPA leak**: `nn.MultiheadAttention` routes through SDPA in PyTorch 2.x even with `eager` flag — use `EagerMultiheadAttention` (manual matmuls) in latent attention layer
2. **LoRA branching in ONNX**: Always call `merge_adapter()` before export — collapses `W + B*A` into single matrix
3. **RoPE compat**: Pin `transformers==4.44.2` — real-valued RoPE ops are stable here; newer versions risk regression
4. **Bidirectional regression**: Add explicit test that token[0] hidden state changes when a future token changes (would fail under causal masking)
5. **int8 attention quantization**: Set `MatMulConstBOnly=True` in quantizer — only quantize weight matrices, not dynamic `Q@K^T` attention scores

---

## Verification

**Architecture correctness**: `pytest tests/unit/` — bidirectionality test, latent attn shape test, MRL loss numerical test

**ONNX parity**: `pytest tests/integration/test_onnx_parity.py` — cosine sim > 0.99 for 50 random inputs

**Training quality**: NDCG@10 on MSMARCO dev (target > 0.55 after Stage 2), full MTEB-Code score (target > 68)

**C++ compatibility**: `tests/cpp/smoke_test.cpp` via ORT C++ API — load int8 model, verify output shape [1, 4096] and unit norm
