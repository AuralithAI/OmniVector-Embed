"""Script to build training datasets with hard negative mining.

Supports text retrieval datasets (MSMARCO, HotpotQA, NQ, BEIR) and
multimodal datasets (LAION, WebVid, CodeSearchNet) for Stage 1/2 training.
Includes synthetic data generation and custom dataset loading for Stage 2
scaling to 55M+ pairs.

Usage:
    python scripts/build_dataset.py --stage 1 --output-dir data/stage1
    python scripts/build_dataset.py --stage 1 --multimodal --target 8000000 --output-dir data/stage1_8M
    python scripts/build_dataset.py --stage 2 --multimodal --target 55000000 \
        --add-synthetic 200000 --add-custom path/to/custom_data --output-dir data/stage2_55M
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


TEXT_DATASETS_STAGE1 = [
    "msmarco",
    "hotpotqa",
    "beir/nfcorpus",
    "beir/fiqa",
    "beir/scifact",
    "beir/arguana",
]

TEXT_DATASETS_STAGE2 = TEXT_DATASETS_STAGE1 + [
    "beir/fever",
    "beir/bioasq",
]


def encode_texts(
    texts: list[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with a teacher model.

    Args:
        texts: List of text strings to encode.
        model_name: HuggingFace model ID for the teacher encoder.
        batch_size: Encoding batch size.

    Returns:
        Numpy array of shape [num_texts, embed_dim].
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    logger.info(f"Encoding {len(texts)} texts with {model_name}")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def load_text_datasets(
    dataset_names: list[str],
    max_samples: Optional[int] = None,
) -> list:
    """Load and combine text retrieval datasets.

    Args:
        dataset_names: List of dataset identifiers.
        max_samples: Per-dataset sample limit.

    Returns:
        Combined list of EmbeddingPair objects.
    """
    from omnivector.data.loaders.base import get_loader

    all_pairs = []
    for name in dataset_names:
        try:
            loader = get_loader(name, max_samples=max_samples)
            pairs = loader.load()
            all_pairs.extend(pairs)
            logger.info(f"Loaded {len(pairs)} pairs from {name}")
        except Exception as e:
            logger.warning(f"Failed to load {name}: {e}")

    logger.info(f"Total text pairs: {len(all_pairs)}")
    return all_pairs


def load_multimodal_datasets(
    max_image_samples: int = 200_000,
    max_video_samples: int = 50_000,
    max_code_samples: Optional[int] = None,
) -> list:
    """Load multimodal datasets (image-text, video-text, code).

    Args:
        max_image_samples: Max image-text pairs from LAION.
        max_video_samples: Max video-text pairs from WebVid.
        max_code_samples: Max code-docstring pairs from CodeSearchNet.

    Returns:
        List of dicts with query, positive, domain, modality keys.
    """
    from datasets import load_dataset

    multimodal_pairs = []

    # LAION aesthetics (image-text)
    try:
        logger.info(f"Loading LAION aesthetics ({max_image_samples} samples)...")
        laion = load_dataset(
            "laion/relaion2B-en-research-safe",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in laion:
            if count >= max_image_samples:
                break
            text = sample.get("TEXT", sample.get("caption", ""))
            url = sample.get("URL", sample.get("url", ""))
            if text and url:
                multimodal_pairs.append({
                    "query": text,
                    "positive": text,
                    "domain": "image_text",
                    "modality": "image",
                    "image_url": url,
                })
                count += 1
        logger.info(f"Loaded {count} LAION image-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load LAION: {e}")

    # WebVid (video-text)
    try:
        logger.info(f"Loading WebVid ({max_video_samples} samples)...")
        webvid = load_dataset(
            "TempoFunk/webvid-10M",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in webvid:
            if count >= max_video_samples:
                break
            text = sample.get("caption", sample.get("name", ""))
            url = sample.get("contentUrl", sample.get("video", ""))
            if text and url:
                multimodal_pairs.append({
                    "query": text,
                    "positive": text,
                    "domain": "video_text",
                    "modality": "video",
                    "video_url": url,
                })
                count += 1
        logger.info(f"Loaded {count} WebVid video-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load WebVid: {e}")

    # AudioSet (audio-text)
    try:
        logger.info("Loading AudioSet (30k samples)...")
        audioset = load_dataset(
            "agkphysics/AudioSet",
            split="train",
            streaming=True,
        )
        count = 0
        for sample in audioset:
            if count >= 30_000:
                break
            labels = sample.get("human_labels", sample.get("labels", ""))
            if isinstance(labels, list):
                labels = ", ".join(labels)
            video_id = sample.get("video_id", "")
            if labels:
                multimodal_pairs.append({
                    "query": labels,
                    "positive": labels,
                    "domain": "audio_text",
                    "modality": "audio",
                    "audio_id": video_id,
                })
                count += 1
        logger.info(f"Loaded {count} AudioSet audio-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load AudioSet: {e}")

    # CodeSearchNet (code-docstring)
    try:
        logger.info("Loading CodeSearchNet...")
        code_ds = load_dataset("code_search_net", "all", split="train")
        if max_code_samples:
            code_ds = code_ds.select(range(min(max_code_samples, len(code_ds))))
        for sample in code_ds:
            docstring = sample.get("func_documentation_string", "")
            code = sample.get("func_code_string", "")
            lang = sample.get("language", "unknown")
            if docstring and code:
                multimodal_pairs.append({
                    "query": docstring,
                    "positive": code,
                    "domain": f"code_{lang}",
                    "modality": "text",
                })
        logger.info(f"Loaded {len(code_ds)} CodeSearchNet pairs")
    except Exception as e:
        logger.warning(f"Failed to load CodeSearchNet: {e}")

    logger.info(f"Total multimodal pairs: {len(multimodal_pairs)}")
    return multimodal_pairs


def generate_synthetic_pairs(
    num_pairs: int = 200_000,
    domains: Optional[list[str]] = None,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic training pairs using template-based augmentation.

    Creates code-diagram description pairs, voiceover-transcript pairs,
    and paraphrased query-passage pairs for data scaling. Uses structured
    templates with domain-specific vocabulary to produce diverse, high-quality
    training signal without requiring LLM API calls at build time.

    For production use with LLM-generated data, set SYNTHETIC_LLM_ENDPOINT
    environment variable to point to a Mixtral/Llama-3.1 inference server.

    Args:
        num_pairs: Number of synthetic pairs to generate.
        domains: List of target domains. Defaults to a balanced mix of
            code, diagram, voiceover, and paraphrase domains.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with query, positive, domain, modality keys.
    """
    import os

    rng = np.random.default_rng(seed)

    if domains is None:
        domains = [
            "code_diagram",
            "voiceover_transcript",
            "paraphrase_query",
            "code_docstring_aug",
        ]

    # Template pools for each domain
    _CODE_TEMPLATES = [
        ("How to {action} in {lang}?", "```{lang}\n{snippet}\n```"),
        ("Implement {pattern} pattern in {lang}", "{description} using {pattern}"),
        ("{lang} function to {action}", "def {func_name}({params}): {body}"),
        ("Convert {source} to {target} format", "Parse {source} and serialize to {target}"),
        ("Optimize {operation} for {constraint}", "Use {technique} to reduce {metric}"),
    ]

    _ACTIONS = [
        "sort a list", "parse JSON", "read a file", "connect to database",
        "handle errors", "create REST API", "implement caching",
        "validate input", "serialize data", "compress files",
        "manage threads", "build a graph", "traverse a tree",
        "implement pagination", "generate reports",
    ]

    _LANGUAGES = ["Python", "TypeScript", "Rust", "Go", "Java", "C++"]

    _PATTERNS = [
        "observer", "factory", "singleton", "strategy", "decorator",
        "adapter", "builder", "iterator", "state machine", "pipeline",
    ]

    _VOICEOVER_TEMPLATES = [
        ("Explain {topic} for beginners", "In this tutorial, we cover {topic} step by step."),
        ("Walkthrough of {tool} setup", "First, install {tool}. Then configure {config}."),
        ("Demo: {feature} in action", "Watch how {feature} handles {scenario}."),
        ("Lecture on {concept}", "Today's topic is {concept}. Key points include {points}."),
    ]

    _TOPICS = [
        "neural network training", "Docker containerization",
        "Kubernetes deployment", "CI/CD pipelines", "microservices",
        "data preprocessing", "model quantization", "ONNX export",
        "embedding models", "attention mechanisms", "vector databases",
        "distributed training", "gradient checkpointing", "mixed precision",
    ]

    _PARAPHRASE_PAIRS = [
        ("What is {concept}?", "Definition and explanation of {concept}"),
        ("How does {concept} work?", "{concept} operates by {mechanism}"),
        ("Best practices for {concept}", "Recommended approaches to {concept}"),
        ("Troubleshoot {issue}", "Common solutions for {issue}"),
        ("{concept} vs {alternative}", "Comparison between {concept} and {alternative}"),
    ]

    # Check for LLM endpoint for high-quality generation
    llm_endpoint = os.environ.get("SYNTHETIC_LLM_ENDPOINT")
    if llm_endpoint:
        logger.info(f"Using LLM endpoint for synthetic generation: {llm_endpoint}")
        return _generate_with_llm(llm_endpoint, num_pairs, domains, seed)

    logger.info(f"Generating {num_pairs} synthetic pairs (template-based)")
    pairs = []
    pairs_per_domain = num_pairs // len(domains)

    for domain in domains:
        for i in range(pairs_per_domain):
            if domain == "code_diagram":
                tmpl_q, tmpl_p = _CODE_TEMPLATES[rng.integers(len(_CODE_TEMPLATES))]
                action = _ACTIONS[rng.integers(len(_ACTIONS))]
                lang = _LANGUAGES[rng.integers(len(_LANGUAGES))]
                pattern = _PATTERNS[rng.integers(len(_PATTERNS))]
                query = tmpl_q.format(
                    action=action, lang=lang, pattern=pattern,
                    source="CSV", target="JSON",
                    operation=action, constraint="memory",
                )
                positive = tmpl_p.format(
                    action=action, lang=lang, snippet=f"# {action}",
                    pattern=pattern, description=f"Implementation of {action}",
                    func_name=action.replace(" ", "_"), params="data",
                    body=f"return {action.replace(' ', '_')}(data)",
                    source="CSV", target="JSON",
                    technique="batching", metric="latency",
                )
                pairs.append({
                    "query": query,
                    "positive": positive,
                    "domain": "code_diagram",
                    "modality": "text",
                })

            elif domain == "voiceover_transcript":
                tmpl_q, tmpl_p = _VOICEOVER_TEMPLATES[
                    rng.integers(len(_VOICEOVER_TEMPLATES))
                ]
                topic = _TOPICS[rng.integers(len(_TOPICS))]
                query = tmpl_q.format(
                    topic=topic, tool=topic, feature=topic, concept=topic,
                )
                positive = tmpl_p.format(
                    topic=topic, tool=topic, config="settings.yaml",
                    feature=topic, scenario="edge cases",
                    concept=topic, points="architecture and implementation",
                )
                pairs.append({
                    "query": query,
                    "positive": positive,
                    "domain": "voiceover_transcript",
                    "modality": "audio",
                })

            elif domain == "paraphrase_query":
                tmpl_q, tmpl_p = _PARAPHRASE_PAIRS[
                    rng.integers(len(_PARAPHRASE_PAIRS))
                ]
                topic_idx = rng.integers(len(_TOPICS))
                concept = _TOPICS[topic_idx]
                alt_idx = (topic_idx + 1) % len(_TOPICS)
                alternative = _TOPICS[alt_idx]
                query = tmpl_q.format(
                    concept=concept, issue=concept, alternative=alternative,
                )
                positive = tmpl_p.format(
                    concept=concept, mechanism="systematic processing",
                    issue=concept, alternative=alternative,
                )
                pairs.append({
                    "query": query,
                    "positive": positive,
                    "domain": "paraphrase",
                    "modality": "text",
                })

            elif domain == "code_docstring_aug":
                action = _ACTIONS[rng.integers(len(_ACTIONS))]
                lang = _LANGUAGES[rng.integers(len(_LANGUAGES))]
                func = action.replace(" ", "_")
                query = f"{lang}: {action}"
                positive = (
                    f"def {func}(data):\n"
                    f'    """{action.capitalize()} using {lang} idioms.\n\n'
                    f"    Args:\n        data: Input data.\n\n"
                    f"    Returns:\n        Processed result.\n"
                    f'    """\n'
                    f"    return process_{func}(data)"
                )
                pairs.append({
                    "query": query,
                    "positive": positive,
                    "domain": f"code_{lang.lower()}",
                    "modality": "text",
                })

    # Fill remaining pairs with random domain selection
    remaining = num_pairs - len(pairs)
    for _ in range(remaining):
        domain = domains[rng.integers(len(domains))]
        topic = _TOPICS[rng.integers(len(_TOPICS))]
        pairs.append({
            "query": f"Explain {topic}",
            "positive": f"An overview of {topic} and its applications.",
            "domain": domain,
            "modality": "text",
        })

    rng.shuffle(pairs)
    logger.info(f"Generated {len(pairs)} synthetic pairs across {len(domains)} domains")
    return pairs


def _generate_with_llm(
    endpoint: str,
    num_pairs: int,
    domains: list[str],
    seed: int,
) -> list[dict]:
    """Generate synthetic pairs using an LLM inference endpoint.

    Sends structured prompts to a Mixtral/Llama-3.1 inference server
    to generate high-quality code-diagram, voiceover-transcript, and
    paraphrased query-passage pairs.

    Args:
        endpoint: URL of the LLM inference server.
        num_pairs: Number of pairs to generate.
        domains: Target domains for generation.
        seed: Random seed.

    Returns:
        List of generated pair dicts.
    """
    import requests

    rng = np.random.default_rng(seed)
    pairs = []
    batch_size = 50  # Pairs per LLM request

    prompts_by_domain = {
        "code_diagram": (
            "Generate {n} pairs of (search query, code snippet) for diverse "
            "programming tasks. Output JSON array of {{query, positive}} objects."
        ),
        "voiceover_transcript": (
            "Generate {n} pairs of (voiceover description, transcript excerpt) "
            "for technical tutorials. Output JSON array of {{query, positive}} objects."
        ),
        "paraphrase_query": (
            "Generate {n} pairs of (question, detailed answer) on technical "
            "topics. Output JSON array of {{query, positive}} objects."
        ),
        "code_docstring_aug": (
            "Generate {n} pairs of (function description, Python docstring + code) "
            "for various utilities. Output JSON array of {{query, positive}} objects."
        ),
    }

    pairs_per_domain = num_pairs // len(domains)

    for domain in domains:
        prompt_template = prompts_by_domain.get(
            domain,
            "Generate {n} pairs of (query, passage) pairs. Output JSON array.",
        )

        generated = 0
        while generated < pairs_per_domain:
            n = min(batch_size, pairs_per_domain - generated)
            prompt = prompt_template.format(n=n)

            try:
                response = requests.post(
                    endpoint,
                    json={
                        "prompt": prompt,
                        "max_tokens": 4096,
                        "temperature": 0.8,
                        "seed": seed + generated,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()

                # Parse generated pairs from response
                text = result.get("choices", [{}])[0].get("text", "")
                import json as _json

                try:
                    batch = _json.loads(text)
                    if isinstance(batch, list):
                        for item in batch:
                            if "query" in item and "positive" in item:
                                pairs.append({
                                    "query": item["query"],
                                    "positive": item["positive"],
                                    "domain": domain,
                                    "modality": "text",
                                })
                                generated += 1
                except _json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response for {domain}")

            except (requests.RequestException, KeyError) as e:
                logger.warning(f"LLM generation failed for {domain}: {e}")
                break

    logger.info(f"LLM-generated {len(pairs)} synthetic pairs")
    return pairs


def load_custom_dataset(
    custom_path: str,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load custom/proprietary datasets from a directory of JSONL files.

    Reads all .jsonl files from the specified directory. Each line must
    contain at minimum 'query' and 'positive' fields. Optional fields:
    'negatives', 'domain', 'modality'.

    Supports nested directory structures — all JSONL files are discovered
    recursively.

    Args:
        custom_path: Path to directory containing JSONL files.
        max_samples: Maximum total samples to load across all files.

    Returns:
        List of dicts with query, positive, domain, modality keys.

    Raises:
        FileNotFoundError: If custom_path doesn't exist.
    """
    custom_dir = Path(custom_path)
    if not custom_dir.exists():
        raise FileNotFoundError(f"Custom dataset path not found: {custom_path}")

    if custom_dir.is_file():
        # Single file
        jsonl_files = [custom_dir]
    else:
        # Directory — find all JSONL files recursively
        jsonl_files = sorted(custom_dir.rglob("*.jsonl"))

    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {custom_path}")
        return []

    pairs = []
    total_loaded = 0

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if max_samples and total_loaded >= max_samples:
                        break

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Skipping malformed JSON at {jsonl_file}:{line_num}"
                        )
                        continue

                    # Validate minimum required fields
                    query = record.get("query", record.get("question", ""))
                    positive = record.get("positive", record.get("answer", record.get("passage", "")))

                    if not query or not positive:
                        continue

                    pairs.append({
                        "query": query,
                        "positive": positive,
                        "negatives": record.get("negatives", []),
                        "domain": record.get("domain", f"custom_{jsonl_file.stem}"),
                        "modality": record.get("modality", "text"),
                    })
                    total_loaded += 1

        except OSError as e:
            logger.warning(f"Failed to read {jsonl_file}: {e}")

        if max_samples and total_loaded >= max_samples:
            break

    logger.info(
        f"Loaded {len(pairs)} custom pairs from {len(jsonl_files)} files "
        f"in {custom_path}"
    )
    return pairs


def mine_hard_negatives(
    pairs: list,
    teacher_model: str,
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
    batch_size: int = 64,
) -> list:
    """Mine hard negatives for a list of pairs using a teacher model.

    Args:
        pairs: List of EmbeddingPair objects.
        teacher_model: HuggingFace model ID for teacher encoder.
        num_negatives: Number of hard negatives per query.
        threshold_ratio: Score threshold relative to positive score.
        batch_size: Encoding batch size.

    Returns:
        Updated list of pairs with negatives populated.
    """
    from omnivector.training.hard_negative_miner import HardNegativeMiner

    queries = [p.query for p in pairs]
    positives = [p.positive for p in pairs]

    logger.info("Encoding queries with teacher model...")
    query_embeddings = encode_texts(queries, teacher_model, batch_size)

    logger.info("Encoding positives with teacher model...")
    positive_embeddings = encode_texts(positives, teacher_model, batch_size)

    corpus_ids = list(range(len(positives)))
    miner = HardNegativeMiner(
        corpus_embeddings=positive_embeddings,
        corpus_ids=corpus_ids,
        num_negatives=num_negatives,
        threshold_ratio=threshold_ratio,
    )

    logger.info("Mining hard negatives...")
    all_negatives = miner.mine_batch(query_embeddings, corpus_ids)

    for pair, neg_ids in zip(pairs, all_negatives):
        pair.negatives = [positives[nid] for nid in neg_ids if nid < len(positives)]

    logger.info(f"Mined hard negatives for {len(pairs)} pairs")
    return pairs


def save_dataset(
    pairs: list,
    output_dir: Path,
    filename: str = "train.jsonl",
) -> Path:
    """Save dataset in JSONL format for training.

    Args:
        pairs: List of dicts or EmbeddingPair objects.
        output_dir: Directory to save dataset.
        filename: Output filename.

    Returns:
        Path to saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in pairs:
            if hasattr(pair, "query"):
                record = {
                    "query": pair.query,
                    "positive": pair.positive,
                    "negatives": getattr(pair, "negatives", []),
                    "domain": getattr(pair, "domain", "retrieval"),
                }
            else:
                record = pair
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(pairs)} pairs to {output_file}")
    return output_file


def build_stage_dataset(
    stage: int,
    output_dir: Path,
    multimodal: bool = False,
    target_size: Optional[int] = None,
    teacher_model: Optional[str] = None,
    max_samples_per_dataset: Optional[int] = None,
    num_negatives: int = 7,
    threshold_ratio: float = 0.95,
    add_synthetic: int = 0,
    add_custom: Optional[str] = None,
) -> None:
    """Build complete training dataset for a given stage.

    Args:
        stage: Training stage (1 or 2).
        output_dir: Output directory.
        multimodal: Include image/video/code data.
        target_size: Target total dataset size.
        teacher_model: Teacher model for hard negative mining.
        max_samples_per_dataset: Per-dataset sample limit.
        num_negatives: Hard negatives per query.
        threshold_ratio: Hard negative threshold.
        add_synthetic: Number of synthetic pairs to generate (0 = disabled).
        add_custom: Path to custom JSONL dataset directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = TEXT_DATASETS_STAGE1 if stage == 1 else TEXT_DATASETS_STAGE2

    # Load text datasets
    logger.info(f"Building Stage {stage} dataset...")
    text_pairs = load_text_datasets(dataset_names, max_samples=max_samples_per_dataset)

    # Mine hard negatives if teacher model provided
    if teacher_model:
        text_pairs = mine_hard_negatives(
            text_pairs,
            teacher_model=teacher_model,
            num_negatives=num_negatives,
            threshold_ratio=threshold_ratio,
        )

    # Save text pairs
    save_dataset(text_pairs, output_dir, "text_pairs.jsonl")

    all_records = []
    for pair in text_pairs:
        all_records.append({
            "query": pair.query,
            "positive": pair.positive,
            "negatives": getattr(pair, "negatives", []),
            "domain": getattr(pair, "domain", "retrieval"),
            "modality": "text",
        })

    # Load multimodal datasets
    if multimodal:
        mm_pairs = load_multimodal_datasets(
            max_image_samples=200_000,
            max_video_samples=50_000,
            max_code_samples=2_000_000 if stage == 1 else 500_000,
        )
        all_records.extend(mm_pairs)
        save_dataset(mm_pairs, output_dir, "multimodal_pairs.jsonl")

    # Generate synthetic data 
    if add_synthetic > 0:
        logger.info(f"Generating {add_synthetic} synthetic training pairs...")
        synthetic_pairs = generate_synthetic_pairs(num_pairs=add_synthetic)
        all_records.extend(synthetic_pairs)
        save_dataset(synthetic_pairs, output_dir, "synthetic_pairs.jsonl")
        logger.info(f"Added {len(synthetic_pairs)} synthetic pairs")

    # Load custom/proprietary datasets
    if add_custom:
        logger.info(f"Loading custom dataset from: {add_custom}")
        custom_pairs = load_custom_dataset(add_custom)
        all_records.extend(custom_pairs)
        save_dataset(custom_pairs, output_dir, "custom_pairs.jsonl")
        logger.info(f"Added {len(custom_pairs)} custom pairs")

    # Upsample to target size if needed (with replacement for scaling)
    if target_size and len(all_records) < target_size:
        logger.info(
            f"Upsampling from {len(all_records)} to {target_size} "
            f"({target_size / len(all_records):.1f}x)"
        )
        rng = np.random.default_rng(42)
        extra_needed = target_size - len(all_records)
        extra_indices = rng.choice(len(all_records), size=extra_needed, replace=True)
        all_records.extend([all_records[i] for i in extra_indices])

    # Subsample to target size if needed
    if target_size and len(all_records) > target_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_records), size=target_size, replace=False)
        all_records = [all_records[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {target_size} records")

    # Shuffle and save combined dataset
    rng = np.random.default_rng(42)
    rng.shuffle(all_records)
    save_dataset(all_records, output_dir, "train.jsonl")

    # Save stats
    modality_counts = {}
    domain_counts = {}
    for r in all_records:
        mod = r.get("modality", "text")
        dom = r.get("domain", "unknown")
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

    stats = {
        "stage": stage,
        "total_samples": len(all_records),
        "multimodal": multimodal,
        "modality_counts": modality_counts,
        "domain_counts": domain_counts,
    }
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Stage {stage} dataset complete: {len(all_records)} total samples")
    logger.info(f"Modality breakdown: {modality_counts}")
    logger.info(f"Stats saved to {stats_file}")


def main():
    """CLI entry point for dataset building."""
    parser = argparse.ArgumentParser(
        description="Build training datasets with hard negative mining"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        required=True,
        help="Training stage (1=retrieval, 2=generalist)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save dataset files",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Include image/video/code multimodal data",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help="Target total dataset size",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        default=None,
        help="Teacher model for hard negative mining (e.g., BAAI/bge-large-en-v1.5)",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Maximum samples per text dataset",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=7,
        help="Number of hard negatives per sample",
    )
    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=0.95,
        help="Threshold ratio for hard negative selection",
    )
    parser.add_argument(
        "--add-synthetic",
        type=int,
        default=0,
        help="Number of synthetic training pairs to generate (0=disabled)."
        " Uses template-based generation by default; set SYNTHETIC_LLM_ENDPOINT"
        " env var for LLM-based generation.",
    )
    parser.add_argument(
        "--add-custom",
        type=str,
        default=None,
        help="Path to directory of custom JSONL files (each line: "
        '{"query": ..., "positive": ...}). Recursively discovers .jsonl files.',
    )

    args = parser.parse_args()

    build_stage_dataset(
        stage=args.stage,
        output_dir=args.output_dir,
        multimodal=args.multimodal,
        target_size=args.target,
        teacher_model=args.teacher_model,
        max_samples_per_dataset=args.max_samples_per_dataset,
        num_negatives=args.num_negatives,
        threshold_ratio=args.threshold_ratio,
        add_synthetic=args.add_synthetic,
        add_custom=args.add_custom,
    )


if __name__ == "__main__":
    main()
