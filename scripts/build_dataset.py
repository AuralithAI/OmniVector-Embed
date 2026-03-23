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
import sys
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
    "beir/trec-covid",
    "beir/scidocs",
    "beir/dbpedia-entity",
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
    import time as _time

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
        _t0 = _time.monotonic()
        _log_interval = 10_000
        for sample in laion:
            if count >= max_image_samples:
                break
            text = sample.get("TEXT", sample.get("caption", ""))
            url = sample.get("URL", sample.get("url", ""))
            if text and url:
                multimodal_pairs.append(
                    {
                        "query": text,
                        "positive": text,
                        "domain": "image_text",
                        "modality": "image",
                        "image_url": url,
                    }
                )
                count += 1
                if count % _log_interval == 0:
                    _elapsed = _time.monotonic() - _t0
                    _rate = count / _elapsed if _elapsed > 0 else 0
                    _eta = (max_image_samples - count) / _rate if _rate > 0 else 0
                    logger.info(
                        f"LAION progress: {count}/{max_image_samples} "
                        f"({100 * count / max_image_samples:.1f}%) — "
                        f"{_rate:.0f} samples/s, ETA={_eta:.0f}s"
                    )
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
        _t0 = _time.monotonic()
        _log_interval = 10_000
        for sample in webvid:
            if count >= max_video_samples:
                break
            text = sample.get("caption", sample.get("name", ""))
            url = sample.get("contentUrl", sample.get("video", ""))
            if text and url:
                multimodal_pairs.append(
                    {
                        "query": text,
                        "positive": text,
                        "domain": "video_text",
                        "modality": "video",
                        "video_url": url,
                    }
                )
                count += 1
                if count % _log_interval == 0:
                    _elapsed = _time.monotonic() - _t0
                    _rate = count / _elapsed if _elapsed > 0 else 0
                    _eta = (max_video_samples - count) / _rate if _rate > 0 else 0
                    logger.info(
                        f"WebVid progress: {count}/{max_video_samples} "
                        f"({100 * count / max_video_samples:.1f}%) — "
                        f"{_rate:.0f} samples/s, ETA={_eta:.0f}s"
                    )
        logger.info(f"Loaded {count} WebVid video-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load WebVid: {e}")

    # AudioSet (audio-text) — load metadata only, skip audio decoding
    try:
        logger.info("Loading AudioSet (30k samples)...")
        # The agkphysics/AudioSet dataset has an Audio() feature that triggers
        # torchcodec loading even in streaming mode. We disable decoding by
        # casting the audio column to a plain string/null before iteration.
        from datasets import Audio as _AudioFeature

        audioset = load_dataset(
            "agkphysics/AudioSet",
            split="train",
            streaming=True,
        )
        # Disable audio decoding by casting audio feature to None/removing it
        # This must happen BEFORE iteration to prevent torchcodec init
        audioset = audioset.cast_column("audio", _AudioFeature(decode=False))
        audioset = audioset.remove_columns(["audio"])
        logger.info("Disabled audio decoding and removed audio column")

        count = 0
        _t0 = _time.monotonic()
        _log_interval = 5_000
        max_audio = 30_000
        for sample in audioset:
            if count >= max_audio:
                break
            labels = sample.get("human_labels", sample.get("labels", ""))
            if isinstance(labels, list):
                labels = ", ".join(labels)
            video_id = sample.get("video_id", "")
            if labels:
                multimodal_pairs.append(
                    {
                        "query": labels,
                        "positive": labels,
                        "domain": "audio_text",
                        "modality": "audio",
                        "audio_id": video_id,
                    }
                )
                count += 1
                if count % _log_interval == 0:
                    _elapsed = _time.monotonic() - _t0
                    _rate = count / _elapsed if _elapsed > 0 else 0
                    _eta = (max_audio - count) / _rate if _rate > 0 else 0
                    logger.info(
                        f"AudioSet progress: {count}/{max_audio} "
                        f"({100 * count / max_audio:.1f}%) — "
                        f"{_rate:.0f} samples/s, ETA={_eta:.0f}s"
                    )
        logger.info(f"Loaded {count} AudioSet audio-text pairs")
    except Exception as e:
        logger.warning(f"Failed to load AudioSet: {e}")

    # CodeSearchNet (code-docstring)
    try:
        logger.info("Loading CodeSearchNet...")
        code_ds = load_dataset("code_search_net", "all", split="train")
        if max_code_samples:
            code_ds = code_ds.select(range(min(max_code_samples, len(code_ds))))
        total_code = len(code_ds)
        _t0 = _time.monotonic()
        _log_interval = 200_000
        code_count = 0
        for i, sample in enumerate(code_ds):
            docstring = sample.get("func_documentation_string", "")
            code = sample.get("func_code_string", "")
            lang = sample.get("language", "unknown")
            if docstring and code:
                multimodal_pairs.append(
                    {
                        "query": docstring,
                        "positive": code,
                        "domain": f"code_{lang}",
                        "modality": "text",
                    }
                )
                code_count += 1
            if (i + 1) % _log_interval == 0:
                _elapsed = _time.monotonic() - _t0
                _rate = (i + 1) / _elapsed if _elapsed > 0 else 0
                _eta = (total_code - i - 1) / _rate if _rate > 0 else 0
                logger.info(
                    f"CodeSearchNet progress: {i + 1}/{total_code} "
                    f"({100 * (i + 1) / total_code:.1f}%) — "
                    f"{_rate:.0f} rows/s, ETA={_eta:.0f}s"
                )
        logger.info(f"Loaded {code_count} CodeSearchNet pairs (from {total_code} rows)")
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
        "sort a list",
        "parse JSON",
        "read a file",
        "connect to database",
        "handle errors",
        "create REST API",
        "implement caching",
        "validate input",
        "serialize data",
        "compress files",
        "manage threads",
        "build a graph",
        "traverse a tree",
        "implement pagination",
        "generate reports",
    ]

    _LANGUAGES = ["Python", "TypeScript", "Rust", "Go", "Java", "C++"]

    _PATTERNS = [
        "observer",
        "factory",
        "singleton",
        "strategy",
        "decorator",
        "adapter",
        "builder",
        "iterator",
        "state machine",
        "pipeline",
    ]

    _VOICEOVER_TEMPLATES = [
        ("Explain {topic} for beginners", "In this tutorial, we cover {topic} step by step."),
        ("Walkthrough of {tool} setup", "First, install {tool}. Then configure {config}."),
        ("Demo: {feature} in action", "Watch how {feature} handles {scenario}."),
        ("Lecture on {concept}", "Today's topic is {concept}. Key points include {points}."),
    ]

    _TOPICS = [
        "neural network training",
        "Docker containerization",
        "Kubernetes deployment",
        "CI/CD pipelines",
        "microservices",
        "data preprocessing",
        "model quantization",
        "ONNX export",
        "embedding models",
        "attention mechanisms",
        "vector databases",
        "distributed training",
        "gradient checkpointing",
        "mixed precision",
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
        for _i in range(pairs_per_domain):
            if domain == "code_diagram":
                tmpl_q, tmpl_p = _CODE_TEMPLATES[rng.integers(len(_CODE_TEMPLATES))]
                action = _ACTIONS[rng.integers(len(_ACTIONS))]
                lang = _LANGUAGES[rng.integers(len(_LANGUAGES))]
                pattern = _PATTERNS[rng.integers(len(_PATTERNS))]
                query = tmpl_q.format(
                    action=action,
                    lang=lang,
                    pattern=pattern,
                    source="CSV",
                    target="JSON",
                    operation=action,
                    constraint="memory",
                )
                positive = tmpl_p.format(
                    action=action,
                    lang=lang,
                    snippet=f"# {action}",
                    pattern=pattern,
                    description=f"Implementation of {action}",
                    func_name=action.replace(" ", "_"),
                    params="data",
                    body=f"return {action.replace(' ', '_')}(data)",
                    source="CSV",
                    target="JSON",
                    technique="batching",
                    metric="latency",
                )
                pairs.append(
                    {
                        "query": query,
                        "positive": positive,
                        "domain": "code_diagram",
                        "modality": "text",
                    }
                )

            elif domain == "voiceover_transcript":
                tmpl_q, tmpl_p = _VOICEOVER_TEMPLATES[rng.integers(len(_VOICEOVER_TEMPLATES))]
                topic = _TOPICS[rng.integers(len(_TOPICS))]
                query = tmpl_q.format(
                    topic=topic,
                    tool=topic,
                    feature=topic,
                    concept=topic,
                )
                positive = tmpl_p.format(
                    topic=topic,
                    tool=topic,
                    config="settings.yaml",
                    feature=topic,
                    scenario="edge cases",
                    concept=topic,
                    points="architecture and implementation",
                )
                pairs.append(
                    {
                        "query": query,
                        "positive": positive,
                        "domain": "voiceover_transcript",
                        "modality": "audio",
                    }
                )

            elif domain == "paraphrase_query":
                tmpl_q, tmpl_p = _PARAPHRASE_PAIRS[rng.integers(len(_PARAPHRASE_PAIRS))]
                topic_idx = rng.integers(len(_TOPICS))
                concept = _TOPICS[topic_idx]
                alt_idx = (topic_idx + 1) % len(_TOPICS)
                alternative = _TOPICS[alt_idx]
                query = tmpl_q.format(
                    concept=concept,
                    issue=concept,
                    alternative=alternative,
                )
                positive = tmpl_p.format(
                    concept=concept,
                    mechanism="systematic processing",
                    issue=concept,
                    alternative=alternative,
                )
                pairs.append(
                    {
                        "query": query,
                        "positive": positive,
                        "domain": "paraphrase",
                        "modality": "text",
                    }
                )

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
                pairs.append(
                    {
                        "query": query,
                        "positive": positive,
                        "domain": f"code_{lang.lower()}",
                        "modality": "text",
                    }
                )

    # Fill remaining pairs with random domain selection
    remaining = num_pairs - len(pairs)
    for _ in range(remaining):
        domain = domains[rng.integers(len(domains))]
        topic = _TOPICS[rng.integers(len(_TOPICS))]
        pairs.append(
            {
                "query": f"Explain {topic}",
                "positive": f"An overview of {topic} and its applications.",
                "domain": domain,
                "modality": "text",
            }
        )

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
                                pairs.append(
                                    {
                                        "query": item["query"],
                                        "positive": item["positive"],
                                        "domain": domain,
                                        "modality": "text",
                                    }
                                )
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
                        logger.warning(f"Skipping malformed JSON at {jsonl_file}:{line_num}")
                        continue

                    # Validate minimum required fields
                    query = record.get("query", record.get("question", ""))
                    positive = record.get(
                        "positive", record.get("answer", record.get("passage", ""))
                    )

                    if not query or not positive:
                        continue

                    pairs.append(
                        {
                            "query": query,
                            "positive": positive,
                            "negatives": record.get("negatives", []),
                            "domain": record.get("domain", f"custom_{jsonl_file.stem}"),
                            "modality": record.get("modality", "text"),
                        }
                    )
                    total_loaded += 1

        except OSError as e:
            logger.warning(f"Failed to read {jsonl_file}: {e}")

        if max_samples and total_loaded >= max_samples:
            break

    logger.info(f"Loaded {len(pairs)} custom pairs from {len(jsonl_files)} files in {custom_path}")
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
    import time as _time

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / filename

    total = len(pairs)
    _log_interval = max(total // 10, 1)
    _t0 = _time.monotonic()

    with open(output_file, "w", encoding="utf-8") as f:
        for i, pair in enumerate(pairs):
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

            if (i + 1) % _log_interval == 0:
                _elapsed = _time.monotonic() - _t0
                _rate = (i + 1) / _elapsed if _elapsed > 0 else 0
                _eta = (total - i - 1) / _rate if _rate > 0 else 0
                logger.info(
                    f"Saving {filename}: {i + 1}/{total} "
                    f"({100 * (i + 1) / total:.0f}%) — "
                    f"{_rate:.0f} records/s, ETA={_eta:.0f}s"
                )

    logger.info(f"Saved {len(pairs)} pairs to {output_file}")
    return output_file


# ───────────────────────────────────────────────────────────────────
# Stage 3: Download multimodal media and produce local-path JSONL
# ───────────────────────────────────────────────────────────────────


def _download_image(url: str, dest: Path, timeout: int = 10) -> bool:
    """Download a single image from URL.

    Args:
        url: Image URL.
        dest: Destination file path.
        timeout: HTTP timeout in seconds.

    Returns:
        True if download succeeded and file is a valid image.
    """
    import requests
    from PIL import Image

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "image" not in content_type and "octet-stream" not in content_type:
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Validate image is readable
        img = Image.open(dest)
        img.verify()
        return True
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


def _download_video(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a single video from URL.

    Args:
        url: Video URL (direct link, e.g. from WebVid).
        dest: Destination file path.
        timeout: HTTP timeout in seconds.

    Returns:
        True if download succeeded and file is non-empty.
    """
    import requests

    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()

        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                size += len(chunk)

        # Basic sanity: video should be at least 1 KB
        if size < 1024:
            dest.unlink()
            return False
        return True
    except Exception:
        if dest.exists():
            dest.unlink()
        return False


def _download_audio_from_youtube(video_id: str, dest: Path, timeout: int = 60) -> bool:
    """Download audio from a YouTube video using yt-dlp.

    AudioSet records reference YouTube video IDs. We extract audio-only
    in WAV format (16 kHz mono) for Whisper-compatible processing.

    Args:
        video_id: YouTube video ID.
        dest: Destination .wav file path.
        timeout: Download timeout in seconds.

    Returns:
        True if download succeeded.
    """
    import subprocess

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # yt-dlp: extract audio as wav, 16kHz mono, max 10s
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000 -t 10",
            "--output", str(dest.with_suffix(".%(ext)s")),
            "--socket-timeout", str(timeout),
            "--quiet",
            f"https://www.youtube.com/watch?v={video_id}",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 30)
        # yt-dlp names the file with the correct extension
        wav_path = dest.with_suffix(".wav")
        if wav_path.exists() and wav_path.stat().st_size > 1024:
            if wav_path != dest:
                wav_path.rename(dest)
            return True
        return False
    except Exception:
        for suffix in (".wav", ".webm", ".m4a", ".mp3"):
            p = dest.with_suffix(suffix)
            if p.exists():
                p.unlink()
        return False


def build_stage3_multimodal(
    source_dir: str,
    output_dir: str,
    max_images: int = 100_000,
    max_audio: int = 20_000,
    max_video: int = 10_000,
    download_workers: int = 16,
    timeout: int = 10,
) -> None:
    """Build Stage 3 multimodal dataset by downloading media.

    Reads ``multimodal_pairs.jsonl`` from *source_dir* (produced by
    ``--stage 2``), downloads images from LAION URLs, videos from
    WebVid URLs, and audio from AudioSet YouTube IDs, then writes
    JSONL files with **local file paths** ready for
    ``train_multimodal.py``.

    Also copies text pairs from *source_dir* for the text branch of
    Stage 3 training.

    Args:
        source_dir: Stage 2 data directory containing multimodal_pairs.jsonl.
        output_dir: Output directory for Stage 3 data.
        max_images: Maximum number of images to download.
        max_audio: Maximum audio samples to download from AudioSet/YouTube.
        max_video: Maximum video samples to download from WebVid.
        download_workers: Number of parallel download threads.
        timeout: HTTP timeout per download in seconds.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    source = Path(source_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mm_file = source / "multimodal_pairs.jsonl"
    if not mm_file.exists():
        logger.error(f"multimodal_pairs.jsonl not found in {source_dir}")
        raise FileNotFoundError(mm_file)

    # ── Parse existing multimodal pairs ──
    image_records = []
    audio_records = []
    video_records = []
    code_records = []

    logger.info(f"Reading multimodal pairs from {mm_file}...")
    with open(mm_file) as f:
        for line in f:
            record = json.loads(line.strip())
            modality = record.get("modality", "text")
            if modality == "image" and record.get("image_url"):
                image_records.append(record)
            elif modality == "audio":
                audio_records.append(record)
            elif modality == "video":
                video_records.append(record)
            else:
                code_records.append(record)

    logger.info(
        f"Found: {len(image_records)} image, {len(audio_records)} audio, "
        f"{len(video_records)} video, {len(code_records)} code/text"
    )

    # ── Download images ──
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    image_records = image_records[:max_images]
    image_text_pairs = []
    failed = 0

    logger.info(f"Downloading {len(image_records)} images with {download_workers} workers...")

    def _download_one(idx_record):
        idx, record = idx_record
        url = record["image_url"]
        ext = Path(url).suffix.split("?")[0]
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
            ext = ".jpg"
        filename = f"{idx:08d}{ext}"
        dest = image_dir / filename
        ok = _download_image(url, dest, timeout=timeout)
        if ok:
            return {
                "image_path": filename,
                "caption": record["query"],
                "domain": record.get("domain", "image_text"),
            }
        return None

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {
            pool.submit(_download_one, (i, rec)): i
            for i, rec in enumerate(image_records)
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                image_text_pairs.append(result)
            else:
                failed += 1

            if (i + 1) % 5000 == 0 or i + 1 == len(futures):
                logger.info(
                    f"Image download progress: {i + 1}/{len(image_records)} "
                    f"({len(image_text_pairs)} ok, {failed} failed)"
                )

    # Save image-text JSONL
    image_jsonl = output / "image_text_pairs.jsonl"
    with open(image_jsonl, "w") as f:
        for pair in image_text_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(image_text_pairs)} image-text pairs to {image_jsonl}")

    # ── Download videos (WebVid direct URLs) ──
    video_text_pairs = []
    video_failed = 0
    if max_video > 0 and video_records:
        vid_dir = output / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)
        video_records = video_records[:max_video]

        logger.info(
            f"Downloading {len(video_records)} videos with {download_workers} workers..."
        )

        def _download_one_video(idx_record):
            idx, record = idx_record
            url = record.get("video_url", "")
            if not url:
                return None
            ext = Path(url).suffix.split("?")[0]
            if ext not in (".mp4", ".webm", ".avi", ".mkv", ".mov"):
                ext = ".mp4"
            filename = f"{idx:08d}{ext}"
            dest = vid_dir / filename
            ok = _download_video(url, dest, timeout=30)
            if ok:
                return {
                    "video_path": filename,
                    "caption": record["query"],
                    "domain": record.get("domain", "video_text"),
                }
            return None

        with ThreadPoolExecutor(max_workers=download_workers) as pool:
            futures = {
                pool.submit(_download_one_video, (i, rec)): i
                for i, rec in enumerate(video_records)
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    video_text_pairs.append(result)
                else:
                    video_failed += 1
                if (i + 1) % 2000 == 0 or i + 1 == len(futures):
                    logger.info(
                        f"Video download progress: {i + 1}/{len(video_records)} "
                        f"({len(video_text_pairs)} ok, {video_failed} failed)"
                    )

        video_jsonl = output / "video_text_pairs.jsonl"
        with open(video_jsonl, "w") as f:
            for pair in video_text_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(video_text_pairs)} video-text pairs to {video_jsonl}")
    else:
        logger.info("Skipping video download (max_video=0 or no video records)")

    # ── Download audio (AudioSet → YouTube via yt-dlp) ──
    audio_text_pairs = []
    audio_failed = 0
    if max_audio > 0 and audio_records:
        aud_dir = output / "audio"
        aud_dir.mkdir(parents=True, exist_ok=True)
        audio_records = audio_records[:max_audio]

        # Check yt-dlp is available
        import shutil as _shutil

        if _shutil.which("yt-dlp") is None:
            logger.warning(
                "yt-dlp not found — installing via pip for AudioSet download..."
            )
            import subprocess

            subprocess.run(
                [sys.executable, "-m", "pip", "install", "yt-dlp"],
                check=True,
                capture_output=True,
            )

        logger.info(
            f"Downloading {len(audio_records)} audio clips from YouTube "
            f"(AudioSet) with {max(download_workers // 2, 1)} workers..."
        )

        def _download_one_audio(idx_record):
            idx, record = idx_record
            video_id = record.get("audio_id", "")
            if not video_id:
                return None
            filename = f"{idx:08d}.wav"
            dest = aud_dir / filename
            ok = _download_audio_from_youtube(video_id, dest, timeout=60)
            if ok:
                return {
                    "audio_path": filename,
                    "caption": record["query"],
                    "domain": record.get("domain", "audio_text"),
                }
            return None

        # Fewer workers for yt-dlp (heavier, rate-limited)
        audio_workers = max(download_workers // 2, 1)
        with ThreadPoolExecutor(max_workers=audio_workers) as pool:
            futures = {
                pool.submit(_download_one_audio, (i, rec)): i
                for i, rec in enumerate(audio_records)
            }
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    audio_text_pairs.append(result)
                else:
                    audio_failed += 1
                if (i + 1) % 1000 == 0 or i + 1 == len(futures):
                    logger.info(
                        f"Audio download progress: {i + 1}/{len(audio_records)} "
                        f"({len(audio_text_pairs)} ok, {audio_failed} failed)"
                    )

        audio_jsonl = output / "audio_text_pairs.jsonl"
        with open(audio_jsonl, "w") as f:
            for pair in audio_text_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(audio_text_pairs)} audio-text pairs to {audio_jsonl}")
    else:
        logger.info("Skipping audio download (max_audio=0 or no audio records)")

    # ── Copy text pairs from Stage 2 (for text branch of training) ──
    text_src = source / "train.jsonl"
    text_dst = output / "text_pairs.jsonl"
    text_count = 0
    if text_src.exists():
        import shutil

        shutil.copy2(text_src, text_dst)
        with open(text_dst) as f:
            text_count = sum(1 for _ in f)
        logger.info(f"Copied {text_count} text pairs from {text_src}")
    else:
        logger.warning(f"No text pairs found at {text_src}")

    # ── Code pairs (already text-only, just save) ──
    if code_records:
        code_jsonl = output / "code_pairs.jsonl"
        with open(code_jsonl, "w") as f:
            for rec in code_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(code_records)} code pairs to {code_jsonl}")

    # ── Summary ──
    stats = {
        "stage": 3,
        "image_text_pairs": len(image_text_pairs),
        "image_download_failed": failed,
        "video_text_pairs": len(video_text_pairs),
        "video_download_failed": video_failed,
        "audio_text_pairs": len(audio_text_pairs),
        "audio_download_failed": audio_failed,
        "code_pairs": len(code_records),
        "text_pairs": text_count,
    }
    stats_file = output / "dataset_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Stage 3 data build complete: {stats}")


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
        all_records.append(
            {
                "query": pair.query,
                "positive": pair.positive,
                "negatives": getattr(pair, "negatives", []),
                "domain": getattr(pair, "domain", "retrieval"),
                "modality": "text",
            }
        )

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

    # Domain-balanced upsampling to target size
    if target_size and len(all_records) < target_size:
        import time as _time

        ratio = target_size / len(all_records)
        logger.info(
            f"Upsampling from {len(all_records)} to {target_size} "
            f"({ratio:.1f}x) using domain-balanced sampling"
        )
        if ratio > 10:
            logger.warning(
                f"Upsample ratio {ratio:.1f}x is very high — consider "
                f"adding more unique data sources or synthetic pairs to "
                f"reduce overfitting risk."
            )
        rng = np.random.default_rng(42)

        # Group records by domain for balanced sampling
        logger.info("Grouping records by domain...")
        domain_indices: dict[str, list[int]] = {}
        for i, rec in enumerate(all_records):
            dom = rec.get("domain", "unknown")
            domain_indices.setdefault(dom, []).append(i)

        logger.info(
            f"Found {len(domain_indices)} domains: "
            f"{', '.join(f'{k}({len(v)})' for k, v in sorted(domain_indices.items(), key=lambda x: -len(x[1])))}"
        )

        extra_needed = target_size - len(all_records)
        n_domains = len(domain_indices)

        # Proportional allocation: each domain gets a share proportional
        # to its size, but we add a floor so tiny domains still get
        # representation, and cap at ratio * original_size.
        total_records = sum(len(v) for v in domain_indices.values())
        max_repeat = max(int(ratio) + 2, 10)  # allow generous repetition

        extra_records: list[dict] = []
        _t0 = _time.monotonic()
        allocated_total = 0
        domain_quotas: dict[str, int] = {}
        for dom, indices in domain_indices.items():
            # Proportional share + small floor for tiny domains
            prop_share = int(extra_needed * len(indices) / total_records)
            floor_share = max(len(indices), 1000)
            quota = max(prop_share, floor_share)
            # Cap at max_repeat * original size
            quota = min(quota, len(indices) * max_repeat)
            domain_quotas[dom] = quota
            allocated_total += quota

        # Scale quotas if we over-allocated
        if allocated_total > extra_needed:
            scale = extra_needed / allocated_total
            domain_quotas = {d: int(q * scale) for d, q in domain_quotas.items()}

        _domain_i = 0
        for dom, indices in domain_indices.items():
            _domain_i += 1
            domain_quota = domain_quotas[dom]
            if domain_quota == 0:
                continue
            sampled = rng.choice(indices, size=domain_quota, replace=True)
            extra_records.extend([all_records[i] for i in sampled])
            if _domain_i % 5 == 0 or _domain_i == n_domains:
                _elapsed = _time.monotonic() - _t0
                logger.info(
                    f"Upsampling progress: domain {_domain_i}/{n_domains} "
                    f"({dom}) — {len(extra_records)}/{extra_needed} extra records "
                    f"generated ({100 * len(extra_records) / extra_needed:.1f}%) — "
                    f"{_elapsed:.1f}s elapsed"
                )

        # Fill the remainder from underrepresented domains (vectorised)
        still_needed = extra_needed - len(extra_records)
        if still_needed > 0:
            logger.info(
                f"Filling {still_needed} remaining records from "
                f"underrepresented domains (vectorised)..."
            )
            # Weight towards smaller domains to improve balance
            domain_names = list(domain_indices.keys())
            domain_weights = np.array([1.0 / max(len(domain_indices[d]), 1) for d in domain_names])
            domain_weights /= domain_weights.sum()

            # Vectorised: draw all domain choices at once, then draw
            # record indices per domain in bulk — avoids 44M Python iterations
            _t_fill = _time.monotonic()
            chosen_domains = rng.choice(len(domain_names), size=still_needed, p=domain_weights)
            domain_counts_fill = np.bincount(chosen_domains, minlength=len(domain_names))

            remainder_indices: list[int] = []
            for _di, (dom, count) in enumerate(zip(domain_names, domain_counts_fill)):
                if count == 0:
                    continue
                idxs = np.array(domain_indices[dom])
                sampled = rng.choice(idxs, size=int(count), replace=True)
                remainder_indices.extend(sampled.tolist())
                if (_di + 1) % 5 == 0 or _di + 1 == len(domain_names):
                    logger.info(
                        f"Remainder fill: domain {_di + 1}/{len(domain_names)} "
                        f"({dom}) — {len(remainder_indices)}/{still_needed} "
                        f"({100 * len(remainder_indices) / still_needed:.1f}%)"
                    )

            extra_records.extend([all_records[i] for i in remainder_indices])
            logger.info(f"Remainder fill complete in {_time.monotonic() - _t_fill:.1f}s")

        all_records.extend(extra_records)
        logger.info(f"Upsampling complete: {len(all_records)} total records")

    # Subsample to target size if needed
    if target_size and len(all_records) > target_size:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(all_records), size=target_size, replace=False)
        all_records = [all_records[i] for i in sorted(indices)]
        logger.info(f"Subsampled to {target_size} records")

    # Shuffle and save combined dataset
    import time as _time

    logger.info(f"Shuffling {len(all_records)} records...")
    _t0 = _time.monotonic()
    rng = np.random.default_rng(42)
    rng.shuffle(all_records)
    logger.info(f"Shuffle complete in {_time.monotonic() - _t0:.1f}s")
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
        choices=[1, 2, 3],
        required=True,
        help="Training stage (1=retrieval, 2=generalist, 3=multimodal)",
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
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Source data directory for Stage 3 (contains multimodal_pairs.jsonl "
        "from Stage 2 build). Required when --stage 3.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=100_000,
        help="Maximum number of images to download for Stage 3.",
    )
    parser.add_argument(
        "--max-video",
        type=int,
        default=10_000,
        help="Maximum number of videos to download for Stage 3 (WebVid).",
    )
    parser.add_argument(
        "--max-audio",
        type=int,
        default=20_000,
        help="Maximum number of audio clips to download for Stage 3 (AudioSet/YouTube).",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Number of parallel download threads for Stage 3 media.",
    )

    args = parser.parse_args()

    if args.stage == 3:
        source = args.source_dir or "data/stage2_55M"
        build_stage3_multimodal(
            source_dir=source,
            output_dir=str(args.output_dir),
            max_images=args.max_images,
            max_video=args.max_video,
            max_audio=args.max_audio,
            download_workers=args.download_workers,
        )
    else:
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
