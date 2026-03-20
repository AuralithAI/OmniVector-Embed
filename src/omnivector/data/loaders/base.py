"""Dataset loaders for retrieval benchmarks via HuggingFace datasets."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from omnivector.data.schema import EmbeddingPair

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, name: str):
        """Initialize dataset loader.
        
        Args:
            name: Human-readable identifier for the dataset.
        """
        self.name = name

    @abstractmethod
    def load(self) -> list[EmbeddingPair]:
        """Load dataset and convert to EmbeddingPair format.
        
        Returns:
            List of EmbeddingPair objects ready for training.
        """
        pass

    @abstractmethod
    def load_corpus(self) -> dict[int, str]:
        """Load corpus for hard negative mining.
        
        Returns:
            Dictionary mapping corpus ID to text.
        """
        pass


class MSMARCOLoader(BaseDataLoader):
    """Loader for MS MARCO dataset from HuggingFace datasets."""

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        use_instruction_prefix: bool = True,
    ):
        """Initialize MSMARCO loader.
        
        Args:
            split: Dataset split ('train', 'validation').
            max_samples: Limit to N samples (None = all).
            use_instruction_prefix: Add instruction prefix to queries.
        """
        super().__init__("msmarco")
        self.split = split
        self.max_samples = max_samples
        self.use_instruction_prefix = use_instruction_prefix

    def load(self) -> list[EmbeddingPair]:
        """Load MS MARCO training pairs."""
        from datasets import load_dataset

        dataset = load_dataset("ms_marco", "v2.1", split=self.split)

        if self.max_samples:
            dataset = dataset.take(self.max_samples)

        pairs = []
        for sample in dataset:
            query = sample["query"]
            if self.use_instruction_prefix:
                query = f"Instruct: Find most relevant passage.\nQuery: {query}"

            for passage_id in sample["passages"]["is_selected"]:
                if passage_id == 1:
                    passage = sample["passages"]["passage_text"][0]
                    pair = EmbeddingPair(
                        query=query,
                        positive=passage,
                        negatives=[],
                        domain="retrieval",
                    )
                    pairs.append(pair)
                    break

        logger.info(f"Loaded {len(pairs)} MS MARCO pairs from split '{self.split}'")
        return pairs

    def load_corpus(self) -> dict[int, str]:
        """Load MS MARCO corpus for negative mining.

        MS MARCO v2.1 does not have a dedicated ``corpus`` split.
        Passages are extracted from the ``train`` split's ``passages``
        field and de-duplicated.
        """
        from datasets import load_dataset

        dataset = load_dataset("ms_marco", "v2.1", split="train")
        corpus = {}
        seen: set[str] = set()
        idx = 0

        for sample in dataset:
            for passage_text in sample["passages"]["passage_text"]:
                text = passage_text.strip()
                if text and text not in seen:
                    corpus[idx] = text
                    seen.add(text)
                    idx += 1

        logger.info(f"Loaded {len(corpus)} MS MARCO corpus items")
        return corpus


class HotpotQALoader(BaseDataLoader):
    """Loader for HotpotQA dataset from HuggingFace datasets.
    
    Uses the canonical ``hotpotqa/hotpot_qa`` repository on the Hub with the
    ``fullwiki`` configuration.  The ``supporting_facts`` field is a dict
    with parallel ``title`` and ``sent_id`` lists.
    """

    def __init__(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        use_instruction_prefix: bool = True,
    ):
        """Initialize HotpotQA loader.
        
        Args:
            split: Dataset split ('train', 'validation').
            max_samples: Limit to N samples (None = all).
            use_instruction_prefix: Add instruction prefix to queries.
        """
        super().__init__("hotpotqa")
        self.split = split
        self.max_samples = max_samples
        self.use_instruction_prefix = use_instruction_prefix

    def load(self) -> list[EmbeddingPair]:
        """Load HotpotQA training pairs."""
        from datasets import load_dataset

        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=self.split)

        if self.max_samples:
            dataset = dataset.take(self.max_samples)

        pairs = []
        for sample in dataset:
            query = sample["question"]
            if self.use_instruction_prefix:
                query = f"Instruct: Find supporting facts.\nQuery: {query}"

            # supporting_facts is a dict with parallel lists:
            #   {"title": [...], "sent_id": [...]}
            sf = sample["supporting_facts"]
            supporting_facts_ids = set(zip(sf["title"], sf["sent_id"]))

            for doc_title, doc_sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
                for sent_idx, sent_text in enumerate(doc_sents):
                    if (doc_title, sent_idx) in supporting_facts_ids:
                        pair = EmbeddingPair(
                            query=query,
                            positive=sent_text,
                            negatives=[],
                            domain="retrieval",
                        )
                        pairs.append(pair)

        logger.info(f"Loaded {len(pairs)} HotpotQA pairs from split '{self.split}'")
        return pairs

    def load_corpus(self) -> dict[int, str]:
        """Load HotpotQA passage corpus."""
        from datasets import load_dataset

        dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
        corpus = {}
        idx = 0

        for sample in dataset:
            for doc_title, doc_sents in zip(sample["context"]["title"], sample["context"]["sentences"]):
                for sent_text in doc_sents:
                    corpus[idx] = sent_text
                    idx += 1

        logger.info(f"Loaded {len(corpus)} HotpotQA corpus items")
        return corpus


class BEIRLoader(BaseDataLoader):
    """Loader for BEIR benchmark datasets.

    Handles two HuggingFace repository formats:

    1. **Parquet repos** (e.g. ``BeIR/nfcorpus``) — have ``corpus`` and
       ``queries`` configs with matching split names.  Loaded via the
       standard ``datasets.load_dataset`` API.
    2. **Legacy-script repos** (e.g. ``BeIR/fiqa``, ``BeIR/scifact``,
       ``BeIR/arguana``) — contain ``corpus.jsonl.gz`` and
       ``queries.jsonl.gz`` but rely on a deprecated ``.py`` loading
       script.  We download the JSONL files directly with
       ``load_dataset("json", …)``.

    In both cases the loader creates (title → text) training pairs from
    the corpus, which is the standard approach for unsupervised BEIR
    pre-training.
    """

    # Benchmarks whose Hub repos already have auto-converted parquet
    _PARQUET_BENCHMARKS = {"nfcorpus"}

    # Benchmarks whose standard corpus repo is unavailable — fall back
    # to the ``BeIR/<name>-generated-queries`` variant which has
    # (id, title, text, query) columns.
    _GENERATED_QUERIES_BENCHMARKS = {"bioasq"}

    def __init__(
        self,
        benchmark: str = "nfcorpus",
        split: str = "train",
        max_samples: Optional[int] = None,
        use_instruction_prefix: bool = True,
    ):
        """Initialize BEIR loader.
        
        Args:
            benchmark: BEIR benchmark name (nfcorpus, scifact, arguana, etc).
            split: Ignored for BEIR (corpus split is used automatically).
            max_samples: Limit to N samples (None = all).
            use_instruction_prefix: Add instruction prefix to queries.
            
        Raises:
            ValueError: If benchmark name is invalid.
        """
        super().__init__(f"beir/{benchmark}")
        self.benchmark = benchmark
        self.split = split
        self.max_samples = max_samples
        self.use_instruction_prefix = use_instruction_prefix

        self._validate_benchmark()

    def _validate_benchmark(self):
        """Validate benchmark name."""
        valid = {
            "nfcorpus",
            "scifact",
            "arguana",
            "fiqa",
            "trec-covid",
            "dbpedia-entity",
            "bioasq",
            "fever",
            "climate-fever",
            "scidocs",
            "quora",
            "nq",
            "hotpotqa",
            "msmarco",
            "cqadupstack",
            "signal1m",
            "robust04",
            "trec-news",
        }
        if self.benchmark not in valid:
            raise ValueError(
                f"Invalid BEIR benchmark '{self.benchmark}'. "
                f"Must be one of: {valid}"
            )

    def _load_corpus_dataset(self):
        """Load the corpus split, handling parquet, legacy, and generated-queries repos."""
        from datasets import load_dataset

        dataset_name = f"BeIR/{self.benchmark}"

        # Generated-queries repos: single table with (query, text, title)
        if self.benchmark in self._GENERATED_QUERIES_BENCHMARKS:
            gq_name = f"BeIR/{self.benchmark}-generated-queries"
            logger.info(
                f"Using generated-queries variant: {gq_name}"
            )
            return load_dataset(gq_name, split="train")

        if self.benchmark in self._PARQUET_BENCHMARKS:
            # Parquet repos: config="corpus", split="corpus"
            return load_dataset(dataset_name, "corpus", split="corpus")

        # Legacy-script repos: load the jsonl.gz directly from the Hub
        url = (
            f"https://huggingface.co/datasets/{dataset_name}"
            f"/resolve/main/corpus.jsonl.gz"
        )
        return load_dataset("json", data_files=url, split="train")

    def _load_queries_dataset(self):
        """Load the queries split, handling both parquet and legacy repos."""
        from datasets import load_dataset

        dataset_name = f"BeIR/{self.benchmark}"

        if self.benchmark in self._PARQUET_BENCHMARKS:
            return load_dataset(dataset_name, "queries", split="queries")

        url = (
            f"https://huggingface.co/datasets/{dataset_name}"
            f"/resolve/main/queries.jsonl.gz"
        )
        return load_dataset("json", data_files=url, split="train")

    def load(self) -> list[EmbeddingPair]:
        """Load BEIR dataset pairs from corpus.

        For generated-queries variants the explicit ``query`` column is
        used directly.  For standard corpora, ``title → text`` is used
        when a title is available; otherwise the first sentence of the
        text is extracted as a pseudo-query.
        """
        dataset = self._load_corpus_dataset()

        if self.max_samples:
            dataset = dataset.take(self.max_samples)

        # Detect if dataset has an explicit query column (generated-queries)
        has_query_col = "query" in (
            dataset.column_names
            if hasattr(dataset, "column_names")
            else (dataset.features or {})
        )

        pairs = []
        for sample in dataset:
            passage = (sample.get("text", "") or "").strip()
            if not passage:
                continue

            # Prefer explicit query column (generated-queries datasets)
            explicit_query = (sample.get("query", "") or "").strip() if has_query_col else ""

            if explicit_query:
                query = explicit_query
            else:
                title = (sample.get("title", "") or "").strip()
                if title:
                    query = title
                else:
                    for sep in (". ", ".\n", "? ", "!\n"):
                        idx = passage.find(sep)
                        if idx != -1:
                            query = passage[: idx + 1].strip()
                            break
                    else:
                        query = passage[:128].strip()

                    if not query:
                        continue

            if self.use_instruction_prefix:
                query = f"Instruct: Find relevant document.\nQuery: {query}"

            pair = EmbeddingPair(
                query=query,
                positive=passage,
                negatives=[],
                domain="retrieval",
            )
            pairs.append(pair)

        logger.info(
            f"Loaded {len(pairs)} BEIR {self.benchmark} pairs"
        )
        return pairs

    def load_corpus(self) -> dict[int, str]:
        """Load BEIR corpus."""
        dataset = self._load_corpus_dataset()

        corpus = {}
        for idx, sample in enumerate(dataset):
            corpus[idx] = sample.get("text", "")

        logger.info(f"Loaded {len(corpus)} BEIR {self.benchmark} corpus items")
        return corpus


def get_loader(dataset_name: str, **kwargs) -> BaseDataLoader:
    """Factory function to get appropriate dataset loader.
    
    Args:
        dataset_name: Name of dataset (msmarco, hotpotqa, beir/nfcorpus, etc).
        **kwargs: Additional arguments passed to loader constructor.
    
    Returns:
        Instance of appropriate BaseDataLoader subclass.
    
    Raises:
        ValueError: If dataset name is not recognized.
    """
    if dataset_name == "msmarco":
        return MSMARCOLoader(**kwargs)
    elif dataset_name == "hotpotqa":
        return HotpotQALoader(**kwargs)
    elif dataset_name.startswith("beir/"):
        benchmark = dataset_name.split("/")[1]
        return BEIRLoader(benchmark=benchmark, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: msmarco, hotpotqa, beir/*"
        )
