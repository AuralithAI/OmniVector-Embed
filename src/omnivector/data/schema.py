"""
Data schema definitions using Pydantic v2.

Defines the canonical format for all training data:
- EmbeddingPair: single query-positive-negatives tuple
- Supports domain tagging and instruction prefixes
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class EmbeddingPairPydantic(BaseModel):
    """Pydantic v2 model for embedding pairs."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)

    query: str = Field(..., description="Query text or code snippet")
    positive: str = Field(..., description="Positive passage/document")
    negatives: list[str] = Field(default_factory=list, description="Hard negative passages")
    query_instruction: Optional[str] = Field(
        default=None, description="Instruction prefix for queries (e.g., 'Search query')"
    )
    domain: str = Field(
        default="general", description="Domain tag (retrieval, clustering, sts, etc.)"
    )


@dataclass
class EmbeddingPair:
    """
    Dataclass representation of an embedding training pair.

    Attributes:
        query: Query text
        positive: Positive passage
        negatives: Hard negative passages
        query_instruction: Instruction prefix for query
        domain: Domain tag
    """

    query: str
    positive: str
    negatives: list[str] = field(default_factory=list)
    query_instruction: Optional[str] = None
    domain: str = "general"

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingPair":
        """Create from dictionary."""
        return cls(
            query=data.get("query", ""),
            positive=data.get("positive", ""),
            negatives=data.get("negatives", []),
            query_instruction=data.get("query_instruction"),
            domain=data.get("domain", "general"),
        )

    @classmethod
    def from_pydantic(cls, pydantic_obj: EmbeddingPairPydantic) -> "EmbeddingPair":
        """Create from Pydantic model."""
        return cls(
            query=pydantic_obj.query,
            positive=pydantic_obj.positive,
            negatives=pydantic_obj.negatives,
            query_instruction=pydantic_obj.query_instruction,
            domain=pydantic_obj.domain,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "positive": self.positive,
            "negatives": self.negatives,
            "query_instruction": self.query_instruction,
            "domain": self.domain,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EmbeddingPair(query='{self.query[:50]}...', "
            f"n_negatives={len(self.negatives)}, domain='{self.domain}')"
        )


def validate_embedding_pair(data: dict) -> EmbeddingPair:
    """
    Validate and create EmbeddingPair from dictionary.

    Args:
        data: Dictionary with required keys: query, positive

    Returns:
        Validated EmbeddingPair

    Raises:
        ValueError: If required fields missing or invalid
    """
    required_keys = {"query", "positive"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - data.keys()
        raise ValueError(f"Missing required keys: {missing}")

    if not isinstance(data["query"], str) or not data["query"].strip():
        raise ValueError("query must be non-empty string")

    if not isinstance(data["positive"], str) or not data["positive"].strip():
        raise ValueError("positive must be non-empty string")

    return EmbeddingPair.from_dict(data)
