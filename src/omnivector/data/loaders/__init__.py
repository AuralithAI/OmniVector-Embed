from omnivector.data.loaders.base import BaseDataLoader
from omnivector.data.loaders.beir import BEIRLoader
from omnivector.data.loaders.hotpotqa import HotpotQALoader
from omnivector.data.loaders.msmarco import MSMARCOLoader

__all__ = [
    "BaseDataLoader",
    "MSMARCOLoader",
    "HotpotQALoader",
    "BEIRLoader",
]
