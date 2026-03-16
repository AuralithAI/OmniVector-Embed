"""
OmniVector-Embed: Production-grade multimodal embedding model with ONNX export.

Key features:
- Bidirectional attention for better semantic understanding
- Matryoshka Representation Learning (MRL) for dynamic dimensionality
- ONNX export with int8 quantization
- Multimodal (text, code, image, video) in unified 4096-dim space
"""

__version__ = "0.1.0"
__author__ = "OmniVector Contributors"
__license__ = "MIT"

from omnivector.model.backbone import MistralEmbeddingBackbone
from omnivector.model.omnivector_model import OmniVectorModel

__all__ = [
    "OmniVectorModel",
    "MistralEmbeddingBackbone",
    "__version__",
]
