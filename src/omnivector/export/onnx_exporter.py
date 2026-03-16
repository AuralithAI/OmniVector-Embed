"""
ONNX export utilities (placeholder).

Will implement:
- OmniVectorONNXWrapper: Wraps model for ONNX export
- torch.onnx.export with opset 17
- Dynamic shape handling
"""

import logging

logger = logging.getLogger(__name__)

# ONNX export will be implemented in Week 4
# Key requirements:
# - LoRA merge before export
# - Dynamic batch size and sequence length
# - Opset 17 for LayerNorm native op support
# - No SDPA/custom ops (use eager attention)
