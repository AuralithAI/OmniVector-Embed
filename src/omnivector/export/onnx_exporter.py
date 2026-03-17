"""ONNX export utilities for OmniVector model.

Provides wrapper and exporter for converting PyTorch model to ONNX format
with opset 18, dynamic shapes, and LoRA merge support.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OmniVectorONNXWrapper(nn.Module):
    """Wrapper that exposes a clean forward signature for ONNX export.

    Strips away tokenizer logic, multimodal routing, and other
    non-exportable components. Accepts raw token IDs and attention mask,
    returns L2-normalized embeddings.
    """

    def __init__(self, backbone: nn.Module, pooling: nn.Module, output_dim: int = 4096):
        """Initialize ONNX wrapper.

        Args:
            backbone: Text encoder backbone (bidirectional Mistral).
            pooling: Latent attention pooling module.
            output_dim: Output embedding dimension.
        """
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.output_dim = output_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for ONNX export.

        Args:
            input_ids: Token IDs [batch_size, seq_length].
            attention_mask: Attention mask [batch_size, seq_length].

        Returns:
            L2-normalized embeddings [batch_size, output_dim].
        """
        hidden_states = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        key_padding_mask = ~attention_mask.bool()
        embeddings = self.pooling(hidden_states=hidden_states, attention_mask=key_padding_mask)
        embeddings = embeddings[:, : self.output_dim]
        norm = torch.clamp(torch.norm(embeddings, p=2, dim=-1, keepdim=True), min=1e-12)
        embeddings = embeddings / norm
        return embeddings


class ONNXExporter:
    """Handles ONNX export with LoRA merge, dynamic shapes, and validation."""

    def __init__(
        self,
        model,
        output_dir: str = "./onnx_export",
        opset_version: int = 18,
        output_dim: int = 4096,
    ):
        """Initialize exporter.

        Args:
            model: OmniVectorModel instance.
            output_dir: Directory for exported ONNX files.
            opset_version: ONNX opset version (18+ for native LayerNorm).
            output_dim: Output embedding dimension.
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.opset_version = opset_version
        self.output_dim = output_dim

    def merge_lora(self):
        """Merge LoRA adapters into base weights before export.

        Collapses W + B*A into a single weight matrix, eliminating
        conditional branching that would break ONNX graph.
        """
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "merge_lora"):
            self.model.backbone.merge_lora()
            logger.info("LoRA adapters merged into base weights")
        else:
            logger.info("No LoRA adapters to merge")

    def export(
        self,
        output_path: Optional[str] = None,
        merge_lora: bool = True,
        max_seq_length: int = 512,
    ) -> str:
        """Export model to ONNX format.

        Args:
            output_path: Custom output path (default: output_dir/omnivector_embed.onnx).
            merge_lora: Whether to merge LoRA before export.
            max_seq_length: Maximum sequence length for dummy input.

        Returns:
            Path to exported ONNX file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if output_path is None:
            output_path = str(self.output_dir / "omnivector_embed.onnx")

        if merge_lora:
            self.merge_lora()

        wrapper = OmniVectorONNXWrapper(
            backbone=self.model.backbone,
            pooling=self.model.pooling,
            output_dim=self.output_dim,
        )
        wrapper.eval()

        device = next(self.model.parameters()).device
        dummy_input_ids = torch.randint(0, 1000, (1, 32), device=device)
        dummy_attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "embedding": {0: "batch_size"},
        }

        logger.info(f"Exporting ONNX model to {output_path} (opset {self.opset_version})")

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                (dummy_input_ids, dummy_attention_mask),
                output_path,
                opset_version=self.opset_version,
                input_names=["input_ids", "attention_mask"],
                output_names=["embedding"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
            )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"ONNX export complete: {output_path} ({file_size_mb:.1f} MB)")

        return output_path

    def optimize(self, onnx_path: Optional[str] = None, output_path: Optional[str] = None) -> str:
        """Apply ORT transformer-specific graph optimizations.

        Fuses LayerNorm, Attention, GELU, and SkipLayerNorm patterns
        for better inference performance.

        Args:
            onnx_path: Input ONNX path. Defaults to last exported path.
            output_path: Output path. Defaults to <output_dir>/omnivector_embed_opt.onnx.

        Returns:
            Path to optimized ONNX model.
        """
        if onnx_path is None:
            onnx_path = str(self.output_dir / "omnivector_embed.onnx")
        if output_path is None:
            output_path = str(self.output_dir / "omnivector_embed_opt.onnx")

        try:
            from onnxruntime.transformers import optimizer

            opt_model = optimizer.optimize_model(
                onnx_path,
                model_type="bert",
                num_heads=32,
                hidden_size=4096,
            )
            opt_model.save_model_to_file(output_path)
            logger.info(f"ORT transformer optimization complete: {output_path}")
        except ImportError:
            logger.warning(
                "onnxruntime.transformers not available, falling back to session-level optimization"
            )
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.optimized_model_filepath = output_path
            ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
            logger.info(f"Session-level optimization complete: {output_path}")

        opt_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Optimized model size: {opt_mb:.1f} MB")
        return output_path

    def validate_onnx(self, onnx_path: str) -> bool:
        """Run basic ONNX model validation.

        Args:
            onnx_path: Path to ONNX model file.

        Returns:
            True if model passes validation.
        """
        import onnx

        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info(f"ONNX model validation passed: {onnx_path}")
        return True

    def export_full_pipeline(
        self,
        merge_lora: bool = True,
        optimize: bool = True,
        quantize: bool = True,
        validate: bool = True,
        num_validation_samples: int = 50,
        validation_threshold: float = 0.99,
    ) -> dict:
        """Run complete export pipeline: export -> optimize -> quantize -> validate.

        Args:
            merge_lora: Whether to merge LoRA before export.
            optimize: Run ORT graph optimization.
            quantize: Run int8 dynamic quantization.
            validate: Run cosine parity validation.
            num_validation_samples: Number of samples for validation.
            validation_threshold: Minimum cosine similarity.

        Returns:
            Dict with paths to all generated artifacts and validation results.
        """
        result = {}

        # Export
        fp32_path = self.export(merge_lora=merge_lora)
        self.validate_onnx(fp32_path)
        result["fp32_path"] = fp32_path

        current_path = fp32_path

        # Optimize
        if optimize:
            opt_path = self.optimize(current_path)
            result["optimized_path"] = opt_path
            current_path = opt_path

        # Quantize
        if quantize:
            from omnivector.export.onnx_quantizer import ONNXQuantizer

            quantizer = ONNXQuantizer(current_path, output_dir=str(self.output_dir))
            int8_path = quantizer.quantize()
            result["int8_path"] = int8_path

        # Validate
        if validate:
            from omnivector.export.onnx_validator import ONNXValidator

            validator = ONNXValidator(fp32_path)
            parity = validator.validate_parity(
                pytorch_model=self.model,
                num_samples=num_validation_samples,
                threshold=validation_threshold,
                output_dim=self.output_dim,
            )
            result["validation"] = parity

        return result
