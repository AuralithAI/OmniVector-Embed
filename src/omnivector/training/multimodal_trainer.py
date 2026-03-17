"""Multimodal trainer extending OmniVectorTrainer for vision-text training.

Handles mixed-modality batches by routing image/video data through the
vision encoder while text goes through the backbone, then computing
both text-retrieval and cross-modal contrastive losses.
"""

import logging
from typing import Any, Optional

import torch

from omnivector.training.trainer import OmniVectorTrainer

logger = logging.getLogger(__name__)


class MultimodalTrainer(OmniVectorTrainer):
    """Trainer for joint text + vision embedding training.

    Extends OmniVectorTrainer with:
    - Image/video batch routing through vision encoder
    - Cross-modal contrastive loss alongside text MRL loss
    - Gradient accumulation across modalities
    - Vision encoder freeze/unfreeze scheduling

    Attributes:
        cross_modal_weight: Weight for cross-modal loss component.
        freeze_vision_steps: Number of steps to freeze vision encoder.
        multimodal_loss: Combined text + cross-modal loss module.
    """

    def __init__(
        self,
        *args,
        cross_modal_weight: float = 1.0,
        freeze_vision_steps: int = 0,
        multimodal_loss: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize multimodal trainer.

        Args:
            *args: Positional args for HF Trainer.
            cross_modal_weight: Weight for cross-modal contrastive loss.
            freeze_vision_steps: Steps to keep vision encoder frozen.
            multimodal_loss: Pre-configured MultimodalMRLLoss instance.
            **kwargs: Keyword args for HF Trainer.
        """
        super().__init__(*args, **kwargs)
        self.cross_modal_weight = cross_modal_weight
        self.freeze_vision_steps = freeze_vision_steps
        self.multimodal_loss_fn = multimodal_loss
        self._vision_frozen = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute combined text + cross-modal loss for mixed batches.

        Routes each modality through the appropriate encoder, then
        computes MRL InfoNCE for text pairs and symmetric contrastive
        loss for vision-text pairs.

        Args:
            model: OmniVectorModel instance.
            inputs: Batch dict from MultimodalCollator.
            return_outputs: Whether to return outputs with loss.

        Returns:
            Loss tensor, or (loss, outputs) if return_outputs=True.
        """
        # Handle vision encoder freeze schedule
        if self.freeze_vision_steps > 0:
            self._handle_vision_freeze(model)

        # Remove labels if HF Trainer injected them
        if "labels" in inputs:
            del inputs["labels"]

        # Check if this is a multimodal batch
        has_images = inputs.get("has_images", False)
        has_videos = inputs.get("has_videos", False)

        # Route text through backbone
        query_input_ids = inputs["query_input_ids"]
        query_attention_mask = inputs["query_attention_mask"]
        positive_input_ids = inputs["positive_input_ids"]
        positive_attention_mask = inputs["positive_attention_mask"]

        # Encode queries
        query_hidden = model.backbone(
            input_ids=query_input_ids,
            attention_mask=query_attention_mask,
        )
        query_embeddings = model.pooling(
            hidden_states=query_hidden,
            attention_mask=~query_attention_mask.bool(),
        )

        # Encode positives
        positive_hidden = model.backbone(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask,
        )
        positive_embeddings = model.pooling(
            hidden_states=positive_hidden,
            attention_mask=~positive_attention_mask.bool(),
        )

        # Encode hard negatives if present
        negative_embeddings = None
        neg_ids = inputs.get("negative_input_ids")
        neg_mask = inputs.get("negative_attention_mask")

        if neg_ids is not None and neg_ids.size(1) > 0:
            batch_size, num_negs, seq_len = neg_ids.shape
            neg_ids_flat = neg_ids.reshape(-1, seq_len)
            neg_mask_flat = neg_mask.reshape(-1, seq_len)

            neg_hidden = model.backbone(
                input_ids=neg_ids_flat,
                attention_mask=neg_mask_flat,
            )
            neg_emb = model.pooling(
                hidden_states=neg_hidden,
                attention_mask=~neg_mask_flat.bool(),
            )
            negative_embeddings = neg_emb.reshape(batch_size, num_negs, -1)

        # Vision embeddings
        visual_embeddings = None
        visual_mask = None
        text_for_visual = None

        if has_images and model.vision_encoder is not None:
            images = inputs.get("images")
            image_mask = inputs.get("image_mask")

            if images is not None:
                visual_embeddings = model.vision_encoder(images)
                visual_mask = image_mask
                text_for_visual = query_embeddings

        if has_videos and model.video_encoder is not None:
            videos = inputs.get("videos")
            video_mask = inputs.get("video_mask")

            if videos is not None:
                video_emb = model.video_encoder(videos)

                if visual_embeddings is not None:
                    # Combine image and video embeddings
                    visual_embeddings = torch.where(
                        video_mask.unsqueeze(-1),
                        video_emb,
                        visual_embeddings,
                    )
                    visual_mask = visual_mask | video_mask
                else:
                    visual_embeddings = video_emb
                    visual_mask = video_mask
                    text_for_visual = query_embeddings

        # Compute loss
        if self.multimodal_loss_fn is not None:
            result = self.multimodal_loss_fn(
                query_embeddings=query_embeddings,
                positive_embeddings=positive_embeddings,
                negative_embeddings=negative_embeddings,
                visual_embeddings=visual_embeddings,
                text_for_visual=text_for_visual,
                visual_mask=visual_mask,
            )
        else:
            # Fallback to parent text-only behavior
            outputs = model(
                input_ids=query_input_ids,
                attention_mask=query_attention_mask,
            )
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            return (loss, outputs) if return_outputs else loss

        loss = result["loss"]
        outputs = {
            "query_embeddings": query_embeddings,
            "positive_embeddings": positive_embeddings,
            "loss_components": {k: v for k, v in result.items() if k != "loss"},
        }

        return (loss, outputs) if return_outputs else loss

    def _handle_vision_freeze(self, model):
        """Freeze/unfreeze vision encoder based on training step.

        Keeps vision encoder frozen for the first N steps to let text
        backbone warm up before cross-modal alignment begins.

        Args:
            model: OmniVectorModel instance.
        """
        current_step = self.state.global_step

        if current_step < self.freeze_vision_steps and not self._vision_frozen:
            # Freeze vision encoder
            if model.vision_encoder is not None:
                for param in model.vision_encoder.parameters():
                    param.requires_grad = False
                self._vision_frozen = True
                logger.info(f"Step {current_step}: Vision encoder frozen")

        elif current_step >= self.freeze_vision_steps and self._vision_frozen:
            # Unfreeze vision encoder (only projection layer)
            if model.vision_encoder is not None:
                for param in model.vision_encoder.projection.parameters():
                    param.requires_grad = True
                self._vision_frozen = False
                logger.info(
                    f"Step {current_step}: Vision encoder projection unfrozen "
                    f"(base vision model stays frozen)"
                )

    def log_modality_stats(self, batch: dict):
        """Log modality distribution in current batch.

        Args:
            batch: Collated batch dict.
        """
        modalities = batch.get("modalities", [])
        if not modalities:
            return

        counts = {}
        for m in modalities:
            counts[m] = counts.get(m, 0) + 1

        if self.state.global_step % self.args.logging_steps == 0:
            logger.info(f"Step {self.state.global_step} batch modalities: {counts}")
