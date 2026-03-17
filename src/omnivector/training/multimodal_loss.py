"""Cross-modal contrastive loss for vision-text alignment.

Implements CLIP-style contrastive learning between image/video embeddings
and text embeddings, with MRL support for multiple dimensions.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CrossModalContrastiveLoss(nn.Module):
    """CLIP-style symmetric contrastive loss for cross-modal alignment.

    Computes bidirectional InfoNCE: image-to-text and text-to-image,
    ensuring both modalities are aligned in the shared embedding space.
    Supports MRL by computing loss at multiple truncation dimensions.

    Attributes:
        mrl_dims: Matryoshka dimensions for multi-resolution loss.
        temperature: Learnable or fixed temperature for softmax scaling.
    """

    def __init__(
        self,
        mrl_dims: tuple[int, ...] = (512, 1024, 2048, 4096),
        initial_temperature: float = 0.07,
        learnable_temperature: bool = True,
    ):
        """Initialize cross-modal contrastive loss.

        Args:
            mrl_dims: Matryoshka dimensions.
            initial_temperature: Initial softmax temperature.
            learnable_temperature: Whether temperature is a learnable parameter.
        """
        super().__init__()
        self.mrl_dims = mrl_dims

        if learnable_temperature:
            # log(1/0.07) = 2.659
            self.log_temperature = nn.Parameter(
                torch.tensor(2.6593, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(2.6593, dtype=torch.float32),
            )

        # Fixed dimension weights matching MRL spec
        self.register_buffer(
            "dim_weights",
            torch.tensor([0.5, 0.75, 1.0, 1.0][: len(mrl_dims)]),
        )

    @property
    def temperature(self) -> torch.Tensor:
        """Current temperature value."""
        return torch.exp(-self.log_temperature)

    def forward(
        self,
        visual_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute symmetric cross-modal contrastive loss.

        Args:
            visual_embeddings: Image or video embeddings [batch_size, max_dim].
            text_embeddings: Text embeddings [batch_size, max_dim].
            visual_mask: Boolean mask indicating which samples have valid
                visual data [batch_size]. If None, all are assumed valid.

        Returns:
            Dict with 'loss' (total), per-dim losses, and temperature.
        """
        batch_size = visual_embeddings.size(0)

        if visual_mask is not None:
            visual_embeddings = visual_embeddings[visual_mask]
            text_embeddings = text_embeddings[visual_mask]
            batch_size = visual_embeddings.size(0)

        if batch_size == 0:
            return {"loss": torch.tensor(0.0, device=visual_embeddings.device)}

        total_loss = torch.tensor(0.0, device=visual_embeddings.device)
        losses = {}

        for dim_idx, dim in enumerate(self.mrl_dims):
            v_slice = F.normalize(visual_embeddings[:, :dim], p=2, dim=-1)
            t_slice = F.normalize(text_embeddings[:, :dim], p=2, dim=-1)

            # Similarity matrix [batch, batch]
            logits = torch.matmul(v_slice, t_slice.T) / self.temperature

            # Symmetric loss: image->text and text->image
            labels = torch.arange(batch_size, device=logits.device)
            loss_v2t = F.cross_entropy(logits, labels)
            loss_t2v = F.cross_entropy(logits.T, labels)
            loss_dim = (loss_v2t + loss_t2v) / 2.0

            losses[f"cross_modal_loss_dim_{dim}"] = loss_dim.item()
            total_loss = total_loss + loss_dim * self.dim_weights[dim_idx]

        losses["cross_modal_loss"] = total_loss.item()
        losses["temperature"] = self.temperature.item()

        return {"loss": total_loss, **losses}


class MultimodalMRLLoss(nn.Module):
    """Combined loss for joint text-retrieval and cross-modal training.

    Computes text-text MRL InfoNCE loss plus cross-modal contrastive loss
    with a configurable mixing weight.

    Attributes:
        text_loss: MRLInfoNCELoss for text-text pairs.
        cross_modal_loss: CrossModalContrastiveLoss for vision-text alignment.
        cross_modal_weight: Weight for cross-modal loss relative to text loss.
    """

    def __init__(
        self,
        mrl_dims: tuple[int, ...] = (512, 1024, 2048, 4096),
        cross_modal_weight: float = 1.0,
        temperature: float = 0.07,
    ):
        """Initialize combined multimodal loss.

        Args:
            mrl_dims: Matryoshka dimensions.
            cross_modal_weight: Scaling factor for cross-modal loss.
            temperature: Initial temperature for cross-modal contrastive.
        """
        super().__init__()

        from omnivector.training.losses import MRLInfoNCELoss

        self.text_loss = MRLInfoNCELoss(mrl_dims=mrl_dims, device="cpu")
        self.cross_modal_loss = CrossModalContrastiveLoss(
            mrl_dims=mrl_dims,
            initial_temperature=temperature,
        )
        self.cross_modal_weight = cross_modal_weight

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        text_for_visual: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute combined text + cross-modal loss.

        Args:
            query_embeddings: Query text embeddings [batch, dim].
            positive_embeddings: Positive text embeddings [batch, dim].
            negative_embeddings: Hard negative embeddings [batch, n_neg, dim].
            visual_embeddings: Image/video embeddings [batch, dim] or None.
            text_for_visual: Text embeddings paired with visuals [batch, dim].
            visual_mask: Mask for valid visual samples [batch].

        Returns:
            Dict with total loss and all component losses.
        """
        # Text-text retrieval loss
        text_result = self.text_loss(
            query_embeddings=query_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
        )

        result = {k: v for k, v in text_result.items()}
        text_loss = text_result["loss"]
        if not isinstance(text_loss, torch.Tensor):
            text_loss = torch.tensor(text_loss, dtype=torch.float32)
        total_loss = text_loss

        # Cross-modal loss (if visual data present)
        if visual_embeddings is not None and text_for_visual is not None:
            cm_result = self.cross_modal_loss(
                visual_embeddings=visual_embeddings,
                text_embeddings=text_for_visual,
                visual_mask=visual_mask,
            )

            cm_loss = cm_result["loss"]
            total_loss = total_loss + self.cross_modal_weight * cm_loss

            for k, v in cm_result.items():
                if k != "loss":
                    result[k] = v

            result["cross_modal_loss_weighted"] = (self.cross_modal_weight * cm_loss).item()

        result["loss"] = total_loss
        result["text_loss"] = text_loss.item() if isinstance(text_loss, torch.Tensor) else float(text_loss)

        return result
