"""Training losses including Matryoshka Representation Learning (MRL)."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MRLInfoNCELoss(nn.Module):
    """
    Matryoshka Representation Learning with InfoNCE loss.

    Applies InfoNCE loss at multiple dimensions [512, 1024, 2048, 4096]
    with learnable weights for each dimension.

    Attributes:
        temperatures: Softmax temperatures for each MRL dimension
        mrl_dims: Tuple of dimensions to apply loss at
    """

    def __init__(
        self,
        mrl_dims: tuple[int, ...] = (512, 1024, 2048, 4096),
        temperatures: Optional[tuple[float, ...]] = None,
        device: str = "cpu",
    ) -> None:
        """
        Initialize MRL loss.

        Args:
            mrl_dims: Matryoshka dimensions
            temperatures: Temperature per dimension (default: 0.07 for all)
            device: Device for buffer creation (buffers move with the model)
        """
        super().__init__()
        self.mrl_dims = mrl_dims

        if temperatures is None:
            temperatures = tuple(0.07 for _ in mrl_dims)

        self.register_buffer(
            "temperatures",
            torch.tensor(temperatures, device=device),
        )

        # Fixed dimension weights per NV-Embed-v2 spec
        self.register_buffer(
            "dim_weights",
            torch.tensor([0.5, 0.75, 1.0, 1.0][: len(mrl_dims)], device=device),
        )

        logger.info(f"MRLInfoNCELoss initialized: dims={mrl_dims}, temps={temperatures}")

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: Optional[torch.Tensor] = None,
        hard_negatives: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute Matryoshka InfoNCE loss.

        Args:
            query_embeddings: Query embeddings [batch_size, max_dim] (e.g., 4096)
            positive_embeddings: Positive passage embeddings [batch_size, max_dim]
            negative_embeddings: In-batch hard negatives [batch_size, num_hard_negs, max_dim]
            hard_negatives: Additional hard negatives [batch_size, num_negs, max_dim]

        Returns:
            Dictionary with loss components and total loss

        Raises:
            ValueError: If embeddings don't match expected shapes
        """
        batch_size = query_embeddings.size(0)

        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=-1)

        losses = {}
        total_loss = 0.0

        # Compute loss at each MRL dimension
        for dim_idx, dim in enumerate(self.mrl_dims):
            # Slice embeddings to this dimension
            q_slice = query_embeddings[:, :dim]
            p_slice = positive_embeddings[:, :dim]

            # Compute similarity scores
            # Positive similarity: [batch_size]
            pos_sim = (q_slice * p_slice).sum(dim=-1, keepdim=True)

            # Gather negatives
            negatives = []

            # In-batch negatives (other queries/positives)
            if negative_embeddings is not None:
                neg_embeddings_slice = negative_embeddings[:, :, :dim]
                neg_sim = torch.bmm(
                    q_slice.unsqueeze(1), neg_embeddings_slice.transpose(-2, -1)
                ).squeeze(1)
                negatives.append(neg_sim)

            # Hard negatives
            if hard_negatives is not None:
                hard_neg_slice = hard_negatives[:, :, :dim]
                hard_sim = torch.bmm(
                    q_slice.unsqueeze(1), hard_neg_slice.transpose(-2, -1)
                ).squeeze(1)
                negatives.append(hard_sim)

            # Concatenate all negatives
            if negatives:
                all_neg_sims = torch.cat(negatives, dim=1)
            else:
                # Fallback: use in-batch negatives
                pos_sims_all = torch.matmul(q_slice, positive_embeddings[:, :dim].transpose(0, 1))
                all_neg_sims = pos_sims_all[~torch.eye(batch_size, dtype=torch.bool, device=q_slice.device)]
                all_neg_sims = all_neg_sims.reshape(batch_size, -1)

            # Compute InfoNCE loss
            # logits = [positive similarity, negative similarities]
            logits = torch.cat([pos_sim, all_neg_sims], dim=1)
            logits = logits / self.temperatures[dim_idx]

            # Cross-entropy loss (label is 0 for positive)
            labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
            loss_dim = F.cross_entropy(logits, labels)

            losses[f"loss_dim_{dim}"] = loss_dim.item()
            total_loss = total_loss + loss_dim * self.dim_weights[dim_idx]

        losses["total_loss_scalar"] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        return {"loss": total_loss, **losses}


def create_mrl_loss(
    mrl_dims: tuple[int, ...] = (512, 1024, 2048, 4096),
) -> MRLInfoNCELoss:
    """
    Convenience function to create MRL loss.

    Args:
        mrl_dims: Matryoshka dimensions

    Returns:
        MRLInfoNCELoss instance
    """
    return MRLInfoNCELoss(mrl_dims=mrl_dims)
