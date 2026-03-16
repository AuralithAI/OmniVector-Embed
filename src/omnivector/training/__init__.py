"""Training package initialization."""

from omnivector.training.losses import MRLInfoNCELoss, create_mrl_loss

__all__ = [
    "MRLInfoNCELoss",
    "create_mrl_loss",
]
