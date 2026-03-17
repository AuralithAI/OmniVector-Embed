"""Training package initialization."""

from omnivector.training.losses import MRLInfoNCELoss, create_mrl_loss
from omnivector.training.multimodal_loss import CrossModalContrastiveLoss, MultimodalMRLLoss
from omnivector.training.multimodal_trainer import MultimodalTrainer

__all__ = [
    "MRLInfoNCELoss",
    "create_mrl_loss",
    "CrossModalContrastiveLoss",
    "MultimodalMRLLoss",
    "MultimodalTrainer",
]
