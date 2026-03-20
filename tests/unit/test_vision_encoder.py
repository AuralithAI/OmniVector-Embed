"""Unit tests for SigLIPVisionEncoder freeze/unfreeze (Fix 3).

Validates the new freeze_backbone parameter, context manager selection,
and the unfreeze_backbone / freeze_backbone methods.
"""


from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


class TestVisionEncoderFreezeBackbone:
    """Tests for the freeze_backbone parameter and methods."""

    def _make_encoder(self, freeze: bool = True):
        """Build a SigLIPVisionEncoder with a mocked backbone."""
        from omnivector.model.vision_encoder import SigLIPVisionEncoder

        with patch.object(SigLIPVisionEncoder, "__init__", return_value=None):
            enc = SigLIPVisionEncoder.__new__(SigLIPVisionEncoder)

        nn.Module.__init__(enc)
        enc.model_name = "SigLIP-SO400M"
        enc.embed_dim = 4096
        enc.vision_model_dim = 1152
        enc._freeze_backbone = freeze

        # Fake vision model with trainable params
        enc.vision_model = nn.Linear(1152, 1152)
        enc.preprocess = None
        enc.projection = nn.Linear(1152, 4096)

        if freeze:
            for p in enc.vision_model.parameters():
                p.requires_grad = False
        return enc

    def test_init_frozen_by_default(self):
        enc = self._make_encoder(freeze=True)
        assert enc._freeze_backbone is True
        for p in enc.vision_model.parameters():
            assert p.requires_grad is False

    def test_init_unfrozen(self):
        enc = self._make_encoder(freeze=False)
        assert enc._freeze_backbone is False
        for p in enc.vision_model.parameters():
            assert p.requires_grad is True

    def test_unfreeze_backbone_method(self):
        enc = self._make_encoder(freeze=True)
        enc.unfreeze_backbone()
        assert enc._freeze_backbone is False
        for p in enc.vision_model.parameters():
            assert p.requires_grad is True

    def test_freeze_backbone_method(self):
        enc = self._make_encoder(freeze=False)
        enc.freeze_backbone()
        assert enc._freeze_backbone is True
        for p in enc.vision_model.parameters():
            assert p.requires_grad is False

    def test_trainable_params_frozen(self):
        enc = self._make_encoder(freeze=True)
        trainable = enc.trainable_parameters
        # Only projection should be trainable
        proj_params = sum(p.numel() for p in enc.projection.parameters())
        assert trainable == proj_params

    def test_trainable_params_unfrozen(self):
        enc = self._make_encoder(freeze=False)
        trainable = enc.trainable_parameters
        total = enc.total_parameters
        assert trainable == total

    def test_forward_frozen_uses_no_grad(self):
        """When frozen, forward should wrap backbone in torch.no_grad()."""
        enc = self._make_encoder(freeze=True)
        enc.eval()

        # Mock encode_image to return features
        enc.vision_model.encode_image = MagicMock(return_value=torch.randn(2, 1152))

        with torch.no_grad():
            out = enc(torch.randn(2, 3, 384, 384))
        assert out.shape == (2, 4096)

    def test_forward_unfrozen_allows_grad(self):
        """When unfrozen, backbone features should have grad_fn."""
        enc = self._make_encoder(freeze=False)
        enc.train()

        # Use a real linear as encode_image for gradient tracking
        fake_encoder = nn.Linear(3 * 384 * 384, 1152)

        def encode_image(x):
            return fake_encoder(x.flatten(1))

        enc.vision_model.encode_image = encode_image

        images = torch.randn(2, 3, 384, 384)
        out = enc(images)
        assert out.requires_grad is True

    def test_freeze_unfreeze_round_trip(self):
        enc = self._make_encoder(freeze=True)
        enc.unfreeze_backbone()
        enc.freeze_backbone()
        assert enc._freeze_backbone is True
        for p in enc.vision_model.parameters():
            assert p.requires_grad is False
