"""Canonical518 DINO features, with normalized pixel-center correspondence to original RGB."""
from torch.nn import functional as F
from src.dino_conditioning import FrozenPatchEncoder,DINOConditionedUNet
class CanonicalFrozenPatchEncoder(FrozenPatchEncoder):
 def forward(self,rgb):
  # align_corners=False: source center u maps to encoded center (u+.5)*518/W-.5.
  return super().forward(F.interpolate(rgb,size=(518,518),mode='bilinear',align_corners=False))
class CanonicalDINOConditionedUNet(DINOConditionedUNet):
 def __init__(self,unet,encoder,feature_dim):
  super().__init__(unet,encoder,feature_dim,14,pad_to_patch=False)
  self.features=CanonicalFrozenPatchEncoder(encoder,14,pad_to_patch=False)
  # Parent samples normalized source coordinates:2*(u+.5)/W-1. Since the encoder
  # resizes rather than pads, this equals the encoded518 normalized coordinate.
