import copy
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from src.dino_conditioning import FrozenPatchEncoder,DINOConditionedUNet,patch_map_at_unet_level,apply_rgb_view
from src.models.model_unet import FaceNormalUNet


class MockPatchEncoder(nn.Module):
    """Contract fixture only, never presented as pretrained DINO validation."""
    def __init__(self):
        super().__init__();self.projection=nn.Conv2d(3,4,1);self.dropout=nn.Dropout(.9);self.last_input=None
    def forward_features(self,x):
        self.last_input=x.detach().clone()
        return {'x_norm_patchtokens':self.dropout(self.projection(F.avg_pool2d(x,16))).flatten(2).transpose(1,2)}


def test_patch_center_registration_matches_actual_strided_convolution_anchors():
    y,x=torch.meshgrid(torch.tensor([7.5,23.5,39.5,55.5]),torch.tensor([7.5,23.5,39.5,55.5]),indexing='ij')
    feature=torch.stack([x,y])[None]
    for level in range(4):
        size=64//(2**level)
        result=patch_map_at_unet_level(feature,(64,64),level,(size,size))
        centers=(torch.arange(size)*2**level).float().clamp(7.5,55.5)
        torch.testing.assert_close(result[0,0],centers[None].expand(size,-1))
        torch.testing.assert_close(result[0,1],centers[:,None].expand(-1,size))


def test_features_see_exact_augmented_rgb_and_encoder_stays_frozen_in_eval():
    torch.manual_seed(9);encoder=MockPatchEncoder();wrapped=FrozenPatchEncoder(encoder)
    wrapped.train();assert not encoder.training and not encoder.dropout.training
    image=torch.rand(2,3,64,96)*2-1
    view=apply_rgb_view(image,(16,0,64,64),True,(32,32))
    explicit=F.interpolate(image[...,0:64,16:80],(32,32),mode='bilinear',align_corners=False).flip(-1)
    torch.testing.assert_close(view,explicit)
    first=wrapped(view);second=wrapped(explicit)
    torch.testing.assert_close(first,second)
    torch.testing.assert_close(encoder.last_input,((view+1)/2-wrapped.mean)/wrapped.std)
    assert not first.requires_grad and all(not p.requires_grad for p in encoder.parameters())
    with pytest.raises(ValueError):wrapped(image[...,:31,:32])


def test_zero_adapter_preserves_baseline_and_gradient_freezing_and_dropout_are_real():
    torch.manual_seed(3)
    base=FaceNormalUNet(channels=(8,16),num_res_blocks=1)
    nn.init.normal_(base.output_conv.weight,std=.1)
    baseline=copy.deepcopy(base)
    model=DINOConditionedUNet(base,MockPatchEncoder(),4).train()
    z=torch.randn(1,3,32,32);rgb=torch.randn_like(z);t=torch.tensor([.3])
    torch.testing.assert_close(model(z,t,rgb),baseline(z,t,rgb))
    output=model(z,t,rgb);output.square().mean().backward()
    assert all(p.grad is None for p in model.features.encoder.parameters())
    assert model.adapters[0].weight.grad.abs().sum()>0
    for adapter in model.adapters:nn.init.normal_(adapter.weight,std=.1)
    zero=torch.tensor([False])
    torch.testing.assert_close(model(z,t,rgb,zero),model(z,t,rgb+10,zero))
