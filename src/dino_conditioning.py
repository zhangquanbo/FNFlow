"""Optional frozen patch encoder, with explicit pixel-center registration.

Use DINOv3's forward_features()['x_norm_patchtokens'] contract. Encoder code and
authorized local weights are supplied by the caller; this module downloads none.
Features are recomputed AFTER RGB crop/flip. ViTs are not assumed equivariant.
UNet stride-2 3x3/pad-1 convolutions anchor level-l centers at 2**l * index.
"""
import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path


def load_local_dino(repo_directory,weights,architecture='dinov3_vits16'):
    repo=Path(repo_directory).resolve();weights=Path(weights).resolve()
    if not (repo/'hubconf.py').is_file() or not weights.is_file():
        raise ValueError('A local reference implementation and authorized local weights are required')
    # Official local Hub contract; the caller must pin source and weights in its run signature.
    return torch.hub.load(str(repo),architecture,source='local',weights=str(weights))


def apply_rgb_view(rgb, crop_xywh=None, horizontal_flip=False, output_hw=None):
    if crop_xywh is not None:
        x,y,w,h=crop_xywh
        if min(x,y)<0 or min(w,h)<=0 or x+w>rgb.shape[-1] or y+h>rgb.shape[-2]:
            raise ValueError('Crop outside RGB')
        rgb=rgb[...,y:y+h,x:x+w]
    if output_hw is not None:rgb=F.interpolate(rgb,size=output_hw,mode='bilinear',align_corners=False)
    return rgb.flip(-1) if horizontal_flip else rgb


def patch_map_at_unet_level(feature, input_hw, level, target_hw):
    """Sample patch-center features at the true convolution feature centers."""
    height,width=input_hw;out_h,out_w=target_hw;stride=2**level
    if out_h!=(height+stride-1)//stride or out_w!=(width+stride-1)//stride:
        raise ValueError('UNet stride grid does not match RGB view')
    ys=torch.arange(out_h,device=feature.device,dtype=torch.float32)*stride
    xs=torch.arange(out_w,device=feature.device,dtype=torch.float32)*stride
    y,x=torch.meshgrid(2*(ys+.5)/height-1,2*(xs+.5)/width-1,indexing='ij')
    grid=torch.stack([x,y],-1)[None].expand(feature.shape[0],-1,-1,-1)
    # Border extension is explicit outside the first/last patch center.
    return F.grid_sample(feature.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)


class FrozenPatchEncoder(nn.Module):
    def __init__(self,encoder,patch_size=16):
        super().__init__();self.encoder=encoder;self.patch_size=patch_size
        self.encoder.requires_grad_(False);self.encoder.eval()
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])

    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self

    @torch.no_grad()
    def forward(self,rgb):
        h,w=rgb.shape[-2:];p=self.patch_size
        if h%p or w%p:raise ValueError('RGB shape must be a multiple of patch size; no implicit crop/resize')
        self.encoder.eval()
        x=((rgb.float()+1)/2-self.mean)/self.std
        tokens=self.encoder.forward_features(x)['x_norm_patchtokens']
        if tokens.ndim!=3 or tokens.shape[1]!=(h//p)*(w//p):
            raise ValueError('Patch token count mismatch; CLS/register tokens must not enter the grid')
        return tokens.transpose(1,2).reshape(len(rgb),tokens.shape[-1],h//p,w//p).detach()


class DINOConditionedUNet(nn.Module):
    """Zero-initialized semantic residual at each encoder level of a supplied UNet."""
    def __init__(self,unet,encoder,feature_dim,patch_size=16):
        super().__init__();self.unet=unet;self.features=FrozenPatchEncoder(encoder,patch_size)
        self.adapters=nn.ModuleList([nn.Conv2d(feature_dim,c,1) for c in unet.channels])
        for layer in self.adapters:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,z_t,t,condition_img,condition_keep=None):
        keep=torch.ones((len(condition_img),1,1,1),device=condition_img.device,dtype=condition_img.dtype)
        if condition_keep is not None:
            keep=condition_keep.to(condition_img).reshape(-1,1,1,1)
            if len(keep)!=len(condition_img):raise ValueError('Condition dropout batch mismatch')
        rgb=condition_img*keep
        feature=self.features(rgb);u=self.unet;embedding=u.time_embed(t)
        x=u.input_conv(torch.cat([z_t,rgb],1));skips=[]
        for level,blocks in enumerate(u.encoder_blocks):
            for block in blocks:x=block(x,embedding)
            registered=patch_map_at_unet_level(feature,rgb.shape[-2:],level,x.shape[-2:])
            x=x+self.adapters[level](registered.to(x.dtype))*keep
            skips.append(x)
            if level<u.num_levels-1:x=u.downsamples[level](x)
        for block in u.bottleneck:x=block(x,embedding)
        for i,level in enumerate(range(u.num_levels-1,-1,-1)):
            x=torch.cat([x,skips.pop()],1)
            for block in u.decoder_blocks[i]:x=block(x,embedding)
            if level>0:x=F.interpolate(u.upsamples[i](x,embedding),scale_factor=2,mode='nearest')
        return u.output_conv(F.silu(u.output_norm(x)))
