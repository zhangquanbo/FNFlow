"""Strict conditioning ablation: D2 direct Xn output,3 extra normal-input components."""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from src.dino_conditioning import load_local_dino,patch_map_at_unet_level
from src.dino_canonical_518 import CanonicalDINOConditionedUNet
from src.models.model_unet import FaceNormalUNet_B
from src.research_io import sha256
from src.e_models_20261106 import ConditionCache as LegacyCache,evaluate as legacy_evaluate
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
class ConditionDirectUNet(CanonicalDINOConditionedUNet):
 def __init__(self,unet,encoder,feature_dim):
  super().__init__(unet,encoder,feature_dim)
  original=unet.input_conv
  with torch.random.fork_rng(devices=[]):expanded=nn.Conv2d(9,original.out_channels,original.kernel_size,original.stride,original.padding,bias=original.bias is not None)
  with torch.no_grad():
   expanded.weight.zero_();expanded.weight[:,:6].copy_(original.weight)
   if original.bias is not None:expanded.bias.copy_(original.bias)
  unet.input_conv=expanded
 def forward(self,z_t,t,condition_img,condition_keep=None):
  assert condition_img.shape[1]==6
  keep=torch.ones((len(condition_img),1,1,1),device=condition_img.device,dtype=condition_img.dtype)
  if condition_keep is not None:keep=condition_keep.to(condition_img).reshape(-1,1,1,1)
  rgb=condition_img[:,:3]*keep;coarse=condition_img[:,3:]*keep
  feature=self.features(rgb);u=self.unet;embedding=u.time_embed(t)
  x=u.input_conv(torch.cat([z_t,rgb,coarse],1));skips=[]
  for level,blocks in enumerate(u.encoder_blocks):
   for block in blocks:x=block(x,embedding)
   registered=patch_map_at_unet_level(feature,rgb.shape[-2:],level,x.shape[-2:],rgb.shape[-2:],(0,0))
   x=x+self.adapters[level](registered.to(x.dtype))*keep;skips.append(x)
   if level<u.num_levels-1:x=u.downsamples[level](x)
  for block in u.bottleneck:x=block(x,embedding)
  for i,level in enumerate(range(u.num_levels-1,-1,-1)):
   x=torch.cat([x,skips.pop()],1)
   for block in u.decoder_blocks[i]:x=block(x,embedding)
   if level>0:x=F.interpolate(u.upsamples[i](x,embedding),scale_factor=2,mode='nearest')
  return u.output_conv(F.silu(u.output_norm(x)))
def make_model(cfg):
 compatibility=read(ROOT/'configs/dinov2_b_518_compatibility_20261029.json');assert sha256(ROOT/'runtime/dinov2/source.zip')==compatibility['source_archive_sha256'];assert sha256(ROOT/cfg['prior_weights'])==cfg['prior_weights_sha256']
 unet=FaceNormalUNet_B();encoder=load_local_dino(ROOT/compatibility['source_directory'],ROOT/cfg['prior_weights'],cfg['prior_architecture']);return ConditionDirectUNet(unet,encoder,768).cuda()
def mapped(cfg):value=dict(cfg);value['stage']='E2_B';return value
class ConditionCache(LegacyCache):
 def __init__(self,records,capacity=128,cfg=None):super().__init__(records,capacity,mapped(cfg))
def evaluate(model,mode,domains,seeds,limit,device,cfg):return legacy_evaluate(model,mode,domains,seeds,limit,device,mapped(cfg))
