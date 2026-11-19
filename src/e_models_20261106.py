"""Frozen geometry priors and additive normal conditioning for registered E fits."""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.models.model_unet import FaceNormalUNet_B
from src.dino_conditioning import load_local_dino,patch_map_at_unet_level
from src.dino_canonical_518 import CanonicalDINOConditionedUNet
from src.research_io import sha256
from scripts.train_parameterization import ImageCache
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
class ResidualNormalUNet(CanonicalDINOConditionedUNet):
 def __init__(self,unet,encoder,feature_dim):
  super().__init__(unet,encoder,feature_dim)
  original=unet.input_conv
  with torch.random.fork_rng(devices=[]):expanded=nn.Conv2d(9,original.out_channels,original.kernel_size,original.stride,original.padding,bias=original.bias is not None)
  with torch.no_grad():
   expanded.weight.zero_();expanded.weight[:,:6].copy_(original.weight)
   if original.bias is not None:expanded.bias.copy_(original.bias)
   unet.output_conv.weight.zero_()
   if unet.output_conv.bias is not None:unet.output_conv.bias.zero_()
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
  return coarse.float()+u.output_conv(F.silu(u.output_norm(x))).float()
def make_model(cfg):
 compatibility=read(ROOT/'configs/dinov2_b_518_compatibility_20261029.json')
 assert sha256(ROOT/'runtime/dinov2/source.zip')==compatibility['source_archive_sha256']
 unet=FaceNormalUNet_B()
 assert sha256(ROOT/cfg['prior_weights'])==cfg['prior_weights_sha256']
 encoder=load_local_dino(ROOT/compatibility['source_directory'],ROOT/cfg['prior_weights'],cfg['prior_architecture'])
 klass=ResidualNormalUNet if cfg['stage']=='E2_B' else CanonicalDINOConditionedUNet
 return klass(unet,encoder,cfg['prior_feature_dim']).cuda()
class ConditionCache:
 def __init__(self,records,capacity=128,cfg=None):
  self.base=ImageCache(records,capacity);self.records=records;self.cfg=cfg;self.capacity=capacity;self.coarse={}
 def get(self,index):
  rgb,normal,valid=self.base.get(index)
  if self.cfg['stage']!='E2_B':return rgb,normal,valid
  row=self.records[index];key=hashlib.sha256(row['image_key'].encode()).hexdigest()
  if index not in self.coarse:
   root=ROOT/self.cfg['coarse_cache_directory'];path=root/(key+'.npz');receipt=read(root/(key+'.json'))
   assert receipt['image_key']==row['image_key'] and receipt['source_rgb_sha256']==row['source_rgb_sha256']
   assert receipt['contract_sha256']==self.cfg['coarse_contract_sha256'] and sha256(path)==receipt['sha256']
   with np.load(path,allow_pickle=False) as data:
    assert str(data['contract_sha256'])==self.cfg['coarse_contract_sha256']
    n=data['normal'].transpose(2,0,1).astype(np.float32)
   assert n.shape==(3,256,256) and np.isfinite(n).all()
   n=n/np.maximum(np.linalg.norm(n,axis=0,keepdims=True),1e-12)
   self.coarse[index]=n
   if len(self.coarse)>self.capacity:self.coarse.pop(next(iter(self.coarse)))
  return np.concatenate([rgb,self.coarse[index]],axis=0),normal,valid
def evaluate(model,mode,domains,seeds,limit,device,cfg):
 import scripts.train_parameterization as original
 previous=original.ImageCache
 try:
  original.ImageCache=lambda records,capacity:ConditionCache(records,capacity,cfg)
  return original.evaluate(model,mode,domains,seeds,limit,device)
 finally:original.ImageCache=previous
