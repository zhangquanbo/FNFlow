"""E1-compatible image-only normal network with optional encoder gradients; teacher never input."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from src.e_models_20261106 import make_model as base_model
from scripts.train_parameterization import ImageCache
from src.research_io import sha256
R=Path(__file__).resolve().parents[1]
class GradientCanonicalEncoder(nn.Module):
 def __init__(self,old):
  super().__init__();self.encoder=old.encoder;self.register_buffer('mean',old.mean.clone());self.register_buffer('std',old.std.clone());self.gradient_enabled=False;self.patch_size=14;self.pad_to_patch=False
 def train(self,mode=True):super().train(mode);self.encoder.eval();return self
 def forward(self,rgb):
  self.encoder.eval();x=F.interpolate(rgb,size=(518,518),mode='bilinear',align_corners=False);x=((x.float()+1)/2-self.mean)/self.std
  with torch.set_grad_enabled(torch.is_grad_enabled() and self.gradient_enabled):tokens=self.encoder.forward_features(x)['x_norm_patchtokens'];assert tokens.shape[1]==1369;f=tokens.transpose(1,2).reshape(len(rgb),tokens.shape[-1],37,37)
  return f if self.gradient_enabled else f.detach()
 def set_gradient(self,enabled):self.gradient_enabled=enabled;self.encoder.requires_grad_(enabled);self.encoder.eval()
def make_model(cfg):
 model=base_model(cfg);model.features=GradientCanonicalEncoder(model.features).cuda();return model
class TeacherCache:
 def __init__(self,records,cfg,capacity=128):self.base=ImageCache(records,capacity);self.records=records;self.cfg=cfg;self.capacity=capacity;self.cache={}
 def get(self,index):
  rgb,normal,mask=self.base.get(index)
  if index not in self.cache:
   row=self.records[index];key=hashlib.sha256(row['image_key'].encode()).hexdigest();folder=R/self.cfg['teacher_cache'];p=folder/(key+'.npz');receipt=json.loads((folder/(key+'.json')).read_text());assert receipt['contract_sha256']==self.cfg['teacher_contract_sha256'] and receipt['image_key']==row['image_key'] and receipt['source_rgb_sha256']==row['source_rgb_sha256'] and sha256(p)==receipt['sha256']
   with np.load(p,allow_pickle=False) as z:assert str(z['contract_sha256'])==self.cfg['teacher_contract_sha256'];teacher=z['normal'].astype(np.float32).transpose(2,0,1)
   assert teacher.shape==normal.shape and np.isfinite(teacher).all();teacher/=np.maximum(np.linalg.norm(teacher,axis=0,keepdims=True),1e-12);self.cache[index]=teacher
   if len(self.cache)>self.capacity:self.cache.pop(next(iter(self.cache)))
  return rgb,normal,mask,self.cache[index]
def distill(pred,teacher,mask):
 if mask.ndim==4:mask=mask[:,0]
 counts=mask.sum((1,2));assert (counts>0).all();return (((1-F.cosine_similarity(pred.float(),teacher.float(),dim=1,eps=1e-4))*mask).sum((1,2))/counts).mean()
def groups(model,cfg):
 head=[];layers={};depth=len(model.features.encoder.blocks)
 for name,p in model.named_parameters():
  if name.startswith('features.encoder.'):
   if not cfg['unfreeze_encoder']:continue
   local=name.removeprefix('features.encoder.');index=int(local.split('.')[1])+1 if local.startswith('blocks.') else (depth if local.startswith('norm.') else 0);decay=cfg['layer_decay']**(depth-index);layers.setdefault(index,{'params':[],'lr':cfg['encoder_learning_rate']*decay,'base_lr':cfg['encoder_learning_rate']*decay,'layer_id':index})['params'].append(p)
  else:head.append(p)
 return [{'params':head,'lr':cfg['learning_rate'],'base_lr':cfg['learning_rate'],'layer_id':'head'}]+[layers[k] for k in sorted(layers)]
