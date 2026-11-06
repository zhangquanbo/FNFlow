
"""Pinned MoGe-2 normal inference and D2 loading for matched measurements.

The normal-only path follows the public MoGe v2 computation graph (Microsoft, MIT),
with the unused geometry heads and focal/shift recovery omitted. Its equivalence is
checked before use. DINOv2 dependencies retain their upstream Apache2 license.
"""
from pathlib import Path
import sys,json,math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.research_io import sha256
from src.noise_protocol import image_noise_seed
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'external_baselines_20261030/MoGe2/source'

def read(path):return json.loads(Path(path).read_text())
def tensor_rgb(image,side=None):
 if side is not None:image=image.resize((side,side),Image.Resampling.BILINEAR)
 return torch.from_numpy(np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()).cuda()/255

def load_moge(size,sdpa=True,verify=True):
 spec=read(ROOT/'configs/moge2_normal_models_20261106.json')['models'][size]
 path=ROOT/spec['weights_relative']
 if verify:assert sha256(path)==spec['sha256']
 sys.path.insert(0,str(SOURCE))
 from moge.model.v2 import MoGeModel
 checkpoint=torch.load(path,map_location='cpu',weights_only=True)
 config=checkpoint['model_config'];model=MoGeModel(**config)
 model.load_state_dict(checkpoint['model'],strict=True);del checkpoint
 if sdpa:model.enable_pytorch_native_sdpa()
 model=model.cuda().eval()
 return model,config

def normal_only(model,image,use_fp16=True,num_tokens=None):
 from moge.utils.geometry_torch import normalized_view_plane_uv
 single=image.ndim==3
 if single:image=image[None]
 image=image.to(device=model.device,dtype=model.dtype)
 height,width=image.shape[-2:];aspect=width/height
 tokens=model.num_tokens_range[1] if num_tokens is None else num_tokens
 rows=round(math.sqrt(tokens/aspect));cols=round(math.sqrt(tokens*aspect))
 with torch.autocast('cuda',dtype=torch.float16,enabled=use_fp16 and model.dtype!=torch.float16):
  encoded,_=model.encoder(image,rows,cols,return_class_token=True)
  features=[]
  for level in range(5):
   uv=normalized_view_plane_uv(width=cols*2**level,height=rows*2**level,aspect_ratio=aspect,dtype=image.dtype,device=image.device)
   uv=uv.permute(2,0,1)[None].expand(len(image),-1,-1,-1)
   features.append(torch.cat([encoded,uv],1) if level==0 else uv)
  features=model.neck(features)
  normal=model.normal_head(features)[-1]
  normal=F.interpolate(normal,(height,width),mode='bilinear',align_corners=False,antialias=False)
  normal=F.normalize(normal.permute(0,2,3,1),dim=-1)
 return normal.float()[0] if single else normal.float()

def full_normal(model,image,use_fp16=True):
 return model.infer(image,resolution_level=9,use_fp16=use_fp16,apply_mask=False)['normal'].float()

def common256(normal,model_name):
 normal=normal.float()
 if model_name!='D2':normal=-normal
 normal=F.interpolate(normal.permute(2,0,1)[None],size=(256,256),mode='bilinear',align_corners=False)
 return F.normalize(normal,dim=1)[0].permute(1,2,0)

def load_d2():
 from src.models.model_unet import FaceNormalUNet_B
 from src.dino_conditioning import load_local_dino
 from src.dino_canonical_518 import CanonicalDINOConditionedUNet
 cfg=read(ROOT/'configs/dinov2_b_518_compatibility_20261029.json')
 weights=ROOT/'runtime/dinov2/dinov2_vitb14_pretrain.pth';assert sha256(weights)==cfg['weights_sha256']
 assert sha256(ROOT/'runtime/dinov2/source.zip')==cfg['source_archive_sha256']
 unet=FaceNormalUNet_B();encoder=load_local_dino(ROOT/cfg['source_directory'],weights,cfg['architecture'])
 model=CanonicalDINOConditionedUNet(unet,encoder,768).cuda().eval()
 path=ROOT/'outputs/D2_train_20261030/D2_Xn_s1247/last_local.pth'
 assert sha256(path)=='f1cbe7e14db11c4856677731c6294d3a55b4068dcbd2f59fecae68e9fa05d170'
 checkpoint=torch.load(path,map_location='cpu',weights_only=False);state=model.state_dict()
 assert set(checkpoint['trainable_model'])=={key for key in state if not key.startswith('features.')}
 state.update(checkpoint['trainable_model']);model.load_state_dict(state,strict=True)
 return model


def d2_normal_tensor(model,rgb,image_key,seed=42):
 if rgb.ndim==3:rgb=rgb[None]
 noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,image_key)))
 with torch.autocast('cuda',dtype=torch.bfloat16):normal=model(noise,torch.zeros(1,device='cuda'),rgb)
 return normal.float()[0].permute(1,2,0)

def d2_normal(model,image,image_key,seed=42):
 return d2_normal_tensor(model,tensor_rgb(image,256)*2-1,image_key,seed)
