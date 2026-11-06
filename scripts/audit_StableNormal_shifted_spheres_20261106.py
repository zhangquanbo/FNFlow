
from pathlib import Path
import sys,json,fcntl,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from PIL import Image
from src.external_normal_adapter_20261106 import load
from src.research_io import write_json,now
OUT=ROOT/'outputs/StableNormal_audit_20261106';OUT.mkdir(exist_ok=True)
def sphere(center,radius=.65,side=512):
 y,x=np.mgrid[:side,:side];uv=np.stack([(x+.5)/side*2-1,1-(y+.5)/side*2,np.full_like(x,-1,dtype=float)],-1);uv[...,:2]*=np.tan(np.deg2rad(38)/2);ray=uv/np.linalg.norm(uv,axis=-1,keepdims=True);eye=np.array([0.,0.,3.6]);center=np.array(center);offset=eye-center;b=(ray*offset).sum(-1);disc=b*b-(offset@offset-radius*radius);valid=disc>0;t=-b-np.sqrt(np.maximum(disc,0));points=eye+ray*t[...,None];normal=(points-center)/radius;normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8)
 light=np.array([-.4,.6,1.]);light/=np.linalg.norm(light);view=eye-points;view/=np.maximum(np.linalg.norm(view,axis=-1,keepdims=True),1e-8);half=view+light;half/=np.maximum(np.linalg.norm(half,axis=-1,keepdims=True),1e-8);diff=np.maximum((normal*light).sum(-1),0);spec=np.maximum((normal*half).sum(-1),0)**32
 linear=np.array([.62,.56,.45])*(.25+.65*diff[...,None])+.08*spec[...,None];rgb=np.where(valid[...,None],np.clip(linear,0,1)**(1/2.2),1.)
 return Image.fromarray(np.rint(rgb*255).astype(np.uint8)),(normal*np.array([-1,1,1])).astype(np.float32),valid
with (ROOT/'runtime/external_round_GPU3.lock').open('a') as lock,torch.inference_mode():
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=load('StableNormal',OUT/'runtime');matrix=np.diag([-1.,-1.,1.]).astype(np.float32);rows=[]
 for index,center in enumerate([[.4,.1,0],[-.4,-.1,0],[.1,.4,0],[-.1,-.4,0]]):
  image,gt,valid=sphere(center);torch.manual_seed(42);raw=model(image);pred=cv2.resize(raw,(512,512))@matrix.T;pred/=np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-8);mae=float(np.degrees(np.arccos(np.clip((pred*gt).sum(-1)[valid],-1,1))).mean());checks={}
  for axis in [0,1]:
   for sign in [-1,1]:
    mask=valid&(gt[...,axis]*sign>.5);mean=float(pred[...,axis][mask].mean());checks[f'{axis}:{sign}']={'mean':mean,'expected_sign':sign,'sign_correct':mean*sign>.05}
  image.save(OUT/f'shifted{index}_RGB.png');np.savez_compressed(OUT/f'shifted{index}_diagnostic.npz',raw=raw,common_prediction=pred,normal=gt,valid=valid);rows.append({'center':center,'MAE_deg':mae,'sign_probes':checks});write_json(OUT/'status.json',{'phase':'independent_controls','done':index+1,'total':4,'updated_at':now()});print('SHIFTED_SPHERE',center,mae,flush=True)
 write_json(OUT/'summary.json',{'status':'completed','purpose':'Off-center analytic spheres break the centered-sphere spatial symmetry. Fixed preregistered matrix only; no axis search or benchmark input/GT used. Geometry diagnostics do not alter frozen benchmark scores.','matrix':matrix.tolist(),'rows':rows,'benchmark_inputs_opened':False,'updated_at':now()});write_json(OUT/'status.json',{'phase':'completed','updated_at':now()})
