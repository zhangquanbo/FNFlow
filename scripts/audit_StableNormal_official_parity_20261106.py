
"""Compare the registered adapter with the pinned official public inference entry on independent controls."""
from pathlib import Path
import sys,os,inspect,json,hashlib,random,fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch,cv2
from PIL import Image
from src.external_normal_adapter_20261106 import load
from src.research_io import write_json,sha256,now
OUT=ROOT/'outputs/StableNormal_official_parity_20261106';DATA=ROOT/'outputs/StableNormal_audit_20261106'
def reset():random.seed(42);np.random.seed(42);torch.manual_seed(42)
def common_metrics(pred,gt,valid):
 pred=cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)@np.diag([-1.,-1.,1.]).astype(np.float32)
 pred/=np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-8)
 mae=float(np.degrees(np.arccos(np.clip((pred*gt).sum(-1)[valid],-1,1))).mean());checks={}
 for axis in [0,1]:
  for sign in [-1,1]:
   region=valid&(gt[...,axis]*sign>.5);mean=float(pred[...,axis][region].mean());checks[f'{axis}:{sign}']={'mean':mean,'expected_sign':sign,'passed':mean*sign>.05}
 return {'MAE_deg':mae,'cardinal_checks':checks}
def main():
 OUT.mkdir(exist_ok=True);write_json(OUT/'status.json',{'phase':'loading','done':0,'total':4,'updated_at':now()})
 with (ROOT/'runtime/external_round_GPU3.lock').open('a') as lock,torch.inference_mode():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  predict=load('StableNormal',OUT/'runtime');closed=inspect.getclosurevars(predict).nonlocals;official=closed['model'];capture=closed['capture'];native=capture.inner
  rows=[]
  for index in range(4):
   image=Image.open(DATA/f'shifted{index}_RGB.png').convert('RGB')
   with np.load(DATA/f'shifted{index}_diagnostic.npz') as z:gt=z['normal'].copy();valid=z['valid'].copy()
   official.model=capture;reset();raw=predict(image)
   reconstructed=Image.fromarray((((raw.clip(-1,1)+1)/2)*255).astype(np.uint8)).resize(image.size,Image.Resampling.LANCZOS)
   official.model=native;reset();direct=official(image,data_type='indoor')
   a=np.asarray(reconstructed).astype(np.int16);b=np.asarray(direct).astype(np.int16);assert a.shape==b.shape
   max_delta=int(np.abs(a-b).max());equal=float(np.mean(a==b));reconstructed.save(OUT/f'shifted{index}_adapter.png');direct.save(OUT/f'shifted{index}_official.png')
   row={'control':index,'max_abs_RGB8_difference':max_delta,'equal_channel_fraction':equal,'byte_identical':max_delta==0,'adapter_float':common_metrics(raw,gt,valid),'official_RGB8_decoded':common_metrics(np.asarray(direct).astype(np.float32)/127.5-1,gt,valid)}
   rows.append(row);write_json(OUT/'status.json',{'phase':'comparing_official_entry','done':index+1,'total':4,'updated_at':now()});print('PARITY_CONTROL',index,'maxRGBdelta',max_delta,'equal',equal,flush=True)
  write_json(OUT/'summary.json',{'status':'completed','adapter_matches_pinned_official_public_RGB_output':all(x['byte_identical'] for x in rows),'controls':rows,'source_commit':'594b934630ab3bc71f35c77d14ec7feb98480cd0','adapter_sha256':sha256(ROOT/'src/external_normal_adapter_20261106.py'),'fixed_matrix_to_common':[-1,-1,1],'benchmark_inputs_or_GT_used':False,'benchmark_scores_changed':False,'method':'Same loaded official model, same weights and public Predictor.__call__(data_type=indoor), same seed42. Adapter floating output is rendered using the official clipping/quantization/Lanczos procedure; compare with direct official entry. No alternate axes or input corrections tried.','updated_at':now()});write_json(OUT/'status.json',{'phase':'completed','done':4,'total':4,'updated_at':now()})
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(OUT/'status.json',{'phase':'failed','error':repr(e),'updated_at':now()});raise
