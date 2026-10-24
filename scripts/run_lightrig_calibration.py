import sys,json,itertools,random,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.lightrig_adapter import load_models
from src.gpu_job_lock import gpu_job_lock
from src.research_io import write_json,sha256
def fit(pred,gt,valid):
 rows=[]
 for perm in itertools.permutations(range(3)):
  for signs in itertools.product([-1,1],repeat=3):
   matrix=np.eye(3)[list(perm)]*np.array(signs)[:,None]
   p=pred[valid]@matrix.T;angles=np.degrees(np.arccos(np.clip((p*gt[valid]).sum(-1),-1,1)))
   rows.append({'matrix':matrix.tolist(),'mae':float(angles.mean())})
 return sorted(rows,key=lambda r:r['mae'])
def main():
 source=ROOT/'data/lightrig_calibration_v1';out=ROOT/'outputs/lightrig_calibration_v1';out.mkdir(parents=True,exist_ok=True)
 module,a,b=load_models();random.seed(511);np.random.seed(511);torch.manual_seed(511);cases=[];allrows=[];shaderrows=[]
 for name in ['sphere','cube']:
  dest=out/(name+'.png');module.inference_impl(a,b,str(source/(name+'.png')),str(dest))
  rgba=np.asarray(Image.open(source/(name+'.png')).convert('RGBA'));data=np.load(source/(name+'.npz'));gt=data['normal_gl'];valid=data['valid']&(rgba[...,3]>252)
  valid=(1-F.max_pool2d(1-torch.from_numpy(valid.astype(np.float32))[None,None],17,1,8))[0,0].numpy()>.5
  pred=np.array(Image.open(dest).crop((2048,0,2560,512)).convert('RGB')).astype(np.float64)/127.5-1;pred/=np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-8)
  shader=np.array(Image.open(source/(name+'_source_shader.png')).convert('RGB')).astype(np.float64)/127.5-1;shader/=np.maximum(np.linalg.norm(shader,axis=-1,keepdims=True),1e-8)
  rows=fit(pred,gt,valid);sr=fit(shader,gt,valid);allrows.append(rows);shaderrows.append(sr)
  cases.append({'name':name,'pixels':int(valid.sum()),'prediction_sha256':sha256(dest),'best':rows[0],'runner_up':rows[1],'source_shader_best':sr[0]});print(cases[-1],flush=True)
 joint=[]
 for r in allrows[0]:
  other=next(q for q in allrows[1] if q['matrix']==r['matrix']);joint.append({'matrix':r['matrix'],'mae':(r['mae']+other['mae'])/2})
 joint.sort(key=lambda r:r['mae']);empirical=all(c['best']['matrix']==joint[0]['matrix'] and c['best']['mae']<35 for c in cases) and joint[1]['mae']-joint[0]['mae']>=5
 source_agreement=all(c['source_shader_best']['matrix']==joint[0]['matrix'] for c in cases)
 result={'status':'closed' if empirical and source_agreement else 'source_mismatch' if empirical else 'ambiguous','empirical_pass':empirical,'source_agreement':source_agreement,'cases':cases,'joint_best':joint[0],'joint_runner_up':joint[1],'nlr_to_common':(np.diag([-1,1,1])@np.array(joint[0]['matrix'])).tolist(),'normal_storage':'RGB uint8, decode /127.5-1 then normalize','amendment_sha256':sha256(ROOT/'docs/object_calibration_intake_amendment_20261023.md'),'script_sha256':sha256(__file__),'test_gt_used':False}
 write_json(out/'result.json',result);(out/'result.json.sha256').write_text(sha256(out/'result.json')+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':
 with gpu_job_lock():main()
