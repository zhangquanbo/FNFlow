"""Independent analytic shapes validate the already registered coordinate candidate; no axis search."""
from pathlib import Path
import numpy as np,cv2,json,time,hashlib
from PIL import Image
def fixture(kind,angles=(0,0,0),side=512):
 y,x=np.mgrid[:side,:side];uv=np.stack([(x+.5)/side*2-1,1-(y+.5)/side*2,np.full_like(x,-1,dtype=float)],-1);uv[...,:2]*=np.tan(np.deg2rad(38)/2);rays=uv/np.linalg.norm(uv,axis=-1,keepdims=True);eye=np.array([0.,0.,3.6]);normal=np.zeros_like(rays)
 if kind=='sphere':
  b=np.sum(rays*eye,-1);disc=b*b-(np.dot(eye,eye)-1);mask=disc>0;t=-b-np.sqrt(np.maximum(disc,0));points=eye+rays*t[...,None];normal[mask]=points[mask]
 else:
  a,b,c=np.deg2rad(angles);rx=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]);ry=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]]);rz=np.array([[np.cos(c),-np.sin(c),0],[np.sin(c),np.cos(c),0],[0,0,1]]);rot=ry@rx@rz;d=rays@rot;o=eye@rot;d=np.where(np.abs(d)<1e-8,1e-8,d);lo=(-.78-o)/d;hi=(.78-o)/d;enter=np.minimum(lo,hi).max(-1);leave=np.maximum(lo,hi).min(-1);mask=(leave>=enter)&(enter>0);points=eye+rays*enter[...,None];hit=o+d*enter[...,None];axis=np.abs(hit).argmax(-1);local=np.zeros_like(rays)
  for k in range(3):local[...,k]=np.where(axis==k,np.sign(hit[...,k]),0)
  normal[mask]=(local@rot.T)[mask]
 normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-8);light=np.array([-.4,.6,1.]);light/=np.linalg.norm(light);view=eye-points;view/=np.maximum(np.linalg.norm(view,axis=-1,keepdims=True),1e-8);half=view+light;half/=np.maximum(np.linalg.norm(half,axis=-1,keepdims=True),1e-8);diff=np.maximum(np.sum(normal*light,-1),0);spec=np.maximum(np.sum(normal*half,-1),0)**32
 linear=np.array([.62,.56,.45])*(.25+.65*diff[...,None])+.08*spec[...,None];rgb=np.where(mask[...,None],np.clip(linear,0,1)**(1/2.2),1.);image=Image.fromarray(np.rint(rgb*255).astype(np.uint8));return image,(normal*np.array([-1,1,1])).astype(np.float32),mask
def run_calibration(name,predict,root,registration):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);reg=json.loads(Path(registration).read_text());matrix=np.asarray(reg['models'][name]['matrix_candidate'],dtype=np.float32);results=[];signs={}
 for title,kind,angles in [('sphere','sphere',(0,0,0)),('cubeA','cube',(18,34,-8)),('cubeB','cube',(-18,-34,8))]:
  image,gt,mask=fixture(kind,angles);image.save(root/(title+'_RGB.png'));np.savez_compressed(root/(title+'_analytic_GT.npz'),normal=gt,valid=mask)
  raw=np.asarray(predict(image),dtype=np.float32);assert raw.ndim==3 and raw.shape[-1]==3 and np.isfinite(raw).all();raw=cv2.resize(raw,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR);pred=raw@matrix.T;pred/=np.maximum(np.linalg.norm(pred,axis=-1,keepdims=True),1e-8);mae=float(np.degrees(np.arccos(np.clip(np.sum(pred*gt,-1)[mask],-1,1))).mean());results.append({'fixture':title,'MAE_deg':mae,'pixels':int(mask.sum())})
  if title=='sphere':
   for k,axis in [(0,'X'),(1,'Y')]:
    for sign in [-1,1]:
     m=mask&(sign*gt[...,k]>.5);value=float(pred[...,k][m].mean());signs[axis+str(sign)]={'mean_prediction':value,'expected_sign':sign,'passed':sign*value>.05}
 passed=np.mean([r['MAE_deg'] for r in results])<40 and all(x['passed'] for x in signs.values())
 doc={'status':'passed' if passed else 'source_contract_needs_review','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'registration_sha256':hashlib.sha256(Path(registration).read_bytes()).hexdigest(),'matrix_tested_only':matrix.tolist(),'fixtures':results,'sphere_cardinal_sign_probes':signs,'acceptance':'Mean3analyticfixtureMAEs<40deg and all4sphere cardinal signmeans match by>0.05; rejectcandidate onfailure, do not search alternatives.','benchmark_GT_or_images_used':False,'benchmark_scoring_allowed':bool(passed)};(root/'coordinate_acceptance.json').write_text(json.dumps(doc,indent=2)+'\n');return doc
