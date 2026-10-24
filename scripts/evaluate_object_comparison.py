"""Frozen object-only comparison; resumable per image, no training."""
import sys,json,hashlib,random,gc,csv,argparse,time,math,ast,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.research_io import sha256,write_json
from src.gpu_job_lock import gpu_job_lock
from src.mesh_normal_gt import gt_angular_gap
from src.angular_metrics import metrics_from_angles
from src.gso_boundary_protocol import angular_band
from src.models.model_unet import FaceNormalUNet_B
from src.parameterization import one_step_endpoint
from src.noise_protocol import image_noise_seed
from src.lightrig_adapter import load_models
OUT=Path(os.environ.get('OBJECT_COMPARISON_OUTPUT',str(ROOT/'outputs/object_comparison_20261023')));OUT.mkdir(parents=True,exist_ok=True)
MP=Path(os.environ.get('OBJECT_COMPARISON_MANIFEST',str(ROOT/'data/object_comparison_20261023/comparison_manifest_private.json')))
MAN=json.loads(MP.read_text());MSHA=sha256(MP)
assert MSHA==MP.with_suffix('.json.sha256').read_text().strip()
LOCK=ROOT/'outputs/lightrig_calibration_v1/coordinate_lock.json';assert sha256(LOCK)==MAN['coordinate_lock_sha256']
COORD=json.loads(LOCK.read_text());assert COORD['status']=='closed'
def progress(**kw):write_json(OUT/'progress.json',{'updated_unix':time.time(),**kw})
def metrics(pred,r):
 assert sha256(r['gt_path'])==r['gt_sha256']
 with np.load(r['gt_path']) as z:
  valid=z['valid'].astype(bool);gt=z['normal'].copy();labels={'main':gt}
  if r['domain']=='gso':labels.update(face=z['normal_face'].copy(),crease45=z['normal_crease45_diagnostic'].copy())
 pred=np.asarray(pred,np.float32)
 if pred.shape[:2]!=valid.shape:
  pred=F.interpolate(torch.from_numpy(pred.copy()).permute(2,0,1)[None],size=valid.shape,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
 norms=np.linalg.norm(pred,axis=-1,keepdims=True)
 if not valid.any() or not np.isfinite(pred[valid]).all() or (norms[...,0][valid]<1e-8).any():raise ValueError('Invalid prediction on GT-valid area')
 pred=pred/np.maximum(norms,1e-12);gt=gt/np.maximum(np.linalg.norm(gt,axis=-1,keepdims=True),1e-12)
 band=angular_band(gt,valid);dest={}
 for label,n in labels.items():
  row=metrics_from_angles(gt_angular_gap(pred,n,valid));row.update(valid_pixels=int(valid.sum()),boundary_pixels=int(band.sum()),outside_pixels=int((valid&~band).sum()),boundary_mean=float(gt_angular_gap(pred,n,band).mean()) if band.any() else None,outside_mean=float(gt_angular_gap(pred,n,valid&~band).mean()) if (valid&~band).any() else None)
  dest[label]=row
 return dest,pred
def rowpath(model,seed,r):return OUT/'per_image_private'/model/str(seed)/(r['id']+'.json')
def save_row(model,seed,r,values,extra=None):
 row={'manifest_sha256':MSHA,'coordinate_lock_sha256':MAN['coordinate_lock_sha256'],'model':model,'seed':seed,'id':r['id'],'image_key':r['image_key'],'domain':r['domain'],'object':r['object'],'metrics':values,**(extra or {})};write_json(rowpath(model,seed,r),row);return row
def cached(model,seed,r):
 p=rowpath(model,seed,r)
 if not p.exists():return None
 row=json.loads(p.read_text());assert row['manifest_sha256']==MSHA and row['coordinate_lock_sha256']==MAN['coordinate_lock_sha256'];return row
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(s):random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def official_vase_preview(model,modelname):
 if set(r['domain'] for r in MAN['records'])=={'diligent'}:return
 source=ROOT/'external/pinned_source/Neural-LightRig-db472e191e7abd3115ced0d5c72d61b75a8412ed/assets/examples/clay_vase.png'
 if not source.exists():return
 image=Image.open(source).convert('RGBA');image=Image.alpha_composite(Image.new('RGBA',image.size,'white'),image).convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
 rgb=torch.from_numpy(np.array(image).transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
 noise=torch.zeros_like(rgb) if modelname=='R' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,'official/clay_vase')))
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
 pred=F.normalize(one_step_endpoint(modelname,raw.float(),noise),dim=1)[0].permute(1,2,0).cpu().numpy()
 dest=OUT/'official_vase_no_gt';dest.mkdir(exist_ok=True);np.savez_compressed(dest/(modelname+'.npz'),normal=pred)
 write_json(dest/(modelname+'.json'),{'source_sha256':sha256(source),'checkpoint':MAN['checkpoints'][modelname],'seed':42,'gt_available':False,'accuracy_metrics_computed':False})
def ours():
 for modelname in ['R','V','Xn']:
  info=MAN['checkpoints'][modelname];assert sha256(info['path'])==info['sha256'];ck=torch.load(info['path'],map_location='cpu',weights_only=False)
  assert ck['step']==52500 and ck['signature']['method']==modelname and ck['signature']['condition_dropout']==0
  model=FaceNormalUNet_B().eval().cuda();model.load_state_dict(ck['model'],strict=True);del ck
  for index,r in enumerate(MAN['records']):
   assert sha256(r['ours_rgb'])==r['ours_rgb_sha256']
   rgb=np.array(Image.open(r['ours_rgb']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR));rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
   for seed in ([42] if modelname=='R' else range(42,50)):
    if cached(modelname,seed,r):continue
    noise=torch.zeros_like(rgb) if modelname=='R' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
    pred=one_step_endpoint(modelname,raw.float(),noise)[0].permute(1,2,0).cpu().numpy();values,pred=metrics(pred,r)
    extra={}
    if seed==42 and r['id'] in MAN['case_ids']:
     p=OUT/'case_predictions_private'/modelname/(r['id']+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,normal=pred);extra={'prediction_path':str(p),'prediction_sha256':sha256(p)}
    save_row(modelname,seed,r,values,extra)
   progress(phase='ours',model=modelname,images_completed=index+1,images_total=len(MAN['records']));print(modelname,index+1,len(MAN['records']),flush=True)
  official_vase_preview(model,modelname)
  assert sha256(info['path'])==info['sha256'];del model;gc.collect();torch.cuda.empty_cache();summarize()
def nlr():
 module,a,b=load_models();matrix=np.array(COORD['nlr_to_common'])
 for modelname,seed,records in [('NeuralLightRig',511,MAN['records'])]+[('NeuralLightRig_variability',s,[r for r in MAN['records'] if r['id'] in MAN['nlr_variability_ids']]) for s in [511,512,513]]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
  for index,r in enumerate(records):
   statepath=OUT/'rng_private'/modelname/str(seed)/(r['id']+'.pt');previous=cached(modelname,seed,r)
   if previous:
    assert statepath.is_file() and sha256(statepath)==previous['post_rng_sha256'];restore(torch.load(statepath,map_location='cpu',weights_only=False));continue
   assert sha256(r['nlr_rgba'])==r['nlr_rgba_sha256'];dest=OUT/'nlr_normals_private'/modelname/str(seed)/(r['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True);concat=dest.with_suffix('.concat.png')
   module.inference_impl(a,b,r['nlr_rgba'],str(concat))
   with Image.open(concat) as im:im.crop((2048,0,2560,512)).convert('RGB').save(dest)
   pred=np.array(Image.open(dest)).astype(np.float32)/127.5-1;pred=pred@matrix.T;values,_=metrics(pred,r)
   statepath.parent.mkdir(parents=True,exist_ok=True);torch.save(rng_state(),statepath)
   save_row(modelname,seed,r,values,{'prediction_path':str(dest),'prediction_sha256':sha256(dest),'post_rng_sha256':sha256(statepath)})
   concat.unlink();progress(phase='nlr',model=modelname,seed=seed,images_completed=index+1,images_total=len(records));print(modelname,seed,index+1,len(records),flush=True)
   if (index+1)%20==0:summarize()
  summarize()
def summarize():
 summaries=[];seedstats=[]
 for model in ['R','V','Xn','NeuralLightRig','NeuralLightRig_variability']:
  seeds=[42] if model=='R' else list(range(42,50)) if model in ['V','Xn'] else [511] if model=='NeuralLightRig' else [511,512,513]
  for domain in sorted(set(r['domain'] for r in MAN['records'])):
   records=[r for r in MAN['records'] if r['domain']==domain and (model!='NeuralLightRig_variability' or r['id'] in MAN['nlr_variability_ids'])]
   image_rows=[];per_seed={s:[] for s in seeds}
   for r in records:
    rows=[cached(model,s,r) for s in seeds]
    if any(x is None for x in rows):continue
    labels={}
    for label in rows[0]['metrics']:
     labels[label]={k:float(np.mean([x['metrics'][label][k] for x in rows])) if rows[0]['metrics'][label][k] is not None else None for k in rows[0]['metrics'][label]}
    image_rows.append({'object':r['object'],'labels':labels})
    for seed,row in zip(seeds,rows):per_seed[seed].append(row['metrics']['main']['mean'])
   if not image_rows:continue
   complete=len(image_rows)==len(records);objects=sorted(set(r['object'] for r in image_rows))
   for label in image_rows[0]['labels']:
    for agg in ['image_macro','object_macro']:
     row={'model':model,'domain':domain,'gt':label,'aggregation':agg,'complete':complete,'images_done':len(image_rows),'images_expected':len(records),'objects_done':len(objects)}
     for key in ['mean','median','lt_11_25','lt_22_5','lt_30','boundary_mean','outside_mean']:
      vals=[r['labels'][label][key] for r in image_rows if r['labels'][label][key] is not None]
      if agg=='object_macro':
       vals=[float(np.mean([r['labels'][label][key] for r in image_rows if r['object']==o and r['labels'][label][key] is not None])) for o in objects if any(r['object']==o and r['labels'][label][key] is not None for r in image_rows)]
      row[key]=float(np.mean(vals)) if vals else None
     row['boundary_images']=sum(r['labels'][label]['boundary_mean'] is not None for r in image_rows);summaries.append(row)
   means=[float(np.mean(per_seed[s])) for s in seeds];seedstats.append({'model':model,'domain':domain,'complete':complete,'seeds':seeds,'image_macro_mae_by_seed':means,'sd_ddof1':float(np.std(means,ddof=1)) if len(means)>1 else None})
 write_json(OUT/'summary.json',{'manifest_sha256':MSHA,'rows':summaries,'seed_variability':seedstats,'arb_overlap_note':MAN['arb_overlap_note'],'metric_note':'mean/median/threshold fractions calculated within GT-valid image pixels, seeds averaged within image, then image/object macro. Partial rows explicitly diagnostic only; no full comparison until complete.'})
 if summaries:
  with (OUT/'comparison_table.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['ours','nlr','all','summarize'],default='all');args=parser.parse_args()
 if args.phase=='summarize':summarize();return
 write_json(OUT/'run_receipt.json',{'manifest_sha256':MSHA,'coordinate_lock_sha256':MAN['coordinate_lock_sha256'],'evaluator_sha256':sha256(__file__),'shared_metrics_sha256':sha256(ROOT/'src/angular_metrics.py'),'angle_code_sha256':sha256(ROOT/'src/mesh_normal_gt.py'),'adapter_sha256':sha256(ROOT/'src/lightrig_adapter.py'),'phase':args.phase,'started_unix':time.time(),'training':False})
 try:
  with gpu_job_lock(),torch.inference_mode():
   if args.phase in ['ours','all']:ours()
   if args.phase in ['nlr','all']:nlr()
  summarize();progress(status='completed',phase=args.phase,training_started=False)
 except Exception as e:
  write_json(OUT/'failure.json',{'error':repr(e),'manifest_sha256':MSHA,'phase':args.phase,'time':time.time()});progress(status='failed',error=repr(e));raise
if __name__=='__main__':main()
