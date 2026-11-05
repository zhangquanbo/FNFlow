"""Co-run bounded-memory C inference ablations; preserve ongoing NLR process and settings."""
from pathlib import Path
import sys,json,time,fcntl,os
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
from scripts.train_object_dino_C_20261029 import make_model
from scripts.evaluate_fresh_ab import assert_registration,metrics
AB=R/'outputs/P3_object_dino_evaluation_fresh_20261027';C=R/'outputs/P3_C_train_20261029';OUT=R/'outputs/C_TTA_20261029';REG=R/'provenance/C_TTA_registration_20261029.json'
PANELS=['primary','diligent','luces'];VARIANTS=['single42','flip_TTA42','mean4','flip_TTA_mean4']
def read(p):return json.loads(Path(p).read_text())
def unit(x):
 n=torch.linalg.vector_norm(x.float(),dim=0,keepdim=True);assert bool(torch.isfinite(n).all()) and bool((n>1e-8).all());return x.float()/n
def reflection(x):
 y=x.flip(-1).clone();y[0]*=-1;return y
def main():
 OUT.mkdir(parents=True,exist_ok=True);assert_registration();reg=read(REG);rs=sha256(REG)
 assert sha256(__file__)==reg['evaluator_sha256'] and sha256(C/'C_Xn_s1247/last_local.pth')==reg['checkpoint_sha256']
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
 with (OUT/'run.lock').open('a') as lock,torch.inference_mode():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Limit this process, not NLR. PyTorch allocator cap leaves room for NLR peaks.
  torch.cuda.set_per_process_memory_fraction(.14)
  free,total=torch.cuda.mem_get_info()
  if free<2*1024**3:raise RuntimeError('Less than2GiB free; no NLR process was paused or signalled.')
  model=make_model().eval();ck=torch.load(C/'C_Xn_s1247/last_local.pth',map_location='cpu',weights_only=False);s=model.state_dict();s.update(ck['trainable_model']);model.load_state_dict(s,strict=True);del ck,s
  torch.cuda.reset_peak_memory_stats();started=time.monotonic();max_control_error=0.
  for panel in PANELS:
   mp=AB/'manifests_private'/(panel+'.json');assert sha256(mp)==reg['manifest_sha256'][panel];records=read(mp)['records']
   for i,r in enumerate(records):
    paths={v:OUT/'per_image_private'/panel/v/(r['id']+'.json') for v in VARIANTS}
    if all(p.exists() for p in paths.values()):
     for p in paths.values():assert read(p)['registration_sha256']==rs
     continue
    assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256']
    assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
    rgb=np.asarray(Image.open(r['white']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR));rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
    originals=[];flipped=[]
    for seed in [42,43,44,45]:
     noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
     with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(noise,torch.zeros(1,device='cuda'),rgb)[0].float()
     originals.append(pred)
     with torch.autocast('cuda',dtype=torch.bfloat16):fp=model(reflection(noise[0])[None],torch.zeros(1,device='cuda'),rgb.flip(-1))[0].float()
     flipped.append(reflection(fp))
    us=[unit(x) for x in originals];fs=[unit(x) for x in flipped]
    outputs={'single42':originals[0],'flip_TTA42':unit(us[0]+fs[0]),'mean4':unit(torch.stack(us).mean(0)),'flip_TTA_mean4':unit(torch.stack(us+fs).mean(0))}
    with np.load(r['gt_path']) as z:gt=z['normal'];valid=z['valid'].astype(bool)
    values={v:metrics(x.permute(1,2,0).cpu().numpy(),gt,valid) for v,x in outputs.items()}
    reference=read(C/'evaluation/per_image_private'/panel/'42'/(r['id']+'.json'))['metrics']['mean_deg'];delta=values['single42']['mean_deg']-reference
    # Same checkpoint, batch1, dtype and image-key noise; any drift is recorded, never hidden.
    max_control_error=max(max_control_error,abs(delta));assert abs(delta)<1e-4,(panel,r['id'],'Single42 control drift',delta)
    for variant,path in paths.items():write_json(path,{'model':'C','variant':variant,'object':r['object'],'id':r['id'],'image_key':r['image_key'],'elevation_bin':r['elevation_bin'],'registration_sha256':rs,'metrics':values[variant],'single42_control_delta_deg':delta})
    if i%10==0 or i+1==len(records):
     status={'phase':'running','panel':panel,'images':i+1,'total_images':len(records),'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU':torch.cuda.get_device_name(),'NLR_pause_requested':False,'max_single42_control_drift_deg':max_control_error,'updated_at':now()};write_json(OUT/'status.json',status);print(json.dumps(status),flush=True)
    del originals,flipped,outputs,us,fs,rgb,gt,valid
    time.sleep(.02)
  results={}
  for panel in PANELS:
   records=read(AB/'manifests_private'/(panel+'.json'))['records'];models={}
   for variant in VARIANTS:
    bins=[0,15,30,45,60] if panel=='primary' else [None];means={}
    for angle in bins:
     rr=[r for r in records if r['elevation_bin']==angle];objects=sorted({r['object'] for r in rr});per=[]
     for obj in objects:per.append(float(np.mean([read(OUT/'per_image_private'/panel/variant/(r['id']+'.json'))['metrics']['mean_deg'] for r in rr if r['object']==obj])))
     means[str(angle)]=float(np.mean(per))
    models[variant]={'MAE_deg':float(np.mean(list(means.values()))),'by_pitch':means,'images':len(records)}
   models['C_original8_error_mean_reference']={'MAE_deg':read(C/'evaluation/summary.json')['panels'][panel]['equal_pitch_object_macro_mae_deg'],'note':'Mean of8inference errors, not vector ensemble.'}
   results[panel]=models
  write_json(OUT/'summary.json',{'status':'completed','updated_at':now(),'registration_sha256':rs,'panels':results,'NLR_paused':False,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started,'max_single42_control_drift_deg':max_control_error,'test_images_scored':0})
  write_json(OUT/'status.json',{'phase':'completed','updated_at':now(),'test_images_scored':0});print(json.dumps(results),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now(),'NLR_was_not_paused':True});write_json(OUT/'status.json',{'phase':'failed','error':str(e),'updated_at':now()});raise
