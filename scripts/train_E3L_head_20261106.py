"""Fixed-exposure MoGe-L normal-head-only control; backbone/neck remain frozen."""
from pathlib import Path
import sys,os,json,hashlib,random,time,argparse,gc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src.research_io import sha256,write_json,now
from src.moge_bridge_20261106 import load_moge,normal_only,common256
from src.e3_native_cache_20261106 import NativeCache
from src.gpu_job_lock import gpu_job_lock
CONFIG=Path(os.environ['FACEFLUX_STAGE_CONFIG'])
def read(p):return json.loads(Path(p).read_text())
def state_hash(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def frozen_hash(model):return state_hash({k:v for k,v in model.state_dict().items() if not k.startswith('normal_head.')})
def validate(model,records):
 cache=NativeCache(records,4);scores={};model.eval()
 with torch.inference_mode():
  for i,r in enumerate(records):
   rgb,gt,mask=cache.get(i);raw=normal_only(model,torch.from_numpy(rgb).cuda(),True);pred=common256(raw,'MoGe-L');target=F.normalize(torch.from_numpy(gt).cuda(),dim=-1);m=torch.from_numpy(mask).cuda();angle=torch.rad2deg(torch.acos((pred[m]*target[m]).sum(-1).clamp(-1,1))).mean();scores.setdefault(r['scene'],[]).append(float(angle))
 model.normal_head.train();return sum(sum(x)/len(x) for x in scores.values())/len(scores)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--group',choices=['E3_L'],default='E3_L');parser.add_argument('--smoke',action='store_true');parser.add_argument('--smoke-reference',action='store_true');parser.add_argument('--resume-smoke',action='store_true');a=parser.parse_args();cfg=read(CONFIG);reg=read(ROOT/cfg['registration_path']);assert sha256(CONFIG)==reg['config_sha256'] and sha256(__file__)==reg['trainer_sha256'];assert all(sha256(ROOT/p)==h for p,h in reg['source_sha256'].items())
 assert sha256(ROOT/cfg['training_manifest'])==cfg['training_manifest_sha256'] and sha256(ROOT/cfg['validation_manifest'])==cfg['validation_manifest_sha256']
 rows=[r for r in read(ROOT/cfg['training_manifest'])['records'] if r['split']=='train'];val=[r for r in read(ROOT/cfg['validation_manifest'])['records'] if r['split']=='val'];assert len(rows)==29976 and len({r['scene'] for r in rows})==10000 and len(val)==594;assert not {r['scene'] for r in rows}&{r['scene'] for r in val};assert not any('/test/' in r['path'] for r in rows+val)
 torch.set_num_threads(4);torch.manual_seed(1247);torch.cuda.manual_seed_all(1247);random.seed(1247);np.random.seed(1247);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 full_precision=cfg.get('acceptance_dtype')=='float32'
 if full_precision:torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 model,_=load_moge('L');model.requires_grad_(False);model.normal_head.requires_grad_(True);model.eval();model.normal_head.train();frozen=frozen_hash(model);initial=state_hash(model.normal_head.state_dict());parameters=list(model.normal_head.parameters());optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scaler=torch.amp.GradScaler('cuda',enabled=not full_precision,init_scale=1024);rng=np.random.default_rng(10247);cache=NativeCache(rows,16);curve=[];step=0;start=time.perf_counter();retries=0
 base=ROOT/cfg['output_root'];base=base/'smoke_generations'/sha256(__file__)[:12] if a.smoke else base;job=base/(('reference_' if a.smoke_reference else '')+'E3_L_s1247');job.mkdir(parents=True,exist_ok=True);last=job/'last_local.pth';end=(4 if a.smoke_reference or a.resume_smoke else 3) if a.smoke else cfg['total_updates']
 signature={'config_sha256':sha256(CONFIG),'trainer_sha256':sha256(__file__),'source_sha256':reg['source_sha256'],'seed':1247,'smoke':a.smoke,'torch':torch.__version__,'base_weights_sha256':cfg['base_weights_sha256']};sp=job/'signature_private.json'
 if sp.exists():assert read(sp)==signature
 else:write_json(sp,signature)
 if last.exists():
  ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['signature']==signature;model.normal_head.load_state_dict(ck['head'],strict=True);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler']);rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);step=ck['step'];curve=ck['curve'];retries=ck['overflow_retries'];del ck
  if step>=end:print('already complete',step,flush=True);return
  log=job/'training_log.jsonl'
  if log.exists():log.write_text(''.join(l+'\n' for l in log.read_text().splitlines() if json.loads(l)['update']<=step))
 write_json(job/'initialization.json',{'head_initial_sha256':initial,'frozen_state_sha256':frozen,'trainable_parameters':sum(p.numel() for p in parameters),'parameters_total':sum(p.numel() for p in model.parameters()),'signature':signature})
 def save():
  temp=last.with_suffix('.partial');torch.save({'head':model.normal_head.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'curve':curve,'overflow_retries':retries,'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'signature':signature},temp);temp.replace(last)
 import utils3d
 try:
  write_json(job/'summary.json',{'status':'running','target_updates':end,'smoke':a.smoke,'started_at':now()})
  with (job/'training_log.jsonl').open('a') as log:
   while step<end:
    nextstep=step+1;factor=min(1.,nextstep/cfg['warmup_updates']) if nextstep<=cfg['stable_end'] else max(0.,(cfg['total_updates']-nextstep)/(cfg['total_updates']-cfg['stable_end']))
    for pg in optimizer.param_groups:pg['lr']=cfg['learning_rate']*factor
    groups=[]
    for _ in range(4):
     rng.random();group=[]
     for _ in range(4):group.append((int(rng.integers(len(rows))),bool(rng.random()<.5)))
     groups.append(group)
    attempt=0
    while True:
     optimizer.zero_grad(set_to_none=True);loss_value=0.
     for group in groups:
      arrays=[]
      for index,flip in group:
       rgb,gt,mask=cache.get(index)
       if flip:rgb=rgb[:,:,::-1].copy();gt=gt[:,::-1,:].copy();gt[:,:,0]*=-1;mask=mask[:,::-1].copy()
       arrays.append((rgb,gt,mask))
      denominator=sum(int(m.sum()) for _,_,m in arrays)
      for rgb,gt,mask in arrays:
       image=torch.from_numpy(rgb).cuda();target=F.normalize(torch.from_numpy(gt).cuda(),dim=-1);valid=torch.from_numpy(mask).cuda();prediction=common256(normal_only(model,image,use_fp16=not full_precision),'MoGe-L');angles=utils3d.pt.angle_between(prediction[valid],target[valid]);loss=angles.square().sum()/denominator/4;assert torch.isfinite(loss);scaler.scale(loss).backward();loss_value+=float(loss.detach())
     scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
     if torch.isfinite(norm):scaler.step(optimizer);scaler.update();break
     assert not full_precision and attempt<4,'Nonfinite head gradient';scaler.step(optimizer);scaler.update();attempt+=1;retries+=1
    assert all(p.grad is None and not p.requires_grad for n,p in model.named_parameters() if not n.startswith('normal_head.'));step=nextstep;record={'update':step,'squared_angular_loss_rad2':loss_value,'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],'overflow_retries':retries};log.write(json.dumps(record)+'\n');log.flush()
    if step==1 or step%10==0 or step==end:progress={'status':'training','group':'E3_L','update':step,'end_update':end,'elapsed_seconds':time.perf_counter()-start,'updated_at':now()};write_json(job/'progress.json',progress);print(json.dumps(progress),flush=True)
    if not a.smoke and step%2500==0:curve.append({'update':step,'Arb_val_object_MAE_deg':validate(model,val)});write_json(job/'learning_curve.json',{'points':curve,'selection':'diagnostic only, fixed final50k checkpoint'})
    if step%250==0 or step==end:save()
  assert frozen_hash(model)==frozen
  result={'status':'completed','last_update':step,'smoke':a.smoke,'frozen_state_unchanged':True,'final_checkpoint_sha256':sha256(last),'elapsed_seconds':time.perf_counter()-start,'overflow_retries':retries};write_json(job/'summary.json',result);write_json(job/'progress.json',result)
 except Exception as error:write_json(job/'failure.json',{'error':repr(error),'update':step,'updated_at':now()});raise
if __name__=='__main__':
 if '--isolated-preflight' in sys.argv:
  assert '--smoke' in sys.argv and os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
  sys.argv.remove('--isolated-preflight');main()
 else:
  with gpu_job_lock():main()
