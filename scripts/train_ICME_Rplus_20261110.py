"""Registered R+ normal fits: final50k only, teacher loss only, encoder warmfreeze and EMA."""
from pathlib import Path
import os,sys,json,time,random,signal,argparse,copy,hashlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from src.research_io import write_json,sha256,now
from src.icme_rplus_20261110 import make_model,TeacherCache,distill,groups
from src.parameterization import parameterization_loss
from src.e_models_20261106 import evaluate
from src.gpu_job_lock import gpu_job_lock

def read(p):return json.loads(Path(p).read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--reference',action='store_true');args=parser.parse_args();path=Path(args.config);cfg=read(path);registration=read(R/cfg['registration_path']);assert sha256(path)==registration['config_sha256'];assert all(sha256(R/k)==v for k,v in registration['source_sha256'].items());rows=[r for r in read(R/cfg['training_manifest'])['records'] if r['split']=='train'];assert len(rows)==29976 and len({r['scene'] for r in rows})==10000;assert sha256(R/cfg['training_manifest'])==cfg['training_manifest_sha256'];assert not {r['scene'] for r in rows}&{r['scene'] for r in read(R/cfg['validation_manifest'])['records'] if r['split']=='val'};out=R/cfg['output_root']
 if args.smoke:out=out/'acceptance'/('reference' if args.reference else 'split')
 out.mkdir(parents=True,exist_ok=True);seed=cfg['seed'];torch.set_num_threads(4);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 if cfg.get('acceptance_precision')=='float32':torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 model=make_model(cfg);model.features.set_gradient(cfg['unfreeze_encoder']);pg=groups(model,cfg);optimizer=torch.optim.AdamW(pg,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);parameters=[p for group in pg for p in group['params']];named=dict(model.named_parameters());tracked=[n for n,p in named.items() if any(p is x for x in parameters)];ema={n:named[n].detach().clone() for n in tracked};generator=torch.Generator(device='cuda').manual_seed(seed+7000);rng=np.random.default_rng(seed+9000);cache=TeacherCache(rows,cfg);step=0;curve=[];elapsed_prior=0;unfreeze_at=cfg['freeze_encoder_updates'];end=cfg['total_updates'];precision=cfg.get('acceptance_precision','bfloat16');amp=precision!='float32';last=out/'last_local.pth';signature={'config_sha256':sha256(path),'registration_sha256':sha256(R/cfg['registration_path']),'smoke':args.smoke,'source_sha256':registration['source_sha256']};stop={'requested':False}
 if args.smoke:unfreeze_at=1;end=(8 if precision=='bfloat16' else 4) if args.reference or args.resume else 2
 if args.resume:
  ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['signature']==signature;model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);ema={n:v.cuda() for n,v in ck['ema'].items()};step=ck['step'];curve=ck.get('curve',[]);rng.bit_generator.state=ck['numpy_rng'];generator.set_state(ck['generator']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_global_rng']);elapsed_prior=ck['elapsed_seconds'];del ck
  log=out/'training_log.jsonl'
  if log.exists():log.write_text(''.join(x+'\n' for x in log.read_text().splitlines() if json.loads(x)['update']<=step))
 signal.signal(signal.SIGTERM,lambda *_:stop.update(requested=True));signal.signal(signal.SIGINT,lambda *_:stop.update(requested=True));started=time.monotonic();encoder_grad_seen=False;encoder_update_seen=False;model.train();torch.cuda.reset_peak_memory_stats()
 def elapsed():return elapsed_prior+time.monotonic()-started
 def factor(n):return min(1.,n/cfg['warmup_updates']) if n<=cfg['stable_end'] else max(0.,(cfg['total_updates']-n)/(cfg['total_updates']-cfg['stable_end']))
 def snapshot(name='last_local.pth'):
  dest=out/name;tmp=dest.with_suffix('.tmp');torch.save({'model':model.state_dict(),'ema':ema,'optimizer':optimizer.state_dict(),'step':step,'curve':curve,'numpy_rng':rng.bit_generator.state,'numpy_global_rng':np.random.get_state(),'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'signature':signature,'elapsed_seconds':elapsed()},tmp);tmp.replace(dest)
 write_json(out/'initialization.json',{'seed':seed,'optimizer_groups':[{'layer_id':p['layer_id'],'base_lr':p['base_lr'],'parameters':sum(x.numel() for x in p['params'])} for p in pg],'EMA_decay':cfg['ema_decay'],'encoder_mode':'eval retained; gradients explicitly enabled after warmfreeze','teacher_is_network_input':False,'teacher_contract_sha256':cfg['teacher_contract_sha256'],'sampling':cfg['sampling_description'],'signature':signature,'updated_at':now()})
 try:
  with (out/'training_log.jsonl').open('a') as log:
   while step<end:
    nextstep=step+1;encoder_on=cfg['unfreeze_encoder'] and nextstep>unfreeze_at;model.features.set_gradient(encoder_on);optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:group['lr']=group['base_lr']*factor(nextstep)
    infos=[];enc_before=None
    if args.smoke and encoder_on:enc_before=model.features.encoder.blocks[-1].attn.qkv.weight.detach().clone()
    for micro in range(cfg['gradient_accumulation']):
     batch=[];rng.random()
     for _ in range(cfg['microbatch']):
      rgb,gt,mask,teacher=[x.copy() for x in cache.get(int(rng.integers(len(rows))))]
      if rng.random()<.5:rgb=rgb[:,:,::-1].copy();gt=gt[:,:,::-1].copy();gt[0]*=-1;mask=mask[:,::-1].copy();teacher=teacher[:,:,::-1].copy();teacher[0]*=-1
      batch.append((rgb,gt,mask,teacher))
     rgb,gt,mask,teacher=[torch.from_numpy(np.stack([x[k] for x in batch])).cuda() for k in range(4)];assert rgb.shape[1]==3 and not teacher.requires_grad;noise=torch.randn(gt.shape,device='cuda',generator=generator);t=torch.rand(len(batch),device='cuda',generator=generator)*.95;z=(1-t[:,None,None,None])*noise+t[:,None,None,None]*gt
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):pred=model(z,t,rgb)
     gt_loss,info=parameterization_loss('Xn',pred,z,gt,noise,t,mask,.25,.25);teacher_loss=distill(pred,teacher,mask);loss=gt_loss+cfg['distill_weight']*teacher_loss;assert torch.isfinite(loss),'Nonfinite training loss';(loss/cfg['gradient_accumulation']).backward();infos.append({**info,'distill':float(teacher_loss.detach()),'loss':float(loss.detach())})
    if encoder_on:assert all(p.requires_grad for p in model.features.encoder.parameters()) and model.features.encoder.blocks[-1].attn.qkv.weight.grad is not None
    else:assert all(p.grad is None for p in model.features.encoder.parameters())
    norm=torch.nn.utils.clip_grad_norm_(parameters,1.);assert torch.isfinite(norm),'Nonfinite gradient';optimizer.step();step=nextstep
    with torch.no_grad():
     torch._foreach_mul_(list(ema.values()),cfg['ema_decay']);torch._foreach_add_(list(ema.values()),[named[n].detach() for n in ema],alpha=1-cfg['ema_decay'])
    if enc_before is not None:
     encoder_grad_seen=encoder_grad_seen or bool(model.features.encoder.blocks[-1].attn.qkv.weight.grad.abs().max()>0);encoder_update_seen=encoder_update_seen or not torch.equal(enc_before,model.features.encoder.blocks[-1].attn.qkv.weight)
     write_json(out/'encoder_gradient_acceptance.json',{'update':step,'nonzero_encoder_gradient_seen':encoder_grad_seen,'encoder_parameter_update_seen':encoder_update_seen,'note':'Zero-initialized semantic adapters can delay backbone gradients in the first few fresh-start smoke updates.'})
    log.write(json.dumps({'update':step,'lr_head':optimizer.param_groups[0]['lr'],'encoder_unfrozen':encoder_on,'gradient_norm_before_clip':float(norm),**{k:float(np.mean([x[k] for x in infos])) for k in infos[0]}})+'\n');log.flush()
    if step%10==0 or step==1 or args.smoke:progress={'status':'training','run':cfg['run'],'update':step,'end_update':end,'elapsed_seconds':elapsed(),'encoder_unfrozen':encoder_on,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'updated_at':now()};write_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    if not args.smoke and step%2500==0:
     validation=[r for r in read(R/cfg['validation_manifest'])['records'] if r['split']=='val'];score,_=evaluate(model,'Xn',{'arb_validation_retrain':validation},[42,43],0,'cuda',cfg);curve.append({'update':step,'scope':'raw Arb validation diagnostic only; no checkpoint selection','metrics':score});write_json(out/'learning_curve.json',{'points':curve,'fixed_final_update':50000});model.train()
    if step%250==0 or step==end or stop['requested']:snapshot()
    if not args.smoke and step==25000:
     snapshot('diagnostic_25000.pth');write_json(out/'gate_status.json',{'phase':'requires_25000_GSO_diagnostic','raw_weights_only':True,'threshold_deg':cfg['early_stop_threshold'],'updated_at':now()});write_json(out/'progress.json',{'status':'awaiting_25000_diagnostic','update':step,'end_update':end,'elapsed_seconds':elapsed(),'updated_at':now()});return
    if stop['requested']:write_json(out/'summary.json',{'status':'paused_by_user','last_update':step,'checkpoint_sha256':sha256(last),'config_sha256':sha256(path),'updated_at':now()});write_json(out/'progress.json',read(out/'summary.json'));return
   if args.smoke and cfg['unfreeze_encoder']:assert encoder_grad_seen and encoder_update_seen,'No encoder gradient/update across complete eight-step BF16 smoke'
   if not args.smoke and not (out/'final_50000.pth').exists():os.link(last,out/'final_50000.pth')
   result={'status':'completed','run':cfg['run'],'last_update':step,'smoke':args.smoke,'config_sha256':sha256(path),'final_checkpoint_sha256':sha256(last),'elapsed_seconds':elapsed(),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'teacher_network_input':False,'EMA_saved':True,'scope':'fit only; raw/EMA five-panel evaluation pending','test_images_scored':0,'updated_at':now()};write_json(out/'summary.json',result);write_json(out/'progress.json',result);print(json.dumps(result),flush=True)
 except Exception as e:write_json(out/'failure.json',{'status':'failed','error':repr(e),'last_completed_update':step,'config_sha256':sha256(path),'updated_at':now()});write_json(out/'summary.json',read(out/'failure.json'));raise
if __name__=='__main__':
 with gpu_job_lock():main()
