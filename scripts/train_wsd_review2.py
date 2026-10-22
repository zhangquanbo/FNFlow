"""Independent WSD trunk/cooldown jobs; parent optimizer/RNG states are immutable."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,random,time
import numpy as np
import torch
from scripts.train_parameterization import ImageCache,load_records,evaluate
from scripts.prepare_training_probe import prepare as prepare_probe
from src.models.model_unet import FaceNormalUNet_B
from src.parameterization import parameterization_loss
from src.training_diagnostics import TimeLossBins,resume_segment_action
from src.wsd_protocol import wsd_factor
from src.research_io import ROOT,write_json,sha256,now


def run_name(phase,method,fork=None):
    return f'cooldown_{method}_k{fork}_s1247' if phase=='cooldown' else f'{phase}_{method}_s1247'


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['R','V','Xv','Xn'],required=True)
    p.add_argument('--phase',choices=['stable','cooldown','clip_diagnostic','smoke'],required=True)
    p.add_argument('--fork-step',type=int);p.add_argument('--stable-limit',type=int,choices=[30000,50000],default=30000)
    p.add_argument('--config',default='configs/parameterization_first_round_v2_review2.json');args=p.parse_args()
    cfg=json.loads((ROOT/args.config).read_text());rule=json.loads((ROOT/cfg['decision_rule']).read_text())
    if cfg['ray_condition'] or cfg['dino_condition'] or cfg['cfg'] or cfg['formal_budget_approved']:raise RuntimeError('WSD diagnostics input/gate changed')
    root=ROOT/cfg['output_root'];phase=args.phase;fork=args.fork_step if phase=='cooldown' else None
    if fork is not None and fork not in [*rule['initial_fork_steps'],50000]:raise RuntimeError('Unregistered cooldown fork')
    if phase=='cooldown' and fork is None:raise RuntimeError('Missing fork checkpoint')
    end=fork+rule['cooldown_updates'] if fork else args.stable_limit if phase=='stable' else cfg['clip_diagnostic_steps'] if phase=='clip_diagnostic' else 3
    if args.stable_limit==50000 or fork==50000:
        decision=json.loads((root/'budget_decision.json').read_text())
        if not decision.get('extension_required'):raise RuntimeError('50k requires preregistered extension trigger')
    data_phase='smoke' if phase=='smoke' else 'formal';object_root=ROOT/f'data/arb_mixed_{data_phase}_v1'
    hyp=load_records(ROOT/cfg['hypersim_root'],'train');obj=load_records(object_root,'train')
    for key in ['hypersim_root','nyu_validation_root']:
        if json.loads((ROOT/cfg[key]/'summary.json').read_text()).get('spatial_protocol')!='center_square_v2':raise RuntimeError('Wrong spatial cache')
    domains={'hypersim':load_records(ROOT/cfg['hypersim_root'],'val'),'arb':load_records(object_root,'val'),
             'nyuv2':load_records(ROOT/cfg['nyu_validation_root'],'val')}
    probes=prepare_probe(cfg) if phase!='smoke' else {}
    manifests=[ROOT/cfg['hypersim_root']/'private_manifest.json',object_root/'private_manifest.json',ROOT/cfg['nyu_validation_root']/'private_manifest.json']
    signature={'config_sha256':sha256(ROOT/args.config),'rule_sha256':sha256(ROOT/cfg['decision_rule']),
        'code_sha256':sha256(__file__),'shared_evaluator_sha256':sha256(ROOT/'scripts/train_parameterization.py'),
        'schedule_sha256':sha256(ROOT/'src/wsd_protocol.py'),'target_sha256':sha256(ROOT/'src/parameterization.py'),
        'model_sha256':sha256(ROOT/'src/models/model_unet.py'),'diagnostics_sha256':sha256(ROOT/'src/training_diagnostics.py'),
        'manifests_sha256':[sha256(x) for x in manifests],'method':args.method,'phase':phase,
        'training_seed':1247,'condition_dropout':0,'fork_step':fork,
        'initial_stable_limit':rule['initial_stable_endpoint'],'maximum_stable_limit':rule['extended_stable_endpoint'],
        'cooldown_updates':rule['cooldown_updates']}
    if probes:signature['training_probe_sha256']=sha256(root/'training_probe_private.json')
    parent=root/run_name('stable',args.method)/f'fork_{fork:06d}_local.pth' if fork else None
    if parent:
        if not parent.exists():raise RuntimeError('Missing immutable stable parent checkpoint')
        signature['parent_checkpoint_sha256']=sha256(parent)
    job_root=root/'smoke_generations'/sha256(__file__)[:12] if phase=='smoke' else root
    out=job_root/run_name(phase,args.method,fork);out.mkdir(parents=True,exist_ok=True)
    sig=out/'signature_private.json'
    if sig.exists():
        if json.loads(sig.read_text())!=signature:raise RuntimeError('Existing WSD job signature differs')
        summary=out/'summary.json'
        if summary.exists():
            previous=json.loads(summary.read_text())
            if previous.get('status')=='completed' and previous.get('last_update',0)>=end:return
    else:write_json(sig,signature)
    torch.set_num_threads(4);torch.manual_seed(1247);torch.cuda.manual_seed_all(1247)
    random.seed(1247);np.random.seed(1247);torch.backends.cudnn.benchmark=False
    model=FaceNormalUNet_B().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator(device='cuda').manual_seed(1247+7000);rng=np.random.default_rng(1247+9000)
    completed=0;norms=[];curve=[];time_bins=TimeLossBins(cfg['time_bin_edges']);resume=out/'last_local.pth'
    source=resume if resume.exists() else parent
    if source:
        state=torch.load(source,map_location='cpu')
        if source==parent:
            if state['step']!=fork:raise RuntimeError('Parent step mismatch')
            shared=['config_sha256','rule_sha256','code_sha256','shared_evaluator_sha256','schedule_sha256',
                    'target_sha256','model_sha256','diagnostics_sha256','manifests_sha256','training_probe_sha256','method']
            if any(state['signature'].get(k)!=signature.get(k) for k in shared):
                raise RuntimeError('Parent code/data/probe/config signature mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for k,v in values.items():
                if torch.is_tensor(v):values[k]=v.cuda()
        completed=state['step'];generator.set_state(state['generator']);rng.bit_generator.state=state['numpy_rng']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        if source==resume:
            norms=state['grad_norms'];time_bins.rows=state['time_bin_rows'];curve=state['curve']
            if resume_segment_action(completed,end,'convergence',50000 if end==50000 else None,curve)=='skip_initial':return
            log=out/'training_log.jsonl'
            if log.exists():log.write_text('\n'.join(x for x in log.read_text().splitlines() if json.loads(x)['update']<=completed)+'\n')
    caches=[ImageCache(hyp),ImageCache(obj)];started=time.perf_counter();model.train();torch.cuda.reset_peak_memory_stats()
    def save(path):
        temporary=path.with_suffix('.tmp')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':completed,
            'generator':generator.get_state(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'grad_norms':norms,'time_bin_rows':time_bins.rows,
            'curve':curve,'signature':signature},temporary);temporary.replace(path)
    write_json(out/'summary.json',{'status':'running','phase':phase,'method':args.method,'end_update':end,'fork_step':fork,'started_at':now()})
    try:
        with (out/'training_log.jsonl').open('a') as log:
            for step in range(completed+1,end+1):
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*wsd_factor(step,rule['warmup_updates'],fork,rule['cooldown_updates'])
                components=[];update_bins=TimeLossBins(cfg['time_bin_edges'])
                rng.random()  # Same condition-dropout RNG draw as the controlled p=0 baseline.
                for micro in range(cfg['gradient_accumulation']):
                    batch=[]
                    for cache in caches:
                        for _ in range(cfg['microbatch']//2):
                            rgb,gt,mask=cache.get(int(rng.integers(len(cache.records))))
                            if rng.random()<.5:
                                rgb=rgb[:,:,::-1].copy();gt=gt[:,:,::-1].copy();gt[0]*=-1;mask=mask[:,::-1].copy()
                            batch.append((rgb,gt,mask))
                    rgb=torch.from_numpy(np.stack([x[0] for x in batch])).cuda()
                    gt=torch.from_numpy(np.stack([x[1] for x in batch])).cuda();mask=torch.from_numpy(np.stack([x[2] for x in batch])).cuda()
                    noise=torch.randn(gt.shape,device='cuda',generator=generator);t=torch.rand(len(batch),device='cuda',generator=generator)*(1-cfg['time_delta'])
                    z=(1-t[:,None,None,None])*noise+t[:,None,None,None]*gt
                    if args.method=='R':z=torch.zeros_like(z);t=torch.zeros_like(t)
                    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(z,t,rgb)
                    loss,info=parameterization_loss(args.method,raw,z,gt,noise,t,mask,cfg['unit_weight'],cfg['cosine_weight'],return_per_image=True)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite WSD loss')
                    (loss/cfg['gradient_accumulation']).backward();components.append(info)
                    time_bins.add(t.cpu().tolist(),info['per_image']);update_bins.add(t.cpu().tolist(),info['per_image'])
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite WSD gradient')
                norms.append(float(grad));optimizer.step();completed=step
                row={'update':step,'phase_updates':step-(fork or 0),'lr':optimizer.param_groups[0]['lr'],
                    'gradient_norm_before_clip':float(grad),'gradient_clipped':float(grad)>cfg['gradient_clip_norm'],
                    'gradient_clip_scale':min(1.,cfg['gradient_clip_norm']/(float(grad)+1e-6)),
                    'time_loss_bins':update_bins.summary(),**{k:float(np.mean([v[k] for v in components])) for k in ['mse','unit','cosine']}}
                log.write(json.dumps(row)+'\n');log.flush()
                if step%10==0 or step==end or step==1:
                    write_json(out/'progress.json',{'status':'running','update':step,'end_update':end,'updated_at':now()})
                if phase=='stable' and step%rule['stable_evaluation_interval']==0:
                    scores,_=evaluate(model,args.method,domains,cfg['curve_inference_seeds'],0,'cuda')
                    probe,_=evaluate(model,args.method,probes,cfg['curve_inference_seeds'],0,'cuda')
                    curve.append({'update':step,'validation':scores,'training_probe':probe})
                    write_json(out/'learning_curve.json',{'status':'running','points':curve})
                if step%250==0 or step==end:save(resume)
                if phase=='stable' and step in [*rule['initial_fork_steps'],50000]:save(out/f'fork_{step:06d}_local.pth')
        limit=cfg['pilot_validation_images_per_domain'] if phase=='smoke' else 0
        scores,details=evaluate(model,args.method,domains,cfg['curve_inference_seeds'],limit,'cuda')
        probescores,_=evaluate(model,args.method,probes,cfg['curve_inference_seeds'],0,'cuda') if probes else ({},{})
        write_json(out/'per_image_private.json',details)
        result={'status':'completed','created_at':now(),'phase':phase,'method':args.method,'last_update':completed,
            'stable_fork_step':fork,'branch_optimizer_updates':completed-fork if fork else None,
            'total_budget_updates':completed,'validation':scores,'training_probe':probescores,
            'inference_seeds':cfg['curve_inference_seeds'],'gradient_updates_recorded':len(norms),
            'clip_fraction':float(np.mean(np.asarray(norms)>cfg['gradient_clip_norm'])),
            'time_loss_bins':time_bins.summary(),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.perf_counter()-started,'formal_grid_locked':True,'dino_used':False,'ray_used':False}
        write_json(out/'summary.json',result)
        if curve:write_json(out/'learning_curve.json',{'status':'completed','points':curve})
        print(json.dumps(result),flush=True)
    except Exception as e:
        write_json(out/'summary.json',{'status':'failed','phase':phase,'method':args.method,'completed_update':completed,'error':repr(e)})
        raise


if __name__=='__main__':
    from src.gpu_job_lock import gpu_job_lock
    with gpu_job_lock():main()
