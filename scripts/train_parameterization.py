"""Controlled no-ray R/V/Xv/Xn training; private details and weights stay local."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,math,random,time
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from src.models.model_unet import FaceNormalUNet_B
from src.general_dataset import PreparedNormalDataset
from src.noise_protocol import image_noise_seed,mae_seed_statistics
from src.parameterization import parameterization_loss,one_step_endpoint
from src.research_io import ROOT,output_path,write_json,sha256,now
from src.training_diagnostics import TimeLossBins,learning_rate_factor,resume_segment_action
from src.material_protocol import validation_material_sensitivity


def require_formal_budget(cfg,phase):
    if phase=='formal' and (not cfg.get('formal_budget_approved') or not isinstance(cfg.get('formal_steps'),int) or cfg['formal_steps']<=0):
        raise RuntimeError('Formal grid locked pending convergence and clipping review')


class ImageCache:
    def __init__(self,records,capacity=256):self.records=records;self.cache=OrderedDict();self.capacity=capacity
    def get(self,index):
        if index not in self.cache:
            r=self.records[index]
            with np.load(r['path']) as f:
                rgb=f['rgb'].transpose(2,0,1).astype(np.float32)/127.5-1
                normal=f['normal'].transpose(2,0,1).astype(np.float32);valid=f['valid'].astype(bool)
            if not valid.any() or not np.isfinite(normal[:,valid]).all():raise RuntimeError('Invalid training label')
            self.cache[index]=(rgb,normal,valid)
            if len(self.cache)>self.capacity:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def load_records(directory,split):
    data=PreparedNormalDataset(directory,split)  # validates group separation without requiring K.
    return data.records


def boundary_mask(gt,mask):
    horizontal=(gt[:,:,:,1:]*gt[:,:,:,:-1]).sum(1)<math.cos(math.radians(20))
    horizontal&=mask[:,:,1:]&mask[:,:,:-1]
    vertical=(gt[:,:,1:,:]*gt[:,:,:-1,:]).sum(1)<math.cos(math.radians(20))
    vertical&=mask[:,1:,:]&mask[:,:-1,:]
    edges=F.pad(horizontal,(0,1))|F.pad(horizontal,(1,0))|F.pad(vertical,(0,0,0,1))|F.pad(vertical,(0,0,1,0))
    return (F.max_pool2d(edges[:,None].float(),5,1,2)[:,0]>0)&mask


@torch.no_grad()
def evaluate(model,mode,domains,seeds,limit,device,material_audit=None):
    model.eval();result={};private={}
    for name,all_records in domains.items():
        records=sorted(all_records,key=lambda r:hashlib.sha256((name+'/'+Path(r['path']).name).encode()).digest())
        if limit:records=records[:limit]
        cache=ImageCache(records,capacity=16);errors=[];boundary_errors=[];dispersion=[];detail=[]
        for start in range(0,len(records),8):
            group=records[start:start+8];arrays=[cache.get(i) for i in range(start,start+len(group))]
            rgb=torch.from_numpy(np.stack([a[0] for a in arrays])).to(device)
            gt=torch.from_numpy(np.stack([a[1] for a in arrays])).to(device)
            mask=torch.from_numpy(np.stack([a[2] for a in arrays])).to(device)
            gt=F.normalize(gt,dim=1);band=boundary_mask(gt,mask)
            predictions=[]
            for inference_seed in (seeds if mode!='R' else seeds[:1]):
                noises=[]
                for row in group:
                    key=row.get('image_key',name+'/'+Path(row['path']).name)
                    generator=torch.Generator(device=device).manual_seed(image_noise_seed(inference_seed,key))
                    noises.append(torch.randn((3,256,256),generator=generator,device=device))
                noise=torch.stack(noises) if mode!='R' else torch.zeros_like(gt)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    raw=model(noise,torch.zeros(len(group),device=device),rgb)
                endpoint=one_step_endpoint(mode,raw.float(),noise)
                if not torch.isfinite(endpoint.permute(0,2,3,1)[mask]).all():raise RuntimeError('Nonfinite validation prediction')
                if (endpoint.norm(dim=1)[mask]<1e-8).any():raise RuntimeError('Zero valid predicted direction')
                predictions.append(F.normalize(endpoint,dim=1))
            if mode=='R':predictions=predictions*len(seeds)
            stack=torch.stack(predictions)
            angle=torch.rad2deg(torch.acos((stack*gt[None]).sum(2).clamp(-1,1)))
            image_mae=(angle*mask[None]).sum((2,3))/mask.sum((1,2))[None]
            band_counts=band.sum((1,2));band_mae=(angle*band[None]).sum((2,3))/band_counts.clamp_min(1)[None]
            center=F.normalize(stack.mean(0),dim=1)
            pixel_disp=torch.rad2deg(torch.acos((stack*center[None]).sum(2).clamp(-1,1))).mean(0)
            pixel_disp=(pixel_disp*mask).sum((1,2))/mask.sum((1,2))
            for j,row in enumerate(group):
                values=image_mae[:,j].cpu().tolist();errors.append(values)
                b=band_mae[:,j].cpu().tolist() if band_counts[j]>0 else None
                boundary_errors.append(b)
                d=0. if mode=='R' else float(pixel_disp[j]);dispersion.append(d)
                detail.append({'image_key':row.get('image_key',name+'/'+Path(row['path']).name),'group':row['scene'],
                    'mae_by_seed':values,'boundary_mae_by_seed':b,'pixel_angular_dispersion_deg':d})
        if len(seeds)>1:stats=mae_seed_statistics(np.asarray(errors))
        else:stats={'mean_mae_deg':float(np.mean(errors)),'num_images':len(errors)}
        available=[v for v in boundary_errors if v is not None]
        stats.update(mean_pixel_angular_dispersion_deg=float(np.mean(dispersion)),
            boundary_mae_deg=float(np.mean(available)) if available else None,
            boundary_definition='GT adjacent normal angle >20deg, dilated by radius 2 pixels, intersect valid mask',
            boundary_images=len(available))
        if name=='arb' and material_audit is not None and all(r.get('split')=='val' for r in records):
            stats['without_blend_or_transmission_sensitivity']=validation_material_sensitivity(records,errors,boundary_errors,dispersion,material_audit)
        if name=='arb' and all(r.get('split')=='val' and Path(r['path']).is_relative_to(ROOT/'data/arb_mixed_formal_v1') for r in records):
            audit=ROOT/'outputs/P1_arb_boundary_audit_formal_v1/summary.json'
            if audit.exists():
                a=json.loads(audit.read_text())
                stats['boundary_mask_coverage_audit']={'phase':a['phase'],'validation_counts_and_fractions':a['by_split']['val'],
                    'includes_empty_views':True,'note':'Error MAE is conditional on retained labels; coverage audit also includes rejected/empty views. Raw excluded normals are not restored as GT.'}
        result[name]=stats;private[name]=detail
    model.train()
    return result,private


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['R','V','Xv','Xn'],required=True)
    p.add_argument('--dropout',type=float,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--phase',choices=['smoke','pilot','formal','convergence','clip_diagnostic'],required=True)
    p.add_argument('--steps',type=int);p.add_argument('--run-id')
    p.add_argument('--config',default='configs/parameterization_first_round_v2_review1.json')
    p.add_argument('--extend-to',type=int,choices=[50000])
    args=p.parse_args()
    config_path=ROOT/args.config;cfg=json.loads(config_path.read_text())
    require_formal_budget(cfg,args.phase)
    if cfg['ray_condition'] or cfg['dino_condition'] or cfg['cfg']:raise RuntimeError('First-round input protocol changed')
    data_phase='formal' if args.phase in {'convergence','clip_diagnostic'} else args.phase
    object_root=ROOT/f'data/arb_mixed_{data_phase}_v1'
    if not (object_root/'summary.json').exists():raise RuntimeError('Prepared object partition missing')
    prepared=json.loads((object_root/'summary.json').read_text())
    expected=cfg['formal_assets'] if data_phase=='formal' else cfg['pilot_min_assets'] if data_phase=='pilot' else 3
    if prepared['assets']!=expected:raise RuntimeError('Asset budget mismatch')
    run_id=args.run_id or f'{args.phase}_{args.method}_p{args.dropout:g}_s{args.seed}'
    out=output_path(ROOT/cfg.get('output_root','outputs/P2_parameterization_v1')/run_id);out.mkdir(parents=True,exist_ok=True)
    steps=args.steps or (cfg['convergence_steps'] if args.phase=='convergence' else cfg['clip_diagnostic_steps'] if args.phase=='clip_diagnostic'
                        else cfg['formal_steps'] if args.phase=='formal' else cfg['pilot_steps'] if args.phase=='pilot' else 3)
    schedule_steps=cfg['convergence_steps'] if args.phase in {'convergence','clip_diagnostic'} else steps
    evaluation_every=cfg.get('evaluation_every',2500) if args.phase=='convergence' else 0
    base_steps=steps
    if args.extend_to:
        if args.phase!='convergence' or not cfg.get('decision_rule'):raise RuntimeError('Only preregistered convergence continuation is allowed')
        steps=args.extend_to
    hyp=load_records(ROOT/cfg['hypersim_root'],'train');obj=load_records(object_root,'train')
    if cfg.get('spatial_protocol')=='center_square_v2':
        for key in ['hypersim_root','nyu_validation_root']:
            summary=json.loads((ROOT/cfg[key]/'summary.json').read_text())
            if summary.get('spatial_protocol')!='center_square_v2':raise RuntimeError('Wrong spatial cache protocol')
    domains={'hypersim':load_records(ROOT/cfg['hypersim_root'],'val'),
             'arb':load_records(object_root,'val'),
             'nyuv2':load_records(ROOT/cfg['nyu_validation_root'],'val')}
    manifests=[ROOT/cfg['hypersim_root']/'private_manifest.json',object_root/'private_manifest.json',
               ROOT/cfg['nyu_validation_root']/'private_manifest.json']
    probes={}
    if cfg.get('curve_training_probe') and args.phase in {'convergence','clip_diagnostic','formal'}:
        from scripts.prepare_training_probe import prepare
        probes=prepare(cfg)
    signature={'config_sha256':sha256(config_path),'data_manifests_sha256':[sha256(x) for x in manifests],
        'method':args.method,'condition_dropout':args.dropout,'training_seed':args.seed,'phase':args.phase,'steps':base_steps,
        'train_code_sha256':sha256(__file__),'target_code_sha256':sha256(ROOT/'src/parameterization.py'),
        'diagnostics_code_sha256':sha256(ROOT/'src/training_diagnostics.py'),
        'schedule_steps':schedule_steps,'evaluation_every':evaluation_every}
    if cfg.get('decision_rule'):
        signature['decision_rule_sha256']=sha256(ROOT/cfg['decision_rule'])
    if probes:signature['training_probe_sha256']=sha256(ROOT/cfg['output_root']/'training_probe_private.json')
    if (out/'signature_private.json').exists():
        if json.loads((out/'signature_private.json').read_text())!=signature:raise RuntimeError('Existing run signature differs')
        existing=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else {}
        if existing.get('status')=='completed' and existing.get('optimizer_updates',0)>=steps:
            print('already completed',run_id,flush=True);return
    else:write_json(out/'signature_private.json',signature)
    if args.extend_to:
        rule=json.loads((ROOT/cfg['decision_rule']).read_text());triggered=False
        for method in rule['methods']:
            curve_file=ROOT/cfg['output_root']/f'convergence_{method}_p0_s{cfg["convergence_seed"]}'/'learning_curve.json'
            points={r['update']:r['validation'] for r in json.loads(curve_file.read_text())['points']}
            for domain in rule['domains']:
                triggered|=points[27500][domain]['mean_mae_deg']-points[30000][domain]['mean_mae_deg']>rule['extension_threshold_deg']+1e-9
        if not triggered or not (out/'last_local.pth').exists():raise RuntimeError('Continuation requires trigger plus optimizer/RNG resume')
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed);random.seed(args.seed);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');model=FaceNormalUNet_B().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator(device=device).manual_seed(args.seed+7000)
    rng=np.random.default_rng(args.seed+9000)
    caches=[ImageCache(hyp),ImageCache(obj)]
    completed=0;resume=out/'last_local.pth';grad_norms=[];curve=[]
    time_bins=TimeLossBins(cfg.get('time_bin_edges',[0,.2,.4,.6,.8,.95]))
    if resume.exists():
        state=torch.load(resume,map_location='cpu');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):values[key]=value.to(device)
        generator.set_state(state['generator']);rng.bit_generator.state=state['numpy_rng'];completed=state['step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        grad_norms=state['grad_norms'];time_bins.rows=state['time_bin_rows'];curve=state['curve']
        if resume_segment_action(completed,steps,args.phase,args.extend_to,curve)=='skip_initial':
            print('initial segment already complete; continuation checkpoint preserved',run_id,completed,flush=True)
            return
        # Discard updates after the durable checkpoint; their exact replay is logged once.
        old_log=out/'training_log.jsonl'
        if old_log.exists():
            kept=[line for line in old_log.read_text().splitlines() if json.loads(line)['update']<=completed]
            old_log.write_text('\n'.join(kept)+'\n',encoding='utf-8')
    write_json(out/'summary.json',{'status':'running','phase':args.phase,'method':args.method,'seed':args.seed,'started_at':now(),'planned_updates':steps})
    model.train();started=time.perf_counter();last_loss=None;max_grad=0.;torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'training_log.jsonl').open('a',encoding='utf-8') as log:
            for step in range(completed+1,steps+1):
                optimizer.zero_grad(set_to_none=True)
                schedule=learning_rate_factor(step,schedule_steps)
                for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*schedule
                drop=bool(rng.random()<args.dropout);components=[]
                update_bins=TimeLossBins(time_bins.edges)
                for micro in range(cfg['gradient_accumulation']):
                    batch=[]
                    for source in range(2):
                        for _ in range(cfg['microbatch']//2):
                            rgb,normal,valid=caches[source].get(int(rng.integers(len(caches[source].records))))
                            if rng.random()<.5:
                                rgb=rgb[:,:,::-1].copy();normal=normal[:,:,::-1].copy();normal[0]*=-1;valid=valid[:,::-1].copy()
                            batch.append((rgb,normal,valid))
                    rgb=torch.from_numpy(np.stack([x[0] for x in batch])).to(device)
                    target=torch.from_numpy(np.stack([x[1] for x in batch])).to(device)
                    valid=torch.from_numpy(np.stack([x[2] for x in batch])).to(device)
                    noise=torch.randn(target.shape,device=device,generator=generator)
                    times=torch.rand(len(batch),device=device,generator=generator)*(1-cfg['time_delta'])
                    state=(1-times[:,None,None,None])*noise+times[:,None,None,None]*target
                    if args.method=='R':state=torch.zeros_like(state);times=torch.zeros_like(times)
                    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(state,times,torch.zeros_like(rgb) if drop else rgb)
                    loss,info=parameterization_loss(args.method,raw,state,target,noise,times,valid,cfg['unit_weight'],cfg['cosine_weight'],return_per_image=True)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
                    ts=times.detach().cpu().tolist()
                    time_bins.add(ts,info['per_image']);update_bins.add(ts,info['per_image'])
                    (loss/cfg['gradient_accumulation']).backward();components.append({'loss':float(loss.detach()),**info})
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient norm')
                max_grad=max(max_grad,float(grad));grad_norms.append(float(grad));optimizer.step();completed=step
                last_loss=float(np.mean([r['loss'] for r in components]))
                if True:  # Every optimizer update: sparse logging would bias clipping fractions.
                    row={'update':step,'loss':last_loss,'gradient_norm_before_clip':float(grad),'condition_dropped':drop,
                         'gradient_clip_threshold':cfg['gradient_clip_norm'],'gradient_clipped':float(grad)>cfg['gradient_clip_norm'],
                         'gradient_clip_scale':min(1.,cfg['gradient_clip_norm']/(float(grad)+1e-6)),
                         'time_loss_bins':update_bins.summary(),
                         'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-started,
                         **{key:float(np.mean([r[key] for r in components])) for key in ['mse','unit','cosine']}}
                    log.write(json.dumps(row)+'\n');log.flush()
                    if step==1 or step%10==0 or step==steps:
                        print(run_id,json.dumps({k:v for k,v in row.items() if k!='time_loss_bins'}),flush=True)
                        write_json(out/'progress.json',{'status':'running','update':step,'planned_updates':steps,'updated_at':now(),'last_loss':last_loss})
                if evaluation_every and step%evaluation_every==0:
                    torch.save({'model':model.state_dict(),'step':step,'signature':signature},out/f'checkpoint_{step:06d}_local.pth')
                    scores,_=evaluate(model,args.method,domains,cfg['curve_inference_seeds'],0,device)
                    train_scores,_=evaluate(model,args.method,probes,cfg['curve_inference_seeds'],0,device) if probes else ({},{})
                    curve.append({'update':step,'validation':scores,'training_probe':train_scores,
                        'validation_minus_training_mae_deg':{d:scores[d]['mean_mae_deg']-train_scores[d]['mean_mae_deg'] for d in train_scores},
                        'inference_seeds':cfg['curve_inference_seeds']})
                    write_json(out/'learning_curve.json',{'status':'running','method':args.method,'points':curve,
                        'selection':'validation curve only; no benchmark test frames','updated_at':now()})
                if cfg.get('spatial_protocol')=='center_square_v2' and (step%2500==0 or step==steps or (step<=2500 and step%250==0)):
                    from scripts.summarize_convergence import summarize
                    summarize(args.config)
                if step%250==0:
                    temporary=out/'last_local.tmp'
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                        'generator':generator.get_state(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'grad_norms':grad_norms,
                        'time_bin_rows':time_bins.rows,'curve':curve},temporary);temporary.replace(resume)
        torch.save({'model':model.state_dict(),'step':steps,'signature':signature},out/'final_local.pth')
        if args.phase=='convergence':torch.save({'model':model.state_dict(),'step':steps,'signature':signature},out/f'endpoint_{steps:06d}_local.pth')
        full_eval=args.phase in {'formal','convergence','clip_diagnostic'}
        seeds=cfg['inference_seeds'] if full_eval else cfg['pilot_inference_seeds']
        scores,detail=evaluate(model,args.method,domains,seeds,0 if full_eval else cfg['pilot_validation_images_per_domain'],device)
        write_json(out/'per_image_private.json',detail)
        result={'status':'completed','created_at':now(),'phase':args.phase,'method':args.method,'condition_dropout':args.dropout,
            'training_seed':args.seed,'optimizer_updates':completed,'trainable_parameters':sum(x.numel() for x in model.parameters()),
            'ray_condition_used':False,'dino_condition_used':False,'cfg_used':False,'precision':'bfloat16',
            'last_loss':last_loss,'max_gradient_norm_before_clip':max(grad_norms),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'gradient_updates_recorded':len(grad_norms),'gradient_clip_fraction':float(np.mean(np.array(grad_norms)>cfg['gradient_clip_norm'])),
            'gradient_norm_quantiles':{str(q):float(np.quantile(grad_norms,q)) for q in [.5,.9,.99,1.]},
            'gradient_clip_threshold':cfg['gradient_clip_norm'],'time_loss_bins':time_bins.summary(),
            'r_time_bins_note':'R has fixed t=0; not a comparison of time dependence' if args.method=='R' else None,
            'elapsed_seconds':time.perf_counter()-started,'validation':scores,
            'mean_domain_mae_deg':float(np.mean([x['mean_mae_deg'] for x in scores.values()])),
            'training_seed_variance_estimated':False,'convergence_established':False,
            'scope':'convergence/clipping diagnostic, one training seed' if args.phase in {'convergence','clip_diagnostic'}
                    else 'fixed-budget formal run' if args.phase=='formal' else 'numerical preflight only; not a method ranking'}
        write_json(out/'summary.json',result);print(json.dumps(result),flush=True)
        if curve:write_json(out/'learning_curve.json',{'status':'completed','method':args.method,'points':curve,'updated_at':now()})
        if resume.exists() and args.phase!='convergence':resume.unlink()
    except Exception as error:
        write_json(out/'summary.json',{'status':'failed','method':args.method,'phase':args.phase,'seed':args.seed,
            'completed_updates':completed,'error':type(error).__name__+': '+str(error),'failed_at':now()})
        raise

if __name__=='__main__':main()
