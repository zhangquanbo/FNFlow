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
def evaluate(model,mode,domains,seeds,limit,device):
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
        result[name]=stats;private[name]=detail
    model.train()
    return result,private


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['R','V','Xv','Xn'],required=True)
    p.add_argument('--dropout',type=float,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--phase',choices=['smoke','pilot','formal'],required=True)
    p.add_argument('--steps',type=int);p.add_argument('--run-id');args=p.parse_args()
    config_path=ROOT/'configs/parameterization_first_round.json';cfg=json.loads(config_path.read_text())
    if cfg['ray_condition'] or cfg['dino_condition'] or cfg['cfg']:raise RuntimeError('First-round input protocol changed')
    object_root=ROOT/f'data/arb_mixed_{args.phase}_v1'
    if not (object_root/'summary.json').exists():raise RuntimeError('Prepared object partition missing')
    prepared=json.loads((object_root/'summary.json').read_text())
    expected=cfg['formal_assets'] if args.phase=='formal' else cfg['pilot_min_assets'] if args.phase=='pilot' else 3
    if prepared['assets']!=expected:raise RuntimeError('Asset budget mismatch')
    run_id=args.run_id or f'{args.phase}_{args.method}_p{args.dropout:g}_s{args.seed}'
    out=output_path(ROOT/'outputs/P2_parameterization_v1'/run_id);out.mkdir(parents=True,exist_ok=True)
    steps=args.steps or (cfg['formal_steps'] if args.phase=='formal' else cfg['pilot_steps'] if args.phase=='pilot' else 3)
    hyp=load_records(ROOT/cfg['hypersim_root'],'train');obj=load_records(object_root,'train')
    domains={'hypersim':load_records(ROOT/cfg['hypersim_root'],'val'),
             'arb':load_records(object_root,'val'),
             'nyuv2':load_records(ROOT/cfg['nyu_validation_root'],'val')}
    manifests=[ROOT/cfg['hypersim_root']/'private_manifest.json',object_root/'private_manifest.json',
               ROOT/cfg['nyu_validation_root']/'private_manifest.json']
    signature={'config_sha256':sha256(config_path),'data_manifests_sha256':[sha256(x) for x in manifests],
        'method':args.method,'condition_dropout':args.dropout,'training_seed':args.seed,'phase':args.phase,'steps':steps,
        'train_code_sha256':sha256(__file__),'target_code_sha256':sha256(ROOT/'src/parameterization.py')}
    if (out/'signature_private.json').exists():
        if json.loads((out/'signature_private.json').read_text())!=signature:raise RuntimeError('Existing run signature differs')
        if (out/'summary.json').exists() and json.loads((out/'summary.json').read_text()).get('status')=='completed':
            print('already completed',run_id,flush=True);return
    else:write_json(out/'signature_private.json',signature)
    write_json(out/'summary.json',{'status':'running','phase':args.phase,'method':args.method,'seed':args.seed,'started_at':now(),'planned_updates':steps})
    torch.set_num_threads(4);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed);random.seed(args.seed);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');model=FaceNormalUNet_B().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator(device=device).manual_seed(args.seed+7000)
    rng=np.random.default_rng(args.seed+9000)
    caches=[ImageCache(hyp),ImageCache(obj)]
    completed=0;resume=out/'last_local.pth'
    if resume.exists():
        state=torch.load(resume,map_location='cpu');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):values[key]=value.to(device)
        generator.set_state(state['generator']);rng.bit_generator.state=state['numpy_rng'];completed=state['step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    model.train();started=time.perf_counter();last_loss=None;max_grad=0.;torch.cuda.reset_peak_memory_stats()
    try:
        with (out/'training_log.jsonl').open('a',encoding='utf-8') as log:
            for step in range(completed+1,steps+1):
                optimizer.zero_grad(set_to_none=True)
                warmup=min(100,steps//10)
                schedule=min(1.,step/max(1,warmup))*(.1+.9*.5*(1+math.cos(math.pi*step/steps)))
                for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*schedule
                drop=bool(rng.random()<args.dropout);components=[]
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
                    loss,info=parameterization_loss(args.method,raw,state,target,noise,times,valid,cfg['unit_weight'],cfg['cosine_weight'])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training objective')
                    (loss/cfg['gradient_accumulation']).backward();components.append({'loss':float(loss.detach()),**info})
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                if not torch.isfinite(grad):raise RuntimeError('Nonfinite gradient norm')
                max_grad=max(max_grad,float(grad));optimizer.step();completed=step
                last_loss=float(np.mean([r['loss'] for r in components]))
                if step==1 or step%10==0 or step==steps:
                    row={'update':step,'loss':last_loss,'gradient_norm_before_clip':float(grad),'condition_dropped':drop,
                         'lr':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-started,
                         **{key:float(np.mean([r[key] for r in components])) for key in ['mse','unit','cosine']}}
                    log.write(json.dumps(row)+'\n');log.flush();print(run_id,json.dumps(row),flush=True)
                    write_json(out/'progress.json',{'status':'running','update':step,'planned_updates':steps,'updated_at':now(),'last_loss':last_loss})
                if step%250==0 and step<steps:
                    temporary=out/'last_local.tmp'
                    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                        'generator':generator.get_state(),'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all()},temporary);temporary.replace(resume)
        torch.save({'model':model.state_dict(),'step':steps,'signature':signature},out/'final_local.pth')
        seeds=cfg['inference_seeds'] if args.phase=='formal' else cfg['pilot_inference_seeds']
        scores,detail=evaluate(model,args.method,domains,seeds,0 if args.phase=='formal' else cfg['pilot_validation_images_per_domain'],device)
        write_json(out/'per_image_private.json',detail)
        result={'status':'completed','created_at':now(),'phase':args.phase,'method':args.method,'condition_dropout':args.dropout,
            'training_seed':args.seed,'optimizer_updates':completed,'trainable_parameters':sum(x.numel() for x in model.parameters()),
            'ray_condition_used':False,'dino_condition_used':False,'cfg_used':False,'precision':'bfloat16',
            'last_loss':last_loss,'max_gradient_norm_before_clip':max_grad,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.perf_counter()-started,'validation':scores,
            'mean_domain_mae_deg':float(np.mean([x['mean_mae_deg'] for x in scores.values()])),
            'training_seed_variance_estimated':False,'convergence_established':False,
            'scope':'fixed-budget formal run' if args.phase=='formal' else 'numerical preflight only; not a method ranking'}
        write_json(out/'summary.json',result);print(json.dumps(result),flush=True)
        if resume.exists():resume.unlink()  # Own obsolete resume only; final weights remain.
    except Exception as error:
        write_json(out/'summary.json',{'status':'failed','method':args.method,'phase':args.phase,'seed':args.seed,
            'completed_updates':completed,'error':type(error).__name__+': '+str(error),'failed_at':now()})
        raise

if __name__=='__main__':main()
