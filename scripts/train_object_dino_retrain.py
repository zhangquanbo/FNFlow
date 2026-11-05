"""Object-only frozen-DINO pilot, fixed WSD50k, resumable full RNG state."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,collections,hashlib,json,random,time,os
import numpy as np
import torch
from src.research_io import write_json,sha256,now
from src.dino_conditioning import load_local_dino,DINOConditionedUNet
from src.models.model_unet import FaceNormalUNet_B
from src.parameterization import parameterization_loss
from src.gpu_job_lock import gpu_job_lock
from scripts.train_parameterization import ImageCache,evaluate

CONFIG=ROOT/'configs/object_dino_retrain_20261026.json'
def read(p):return json.loads(Path(p).read_text())
def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def factor(step):
    if step<=47500:return min(1.,step/100)
    if step>50000:raise ValueError('Outside fixed budget')
    return max(0.,1-(step-47500)/2500)

def make_model():
    cfg=read(ROOT/'configs/dinov2_s_compatibility.json')
    assert sha256(ROOT/cfg['weights'])==cfg['weights_sha256']
    assert sha256(ROOT/'runtime/dinov2/source.zip')==cfg['source_archive_sha256']
    # Instantiate backbone first, identically for both methods.
    backbone=FaceNormalUNet_B()
    encoder=load_local_dino(ROOT/cfg['source_directory'],ROOT/cfg['weights'],cfg['architecture'])
    return DINOConditionedUNet(backbone,encoder,384,14,pad_to_patch=True).cuda()

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['A','B'],required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume-smoke',action='store_true');p.add_argument('--smoke-reference',action='store_true');a=p.parse_args()
    cfg=read(CONFIG);method=cfg['groups'][a.group]['method'];reg=read(ROOT/'provenance/object_dino_retrain_registration.json')
    assert sha256(CONFIG)==reg['config_sha256']
    manifest=ROOT/cfg['training_manifest'];assert sha256(manifest)==cfg['training_manifest_sha256']
    rows=read(manifest)['records'];records=[r for r in rows if r['split']=='train'];val={r['scene'] for r in rows if r['split']=='val'}
    assert len(records)==cfg['training_images'] and len({r['scene'] for r in records})==cfg['training_assets']
    assert not val&{r['scene'] for r in records}
    for r in records:assert '/arb_mixed_formal_v1/train/' in r['path'] and '/test/' not in r['path']
    validation=[r for r in rows if r['split']=='val']
    assert validation and all('/val/' in r['path'] for r in validation)
    base=ROOT/cfg['output_root']
    if a.smoke:base=base/'smoke_generations'/sha256(__file__)[:12]
    job=base/(('reference_' if a.smoke_reference else '')+a.group+'_'+method+'_s1247')
    job.mkdir(parents=True,exist_ok=True)
    signature={'config_sha256':sha256(CONFIG),'trainer_sha256':sha256(__file__),
      'data_manifest_sha256':sha256(manifest),'group':a.group,'method':method,'smoke':a.smoke,
      'model_code_sha256':sha256(ROOT/'src/models/model_unet.py'),'dino_code_sha256':sha256(ROOT/'src/dino_conditioning.py'),
      'target_code_sha256':sha256(ROOT/'src/parameterization.py'),'seed':1247,'torch':torch.__version__}
    sp=job/'signature_private.json'
    if sp.exists():assert read(sp)==signature,'Run signature changed'
    else:write_json(sp,signature)
    torch.set_num_threads(4);torch.manual_seed(1247);torch.cuda.manual_seed_all(1247)
    random.seed(1247);np.random.seed(1247);torch.backends.cudnn.benchmark=False
    model=make_model();initial=state_hash(model.unet.state_dict());frozen=state_hash(model.features.encoder.state_dict())
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=.01)
    generator=torch.Generator(device='cuda').manual_seed(8247);rng=np.random.default_rng(10247)
    cache=ImageCache(records,capacity=128);step=0;curve=[];started=time.perf_counter();end=3 if a.smoke else 50000
    if a.smoke_reference:
        assert a.smoke
        end=4
    last=job/'last_local.pth'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False);assert ck['signature']==signature
        current=model.state_dict();current.update(ck['trainable_model']);model.load_state_dict(current,strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if torch.is_tensor(value):values[key]=value.cuda()
        step=ck['step'];curve=ck['curve'];rng.bit_generator.state=ck['numpy_rng'];generator.set_state(ck['generator'])
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);random.setstate(ck['python_rng'])
        if a.resume_smoke:end=4
        if step>=end:
            print('already complete',a.group,step,flush=True);return
        log=job/'training_log.jsonl'
        if log.exists():log.write_text(''.join(x+'\n' for x in log.read_text().splitlines() if json.loads(x)['update']<=step))
    write_json(job/'initialization.json',{'unet_initial_sha256':initial,'frozen_encoder_state_sha256':frozen,
       'trainable_parameters':sum(p.numel() for p in parameters),'frozen_parameters':sum(p.numel() for p in model.parameters() if not p.requires_grad),
       'signature':signature})
    def save():
        # Frozen encoder bytes are already pinned by SHA; do not duplicate them in every checkpoint.
        trainable={k:v for k,v in model.state_dict().items() if not k.startswith('features.')}
        tmp=last.with_suffix('.tmp')
        torch.save({'trainable_model':trainable,'optimizer':optimizer.state_dict(),'step':step,'curve':curve,
           'numpy_rng':rng.bit_generator.state,'generator':generator.get_state(),'torch_rng':torch.get_rng_state(),
           'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'signature':signature},tmp)
        tmp.replace(last)
    model.train();torch.cuda.reset_peak_memory_stats()
    write_json(job/'summary.json',{'status':'running','started_at':now(),'group':a.group,'method':method,'target_updates':end,'smoke':a.smoke})
    try:
        with (job/'training_log.jsonl').open('a') as log:
            while step<end:
                nextstep=step+1;optimizer.zero_grad(set_to_none=True)
                for pg in optimizer.param_groups:pg['lr']=2e-4*factor(nextstep)
                infos=[]
                for micro in range(4):
                    batch=[]
                    rng.random() # retain fixed dropout draw without applying dropout
                    for _ in range(4):
                        rgb,normal,mask=cache.get(int(rng.integers(len(records))))
                        if rng.random()<.5:
                            rgb=rgb[:,:,::-1].copy();normal=normal[:,:,::-1].copy();normal[0]*=-1;mask=mask[:,::-1].copy()
                        batch.append((rgb,normal,mask))
                    rgb=torch.from_numpy(np.stack([x[0] for x in batch])).cuda()
                    normal=torch.from_numpy(np.stack([x[1] for x in batch])).cuda();mask=torch.from_numpy(np.stack([x[2] for x in batch])).cuda()
                    noise=torch.randn(normal.shape,device='cuda',generator=generator)
                    t=torch.rand(len(batch),device='cuda',generator=generator)*.95
                    z=(1-t[:,None,None,None])*noise+t[:,None,None,None]*normal
                    if method=='R':z=torch.zeros_like(z);t=torch.zeros_like(t)
                    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,t,rgb)
                    loss,info=parameterization_loss(method,pred,z,normal,noise,t,mask,.25,.25)
                    assert torch.isfinite(loss),'Nonfinite loss'
                    (loss/4).backward();infos.append(info)
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(norm),'Nonfinite gradient'
                assert not model.features.encoder.training
                assert all(p.grad is None and not p.requires_grad for p in model.features.encoder.parameters())
                optimizer.step();step=nextstep
                row={'update':step,'lr':optimizer.param_groups[0]['lr'],'gradient_norm_before_clip':float(norm),
                   'gradient_clipped':bool(norm>1),**{k:float(np.mean([i[k] for i in infos])) for k in ['mse','unit','cosine']}}
                log.write(json.dumps(row)+'\n');log.flush()
                if step%10==0 or step==1 or step==end:
                    elapsed=time.perf_counter()-started
                    progress={'status':'training','group':a.group,'method':method,'update':step,'end_update':end,'elapsed_seconds':elapsed,
                       'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'updated_at':now()}
                    write_json(job/'progress.json',progress);print(json.dumps(progress),flush=True)
                if not a.smoke and step%2500==0:
                    score,_=evaluate(model,method,{'arb_validation_retrain':validation},[42,43],0,'cuda')
                    curve.append({'update':step,'arb_validation_curve_only_2_seeds':score})
                    write_json(job/'learning_curve.json',{'points':curve,'selection':'diagnostic only; fixed final checkpoint remains primary; GSO must be rebuilt separately; Arb curve is not GSO and does not select checkpoints'})
                if step%250==0 or step==end:save()
        assert state_hash(model.features.encoder.state_dict())==frozen
        result={'status':'completed','group':a.group,'method':method,'last_update':step,'smoke':a.smoke,
          'elapsed_seconds':time.perf_counter()-started,'encoder_frozen_unchanged':True,
          'unet_initial_sha256':initial,'final_checkpoint_sha256':sha256(last),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
          'final_evaluation_status':'fresh A/B run; GSO/NLR evaluation not yet rebuilt; fixed final checkpoint'}
        write_json(job/'summary.json',result);write_json(job/'progress.json',result);print(json.dumps(result),flush=True)
    except Exception as e:
        write_json(job/'failure.json',{'error':repr(e),'last_completed_update':step,'updated_at':now()});raise

if __name__=='__main__':
    with gpu_job_lock():main()

