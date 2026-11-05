"""Official NLR default inference with fresh, replayable matched-view RNG states."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,os,random,gc,fcntl,hashlib,math
import numpy as np
import torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.lightrig_adapter import load_models
from scripts.evaluate_fresh_ab import OUT,PANELS,prepare,read,save,cached,assert_registration
def state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def save_state(path,s):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.partial');torch.save(s,tmp);tmp.replace(path)
def verify_weights():
    cfg=read(ROOT/'configs/object_comparison_release_20261024.json')
    for field,folder in [('nlr_weights',ROOT/'external/lightrig_weights_5619cfe'),('base_components',ROOT/'external/base_models')]:
        rows=cfg[field]['components'] if field=='nlr_weights' else cfg[field]
        for row in rows:
            p=folder/row['component'];assert p.exists() and p.stat().st_size==row['bytes'] and sha256(p)==row['sha256'],('NLR component unavailable/changed',str(p))
    return cfg
def infer(module,m1,m2,panel,r,mh,before,model='NeuralLightRig',seed=511):
    dest=OUT/'nlr_predictions_private'/model/str(seed)/panel/(r['id']+'.png');dest.parent.mkdir(parents=True,exist_ok=True)
    concat=dest.with_suffix('.concat.png')
    restore(torch.load(before,map_location='cpu',weights_only=False))
    module.inference_impl(m1,m2,r['rgba'],str(concat),input_res=512,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
    with Image.open(concat) as image:
        assert image.size==(2560,512);image.crop((2048,0,2560,512)).convert('RGB').save(dest)
    after=OUT/'rng_private'/model/str(seed)/panel/(r['id']+'.post.pt');save_state(after,state())
    pred=(np.asarray(Image.open(dest)).astype(np.float32)/127.5-1)*np.array([-1,1,1],np.float32)
    save(panel,model,seed,r,pred,mh,{'prediction_path':str(dest),'prediction_sha256':sha256(dest),
         'pre_rng_path':str(before),'pre_rng_sha256':sha256(before),'post_rng_path':str(after),'post_rng_sha256':sha256(after)})
    concat.unlink()
def main():
    reg=assert_registration();verify_weights();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    with (OUT/'nlr.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        random.seed(511);np.random.seed(511);torch.manual_seed(511)
        module,m1,m2=load_models()
        initial=OUT/'rng_private/initial_after_model_loading.pt'
        if not initial.exists():save_state(initial,state())
        restore(torch.load(initial,map_location='cpu',weights_only=False))
        primary=prepare('primary');pmh=sha256(OUT/'manifests_private/primary.json')
        baseline=[r for r in primary['records'] if r['elevation_bin']==30];assert len(baseline)==372
        baseline.sort(key=lambda r:r['image_key'])
        base={(r['object'],int(r['image_key'].rsplit('/',1)[1])):r for r in baseline}
        sequence=[('primary',r,pmh) for r in baseline]
        for panel in ['diligent','luces']:
            man=prepare(panel);mh=sha256(OUT/'manifests_private'/(panel+'.json'))
            sequence.extend((panel,r,mh) for r in man['records'])
        for i,(panel,r,mh) in enumerate(sequence):
            if cached(panel,'NeuralLightRig',511,r,mh):
                row=read(OUT/'per_image_private'/panel/'NeuralLightRig/511'/(r['id']+'.json'))
                assert sha256(row['post_rng_path'])==row['post_rng_sha256'] and sha256(row['prediction_path'])==row['prediction_sha256']
                restore(torch.load(row['post_rng_path'],map_location='cpu',weights_only=False));continue
            before=OUT/'rng_private'/panel/(r['id']+'.pre.pt');save_state(before,state())
            infer(module,m1,m2,panel,r,mh,before)
            status={'phase':'baseline_and_real','images':i+1,'total_images':len(sequence),'updated_at':now()}
            write_json(OUT/'nlr_status.json',status);print(json.dumps(status),flush=True)
        # Each angle/light replays its fresh baseline30 pre-state. No historical RNG claims.
        for panel in ['primary','original','arb_style']:
            man=prepare(panel);mh=sha256(OUT/'manifests_private'/(panel+'.json'))
            records=[r for r in man['records'] if not(panel=='primary' and r['elevation_bin']==30)]
            for i,r in enumerate(records):
                if cached(panel,'NeuralLightRig',511,r,mh):continue
                ref=base[(r['object'],int(r['image_key'].rsplit('/',1)[1]))]
                before=OUT/'rng_private/primary'/(ref['id']+'.pre.pt')
                assert before.exists();infer(module,m1,m2,panel,r,mh,before)
                status={'phase':'matched_camera_and_light','panel':panel,'images':i+1,'total_images':len(records),'updated_at':now()}
                write_json(OUT/'nlr_status.json',status);print(json.dumps(status),flush=True)
        write_json(OUT/'nlr_default_completed.json',{'phase':'completed_default','updated_at':now(),'test_images_scored':0})
        subset=sorted(sequence,key=lambda x:x[1]['id'])[:math.ceil(.1*len(sequence))]
        assert len(subset)==40
        write_json(OUT/'nlr_variability_selection_private.json',{'ids':[r['id'] for _,r,_ in subset],'rule':'first ceil(10%) by image-key SHA256 among fresh primary30 plus 24 real controls; no errors used'})
        for seed in [511,512,513]:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            for i,(panel,r,mh) in enumerate(subset):
                model='NeuralLightRig_variability'
                if cached(panel,model,seed,r,mh):
                    row=read(OUT/'per_image_private'/panel/model/str(seed)/(r['id']+'.json'))
                    assert sha256(row['post_rng_path'])==row['post_rng_sha256']
                    restore(torch.load(row['post_rng_path'],map_location='cpu',weights_only=False));continue
                before=OUT/'rng_private'/model/str(seed)/panel/(r['id']+'.pre.pt');save_state(before,state())
                infer(module,m1,m2,panel,r,mh,before,model=model,seed=seed)
                write_json(OUT/'nlr_status.json',{'phase':'variability','seed':seed,'images':i+1,'total_images':40,'updated_at':now()})
        write_json(OUT/'nlr_status.json',{'phase':'completed_default_and_variability','updated_at':now(),'test_images_scored':0})
if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'nlr_failure.json',{'error':repr(e),'updated_at':now()});raise
