"""All24 real inputs x5 predeclared legacy-protocol RNG seeds; resumable and yields to D1."""
from pathlib import Path
import sys,json,random,time,fcntl
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));sys.dont_write_bytecode=True
import numpy as np
import torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.lightrig_adapter import load_models
from scripts.evaluate_fresh_ab import assert_registration,metrics
from scripts.evaluate_fresh_nlr import verify_weights,state,restore,save_state
AB=R/'outputs/P3_object_dino_evaluation_fresh_20261027';OLD=R/'outputs/real_protocol_audit_20261029/legacy_rng_replay';OUT=R/'outputs/NLR_real_5seeds_20261030';REG=R/'provenance/NLR_real_5seeds_registration_20261030.json';CONTROL=AB/'queue_control_20261029.json'
def read(p):return json.loads(Path(p).read_text())
def pausing():return read(CONTROL).get('pause_for_C',False) or bool(read(CONTROL).get('priority_training'))
def main():
 OUT.mkdir(parents=True,exist_ok=True);reg=read(REG);rs=sha256(REG);assert sha256(__file__)==reg['evaluator_sha256'];assert_registration()
 for d,h in reg['manifest_sha256'].items():assert sha256(AB/'manifests_private'/(d+'.json'))==h
 if pausing():return 75
 verify_weights();torch.set_num_threads(3);torch.backends.cudnn.benchmark=False
 with (OUT/'worker.lock').open('a') as lock,torch.inference_mode():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);module,m1,m2=load_models()
  for seed in [511,512,513,514,515]:
   for domain in ['diligent','luces']:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rows=sorted(read(AB/'manifests_private'/(domain+'.json'))['records'],key=lambda r:r['id'])
    for i,r in enumerate(rows):
     p=OUT/'per_image_private'/domain/str(seed)/(r['id']+'.json')
     if p.exists():
      v=read(p);assert v['registration_sha256']==rs and sha256(v['post_rng_path'])==v['post_rng_sha256'];restore(torch.load(v['post_rng_path'],map_location='cpu',weights_only=False));continue
     if seed==511:
      v=read(OLD/'per_image_private'/domain/(r['id']+'.json'));assert sha256(v['prediction_path'])==v['prediction_sha256'] and sha256(v['post_rng_path'])==v['post_rng_sha256'];v.update(registration_sha256=rs,reused_completed_legacy511=True,seed=511);write_json(p,v);restore(torch.load(v['post_rng_path'],map_location='cpu',weights_only=False));continue
     if pausing():write_json(OUT/'status.json',{'phase':'paused_for_priority_training','seed':seed,'domain':domain,'updated_at':now()});return 75
     assert sha256(r['rgba'])==r['nlr_rgba_sha256']
     image=OUT/'predictions_private'/domain/str(seed)/(r['id']+'.png');image.parent.mkdir(parents=True,exist_ok=True);concat=image.with_suffix('.concat.png');pre=OUT/'rng_private'/domain/str(seed)/(r['id']+'.pre.pt');post=pre.with_name(r['id']+'.post.pt');save_state(pre,state())
     module.inference_impl(m1,m2,r['rgba'],str(concat),input_res=512,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
     with Image.open(concat) as im:assert im.size==(2560,512);im.crop((2048,0,2560,512)).convert('RGB').save(image)
     save_state(post,state());pred=(np.array(Image.open(image)).astype(np.float32)/127.5-1)*np.array([-1,1,1],np.float32)
     with np.load(r['gt_path']) as z:m=metrics(pred,z['normal'],z['valid'].astype(bool))
     write_json(p,{'domain':domain,'object':r['object'],'id':r['id'],'image_key':r['image_key'],'seed':seed,'registration_sha256':rs,'metrics':m,'prediction_path':str(image),'prediction_sha256':sha256(image),'pre_rng_path':str(pre),'pre_rng_sha256':sha256(pre),'post_rng_path':str(post),'post_rng_sha256':sha256(post),'reused_completed_legacy511':False});concat.unlink()
     s={'phase':'running','seed':seed,'domain':domain,'images':i+1,'total_images':len(rows),'updated_at':now()};write_json(OUT/'status.json',s);print(json.dumps(s),flush=True)
  results={}
  for domain in ['diligent','luces']:
   rows=read(AB/'manifests_private'/(domain+'.json'))['records'];means=[]
   for seed in [511,512,513,514,515]:means.append(float(np.mean([read(OUT/'per_image_private'/domain/str(seed)/(r['id']+'.json'))['metrics']['mean_deg'] for r in rows])))
   results[domain]={'MAE_by_seed':dict(zip([511,512,513,514,515],means)),'mean_deg':float(np.mean(means)),'SD_across_NLR_inference_seeds_ddof1_deg':float(np.std(means,ddof=1)),'minimum_deg':min(means),'maximum_deg':max(means),'images_per_seed':len(rows)}
  write_json(OUT/'summary.json',{'status':'completed','updated_at':now(),'datasets':results,'registration_sha256':rs,'seeds':[511,512,513,514,515],'scope':'Five NLR inference RNG sequences, not training replicates. Fixed1image per object, old reset/order rules.','test_images_scored':0});write_json(OUT/'status.json',{'phase':'completed','updated_at':now()});return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception as e:write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now()});raise
