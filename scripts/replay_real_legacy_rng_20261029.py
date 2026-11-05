"""Isolate historical real-data RNG recipe on frozen current inputs; official FP32 defaults."""
from pathlib import Path
import sys,json,random,fcntl,time
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
import torch
from PIL import Image
from src.research_io import sha256,write_json,now
from src.gpu_job_lock import gpu_job_lock
from src.lightrig_adapter import load_models
from scripts.evaluate_fresh_ab import assert_registration,metrics
from scripts.evaluate_fresh_nlr import verify_weights,state,restore,save_state
AB=R/'outputs/P3_object_dino_evaluation_fresh_20261027';OUT=R/'outputs/real_protocol_audit_20261029/legacy_rng_replay'
REG=R/'provenance/real_legacy_rng_replay_registration_20261029.json'
def read(p):return json.loads(Path(p).read_text())
def validate():
 reg=read(REG);assert sha256(__file__)==reg['evaluator_sha256'];assert_registration()
 for p,h in reg['source_sha256'].items():assert sha256(R/p)==h,p
 for domain,h in reg['manifest_sha256'].items():assert sha256(AB/'manifests_private'/(domain+'.json'))==h
 audit=read(R/'outputs/real_protocol_audit_20261029/summary.json');assert audit['status']=='CPU_audit_completed'
 assert audit['all_prepared_RGB_GT_valid_official_mask_and_RGBA_pixels_match_pinned_89f_recipe']
 return reg
def main():
 reg=validate();verify_weights();OUT.mkdir(parents=True,exist_ok=True);rs=sha256(REG)
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 with (OUT/'run.lock').open('a') as lock,gpu_job_lock(),torch.inference_mode():
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  # Pinned89f evaluator resets RNG AFTER model creation, independently per dataset.
  module,m1,m2=load_models()
  write_json(OUT/'runtime.json',{'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'registration_sha256':rs,'updated_at':now()})
  for domain in ['diligent','luces']:
   random.seed(511);np.random.seed(511);torch.manual_seed(511)
   records=sorted(read(AB/'manifests_private'/(domain+'.json'))['records'],key=lambda r:r['id'])
   for i,r in enumerate(records):
    rowpath=OUT/'per_image_private'/domain/(r['id']+'.json')
    if rowpath.exists():
     row=read(rowpath);assert row['registration_sha256']==rs and sha256(row['prediction_path'])==row['prediction_sha256']
     assert sha256(row['post_rng_path'])==row['post_rng_sha256'];restore(torch.load(row['post_rng_path'],map_location='cpu',weights_only=False));continue
    assert sha256(r['rgba'])==r['nlr_rgba_sha256'] and sha256(r['gt_path'])==r['gt_sha256']
    predpath=OUT/'predictions_private'/domain/(r['id']+'.png');predpath.parent.mkdir(parents=True,exist_ok=True);concat=predpath.with_suffix('.concat.png')
    prepath=OUT/'rng_private'/domain/(r['id']+'.pre.pt');save_state(prepath,state())
    module.inference_impl(m1,m2,r['rgba'],str(concat),input_res=512,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
    with Image.open(concat) as im:assert im.size==(2560,512);im.crop((2048,0,2560,512)).convert('RGB').save(predpath)
    post=OUT/'rng_private'/domain/(r['id']+'.post.pt');save_state(post,state())
    pred=(np.array(Image.open(predpath)).astype(np.float32)/127.5-1)*np.array([-1,1,1],np.float32)
    with np.load(r['gt_path']) as z:value=metrics(pred,z['normal'],z['valid'].astype(bool))
    write_json(rowpath,{'domain':domain,'object':r['object'],'id':r['id'],'image_key':r['image_key'],'registration_sha256':rs,'metrics':value,'prediction_path':str(predpath),'prediction_sha256':sha256(predpath),'pre_rng_path':str(prepath),'pre_rng_sha256':sha256(prepath),'post_rng_path':str(post),'post_rng_sha256':sha256(post)})
    concat.unlink();s={'phase':'legacy_real_RNG_replay','domain':domain,'images':i+1,'total_images':len(records),'updated_at':now()};write_json(OUT/'status.json',s);print(json.dumps(s),flush=True)
  results={}
  for domain in ['diligent','luces']:
   records=read(AB/'manifests_private'/(domain+'.json'))['records'];values=[]
   for r in records:
    row=read(OUT/'per_image_private'/domain/(r['id']+'.json'));fresh=read(AB/'per_image_private'/domain/'NeuralLightRig/511'/(r['id']+'.json'))
    values.append({'object':r['object'],'legacy_rng_MAE_deg':row['metrics']['mean_deg'],'fresh_rng_MAE_deg':fresh['metrics']['mean_deg']})
   legacy=float(np.mean([r['legacy_rng_MAE_deg'] for r in values]));fresh=float(np.mean([r['fresh_rng_MAE_deg'] for r in values]))
   results[domain]={'images':len(values),'legacy_rng_MAE_deg':legacy,'fresh_rng_MAE_deg':fresh,'legacy_minus_fresh_deg':legacy-fresh,'historical_reported_approx_deg':reg['historical_reported_approx_deg'][domain],'difference_from_historical_approx_deg':legacy-reg['historical_reported_approx_deg'][domain],'per_object':values}
  summary={'status':'completed','updated_at':now(),'registration_sha256':rs,'datasets':results,'interpretation':'Same current RGB/GT/masks, weights, FP32,75steps and3090; only RNG reset timing/domain scope/order returns to89f recipe. Legacy manifests/RNG states are lost: this is a recipe replay, not bitwise recovery of4090 outputs. Preserve both protocols; do not select the more favorable score.','real_data_superiority_claims':'Historical discrepancy quantified here; any superiority statement must name protocol and single-image/single-NLR-seed limits.','test_images_opened':0}
  write_json(OUT/'summary.json',summary);write_json(OUT/'status.json',{'phase':'completed_legacy_real_RNG_replay','updated_at':now()});print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now()});raise
