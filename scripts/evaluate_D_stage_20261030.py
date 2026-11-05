"""Fixed-final C comparison on the registered A/B val inputs; no test access."""
from pathlib import Path
import sys,json,time,gc,os
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
import torch
from PIL import Image
from src.research_io import sha256,write_json,now
from src.noise_protocol import image_noise_seed
from src.gpu_job_lock import gpu_job_lock
from scripts.train_D_stage_20261030 import make_model
from scripts.evaluate_fresh_ab import assert_registration,metrics,PANELS
CFG=json.loads(Path(os.environ['FACEFLUX_STAGE_CONFIG']).read_text());GROUP=CFG['stage'];AB=R/'outputs/P3_object_dino_evaluation_fresh_20261027';C=R/CFG['output_root'];OUT=C/'evaluation'
REG=R/CFG['evaluation_registration_path']
def read(p):return json.loads(Path(p).read_text())
def validate():
 assert_registration();reg=read(REG)
 assert sha256(__file__)==reg['evaluator_sha256']
 assert sha256(R/'provenance/D_recipe_ladder_amendment_20261029.json')==reg['routing_sha256']
 for panel,h in reg['manifest_sha256'].items():assert sha256(AB/'manifests_private'/(panel+'.json'))==h
 return reg
def main():
 reg=validate();OUT.mkdir(parents=True,exist_ok=True);job=C/(GROUP+'_Xn_s1247');completed=read(job/'summary.json')
 assert completed['status']=='completed' and completed['last_update']==CFG['total_updates'] and completed['smoke'] is False
 ckpath=job/'last_local.pth';cksha=sha256(ckpath);assert cksha==completed['final_checkpoint_sha256']
 stamp={'registration_sha256':sha256(REG),'checkpoint_sha256':cksha}
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 with gpu_job_lock(),torch.inference_mode():
  model=make_model().eval();ck=torch.load(ckpath,map_location='cpu',weights_only=False)
  assert ck['step']==CFG['total_updates'] and ck['signature']['group']==GROUP and ck['signature']['method']=='Xn'
  assert ck['signature']['config_sha256']==reg['config_sha256'] and ck['signature']['trainer_sha256']==reg['trainer_sha256']
  state=model.state_dict();assert set(ck['trainable_model'])=={k for k in state if not k.startswith('features.')}
  state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del ck,state
  assert not model.features.encoder.training and all(not p.requires_grad for p in model.features.encoder.parameters())
  for panel in PANELS:
   records=read(AB/'manifests_private'/(panel+'.json'))['records']
   for i,r in enumerate(records):
    paths=[OUT/'per_image_private'/panel/str(seed)/(r['id']+'.json') for seed in range(42,50)]
    for p in paths:
     if p.exists():
      row=read(p);assert row['registration_sha256']==stamp['registration_sha256'] and row['checkpoint_sha256']==cksha
    if all(p.exists() for p in paths):continue
    assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256']
    assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
    with np.load(r['gt_path']) as z:
     gt=z['normal'];valid=z['valid'].astype(bool)
     extra={f:z[f] for f in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if f in z.files}
    rgb=np.asarray(Image.open(r['white']).convert('RGB').resize((CFG['image_size'],CFG['image_size']),Image.Resampling.BILINEAR))
    rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
    for seed,path in zip(range(42,50),paths):
     if path.exists():continue
     noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
     with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
     pred=raw.float()[0].permute(1,2,0).cpu().numpy()
     write_json(path,{'model':GROUP,'method':'Xn','seed':seed,'id':r['id'],'object':r['object'],'panel':panel,'elevation_bin':r['elevation_bin'],'image_key':r['image_key'],'metrics':metrics(pred,gt,valid,extra),**stamp})
    if i%20==0 or i+1==len(records):
     s={'phase':'evaluating_'+GROUP,'panel':panel,'images':i+1,'total_images':len(records),'updated_at':now()};write_json(OUT/'status.json',s);print(json.dumps(s),flush=True)
  del model;gc.collect();torch.cuda.empty_cache()
 panels={}
 for panel in PANELS:
  records=read(AB/'manifests_private'/(panel+'.json'))['records'];images=[]
  for r in records:
   vals=[read(OUT/'per_image_private'/panel/str(seed)/(r['id']+'.json'))['metrics'] for seed in range(42,50)]
   images.append({'object':r['object'],'pitch':r['elevation_bin'],**{k:float(np.mean([v[k] for v in vals])) if vals[0][k] is not None else None for k in vals[0]}})
  bins=[0,15,30,45,60] if panel=='primary' else [30] if panel in ['original','arb_style'] else [None];pitches={}
  for pitch in bins:
   rows=[r for r in images if r['pitch']==pitch];objects=sorted({r['object'] for r in rows})
   mae=float(np.mean([np.mean([r['mean_deg'] for r in rows if r['object']==obj]) for obj in objects]));n=sum(r['up_pixels'] for r in rows)
   pitches[str(pitch)]={'objects':len(objects),'images':len(rows),'mean_deg':mae,'up_z_signed_error_pixel_pooled':sum(r['up_z_signed_error_sum'] for r in rows)/n if n else None}
  panels[panel]={'by_pitch':pitches,'equal_pitch_object_macro_mae_deg':float(np.mean([v['mean_deg'] for v in pitches.values()]))}
 previous=reg['previous_recipe'];value=panels['primary']['equal_pitch_object_macro_mae_deg'];improvement=previous['five_angle_MAE_deg']-value;retained=improvement>=.5
 result={'status':'completed','updated_at':now(),'candidate':GROUP,'panels':panels,'previous_recipe':previous,'improvement_in_five_angle_MAE_deg':improvement,'retained':retained,'retained_recipe':GROUP if retained else previous['name'],'next_step':'D3' if GROUP=='D2' else 'freeze_final_recipe','rule':'Retain iff unrounded improvement>=0.5deg; otherwise revert.','test_images_scored':0,**stamp}
 write_json(OUT/'summary.json',result);write_json(OUT/'status.json',{'phase':'completed_'+GROUP+'_retention_decision','updated_at':now(),'retained':retained})
 (OUT/'report.md').write_text(GROUP+'五角度MAE：'+str(value)+'度；相对'+previous['name']+'改善'+str(improvement)+'度。保留：'+str(retained)+'。按登记阶梯继续。\n')
 print(json.dumps(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now()});raise
