from pathlib import Path
import sys,os,json,fcntl,gc,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from PIL import Image
from src.research_io import sha256,write_json,now
from src.moge_bridge_20261106 import load_moge,normal_only,tensor_rgb,common256
from src.e2l_models_20261106 import make_model
from src.noise_protocol import image_noise_seed
from scripts.evaluate_fresh_ab import metrics
OUT=ROOT/'outputs/Gbuffer_round_20261110/Arb_normal_diagnostic'
def read(p):return json.loads(Path(p).read_text())
def main():
 reg=read(OUT/'registration_private.json');assert sha256(__file__)==reg['worker_sha256'];assert sha256(OUT/'manifest_private.json')==reg['manifest_sha256'];rows=read(OUT/'manifest_private.json')['records'];torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 coarse,_=load_moge('L');cfg=read(ROOT/'configs/E2_L_actual_20261106.json');model=make_model(cfg).eval();path=ROOT/reg['E2_checkpoint'];assert sha256(path)==reg['E2_checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del ck,state
 with torch.inference_mode():
  for i,r in enumerate(rows):
   baseline=OUT/'metrics/MoGe-L'/(r['id']+'.json');refined=OUT/'metrics/E2_L'/(r['id']+'.json')
   if baseline.exists() and refined.exists():continue
   assert sha256(r['path'])==r['npz_sha256'] and sha256(r['native_RGB'])==r['source_rgb_sha256']
   with np.load(r['path']) as d:gt=d['normal'].copy();valid=d['valid'].astype(bool)
   rgba=np.asarray(Image.open(r['native_RGB']).convert('RGBA')).astype(np.float32)/255;white=np.rint((rgba[...,:3]*rgba[...,3:4]+1-rgba[...,3:4])*255).clip(0,255).astype(np.uint8);image=Image.fromarray(white);raw=normal_only(coarse,tensor_rgb(image),True);bmetrics=metrics(-raw.cpu().numpy(),gt,valid)
   write_json(baseline,{'model':'MoGe-L','scene':r['scene'],'image_key':r['image_key'],'metrics':bmetrics,'registration_sha256':sha256(OUT/'registration_private.json')})
   condition=common256(raw,'MoGe-L').permute(2,0,1);condition=torch.nn.functional.normalize(condition.half().float(),dim=0);condition=torch.cat([tensor_rgb(image,256)*2-1,condition],0)[None];values=[]
   for seed in range(42,50):
    noise=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
    with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(noise,torch.zeros(1,device='cuda'),condition)
    values.append(metrics(pred.float()[0].permute(1,2,0).cpu().numpy(),gt,valid))
   reduced={k:sum(x[k] for x in values)/8 if all(x[k] is not None for x in values) else None for k in values[0]};write_json(refined,{'model':'E2_L','scene':r['scene'],'image_key':r['image_key'],'metrics':reduced,'registration_sha256':sha256(OUT/'registration_private.json')});write_json(OUT/'status.json',{'phase':'MoGe_L_and_E2_L','images':i+1,'total':len(rows),'updated_at':now()})
  del model;gc.collect();torch.cuda.empty_cache();path=ROOT/reg['E3_checkpoint'];assert sha256(path)==reg['E3_checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);coarse.normal_head.load_state_dict(ck['head'],strict=True);del ck
  for i,r in enumerate(rows):
   dest=OUT/'metrics/E3_L'/(r['id']+'.json')
   if dest.exists():continue
   assert sha256(r['path'])==r['npz_sha256'] and sha256(r['native_RGB'])==r['source_rgb_sha256']
   with np.load(r['path']) as d:gt=d['normal'].copy();valid=d['valid'].astype(bool)
   rgba=np.asarray(Image.open(r['native_RGB']).convert('RGBA')).astype(np.float32)/255;image=Image.fromarray(np.rint((rgba[...,:3]*rgba[...,3:4]+1-rgba[...,3:4])*255).clip(0,255).astype(np.uint8));pred=-normal_only(coarse,tensor_rgb(image),True).cpu().numpy();write_json(dest,{'model':'E3_L','scene':r['scene'],'image_key':r['image_key'],'metrics':metrics(pred,gt,valid),'registration_sha256':sha256(OUT/'registration_private.json')});write_json(OUT/'status.json',{'phase':'E3_L','images':i+1,'total':len(rows),'updated_at':now()})
 results={}
 for name in ['MoGe-L','E2_L','E3_L']:
  files=list((OUT/'metrics'/name).glob('*.json'));assert {p.stem for p in files}=={r['id'] for r in rows};objects={}
  for p in files:
   j=read(p);objects.setdefault(j['scene'],[]).append(j['metrics']['mean_deg'])
  results[name]={'assets':len(objects),'views':len(files),'object_macro_MAE_deg':sum(sum(v)/len(v) for v in objects.values())/len(objects)}
 write_json(OUT/'summary_all198.json',{'status':'completed','scope':'all198 heldout assets,594views; requested128 LightProp-excluded subset must be aggregated separately','models':results,'GT':'existing Cycles BSDF shading normals','training_UID_overlap':0,'test_images_scored':0,'updated_at':now()});write_json(OUT/'status.json',{'phase':'completed_all198','subset128':'pending exact metadata-based selection','updated_at':now()});print('ARB_DIAGNOSTIC_COMPLETE',json.dumps(results),flush=True)
if __name__=='__main__':
 with (ROOT/'runtime/external_round_GPU0.lock').open('a') as lease:
  fcntl.flock(lease,fcntl.LOCK_EX)
  try:main()
  except Exception as e:write_json(OUT/'status.json',{'phase':'failed','error':repr(e),'updated_at':now()});raise
