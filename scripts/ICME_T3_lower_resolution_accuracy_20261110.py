"""E1_B448/392 inference-only accuracy, original518weights unchanged; no test access."""
from pathlib import Path
import os,sys,json,fcntl,gc,types,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from src.research_io import write_json,sha256,now
from src.e_models_20261106 import make_model
from src.dino_conditioning import FrozenPatchEncoder
from src.moge_bridge_20261106 import tensor_rgb
from src.noise_protocol import image_noise_seed
from src.icme_metric_aggregation_20261110 import aggregate
from scripts.evaluate_fresh_ab import metrics
O=R/'outputs/T3_efficiency/lower_encoder_resolution';O.mkdir(exist_ok=True,parents=True)
def read(p):return json.loads(Path(p).read_text())
def main():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert read(R/'outputs/T2_zero_noise/default_inference.json')['default']=='random';spec=read(R/'provenance/E_evaluation_registration_20261106.json')['models']['E1_B'];cfg=read(R/spec['config_path']);ckpath=R/spec['checkpoint_path'];checkpoint_sha=sha256(ckpath);mp=R/'data/external_round_20261106/gso_primary.json';records=read(mp)['records'];results={}
 for side in [448,392]:
  model=make_model(cfg).eval();ck=torch.load(ckpath,map_location='cpu',weights_only=False);state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del ck,state
  def resized_forward(self,rgb):return FrozenPatchEncoder.forward(self,F.interpolate(rgb,(side,side),mode='bilinear',align_corners=False))
  model.features.forward=types.MethodType(resized_forward,model.features);native=model.features.forward;sig={'worker_sha256':sha256(__file__),'config_sha256':sha256(R/spec['config_path']),'checkpoint_sha256':checkpoint_sha,'manifest_sha256':sha256(mp),'encoder_side':side,'tokens':(side//14)**2,'weights_side_during_training':518,'weights_or_recipe_retrained':False,'inference_seeds':list(range(42,50)),'precision':'BF16 autocast FP32 parameterstorage','test_images_scored':0};folder=O/str(side);rows=[];parity=False
  for i,r in enumerate(records):
   p=folder/'metrics_private'/(r['id']+'.json')
   if p.exists():row=read(p);assert row['signature']==sig;rows.append(row);continue
   assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
   with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
   x=tensor_rgb(Image.open(r['white']).convert('RGB'),256)[None]*2-1;values=[]
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):cached=native(x)
   for seed in range(42,50):
    noise=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])));model.features.forward=lambda _:cached
    try:
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(noise,torch.zeros(1,device='cuda'),x)[0].float()
    finally:model.features.forward=native
    if not parity:
     with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):full=model(noise,torch.zeros(1,device='cuda'),x)[0].float()
     assert torch.equal(out,full);parity=True
    v=metrics(out.permute(1,2,0).cpu().numpy(),gt,valid,extra);values.append(v);write_json(folder/'per_seed_private'/str(seed)/(r['id']+'.json'),{'object':r['object'],'elevation_bin':r['elevation_bin'],'metrics':v,'signature':sig})
   reduced={key:float(np.mean([v[key] for v in values])) if all(v[key] is not None for v in values) else None for key in values[0]};row={'object':r['object'],'elevation_bin':r['elevation_bin'],'metrics':reduced,'signature':sig};write_json(p,row);rows.append(row);write_json(O/'status.json',{'phase':'accuracy_shared3090','encoder_side':side,'images':i+1,'total':len(records),'updated_at':now()})
  assert len(rows)==len(records);results[str(side)]={'signature':sig,'gso_primary':aggregate(rows)};write_json(folder/'summary.json',{'status':'completed','encoder_side':side,**results[str(side)],'test_images_scored':0,'updated_at':now()});del model;gc.collect();torch.cuda.empty_cache()
 result={'status':'completed_accuracy_latency_pending','variants':results,'default_inference':'random42-49 error mean','timings':'not measured while sharingGPU; independentexclusive3090worker required','test_images_scored':0,'updated_at':now()};write_json(O/'summary.json',result);write_json(O/'status.json',{'phase':'completed_accuracy','updated_at':now()});(O/'REPORT.md').write_text('# T3 编码器输入尺寸精度\n\n只改E1_B推理编码器输入518→448/392，训练权重、256UNet、mask和42–49误差均值协议保持。完整GSO-val指标与物体区间见summary.json。此进程并行训练，只报告精度；延迟另在独占3090上测。\n\n'+ '\n'.join(f"{s}: {d['gso_primary']['metrics']['mean_deg']['mean']:.6f}°; tokens{d['signature']['tokens']}" for s,d in results.items())+'\n');print('LOWER_RES_ACCURACY_COMPLETE',flush=True)
try:
 with (R/'runtime/ICME_T2_aux_GPU0.lock').open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX);main()
except Exception as e:write_json(O/'status.json',{'phase':'needs_attention','error':repr(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
