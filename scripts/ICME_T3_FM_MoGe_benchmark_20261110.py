"""T3 native and symmetric optimization variants, exclusiveRTX3090 only."""
from pathlib import Path
import sys,os,json,time,gc,statistics,types,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from src.research_io import write_json,sha256,now
from src.e_models_20261106 import make_model
from src.dino_conditioning import FrozenPatchEncoder
from src.moge_bridge_20261106 import load_moge,load_d2,normal_only,common256,tensor_rgb
from src.noise_protocol import image_noise_seed
from src.icme_gpu_exclusivity_20261110 import assert_exclusive
NAME=sys.argv[1];VARIANT=sys.argv[2];SIDE=int(sys.argv[3]) if len(sys.argv)>3 else 518
assert NAME in ['E1_B','D2','MoGe-S','MoGe-B','MoGe-L'];assert VARIANT in ['native','compile','channels_last','compile_channels_last'];assert SIDE in [518,448,392];assert VARIANT=='native' or NAME in ['E1_B','MoGe-S'];assert SIDE==518 or NAME=='E1_B'
O=R/'outputs/T3_efficiency';CASE=NAME+'_'+VARIANT+(('_'+str(SIDE)) if NAME=='E1_B' else '');P=O/'models'/(CASE+'.json');P.parent.mkdir(exist_ok=True);S=O/'status.json'
def read(p):return json.loads(Path(p).read_text())
def measure(fn,batch):
 with torch.inference_mode():
  for _ in range(5):y=fn();assert torch.isfinite(y).all();del y
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall=[];event=[]
  for i in range(30):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();t=time.perf_counter();a.record();y=fn();b.record();torch.cuda.synchronize();wall.append(time.perf_counter()-t);event.append(a.elapsed_time(b));del y
   write_json(S,{'phase':'exclusive3090_timing','case':CASE,'batch':batch,'done':i+1,'total':30,'updated_at':now()})
 return {'status':'completed','batch':batch,'warmup':5,'timed':30,'median_ms':statistics.median(wall)*1000,'p90_ms':float(np.quantile(wall,.9))*1000,'CUDA_event_median_ms':statistics.median(event),'images_per_second':batch/statistics.median(wall),'per_image_ms':statistics.median(wall)*1000/batch,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds_private':wall}
try:
 if P.exists() and read(P).get('status')=='completed':assert read(P)['worker_sha256']==sha256(__file__);sys.exit(0)
 exclusive=assert_exclusive();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 mp=R/'data/external_round_20261106/gso_primary.json';r=min([r for r in read(mp)['records'] if r['elevation_bin']==30],key=lambda x:x['id']);assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'));im=Image.open(r['white']).convert('RGB');signature={}
 if NAME=='E1_B':
  spec=read(R/'provenance/E_evaluation_registration_20261106.json')['models'][NAME];assert sha256(R/spec['config_path'])==spec['config_sha256'];assert sha256(R/spec['checkpoint_path'])=='ec13128db5c93d4731964b367b0fb294df716249fa03820194d5ed8f08ab5a66';cfg=read(R/spec['config_path']);model=make_model(cfg).eval();ck=torch.load(R/spec['checkpoint_path'],map_location='cpu',weights_only=False);assert ck['step']==50000;state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del ck,state;signature={'config_sha256':sha256(R/spec['config_path']),'checkpoint_sha256':sha256(R/spec['checkpoint_path'])}
  if SIDE!=518:
   def resized(self,rgb):return FrozenPatchEncoder.forward(self,F.interpolate(rgb,(SIDE,SIDE),mode='bilinear',align_corners=False))
   model.features.forward=types.MethodType(resized,model.features)
 elif NAME=='D2':model=load_d2()
 else:model,cfg=load_moge(NAME[-1]);signature={'weights':read(R/'provenance/MoGe_E_round_order_20261106.json')['models'][NAME[-1]]}
 rows=[];parity=[];parameters=sum(p.numel() for p in model.parameters());native_rgb=tensor_rgb(im)[None] if NAME.startswith('MoGe-') else tensor_rgb(im,256)[None]*2-1
 for batch in [1,4]:
  x=native_rgb.repeat(batch,1,1,1);noise=None if NAME.startswith('MoGe-') else torch.randn((batch,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));t=None if NAME.startswith('MoGe-') else torch.zeros(batch,device='cuda')
  def native():
   if NAME.startswith('MoGe-'):return normal_only(model,x,True)
   with torch.autocast('cuda',dtype=torch.bfloat16):return model(noise,t,x)
  try:
   if 'channels_last' in VARIANT:model.to(memory_format=torch.contiguous_format)
   with torch.inference_mode():reference=native().float()
   if 'channels_last' in VARIANT:model.to(memory_format=torch.channels_last);x=x.contiguous(memory_format=torch.channels_last);noise=noise.contiguous(memory_format=torch.channels_last) if noise is not None else None
   fn=torch.compile(native) if 'compile' in VARIANT else native
   with torch.inference_mode():out=fn().float()
   assert torch.isfinite(out).all();dim=-1 if NAME.startswith('MoGe-') else 1;angular=torch.rad2deg(torch.acos((F.normalize(reference.double(),dim=dim)*F.normalize(out.double(),dim=dim)).sum(dim).clamp(-1,1)));parity.append({'batch':batch,'max_abs_raw_normal_difference':float((reference-out).abs().max()),'mean_angular_output_drift_deg':float(angular.mean()),'max_angular_output_drift_deg':float(angular.max()),'accuracy_scope':'optimization latency variant only, not silently paired with fullnativeaccuracy; fullGSO accuracy specificallyprovided for448/392'});del reference,out,angular
   row=measure(fn,batch)
  except torch.cuda.OutOfMemoryError as e:row={'status':'out_of_memory','batch':batch,'error':str(e)}
  rows.append(row);del x,noise,t;gc.collect();torch.cuda.empty_cache()
 end_to_end=None
 if NAME=='E1_B' and VARIANT=='native' and SIDE==518:
  path=O/'models/E1_B_e2e_timed_private.png'
  def e2e():
   with Image.open(r['white']) as src:im=src.convert('RGB')
   rgb=tensor_rgb(im,256)[None]*2-1;z=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,r['image_key'])))
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):y=model(z,torch.zeros(1,device='cuda'),rgb)
   n=common256(y.float()[0].permute(1,2,0),'D2').cpu().numpy();Image.fromarray(np.rint((n+1)*127.5).clip(0,255).astype(np.uint8)).save(path)
  for _ in range(5):e2e()
  samples=[]
  for _ in range(30):torch.cuda.synchronize();start=time.perf_counter();e2e();torch.cuda.synchronize();samples.append(time.perf_counter()-start)
  end_to_end={'median_ms':statistics.median(samples)*1000,'p90_ms':float(np.quantile(samples,.9))*1000,'wall_seconds_private':samples,'scope':'PNGread/preprocess/predict/common256unitnormal/PNGwrite, excludesmodelinitialization'}
 assert_exclusive();result={'status':'completed','model':NAME,'variant':VARIANT,'compile_mode':'default' if 'compile' in VARIANT else None,'compiler_workers':os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'),'encoder_input_side':840 if NAME.startswith('MoGe-') else SIDE,'encoder_tokens':3600 if NAME.startswith('MoGe-') else (SIDE//14)**2,'UNet_side':None if NAME.startswith('MoGe-') else 256,'raw_input_hw':list(native_rgb.shape[-2:]),'compute_precision':'FP16 autocast' if NAME.startswith('MoGe-') else 'BF16 autocast','weight_storage_precision':'FP32','rows':rows,'numerical_parity':parity,'end_to_end':end_to_end,'parameters_total':parameters,'model_signature':signature,'pure_forward_scope':'GPUresidentRGB/noise/t; nativeE-roundnormal-onlygraph; noIO/transfer/noisecreation; compiledefaultmode, firstcompilationexcluded','exclusive':exclusive,'TF32':False,'torch':torch.__version__,'CUDA':torch.version.cuda,'source_sha256':{p:sha256(R/p) for p in ['src/moge_bridge_20261106.py','src/e_models_20261106.py','src/dino_conditioning.py','src/dino_canonical_518.py','src/icme_gpu_exclusivity_20261110.py']},'worker_sha256':sha256(__file__),'test_images_scored':0,'updated_at':now()};write_json(P,result);print('T3_MODEL_COMPLETE',CASE,flush=True)
except Exception as e:write_json(P,{'status':'needs_attention','model':NAME,'variant':VARIANT,'compile_mode':'default' if 'compile' in VARIANT else None,'compiler_workers':os.environ.get('TORCHINDUCTOR_COMPILE_THREADS'),'encoder_input_side':SIDE,'error':repr(e),'traceback':traceback.format_exc(),'worker_sha256':sha256(__file__),'updated_at':now()});raise
