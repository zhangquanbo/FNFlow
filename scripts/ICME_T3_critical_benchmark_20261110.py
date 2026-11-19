"""Critical S1/B1 and MoGe-S RTX3090 timings, after fit exits and before the next fit."""
from pathlib import Path
import os,sys,json,time,gc,statistics,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.icme_rplus_20261110 import make_model
from src.e_models_20261106 import make_model as legacy_model
from src.moge_bridge_20261106 import load_moge,normal_only,common256,tensor_rgb
from src.noise_protocol import image_noise_seed
from src.icme_gpu_exclusivity_20261110 import assert_exclusive
RUN=sys.argv[1];assert RUN in ['S1','B1','S2','S0'];O=R/'outputs/T3_efficiency';(O/'critical').mkdir(exist_ok=True);assert '3090' in torch.cuda.get_device_name();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def measure(fn):
 with torch.inference_mode():
  for _ in range(5):v=fn();assert torch.isfinite(v).all();del v
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();samples=[];events=[]
  for _ in range(30):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter();a.record();v=fn();b.record();torch.cuda.synchronize();samples.append(time.perf_counter()-start);events.append(a.elapsed_time(b));del v
 return {'status':'completed','warmup':5,'timed':30,'median_ms':statistics.median(samples)*1000,'p90_ms':float(np.quantile(samples,.9))*1000,'CUDA_event_median_ms':statistics.median(events),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timings_seconds':samples}
try:
 assert_exclusive()
 manifest=R/'data/external_round_20261106/gso_primary.json';record=min([r for r in json.loads(manifest.read_text())['records'] if r['elevation_bin']==30],key=lambda x:x['id']);assert sha256(record['white'])==record.get('white_sha256',record.get('ours_rgb_sha256'));results={}
 for name in [RUN,'MoGe-S']:
  output=O/'critical'/(name+'.json')
  if name=='MoGe-S' and output.exists():assert json.loads(output.read_text())['status']=='completed';continue
  if name==RUN:
   cfgpath=R/'outputs/T5_Rplus'/RUN/'actual.json';cfg=json.loads(cfgpath.read_text());checkpoint=R/cfg['checkpoint_relative'] if name=='S0' else R/cfg['output_root']/'final_50000.pth';ck=torch.load(str(checkpoint),map_location='cpu',weights_only=False);assert ck['step']==50000;model=(legacy_model(cfg) if name=='S0' else make_model(cfg)).eval();state=model.state_dict() if name=='S0' else dict(ck['model']);state.update(ck['trainable_model'] if name=='S0' else ck['ema']);model.load_state_dict(state,strict=True)
   if name!='S0':
    registered=json.loads((R/cfg['registration_path']).read_text());assert sha256(cfgpath)==registered['config_sha256'];assert ck['signature']['config_sha256']==registered['config_sha256'] and ck['signature']['registration_sha256']==sha256(R/cfg['registration_path']) and not ck['signature']['smoke'];model.features.set_gradient(False)
   del ck,state;signature={'configuration_sha256':sha256(cfgpath),'checkpoint_sha256':sha256(checkpoint),'weights':'raw' if name=='S0' else 'EMA','precision':'BF16 autocast,FP32 parameter storage','encoder_input_side':518,'tokens':1369};rgb=tensor_rgb(Image.open(record['white']).convert('RGB'),256)*2-1
  else:
   model,_=load_moge('S');rgb=tensor_rgb(Image.open(record['white']).convert('RGB'));signature={'precision':'FP16 autocast,FP32 parameter storage','normal_path':'verified normal-only forward,SDPA','encoder_input_side':840,'tokens':3600,'weights_sha256':'79a16621928c2bf0ed04659218c55c01075e950507f40bb3332fb4c873d3e1dc'}
  total=sum(p.numel() for p in model.parameters());rows=[]
  for batch in [1,4]:
   x=rgb[None].repeat(batch,1,1,1);noise=None if name=='MoGe-S' else torch.randn((batch,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));t=None if name=='MoGe-S' else torch.zeros(batch,device='cuda')
   def forward():
    if name=='MoGe-S':return normal_only(model,x,True)
    with torch.autocast('cuda',dtype=torch.bfloat16):return model(noise,t,x)
   try:row=measure(forward);row.update(batch=batch,images_per_second=batch/(row['median_ms']/1000))
   except torch.cuda.OutOfMemoryError:row={'status':'out_of_memory','batch':batch}
   rows.append(row);del x,noise,t;gc.collect();torch.cuda.empty_cache()
  timings=[]
  if name==RUN:
   def e2e():
    with Image.open(record['white']) as opened:im=opened.convert('RGB')
    x=tensor_rgb(im,256)[None]*2-1;z=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,record['image_key'])))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model(z,torch.zeros(1,device='cuda'),x)
    n=common256(p.float()[0].permute(1,2,0),'D2').cpu().numpy();Image.fromarray(np.rint((n+1)*127.5).clip(0,255).astype(np.uint8)).save(O/'critical'/(RUN+'_timed_output_private.png'))
   for _ in range(5):e2e()
   for _ in range(30):torch.cuda.synchronize();a=time.perf_counter();e2e();torch.cuda.synchronize();timings.append(time.perf_counter()-a)
  result={'status':'completed','model':name,'rows':rows,'parameters_total':total,'end_to_end':{'median_ms':statistics.median(timings)*1000,'p90_ms':float(np.quantile(timings,.9))*1000,'timings_seconds':timings,'scope':'PNG read/preprocess/predict/common256 normal/PNG write; excludes model load'} if timings else None,'model_signature':signature,'native_raw_input_hw':list(rgb.shape[-2:]),'pure_forward_scope':'GPU-resident input/noise/t, noIO/host transfer or noise generation; native E-round graph, FM raw output before common normal conversion; MoGe normal-only graph contains its intrinsic normal normalization','T6_latency':'batch1 native-forward median_ms, one forward per inference; accuracy averages8 error draws, not an8-forward ensemble','GPU':torch.cuda.get_device_name(),'TF32':False,'worker_sha256':sha256(__file__),'test_images_scored':0,'updated_at':now()};assert_exclusive();write_json(output,result);results[name]=result;del model,rgb;gc.collect();torch.cuda.empty_cache()
 p=(R/cfg['checkpoint_relative']).parent/'REPORT.md' if RUN=='S0' else R/'outputs/T5_Rplus'/RUN/'REPORT.md'
 with p.open('a') as stream:stream.write('\n\n3090纯前向batch1：'+str(results[RUN]['rows'][0].get('median_ms'))+' ms；端到端：'+str(results[RUN]['end_to_end']['median_ms'])+' ms；参数量'+str(results[RUN]['parameters_total'])+'。原始/EMA权重同架构，S0使用原始权重，其余使用登记EMA测量。完整记录见outputs/T3_efficiency/critical。\n')
 write_json(O/'critical'/('status_'+RUN+'.json'),{'status':'completed','run':RUN,'updated_at':now()});print('CRITICAL_3090_BENCHMARK_COMPLETE',RUN,flush=True)
except Exception as e:write_json(O/'critical'/('status_'+RUN+'.json'),{'status':'needs_attention','error':repr(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
