"""Exclusive3090 GPU-resident forward latency and fixed-batch throughput."""
from pathlib import Path
import sys,os,json,time,gc,argparse,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
import torch.nn.functional as F
from PIL import Image
from src.moge_bridge_20261106 import load_moge,load_d2,normal_only,tensor_rgb
from src.research_io import sha256,write_json,now
OUT=ROOT/'outputs/MoGe_E_round_20261106/forward_throughput';REG=ROOT/'provenance/GPU_forward_throughput_registration_20261106.json'
def read(p):return json.loads(Path(p).read_text())
def trained_fm(name):
 if name=='E2_L_direct':from src.e2l_direct_models_20261106 import make_model
 elif name=='E2_L':from src.e2l_models_20261106 import make_model
 else:from src.e_models_20261106 import make_model
 cfg=read(ROOT/'configs'/(name+'_actual_20261106.json'));model=make_model(cfg).eval();path=ROOT/cfg['output_root']/(name+'_Xn_s1247')/'last_local.pth';summary=read(path.parent/'summary.json');assert summary['status']=='completed' and summary['last_update']==50000 and not summary['smoke'] and sha256(path)==summary['final_checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);return model,summary['final_checkpoint_sha256']
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args();reg=read(REG);assert sha256(__file__)==reg['worker_sha256'];assert all(sha256(ROOT/p)==h for p,h in reg['source_sha256'].items());device=torch.cuda.get_device_name();assert args.smoke or '3090' in device
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 record=min([r for r in read(ROOT/'data/external_round_20261106/gso_primary.json')['records'] if r['elevation_bin']==30],key=lambda r:r['id']);assert sha256(record['white'])==record.get('white_sha256',record.get('ours_rgb_sha256'));image=Image.open(record['white']).convert('RGB');out=OUT/'smoke'/sha256(__file__)[:12] if args.smoke else OUT/'completed';out.mkdir(parents=True,exist_ok=True)
 with torch.inference_mode():
  for name in args.models:
   existing=out/(name+'.json')
   if existing.exists():
    d=read(existing);assert d['registration_sha256']==sha256(REG)
    if d['status']=='completed':continue
   auxiliary=None;checkpoint=None
   if name=='D2':model=load_d2();precision='bfloat16'
   elif name.startswith('MoGe-'):model,_=load_moge(name[-1]);precision='float16'
   elif name=='E3_L':
    model,_=load_moge('L');cfg=read(ROOT/'configs/E3_L_actual_20261106.json');path=ROOT/cfg['output_root']/'E3_L_s1247/last_local.pth';summary=read(path.parent/'summary.json');assert summary['status']=='completed' and summary['last_update']==50000 and not summary['smoke'] and sha256(path)==summary['final_checkpoint_sha256'];checkpoint=summary['final_checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);model.normal_head.load_state_dict(ck['head'],strict=True);del ck;precision='float16'
   else:
    model,checkpoint=trained_fm(name);precision='bfloat16'
    if name.startswith('E2_'):auxiliary,_=load_moge('L' if name.startswith('E2_L') else 'B')
   if args.smoke and precision=='bfloat16':precision='float32_smoke'
   parameters=sum(p.numel() for p in model.parameters())+(sum(p.numel() for p in auxiliary.parameters()) if auxiliary is not None else 0);rows=[]
   for batch in ([1,2] if args.smoke else reg['batch_sizes']):
    write_json(OUT/'status.json',{'status':'running','model':name,'batch_size':batch,'updated_at':now()});x=tensor_rgb(image)[None].repeat(batch,1,1,1);rgb=(tensor_rgb(image,256)*2-1)[None].repeat(batch,1,1,1);noise=torch.randn((batch,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));t=torch.zeros(batch,device='cuda')
    def forward():
     if name.startswith('MoGe-') or name=='E3_L':return normal_only(model,x,use_fp16=True)
     condition=rgb
     if auxiliary is not None:
      coarse=normal_only(auxiliary,x,use_fp16=True).permute(0,3,1,2);coarse=F.normalize(-F.interpolate(coarse,(256,256),mode='bilinear',align_corners=False),dim=1);coarse=F.normalize(coarse.half().float(),dim=1);condition=torch.cat([rgb,coarse],dim=1)
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=not args.smoke):return model(noise,t,condition)
    try:
     for _ in range(1 if args.smoke else reg['warmup']):output=forward();assert torch.isfinite(output).all();del output
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall=[];event=[]
     for trial in range(2 if args.smoke else reg['timed']):
      a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter();a.record();output=forward();b.record();torch.cuda.synchronize();elapsed=time.perf_counter()-start;wall.append(elapsed);event.append(a.elapsed_time(b));del output
     row={'batch_size':batch,'status':'completed','synchronized_wall_median_ms':statistics.median(wall)*1000,'GPU_event_median_ms':statistics.median(event),'p90_wall_ms':float(np.quantile(wall,.9))*1000,'throughput_images_per_second':batch/statistics.median(wall),'per_image_median_ms':statistics.median(wall)*1000/batch,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'wall_seconds_private':wall,'GPU_event_ms_private':event}
    except torch.cuda.OutOfMemoryError:
     row={'batch_size':batch,'status':'out_of_memory','throughput_images_per_second':None};torch.cuda.empty_cache()
    rows.append(row);del x,rgb,noise,t;gc.collect();torch.cuda.empty_cache()
    write_json(existing,{'status':'partial','model':name,'rows':rows,'parameters_total_including_condition_model':parameters,'compute_precision':precision,'condition_model_precision':'float16' if auxiliary is not None else None,'registration_sha256':sha256(REG),'checkpoint_sha256':checkpoint,'GPU':device,'torch':torch.__version__,'smoke':args.smoke})
   d=read(existing);d['status']='completed';d['input_output_io_included']=False;d['host_to_device_transfer_included']=False;d['random_tensor_generation_included']=False;d['preloaded_input_shapes']={'MoGe':[3,512,512],'FM':[3,256,256],'FM_DINO_prior_side':518,'MoGe_encoder_max_tokens':3600};write_json(existing,d);del model,auxiliary;gc.collect();torch.cuda.empty_cache();print('FORWARD_THROUGHPUT_COMPLETED',name,flush=True)
 write_json(OUT/'status.json',{'status':'completed_requested_models','models':args.models,'smoke':args.smoke,'updated_at':now()})
if __name__=='__main__':main()
