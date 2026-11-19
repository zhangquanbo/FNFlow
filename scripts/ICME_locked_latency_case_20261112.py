"""Pinned, training-free batch1 timing with external CPU/NVML telemetry; source/weights immutable."""
from pathlib import Path
import os,sys,json,time,subprocess,argparse,statistics,gc,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.icme_idle_guard_20261112 import assert_no_training
from src.icme_gpu_exclusivity_20261110 import assert_exclusive
from src.research_io import sha256,write_json,now
P=argparse.ArgumentParser();P.add_argument('--model',required=True,choices=['S1','MoGe-S','MoGe-B','E1_B']);P.add_argument('--variant',default='native',choices=['native','compile_channels_last']);P.add_argument('--repeat',type=int,required=True);P.add_argument('--components',action='store_true');P.add_argument('--affinity-control',action='store_true');args=P.parse_args();assert args.variant=='native' or args.model in ['S1','MoGe-S'];O=R/'outputs/T6_S1_locked_20261112/timing';case=args.model+'_'+args.variant+'_rep'+str(args.repeat);D=O/'cases'/case;D.mkdir(parents=True,exist_ok=True);resultpath=D/'summary.json';worker_sha=sha256(__file__);(D/'worker_source_private.py').write_bytes(Path(__file__).read_bytes());PIN={8,9,10,11};assert set(os.sched_getaffinity(0))==PIN;monitor=None;blocks=[]
def read(p):return json.loads(Path(p).read_text())
def state_write(phase,**k):write_json(O/'status.json',dict(phase=phase,case=case,model=args.model,variant=args.variant,repeat=args.repeat,updated_at=now(),**k))
def measure(fn,label,warmup=5):
 before=assert_no_training();exclusive=assert_exclusive();assert set(os.sched_getaffinity(0))==PIN or label.startswith('affinity_control_');state_write('measuring',block=label,warmup=warmup,timed=30);wall=[];events=[]
 with torch.inference_mode():
  for _ in range(warmup):value=fn();assert torch.isfinite(value).all();del value
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.monotonic_ns()
  for _ in range(30):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter_ns();a.record();value=fn();b.record();torch.cuda.synchronize();wall.append((time.perf_counter_ns()-start)/1e6);events.append(a.elapsed_time(b));del value
  finish=time.monotonic_ns()
 after=assert_no_training();assert_exclusive();row={'label':label,'warmup':warmup,'timed':30,'wall_ms':wall,'GPU_event_ms':events,'wall_median_ms':statistics.median(wall),'wall_p90_ms':float(np.quantile(wall,.9)),'GPU_event_median_ms':statistics.median(events),'GPU_event_p90_ms':float(np.quantile(events,.9)),'start_monotonic_ns':begin,'end_monotonic_ns':finish,'CPU_affinity':sorted(os.sched_getaffinity(0)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'global_training_scan_before':before,'global_training_scan_after':after,'exclusive_GPU':exclusive};blocks.append(row);return row
try:
 if resultpath.exists():raise RuntimeError('caseoutputexists,useanotherrepeatorpreserveandregisternewattempt')
 initial=assert_no_training();assert_exclusive();lock=read(R/'outputs/T6_S1_locked_20261112/model_lock_private.json');assert all(sha256(R/p)==h for p,h in lock['source_sha256'].items());tele=D/'telemetry_private';tele.mkdir(exist_ok=True);(tele/'stop').unlink(missing_ok=True)
 with (D/'telemetry.log').open('ab') as log:monitor=subprocess.Popen(['/usr/bin/python3','-u',str(R/'scripts/ICME_latency_telemetry_20261112.py'),str(os.getpid()),str(tele),'12'],stdout=log,stderr=subprocess.STDOUT)
 import numpy as np,torch
 from torch.nn import functional as F
 from PIL import Image
 from src.e_models_20261106 import make_model as legacy_model
 from src.icme_rplus_20261110 import make_model as s1_model
 from src.moge_bridge_20261106 import load_moge,normal_only,tensor_rgb
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42);assert '3090' in torch.cuda.get_device_name();record=min([x for x in read(R/'data/external_round_20261106/gso_primary.json')['records'] if x['elevation_bin']==30],key=lambda x:x['id']);assert sha256(record['white'])==record['white_sha256'];signature={}
 if args.model=='S1':
  cfg=read(R/lock['configuration_relative']);assert sha256(R/lock['configuration_relative'])==lock['configuration_sha256'];assert sha256(R/lock['checkpoint_relative'])==lock['checkpoint_sha256'];model=s1_model(cfg).eval();ck=torch.load(R/lock['checkpoint_relative'],map_location='cpu',weights_only=False);assert ck['step']==50000;state=dict(ck['model']);state.update(ck['ema']);model.load_state_dict(state,strict=True);model.features.set_gradient(False);del ck,state;signature={'checkpoint_sha256':lock['checkpoint_sha256'],'config_sha256':lock['configuration_sha256'],'variant':'EMA','effective_state_sha256':lock['effective_state_sha256']}
 elif args.model=='E1_B':
  spec=read(R/'provenance/E_evaluation_registration_20261106.json')['models']['E1_B'];cfg=read(R/spec['config_path']);assert sha256(R/spec['checkpoint_path'])=='ec13128db5c93d4731964b367b0fb294df716249fa03820194d5ed8f08ab5a66';model=legacy_model(cfg).eval();ck=torch.load(R/spec['checkpoint_path'],map_location='cpu',weights_only=False);state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del ck,state;signature={'checkpoint_sha256':sha256(R/spec['checkpoint_path']),'config_sha256':sha256(R/spec['config_path'])}
 else:
  model,_=load_moge(args.model[-1]);signature={'weights':read(R/'provenance/MoGe_E_round_order_20261106.json')['models'][args.model[-1]]}
 rgb=tensor_rgb(Image.open(record['white']).convert('RGB'))[None] if args.model.startswith('MoGe-') else (tensor_rgb(Image.open(record['white']).convert('RGB'),256)*2-1)[None];x=rgb.repeat(1,1,1,1);z=None if args.model.startswith('MoGe-') else torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(42));t=None if z is None else torch.zeros(1,device='cuda')
 def native():
  if args.model.startswith('MoGe-'):return normal_only(model,x,True)
  with torch.autocast('cuda',dtype=torch.bfloat16):return model(z,t,x)
 with torch.inference_mode():reference=native().float().clone()
 compile_seconds=0.;fn=native
 if args.variant=='compile_channels_last':
  model.to(memory_format=torch.channels_last);x=x.contiguous(memory_format=torch.channels_last)
  if z is not None:z=z.contiguous(memory_format=torch.channels_last)
  state_write('compiling_first_call_excluded');fn=torch.compile(native,mode='default');start=time.monotonic()
  with torch.inference_mode():out=fn().float();torch.cuda.synchronize()
  compile_seconds=time.monotonic()-start
 else:
  with torch.inference_mode():out=native().float()
 dim=-1 if args.model.startswith('MoGe-') else 1;angular=torch.rad2deg(torch.acos((F.normalize(reference.double(),dim=dim)*F.normalize(out.double(),dim=dim)).sum(dim).clamp(-1,1)));parity={'max_abs_normal_difference':float((reference-out).abs().max()),'mean_angular_output_difference_deg':float(angular.mean()),'max_angular_output_difference_deg':float(angular.max()),'full_GSO_accuracy_remeasured':False};del out,angular
 assert monitor.poll() is None,'telemetrysamplerfailed';primary=measure(fn,'full_model_primary');component={}
 if args.components:
  assert args.model=='S1' and args.variant=='native';original=model.features.forward
  def encoder():
   with torch.autocast('cuda',dtype=torch.bfloat16):return model.features(x)
  encoder_measure=measure(encoder,'encoder_only_including_518_preprocessing')
  with torch.inference_mode():features=encoder().detach()
  try:
   model.features.forward=lambda *unused,**kw:features
   with torch.inference_mode():head_output=native().float();head_delta=float((head_output-reference).abs().max());assert head_delta==0
   head_measure=measure(native,'prediction_head_only_cached_intermediate')
  finally:model.features.forward=original
  del head_output,features
  stages=[];current={};hooks=[]
  def model_pre(*unused):current.clear();current['a']=torch.cuda.Event(enable_timing=True);current['a'].record()
  def encoder_pre(*unused):current['b']=torch.cuda.Event(enable_timing=True);current['b'].record()
  def encoder_post(*unused):current['c']=torch.cuda.Event(enable_timing=True);current['c'].record()
  def model_post(*unused):current['d']=torch.cuda.Event(enable_timing=True);current['d'].record();stages.append(dict(current))
  hooks=[model.register_forward_pre_hook(model_pre),model.features.register_forward_pre_hook(encoder_pre),model.features.register_forward_hook(encoder_post),model.register_forward_hook(model_post)]
  try:profile_measure=measure(native,'full_forward_stage_event_profile')
  finally:
   for h in hooks:h.remove()
  gpu_parts=[{'total_ms':v['a'].elapsed_time(v['d']),'encoder_ms':v['b'].elapsed_time(v['c']),'head_ms':v['c'].elapsed_time(v['d']),'pre_encoder_ms':v['a'].elapsed_time(v['b'])} for v in stages[-30:]];component={'standalone_encoder':encoder_measure,'standalone_prediction_head':head_measure,'head_output_abs_difference':head_delta,'within_full_forward_GPU_events':gpu_parts,'stage_GPU_medians_ms':{key:statistics.median(v[key] for v in gpu_parts) for key in gpu_parts[0]},'scope':'primarycontainsfullencoder+head;cachedintermediateusedONLYforstandaloneheadmeasurement;stageeventsprofileisaseparatediagnostic'}
 if args.affinity_control:
  assert args.variant=='native';allcores=set(range(80));affinity_blocks=[]
  def bind(mask):
   for task in Path('/proc/self/task').iterdir():
    try:os.sched_setaffinity(int(task.name),mask)
    except ProcessLookupError:pass
  try:
   for label,mask in [('affinity_control_pinned_A',PIN),('affinity_control_legacy_all_B',allcores),('affinity_control_legacy_all_B2',allcores),('affinity_control_pinned_A2',PIN)]:bind(mask);affinity_blocks.append(measure(native,label))
  finally:bind(PIN)
 else:affinity_blocks=[]
 torch.cuda.synchronize();assert_no_training();assert_exclusive();(tele/'stop').touch();monitor.wait(timeout=10);assert monitor.returncode==0;telemetry_summary=read(tele/'summary.json');assert telemetry_summary['NVML_available'] and not telemetry_summary['training_seen'];samples=[json.loads(v) for v in (tele/'samples_private.jsonl').read_text().splitlines()]
 for block in blocks:
  values=[s for s in samples if block['start_monotonic_ns']<=s['monotonic_ns']<=block['end_monotonic_ns']];assert values,'notelemetrysampleinsideblock'
  fields=['CPU_one_core_percent','CPU_four_core_percent','GPU_power_W','GPU_temp_C','GPU_SM_MHz','GPU_mem_MHz','GPU_util_percent'];block['telemetry']={field:{'mean':float(np.mean([v[field] for v in values if v[field] is not None])),'min':min(v[field] for v in values if v[field] is not None),'max':max(v[field] for v in values if v[field] is not None)} for field in fields};block['telemetry']['samples']=len(values);block['telemetry']['GPU_Pstates']=sorted({v['GPU_Pstate'] for v in values});block['telemetry']['GPU_throttle_bits']=sorted({v['GPU_throttle_bits'] for v in values if v['GPU_throttle_bits'] is not None})
 result={'status':'completed','model':args.model,'variant':args.variant,'repeat':args.repeat,'CPU_affinity':sorted(PIN),'CPU_intra_threads':torch.get_num_threads(),'CPU_interop_threads':torch.get_num_interop_threads(),'primary':primary,'components':component,'affinity_control':affinity_blocks,'numerical_parity':parity,'compile_first_call_seconds_excluded':compile_seconds,'parameters_total':sum(p.numel() for p in model.parameters()),'precision':'FP16autocast/FP32weights' if args.model.startswith('MoGe-') else 'BF16autocast/FP32weights','encoder_input':840 if args.model.startswith('MoGe-') else 518,'encoder_tokens':3600 if args.model.startswith('MoGe-') else 1369,'model_signature':signature,'input_record_id':record['id'],'input_RGB_sha256':record['white_sha256'],'input_and_noise_and_t_GPU_resident':True,'TF32':False,'cudnn_benchmark':False,'cudnn_deterministic':False,'torch':torch.__version__,'CUDA':torch.version.cuda,'cuDNN':torch.backends.cudnn.version(),'GPU':torch.cuda.get_device_name(),'initial_global_training_scan':initial,'telemetry_summary':telemetry_summary,'worker_sha256':worker_sha,'model_lock_sha256':sha256(R/'outputs/T6_S1_locked_20261112/model_lock_private.json'),'test_images_scored':0,'updated_at':now()};write_json(resultpath,result);state_write('case_completed');print('PINNED_LATENCY_CASE_COMPLETE',case,primary['wall_median_ms'],flush=True)
except Exception as e:
 write_json(resultpath,{'status':'failed','model':args.model,'variant':args.variant,'repeat':args.repeat,'error':repr(e),'traceback':traceback.format_exc(),'worker_sha256':worker_sha,'test_images_scored':0,'updated_at':now()});raise
finally:
 if monitor is not None and monitor.poll() is None:
  (D/'telemetry_private/stop').touch()
  try:monitor.wait(timeout=10)
  except subprocess.TimeoutExpired:monitor.terminate()
