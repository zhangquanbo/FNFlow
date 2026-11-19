"""External sampler: CPU on a separate pinned core, GPU NVML, no model execution."""
from pathlib import Path
import os,sys,time,json,ctypes,statistics
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.icme_idle_guard_20261112 import training_processes
pid=int(sys.argv[1]);folder=Path(sys.argv[2]);core=int(sys.argv[3]);os.sched_setaffinity(0,{core});folder.mkdir(parents=True,exist_ok=True);stop=folder/'stop';hz=os.sysconf('SC_CLK_TCK');lib=ctypes.CDLL('libnvidia-ml.so.1');assert lib.nvmlInit_v2()==0;handle=ctypes.c_void_p();assert lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(handle))==0
class Util(ctypes.Structure):_fields_=[('gpu',ctypes.c_uint),('memory',ctypes.c_uint)]
def scalar(name,kind=ctypes.c_uint,*extra):
 value=kind();rc=getattr(lib,name)(handle,*extra,ctypes.byref(value));return value.value if rc==0 else None
def ticks():
 try:
  a=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split();return (None,None) if a[0] in ['Z','X'] else (int(a[11])+int(a[12]),int(a[36]))
 except OSError:return None,None
samples=[];previous=None;scan={};next_scan=0;last_tick,last_cpu=ticks();last_time=time.monotonic_ns()
try:
 with (folder/'samples_private.jsonl').open('w') as f:
  while not stop.exists():
   tick,cpu=ticks()
   if tick is None:break
   ts=time.monotonic_ns();dt=(ts-last_time)/1e9;usage=(tick-last_tick)/hz/dt*100 if dt>0 and last_tick is not None else None;last_time=ts;last_tick=tick
   if time.monotonic()>=next_scan:scan=training_processes();next_scan=time.monotonic()+1
   util=Util();rc=lib.nvmlDeviceGetUtilizationRates(handle,ctypes.byref(util));row={'monotonic_ns':ts,'unix':time.time(),'target_pid':pid,'CPU_one_core_percent':usage,'CPU_four_core_percent':usage/4 if usage is not None else None,'last_CPU':cpu,'GPU_power_W':(lambda x:x/1000 if x is not None else None)(scalar('nvmlDeviceGetPowerUsage')),'GPU_power_limit_W':(lambda x:x/1000 if x is not None else None)(scalar('nvmlDeviceGetEnforcedPowerLimit')),'GPU_temp_C':scalar('nvmlDeviceGetTemperature',ctypes.c_uint,0),'GPU_SM_MHz':scalar('nvmlDeviceGetClockInfo',ctypes.c_uint,1),'GPU_mem_MHz':scalar('nvmlDeviceGetClockInfo',ctypes.c_uint,2),'GPU_Pstate':scalar('nvmlDeviceGetPerformanceState'),'GPU_throttle_bits':scalar('nvmlDeviceGetCurrentClocksThrottleReasons',ctypes.c_ulonglong),'GPU_util_percent':util.gpu if rc==0 else None,'GPU_mem_util_percent':util.memory if rc==0 else None,'training_scan':scan};samples.append(row);f.write(json.dumps(row)+'\n');f.flush();time.sleep(.2)
finally:
 lib.nvmlShutdown();(folder/'summary.json').write_text(json.dumps({'status':'completed','samples':len(samples),'sampler_CPU_affinity':[core],'period_seconds':.2,'CPU_percent_definition':'100percent=onecore;four_core_percentdivide4','training_seen':any(x.get('training_scan',{}).get('matches') for x in samples),'NVML_available':bool(samples and samples[0]['GPU_power_W'] is not None)},indent=2)+'\n')
