
"""Versioned model dispatch; complete registered real-domain retries before GSO jobs."""
from pathlib import Path
import sys,os,json,subprocess,time,fcntl
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/external_round_20261106';args=sys.argv[1:]
name=args[args.index('--model')+1];scope=args[args.index('--scope')+1]
def worker(model):
 version='v2' if model=='MarigoldNormal' else 'v3' if model in ('StableNormal','NiRNE') else 'v1'
 return ROOT/'scripts'/f'evaluate_external_round_{version}_20261106.py'
if scope=='gso':
 with (OUT/'real_retry_dispatch.lock').open('a') as lock:
  owns=False
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);owns=True
  except BlockingIOError:pass
  if owns:
   for retry_model in ('NiRNE','StableNormal','MarigoldNormal'):
    final=OUT/retry_model/'real/summary.json'
    if final.exists():continue
    state=OUT/'real_priority_retry.json'
    def write(phase,**extra):
     temp=state.with_suffix('.partial');temp.write_text(json.dumps({'phase':phase,'model':retry_model,'scope':'real','physical_GPU':int(os.environ['CUDA_VISIBLE_DEVICES']),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra},indent=2)+'\n');temp.replace(state)
    write('running')
    with (OUT/(retry_model+'_real_retry.log')).open('ab') as log:
     process=subprocess.Popen([sys.executable,'-u','-B',str(worker(retry_model)),'--model',retry_model,'--scope','real'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
     write('running',pid=process.pid);code=process.wait()
    write('completed' if code==0 else 'failed',exit_code=code)
if name=='StableNormal' and scope=='gso' and not (OUT/'DSINE/gso/summary.json').exists():
 status=OUT/'gso_priority_retry.json'
 def mark(phase,**extra):
  temp=status.with_suffix('.partial');temp.write_text(json.dumps({'phase':phase,'model':'DSINE','scope':'gso','physical_GPU':int(os.environ['CUDA_VISIBLE_DEVICES']),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra},indent=2)+'\n');temp.replace(status)
 mark('running')
 with (OUT/'DSINE_gso_retry.log').open('ab') as log:
  process=subprocess.Popen([sys.executable,'-u','-B',str(worker('DSINE')),'--model','DSINE','--scope','gso'],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  mark('running',pid=process.pid);code=process.wait()
 mark('completed' if code==0 else 'failed',exit_code=code)
target=worker(name)
os.execv(sys.executable,[sys.executable,'-u','-B',str(target),*args])
