
from pathlib import Path
import fcntl,subprocess,json,os,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/D2_followups_20261106';OUT.mkdir(exist_ok=True)
def status(value):
 p=OUT/'status.json';t=p.with_suffix('.partial');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(p)
with (OUT/'controller.lock').open('a') as own:
 fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (ROOT/'runtime/external_round_GPU0.lock').open('a') as lock:
  while True:
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:status({'phase':'waiting_for_priority_real_comparison','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')});time.sleep(10)
  jobs=[('D2_efficiency','train','benchmark_D2_3090_20261106.py',['--model','D2']),('NLR_efficiency','eval','benchmark_D2_3090_20261106.py',['--model','NeuralLightRig']),('D2_TTA','train','evaluate_D2_TTA_20261106.py',[])]
  completed=[]
  for name,env,script,args in jobs:
   status({'phase':'running','job':name,'previous_jobs':completed,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
   with (OUT/(name+'.log')).open('ab') as log:
    p=subprocess.Popen([str(ROOT.parent/'envs'/env/'bin/python'),'-u','-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'))
    code=p.wait()
   completed.append({'job':name,'exit_code':code})
  status({'phase':'completed' if all(x['exit_code']==0 for x in completed) else 'needs_attention','jobs':completed,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
