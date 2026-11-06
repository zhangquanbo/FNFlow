"""Finite per-device queue. Uses only preregistered idle GPUs0,1,3."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
import subprocess
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/external_round_20261106'
QUEUES={
 0:[('D2','real'),('NeuralLightRig','real')],
 1:[('DSINE','real'),('MoGe2','real'),('StableNormal','real'),('DSINE','gso'),('MoGe2','gso'),('StableNormal','gso')],
 3:[('Lotus-D','real'),('Lotus-G','real'),('MarigoldNormal','real'),('NiRNE','real'),('Lotus-D','gso'),('Lotus-G','gso'),('MarigoldNormal','gso'),('NiRNE','gso')]
}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_suffix('.partial')
 temporary.write_text(json.dumps(value,indent=2)+'\n')
 temporary.replace(path)
def run_device(device):
 with (ROOT/'runtime'/f'external_round_GPU{device}.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  results=[]
  for name,scope in QUEUES[device]:
   final=OUT/name/scope/'summary.json'
   if final.exists() and json.loads(final.read_text()).get('status')=='completed':
    results.append({'model':name,'scope':scope,'status':'already_completed'})
    continue
   family='Lotus' if name.startswith('Lotus-') else name
   python=ROOT.parent/'envs/train/bin/python' if name=='D2' else ROOT.parent/'envs/eval/bin/python' if name=='NeuralLightRig' else ROOT/'external_baselines_20261030'/family/'env/bin/python'
   status={'phase':'running','model':name,'scope':scope,'physical_GPU':device,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'previous_jobs':results}
   write(OUT/f'GPU{device}_queue.json',status)
   with (OUT/f'{name}_{scope}.log').open('ab') as log:
    process=subprocess.Popen([str(python),'-u','-B',str(ROOT/'scripts/evaluate_external_round_20261106.py'),'--model',name,'--scope',scope],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(device),PYTHONDONTWRITEBYTECODE='1'))
    status['pid']=process.pid;write(OUT/f'GPU{device}_queue.json',status)
    code=process.wait()
   results.append({'model':name,'scope':scope,'status':'completed' if code==0 else 'failed','exit_code':code})
  write(OUT/f'GPU{device}_queue.json',{'phase':'completed' if all(x['status']!='failed' for x in results) else 'needs_attention','jobs':results,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
def main():
 OUT.mkdir(exist_ok=True)
 with (OUT/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with ThreadPoolExecutor(max_workers=3) as pool:
   list(pool.map(run_device,QUEUES))
if __name__=='__main__':main()
