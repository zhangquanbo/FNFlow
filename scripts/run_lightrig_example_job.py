import json,time,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/lightrig_official_example_job';out.mkdir(parents=True,exist_ok=True)
status=out/'status.json'
def write(state,**kwargs):status.write_text(json.dumps({'state':state,'updated_unix':time.time(),**kwargs},indent=2))
write('waiting_for_pinned_base_components')
try:
 for attempt in range(1440):
  p=ROOT/'provenance/lightrig_base_models.json'
  if p.exists() and len(json.loads(p.read_text()))==3:break
  time.sleep(30)
 else:raise TimeoutError('Base model download not complete after 12h')
 for repo in json.loads(p.read_text()):
  for entry in repo['files']:
   file=ROOT/'external/base_models'/repo['repo'].replace('/','__')/entry['path'];h=hashlib.sha256()
   with file.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   assert h.hexdigest()==entry['sha256'],str(file)
 write('running_official_examples',parameters='512px,75steps,CFG2,rescale0.7,seed511')
 with (out/'inference.log').open('w') as log:
  subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_lightrig_official_examples.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write('examples_generated_pending_visual_review',model_scores_computed=False)
except Exception as e:
 write('failed',error=str(e));raise