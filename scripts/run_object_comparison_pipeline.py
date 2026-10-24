"""Single execution pipeline, serialized with the efficiency job; no training."""
import os,sys,subprocess,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/object_comparison_pipeline_20261023';OUT.mkdir(exist_ok=True)
def status(**kw):(OUT/'status.json').write_text(json.dumps({'updated_unix':time.time(),**kw},indent=2))
def run(script,env=None):
 subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,check=True)
try:
 status(phase='gso_arb_evaluation_waiting_for_GPU',training=False);run('evaluate_object_comparison.py');run('render_object_comparison_cases.py')
 status(phase='diligent_evaluation',training=False);env=os.environ.copy();env['OBJECT_COMPARISON_MANIFEST']=str(ROOT/'data/object_comparison_diligent_20261023/comparison_manifest_private.json');env['OBJECT_COMPARISON_OUTPUT']=str(ROOT/'outputs/object_comparison_diligent_20261023')
 run('evaluate_object_comparison.py',env);run('render_object_comparison_cases.py',env)
 status(phase='available_datasets_completed',luces='blocked_official_link_404',training=False)
except Exception as e:status(phase='failed',error=repr(e),training=False);raise
