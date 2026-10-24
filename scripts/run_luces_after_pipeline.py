"""Continue the existing finite evaluation queue with LUCES; no scheduled task."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.research_io import write_json,sha256
OUT=ROOT/'outputs/object_comparison_luces_pipeline_20261023';OUT.mkdir(exist_ok=True)
def status(**kw):write_json(OUT/'status.json',{'updated_unix':time.time(),'training':False,**kw})
def run(script,env):subprocess.run([sys.executable,'-u','-B',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,check=True)
def main():
    with (ROOT/'runtime/luces_pipeline.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=ROOT/'data/object_comparison_luces_20261023/comparison_manifest_private.json'
        validation=ROOT/'data/object_comparison_luces_20261023/validation.json'
        assert json.loads(validation.read_text())['manifest_sha256']==sha256(manifest)
        assert json.loads(validation.read_text())['status']=='passed'
        status(phase='queued_after_gso_arb_diligent',manifest_sha256=sha256(manifest))
        parent=ROOT/'outputs/object_comparison_pipeline_20261023/status.json'
        while True:
            state=json.loads(parent.read_text())
            if state.get('phase')=='failed':raise RuntimeError('Previous evaluation pipeline failed: '+str(state))
            if state.get('phase') in {'available_datasets_completed','completed_all_registered_datasets'}:break
            time.sleep(30)
        # Parent status predates LUCES intake; its obsolete missing-data label is
        # superseded by this stage's verified manifest, not treated as a new failure.
        for name in ['object_comparison_20261023','object_comparison_diligent_20261023']:
            summary=json.loads((ROOT/'outputs'/name/'summary.json').read_text())
            assert all(r['complete'] for r in summary['rows']),name
            assert {r['model'] for r in summary['rows']}=={'R','V','Xn','NeuralLightRig','NeuralLightRig_variability'},name
        env=os.environ.copy();env['OBJECT_COMPARISON_MANIFEST']=str(manifest);env['OBJECT_COMPARISON_OUTPUT']=str(ROOT/'outputs/object_comparison_luces_20261023')
        status(phase='luces_evaluation',manifest_sha256=sha256(manifest));run('evaluate_object_comparison.py',env)
        status(phase='luces_case_panels');run('render_object_comparison_cases.py',env)
        summary=json.loads((Path(env['OBJECT_COMPARISON_OUTPUT'])/'summary.json').read_text());assert all(r['complete'] for r in summary['rows'])
        status(phase='completed',manifest_sha256=sha256(manifest),comparison_table=str(Path(env['OBJECT_COMPARISON_OUTPUT'])/'comparison_table.csv'))
if __name__=='__main__':
    try:main()
    except Exception as e:status(phase='failed',error=repr(e));raise
