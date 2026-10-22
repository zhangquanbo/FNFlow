"""Review2 queue: WSD diagnostics only; never launches the formal 24-run grid."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os,json,time,subprocess
from src.research_io import ROOT,write_json,sha256,now
from scripts.summarize_wsd_review2 import summarize

CONFIG='configs/parameterization_first_round_v2_review2.json'


def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,check=True,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENCV_IO_ENABLE_OPENEXR='1'))


def main():
    cfg=json.loads((ROOT/CONFIG).read_text());root=ROOT/cfg['output_root']
    rule=json.loads((ROOT/cfg['decision_rule']).read_text());receipt=json.loads((ROOT/'provenance/budget_preregistration_review2.json').read_text())
    if receipt['rule_sha256']!=sha256(ROOT/cfg['decision_rule']) or receipt['config_sha256']!=sha256(ROOT/CONFIG):
        raise RuntimeError('Review2 registration changed')
    def status(state,**detail):write_json(root/'queue_status.json',{'status':state,'updated_at':now(),'formal_grid_locked':True,**detail})
    def train(phase,method,*extra):
        status('running',phase=phase,method=method)
        call('train_wsd_review2.py','--phase',phase,'--method',method,'--config',CONFIG,*extra)
        summarize()
    try:
        for method in cfg['methods']:train('smoke',method)
        ready=ROOT/'outputs/P1_arb_acquisition_v1/formal_ready.json'
        while not ready.exists():
            state=json.loads((ROOT/'outputs/P1_arb_acquisition_v1/status.json').read_text())
            if state.get('status')=='failed':raise RuntimeError('Object acquisition failed; preserve error record')
            status('waiting_for_data',phase='wsd_convergence');time.sleep(20)
        status('preparing_data',phase='arb_full')
        call('prepare_arb_training.py','--phase','formal');call('audit_arb_boundary_mask.py','--phase','formal')
        call('prepare_training_probe.py','--config',CONFIG)
        train('clip_diagnostic','Xv')
        for method in rule['methods']:
            train('stable',method)
            for step in rule['initial_fork_steps']:train('cooldown',method,'--fork-step',step)
        decision=summarize()
        if decision.get('extension_required'):
            for method in rule['methods']:
                train('stable',method,'--stable-limit',50000)
                train('cooldown',method,'--fork-step',50000)
            decision=summarize()
        status('awaiting_convergence_review',phase='formal_budget',budget_decision=decision['status'])
    except Exception as e:
        status('failed',error=repr(e));raise


if __name__=='__main__':main()
