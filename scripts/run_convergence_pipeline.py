"""Prepare aspect-preserving caches and run diagnostics; formal grid stays locked."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,subprocess,time
from src.research_io import ROOT,write_json,now

CONFIG='configs/parameterization_first_round_v2_review1.json'


def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENCV_IO_ENABLE_OPENEXR='1'),check=True)


def cache_ready(path):
    summary=ROOT/path/'summary.json'
    if not summary.exists():return False
    value=json.loads(summary.read_text())
    if value.get('status')!='completed' or value.get('spatial_protocol')!='center_square_v2':
        raise RuntimeError('Prepared cache does not match center-square v2')
    return True


def main():
    cfg=json.loads((ROOT/CONFIG).read_text());out=ROOT/cfg['output_root']
    def status(stage,**kwargs):
        write_json(out/'queue_status.json',{'status':stage,'updated_at':now(),'formal_grid_locked':True,**kwargs})
    try:
        status('preparing_center_square_caches',dataset='hypersim')
        if not cache_ready(cfg['hypersim_root']):
            call('prepare_hypersim_pilot.py','--per-scene',1000000,'--center-crop','--output',cfg['hypersim_root'])
        status('checking_center_square_caches',dataset='hypersim')
        call('check_prepared_data.py','--prepared-root',cfg['hypersim_root'],'--run-id','P1_hypersim_center_checked_v2')
        status('preparing_center_square_caches',dataset='nyuv2')
        if not cache_ready(cfg['nyu_validation_root']):
            call('prepare_nyu_validation.py','--center-crop','--output',cfg['nyu_validation_root'])
        call('check_prepared_data.py','--prepared-root',cfg['nyu_validation_root'],
             '--run-id','P1_nyu_center_checked_v2','--splits','val')
        # Preserve the completed v2 numerical checks; no long training started there.
        from src.research_io import sha256
        expected_smoke_manifests=[sha256(ROOT/cfg['hypersim_root']/'private_manifest.json'),
            sha256(ROOT/'data/arb_mixed_smoke_v1/private_manifest.json'),sha256(ROOT/cfg['nyu_validation_root']/'private_manifest.json')]
        for method in cfg['methods']:
            previous=ROOT/cfg['validated_smoke_reference_root']/f'smoke_center_{method}_s1247'
            summary=json.loads((previous/'summary.json').read_text())
            sig=json.loads((previous/'signature_private.json').read_text())
            if (summary['status']!='completed' or sig['config_sha256']!=sha256(ROOT/'configs/parameterization_first_round_v2.json')
                or sig['data_manifests_sha256']!=expected_smoke_manifests):
                raise RuntimeError('Original center-square numerical smoke reference failed')
        ready=ROOT/'outputs/P1_arb_acquisition_v1/formal_ready.json'
        while not ready.exists():
            acquisition=ROOT/'outputs/P1_arb_acquisition_v1/status.json'
            if acquisition.exists() and json.loads(acquisition.read_text()).get('status')=='failed':
                raise RuntimeError('Object acquisition failed; inspect its error record')
            status('waiting_for_data',phase='convergence')
            time.sleep(20)
        status('preparing_data',dataset='arb')
        call('prepare_arb_training.py','--phase','formal')
        call('audit_arb_boundary_mask.py','--phase','formal')
        call('prepare_training_probe.py','--config',CONFIG)
        # Xv and the first 2500 updates of the three curves share the same 30k LR schedule.
        method=cfg['clip_diagnostic_method']
        status('running',phase='clip_diagnostic',method=method)
        call('train_parameterization.py','--method',method,'--dropout',0,'--seed',cfg['convergence_seed'],
             '--phase','clip_diagnostic','--config',CONFIG)
        for method in cfg['convergence_methods']:
            status('running',phase='convergence',method=method)
            call('train_parameterization.py','--method',method,'--dropout',cfg['convergence_dropout'],
                 '--seed',cfg['convergence_seed'],'--phase','convergence','--config',CONFIG)
        call('summarize_convergence.py')
        decision=json.loads((out/'budget_decision.json').read_text())
        if (decision.get('extend_to')==50000 or decision.get('extension_triggered')
            or decision.get('reference_endpoint')==50000):
            for method in cfg['convergence_methods']:
                status('running',phase='convergence_continuation',method=method)
                call('train_parameterization.py','--method',method,'--dropout',0,'--seed',cfg['convergence_seed'],
                     '--phase','convergence','--config',CONFIG,'--extend-to',50000)
            call('summarize_convergence.py')
        status('awaiting_convergence_review',phase='formal_budget',
               next_action='Review per-domain curves and common-window clipping; set one shared budget/threshold before any formal run')
    except Exception as error:
        status('failed',error=type(error).__name__+': '+str(error))
        raise


if __name__=='__main__':main()
