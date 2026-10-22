"""Single-GPU queue: real-data smoke, balanced pilot, then predeclared formal grid."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,subprocess,time
from src.research_io import ROOT,write_json,now

def call(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENCV_IO_ENABLE_OPENEXR='1'),check=True)

def wait_acquisition(phase):
    ready=ROOT/f'outputs/P1_arb_acquisition_v1/{phase}_ready.json'
    while not ready.exists():
        status=ROOT/'outputs/P1_arb_acquisition_v1/status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='failed':raise RuntimeError('Object acquisition failed; inspect its error record')
        write_json(ROOT/'outputs/P2_parameterization_v1/queue_status.json',{'status':'waiting_for_data','phase':phase,'updated_at':now()})
        time.sleep(20)

def train(method,dropout,seed,phase,steps=None):
    args=['--method',method,'--dropout',dropout,'--seed',seed,'--phase',phase]
    if steps:args+=['--steps',steps]
    call('train_parameterization.py',*args)
    call('summarize_parameterization.py')

def main():
    cfg=json.loads((ROOT/'configs/parameterization_first_round.json').read_text())
    out=ROOT/'outputs/P2_parameterization_v1'
    try:
        call('prepare_arb_training.py','--phase','smoke')
        for method in cfg['methods']:
            write_json(out/'queue_status.json',{'status':'running','phase':'smoke','method':method,'updated_at':now()})
            train(method,0,1247,'smoke',3)
        wait_acquisition('pilot');call('prepare_arb_training.py','--phase','pilot')
        for method in cfg['methods']:
            for dropout in cfg['condition_dropout']:
                write_json(out/'queue_status.json',{'status':'running','phase':'pilot','method':method,'dropout':dropout,'updated_at':now()})
                train(method,dropout,1247,'pilot')
        # Same budget/configuration even when a pilot performs poorly; it is solely
        # a finite-gradient and loader gate. Any exception stops the whole queue.
        wait_acquisition('formal');call('prepare_arb_training.py','--phase','formal')
        for seed in cfg['training_seeds']:
            for method in cfg['methods']:
                for dropout in cfg['condition_dropout']:
                    write_json(out/'queue_status.json',{'status':'running','phase':'formal','method':method,
                        'dropout':dropout,'seed':seed,'updated_at':now()})
                    train(method,dropout,seed,'formal')
        write_json(out/'queue_status.json',{'status':'completed','updated_at':now(),'formal_runs':24})
    except Exception as error:
        write_json(out/'queue_status.json',{'status':'failed','updated_at':now(),'error':type(error).__name__+': '+str(error)})
        raise
if __name__=='__main__':main()
