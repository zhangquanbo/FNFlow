"""Bounded restart of failed network acquisition, retaining exact asset/CRC/hash rules."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os,json,time,subprocess,shutil
from src.research_io import ROOT,write_json,now


def main():
    directory=ROOT/'outputs/P1_arb_acquisition_v1';job=ROOT/'publish/arb_acquisition_server.json';rounds=0
    report=ROOT/'outputs/P1_arb_acquisition_recovery_review4/status.json'
    while True:
        state=json.loads((directory/'status.json').read_text());pid=json.loads(job.read_text())['pid']
        proc=Path(f'/proc/{pid}/cmdline');live=proc.exists() and b'download_arb_training.py' in proc.read_bytes()
        if live:
            write_json(report,{'status':'waiting_for_acquisition','repair_round':rounds,'updated_at':now()});time.sleep(20);continue
        if state['status']=='completed':
            write_json(report,{'status':'completed','repair_round':rounds,'updated_at':now()});return
        error=state.get('error','')
        if rounds>=2 or not any(s in error for s in ['TimeoutError','URLError','IncompleteRangeError','timed out']):
            write_json(report,{'status':'failed','repair_round':rounds,'error':error,'updated_at':now()});return
        rounds+=1;history=directory/'network_repair_history'/str(time.time_ns());history.mkdir(parents=True)
        shutil.copy2(directory/'status.json',history/'status.json')
        write_json(directory/'status.json',{'status':'resuming_network_failure','repair_round':rounds,'updated_at':now()})
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/download_arb_training.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FNFLOW_PUBLIC_FETCH_GATEWAY='http://127.0.0.1:18765'))
        write_json(job,{'pid':child.pid,'log':str(ROOT/'publish/arb_recovery_review4.log'),'repair_round':rounds,
            'protocol':'same fixed assets and strict file CRC/SHA verification; cached receipts reused'})
        write_json(report,{'status':'restarted','repair_round':rounds,'pid':child.pid,'updated_at':now()});time.sleep(1)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(ROOT/'outputs/P1_arb_acquisition_recovery_review4/status.json',{'status':'failed','error':repr(e),'updated_at':now()});raise
