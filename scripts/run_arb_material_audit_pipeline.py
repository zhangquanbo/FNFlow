"""Finish/resume the same pinned metadata census, retaining failures and strict checks."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,time,os,subprocess,shutil
from src.research_io import ROOT,write_json,sha256,now


def main():
    folder=ROOT/'outputs/P1_arb_materials_review4';job=ROOT/'publish/arb_material_audit_review4_job.json'
    output=ROOT/'outputs/P1_arb_materials_pipeline_review4';rounds=0
    while True:
        pid=json.loads(job.read_text())['pid'];proc=Path(f'/proc/{pid}/cmdline')
        live=proc.exists() and b'audit_arb_materials_review4.py' in proc.read_bytes()
        if live:
            state=json.loads((folder/'status.json').read_text())
            write_json(output/'status.json',{'status':'waiting_for_census','classified_assets':state.get('classified_assets'),
                'failed_assets':state.get('failed_assets'),'repair_round':rounds,'updated_at':now()})
            time.sleep(20);continue
        summary=json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
        if summary.get('status')=='completed':
            if summary['classified_assets']!=2000 or summary['failed_assets'] or summary['material_index_sha256']!=sha256(folder/'material_index_private.json'):
                raise RuntimeError('Census completeness/hash mismatch')
            write_json(output/'status.json',{'status':'completed','classified_assets':2000,'repair_round':rounds,'updated_at':now()});return
        if rounds>=2:
            write_json(output/'status.json',{'status':'incomplete_unknown_assets_remain','updated_at':now(),'formal_quality_rule_not_ready':True})
            return
        rounds+=1;history=folder/'repair_history'/str(time.time_ns());history.mkdir(parents=True)
        for name in ['status.json','summary.json','failures_private.json']:
            if (folder/name).exists():shutil.copy2(folder/name,history/name)
        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/audit_arb_materials_review4.py')],cwd=ROOT,
            stdin=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',FNFLOW_PUBLIC_FETCH_GATEWAY='http://127.0.0.1:18765'))
        write_json(job,{'pid':child.pid,'log':str(ROOT/'publish/arb_material_pipeline_review4.log'),'repair_round':rounds})
        time.sleep(1)


if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(ROOT/'outputs/P1_arb_materials_pipeline_review4/status.json',{'status':'failed','error':repr(e),'updated_at':now()});raise
