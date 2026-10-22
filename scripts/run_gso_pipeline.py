"""Wait for complete source intake, then render candidate holdouts without model scoring."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,os,time,subprocess
from src.research_io import ROOT,write_json,now,sha256


def main():
    output=ROOT/'outputs/P1_gso_pipeline_v1'
    intake=ROOT/'outputs/P1_gso_intake_v1'
    def status(state,**detail):
        write_json(output/'status.json',{'status':state,'updated_at':now(),
            'training_allowed':False,'model_scoring_allowed':False,**detail})
    try:
        while True:
            source=json.loads((intake/'status.json').read_text()) if (intake/'status.json').exists() else {}
            if source.get('status')=='completed':break
            if source.get('status') in {'failed','completed_with_download_failures'}:
                raise RuntimeError('GSO intake incomplete; preserve failure details and repair before rendering')
            if source.get('failed',0):
                raise RuntimeError('GSO source download failure; no silent partial render')
            job=ROOT/'publish/gso_intake_job.json'
            if job.exists():
                pid=json.loads(job.read_text())['pid'];cmdline=Path(f'/proc/{pid}/cmdline')
                if not cmdline.exists() or b'prepare_gso_holdout.py' not in cmdline.read_bytes():
                    raise RuntimeError('GSO source worker exited before complete intake')
            status('waiting_for_sources',completed=source.get('completed',0),target=source.get('target',200))
            time.sleep(20)
        summary=json.loads((intake/'summary.json').read_text())
        manifest=intake/'manifest_private.json'
        if summary['downloaded_assets']!=summary['selected_assets'] or summary['failed_assets']:
            raise RuntimeError('GSO source completeness mismatch')
        if summary['planned_objaverse_assets_screened']!=2000 or not summary['accepted_assets']:
            raise RuntimeError('GSO candidate duplicate screen missing')
        status('rendering_or_waiting_for_gpu',candidate_assets=summary['accepted_assets'],
            manifest_sha256=sha256(manifest),cross_source_exhaustive_check=False)
        blender=ROOT/'runtime/blender/blender-3.3.0-linux-x64/blender'
        subprocess.run([str(blender),'-b','--python-exit-code','1','--python',
            str(ROOT/'scripts/render_gso_holdout_blender.py'),'--','--device','GPU'],
            cwd=ROOT,check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        result=json.loads((ROOT/'outputs/P1_gso_render_dual_v2/summary.json').read_text())
        if result['status']!='completed' or result['assets']!=summary['accepted_assets']:
            raise RuntimeError('GSO render completion mismatch')
        status('completed',assets=result['assets'],images=result['images'],
            cross_source_exhaustive_check=False,
            scope='candidate holdouts under declared source/count screening; no unseen-asset performance claim')
    except Exception as e:
        status('failed',error=repr(e));raise


if __name__=='__main__':main()
