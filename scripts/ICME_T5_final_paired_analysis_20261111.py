"""Add pairedobjectCIvsMoGeS tofixed50kraw/EMAresults; noGPU/noinference/noselection."""
from pathlib import Path
import sys,json,time,collections,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from src.cluster_bootstrap import cluster_percentile_intervals
from src.research_io import write_json,sha256,now
T=R/'outputs/T5_Rplus';I=R/'outputs/ICME2027_20261110';PANELS=['gso_primary','single_diligent','single_luces','rose10_diligent','rose10_luces'];MAN=R/'data/external_round_20261106';BASE=R/'outputs/MoGe_E_round_20261106/evaluation/MoGe-S'
def read(p):return json.loads(Path(p).read_text())
def object_means(rows):
 groups=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rows:groups[r['object']][r.get('elevation_bin')].append(float(r['mean_deg']))
 return {k:float(np.mean([np.mean(v) for v in g.values()])) for k,g in groups.items()}
def pair(left,right):
 a=object_means(left);b=object_means(right);assert set(a)==set(b) and len(a)>1;ids=sorted(a);values=[a[k]-b[k] for k in ids];out=cluster_percentile_intervals(values,ids,draws=10000,seed=20261110);out['estimand']='equalobjectmean of new-minusMoGeS; withinobject equalpitchmeans of viewmeans';out['asset_pairs']=len(ids);return out

def acceptance():
 a=[{'object':'a','elevation_bin':0,'mean_deg':1},{'object':'a','elevation_bin':0,'mean_deg':5},{'object':'a','elevation_bin':60,'mean_deg':9},{'object':'b','elevation_bin':0,'mean_deg':3}];b=[{**r,'mean_deg':v} for r,v in zip(a,[0,0,1,1])];p=pair(a,b);assert p['metrics'][0]=={'estimate':3.75,'lower_95':2.,'upper_95':5.5};write_json(T/'paired_analysis_acceptance.json',{'status':'passed','synthetic_unequalview_pitch_counts':'objectpair3.75vsincorrectrawimagemean4.0;95CI[2,5.5]','worker_sha256':sha256(__file__),'updated_at':now()})

def prepare(run):
 final=T/run/'evaluation/50000/summary.json';folder=T/run/'paired_final_50000'
 if not final.exists():return False
 original=read(final);assert original['status']=='completed' and original['stage']==50000;signature={'fixed_final_summary_sha256':sha256(final),'checkpoint_sha256':original['signature']['checkpoint_sha256'],'configuration_sha256':original['signature']['config_sha256'],'worker_sha256':sha256(__file__),'bootstrap_source_sha256':sha256(R/'src/cluster_bootstrap.py')}
 if (folder/'summary.json').exists():assert read(folder/'summary.json')['signature']==signature;return True
 results={};manifest_shas={}
 for variant in ['raw','EMA']:
  results[variant]={}
  for panel in PANELS:
   mp=MAN/(panel+'.json');records=read(mp)['records'];manifest_shas[panel]=sha256(mp);left=[];right=[];scope='gso' if panel=='gso_primary' else 'real'
   for r in records:
    n=read(T/run/'evaluation/50000'/variant/panel/'metrics_private'/(r['id']+'.json'));b=read(BASE/scope/'metrics'/panel/(r['id']+'.json'));assert n['image_key']==r['image_key'] and n['signature']==original['signature'];assert b.get('image_key',r['image_key'])==r['image_key'];assert n['object']==r['object'] and b.get('object',r['object'])==r['object'];meta={'object':r['object'],'elevation_bin':r.get('elevation_bin')};left.append({**meta,'mean_deg':n['metrics']['mean_deg']});right.append({**meta,'mean_deg':b['metrics']['mean_deg']})
   paired=pair(left,right);expected=original['raw_and_EMA'][variant][panel]['metrics']['mean_deg']['mean']-float(np.mean(list(object_means(right).values())));assert abs(paired['metrics'][0]['estimate']-expected)<1e-10;angles=sorted({r['elevation_bin'] for r in left if r['elevation_bin'] is not None});results[variant][panel]={'overall':paired,'by_pitch':{str(p):pair([r for r in left if r['elevation_bin']==p],[r for r in right if r['elevation_bin']==p]) for p in angles}}
 result={'status':'completed_fixed50k_paired_analysis','run':run,'new_minus_MoGe_S':results,'signature':signature,'manifest_sha256':manifest_shas,'protocol':'T2random42-49erroraverage; deterministicMoGeSoneuniqueoutput;10kpairedobjectresamples,seed20261110','no_weight_model_or_checkpoint_selected':True,'scope':'conditionalonfixedtrainedweights; doesnotreplaceformaltrainingseedstd','test_images_scored':0,'updated_at':now()};write_json(folder/'summary.json',result);lines=['# 固定50k：相对MoGe-S的配对区间','','原始与EMA同时报告，10,000次物体级配对重采样；负值表示误差更低。不选择权重、检查点或论文叙事。','','|权重|面板|差值°|95%下界|95%上界|','|---|---|---:|---:|---:|']
 for v,panels in results.items():
  for p,x in panels.items():m=x['overall']['metrics'][0];lines.append(f"|{v}|{p}|{m['estimate']:.6f}|{m['lower_95']:.6f}|{m['upper_95']:.6f}|")
 lines+=['','五个GSO俯仰角的配对区间、源配置/最终检查点/清单SHA均见summary.json。先在物体内按视图和俯仰角等权，再以物体等权；没有将八个推理种子当作独立物体。GSO-test评分0。'];(folder/'REPORT.md').write_text('\n'.join(lines)+'\n');print('FIXED50K_PAIRED_COMPLETE',run,flush=True);return True

def main():
 acceptance()
 while True:
  completed={run:prepare(run) for run in ['S1','B1','S2']};write_json(I/'paired_analysis_status.json',{'phase':'watching_fixed50k_final_results','completed':completed,'updated_at':now()});outcomes=read(T/'decisive_training_status.json')['outcomes'] if (T/'decisive_training_status.json').exists() else {};failure=T/'S1/failure.json';needsS2=failure.exists() and 'Nonfinite' in read(failure).get('error','')
  if all(outcomes.get(r) in ['completed','failed','abandoned'] and (completed[r] or outcomes[r]!='completed') for r in ['S1','B1']) and (not needsS2 or completed['S2']):
   write_json(I/'paired_analysis_status.json',{'phase':'completed_decisive_run_paired_reports','completed':completed,'outcomes':outcomes,'abandoned_or_failed':'no25kdiagnosticcheckpointsubstitutedforfinal50k','updated_at':now()});return
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(I/'paired_analysis_status.json',{'phase':'needs_attention','error':repr(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
