"""CPU-only paired/bootstrap and seed reports for user-locked final50k experiments."""
from pathlib import Path
import os,sys,time,json,statistics,traceback
os.environ['CUDA_VISIBLE_DEVICES']='';R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.research_io import write_json,sha256,now
from scripts import ICME_T5_final_paired_analysis_20261111 as paired_helper
O=R/'outputs/T6_S1_locked_20261112';paired_helper.T=O;LABELS=['S1_s1248','S1_no_distill_s1247','S1_s1249'];PANELS=paired_helper.PANELS
def read(p):return json.loads(Path(p).read_text())

def prepare_final(run):
 final=O/run/'evaluation/50000/summary.json';folder=O/run/'paired_final_50000'
 if not final.exists():return False
 current=read(final);source_sha=sha256(final);paired=folder/'summary.json';snapshot=folder/'source_evaluation_summary_private.json'
 if paired.exists():
  old=read(paired)['signature'];expected={'fixed_final_summary_sha256':source_sha,'checkpoint_sha256':current['signature']['checkpoint_sha256'],'configuration_sha256':current['signature']['config_sha256'],'worker_sha256':sha256(R/'scripts/ICME_T5_final_paired_analysis_20261111.py'),'bootstrap_source_sha256':sha256(R/'src/cluster_bootstrap.py')}
  if old!=expected:
   changed=[k for k in set(old)|set(expected) if old.get(k)!=expected.get(k)]
   assert changed==['fixed_final_summary_sha256'],('Nonmetadataprovenancemismatch',run,changed)
   candidates=[snapshot,O/'restart_after_shutdown_20261117/completed_before_restart'/run/'evaluation/50000/summary.json'];previous=None
   for candidate in candidates:
    if candidate.exists() and sha256(candidate)==old['fixed_final_summary_sha256']:previous=read(candidate);break
   assert previous is not None,('Missingprevioussourcesnapshot',run)
   differences=[k for k in set(previous)|set(current) if previous.get(k)!=current.get(k)]
   assert differences==['updated_at'],('Evaluationcontentschanged',run,differences)
   archive=O/'CPU_collector_repair_20261118'/'archived_pair_reports'/(run+'_'+str(time.time_ns()));archive.parent.mkdir(parents=True,exist_ok=True)
   assert folder.resolve().is_relative_to(O.resolve()) and archive.resolve().is_relative_to(O.resolve())
   folder.rename(archive)
   write_json(archive/'metadata_only_rebuild_receipt_private.json',{'run':run,'reason':'completed_evaluation_replayed_after_shutdown_changed_only_updated_at','old_signature':old,'new_signature':expected,'changed_evaluation_fields':differences,'training_and_metrics_changed':False,'updated_at':now()})
   print('CPU_PAIR_METADATA_ONLY_ARCHIVED_REBUILD',run,flush=True)
 ready=paired_helper.prepare(run)
 if ready:write_json(snapshot,current)
 return ready

def ablation():
 source=O/'S1_no_distill_s1247/evaluation/50000/summary.json'
 if not source.exists():return False
 folder=O/'no_distillation_ablation';folder.mkdir(exist_ok=True)
 if (folder/'summary.json').exists():return True
 result={}
 for panel in PANELS:
  records=read(R/'data/external_round_20261106'/(panel+'.json'))['records'];left=[];right=[]
  for rec in records:
   meta={'object':rec['object'],'elevation_bin':rec.get('elevation_bin')};x=read(O/'S1_no_distill_s1247/evaluation/50000/EMA'/panel/'metrics_private'/(rec['id']+'.json'));y=read(R/'outputs/T5_Rplus/S1/evaluation/50000/EMA'/panel/'metrics_private'/(rec['id']+'.json'));left.append({**meta,'mean_deg':x['metrics']['mean_deg']});right.append({**meta,'mean_deg':y['metrics']['mean_deg']})
  overall=paired_helper.pair(left,right);overall['estimand']='lambda0minuslockedlambda0.5S1EMA;equalobjectmean,withinobjectequalpitchmean';angles=sorted({x['elevation_bin'] for x in left if x['elevation_bin'] is not None});parts={str(a):paired_helper.pair([x for x in left if x['elevation_bin']==a],[x for x in right if x['elevation_bin']==a]) for a in angles}
  for v in parts.values():v['estimand']=overall['estimand']
  result[panel]={'overall':overall,'by_pitch':parts}
 summary={'status':'completed_fixed50000_lambda0_ablation','comparison':'no_distilllambda0minuslockedfullS1lambda0.5;seed1247both;EMA;fixed50k','paired_delta':result,'no_recipe_or_final_checkpoint_selection':True,'source_summary_sha256':sha256(source),'test_images_scored':0,'updated_at':now()};write_json(folder/'summary.json',summary);lines=['# 去蒸馏单因素消融','','固定50k/seed1247/EMA，其余配方不变；差值为λ0减λ0.5，负值表示去蒸馏误差更低。完整逐俯仰配对bootstrap在summary.json；不会根据消融结果改动已锁定配方。','','|面板|MAE差值°|95%下界|95%上界|','|---|---:|---:|---:|']
 for name,a in result.items():v=a['overall']['metrics'][0];lines.append(f'|{name}|{v["estimate"]:.6f}|{v["lower_95"]:.6f}|{v["upper_95"]:.6f}|')
 (folder/'REPORT.md').write_text('\n'.join(lines)+'\n');return True
def seed_summary():
 values=[{'seed':1247,'GSO_val_MAE_deg':read(R/'outputs/T5_Rplus/S1/evaluation/50000/summary.json')['raw_and_EMA']['EMA']['gso_primary']['metrics']['mean_deg']['mean'],'checkpoint_sha256':'ebf6a113cd9e6a15edba0e1282fd469550f8d33f33012e8818e11b8dcfda9822'}]
 for name,seed in [('S1_s1248',1248),('S1_s1249',1249)]:
  p=O/name/'evaluation/50000/summary.json'
  if p.exists():a=read(p);values.append({'seed':seed,'GSO_val_MAE_deg':a['raw_and_EMA']['EMA']['gso_primary']['metrics']['mean_deg']['mean'],'checkpoint_sha256':a['signature']['checkpoint_sha256']})
 folder=O/'final_seed_analysis';folder.mkdir(exist_ok=True);arr=[x['GSO_val_MAE_deg'] for x in values];summary={'status':'completed' if len(values)==3 else 'partial','seeds':values,'mean_GSO_val_MAE_deg':statistics.mean(arr),'sample_std_ddof1_deg':statistics.stdev(arr) if len(arr)>1 else None,'final_checkpoint_stays_user_locked_seed1247_ebf6a113':True,'no_best_seed_selection':True,'test_images_scored':0,'updated_at':now()};write_json(folder/'summary.json',summary)
 if len(values)==3:(folder/'REPORT.md').write_text('# 锁定S1 EMA三训练种子\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
try:
 while True:
  completed={name:prepare_final(name) for name in LABELS};done=ablation();seed_summary();write_json(O/'CPU_collector_status.json',{'phase':'watching_locked_fixed50k_reports','paired_complete':completed,'no_distill_ablation_complete':done,'updated_at':now()});q=read(O/'queue_status.json') if (O/'queue_status.json').exists() else {}
  if q.get('phase')=='completed' and all(completed.values()) and done:break
  time.sleep(60)
except Exception as e:write_json(O/'CPU_collector_status.json',{'phase':'needs_attention','error':repr(e),'traceback':traceback.format_exc(),'updated_at':now()});raise
