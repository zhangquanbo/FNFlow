from pathlib import Path
import sys,json,csv,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from src.cluster_bootstrap import cluster_percentile_intervals
from src.research_io import write_json,sha256,now
O=R/'outputs/T1_metrics';DATA=R/'data/external_round_20261106';PANELS=['gso_primary','single_diligent','single_luces','rose10_diligent','rose10_luces'];FIELDS=['mean_deg','median_deg','fraction_below_11.25','fraction_below_22.5','fraction_below_30'];BOUND=['boundary_primary_angular_mae_deg','boundary_primary_angular_outside_mae_deg'];ROOTS={'E1_B':R/'outputs/MoGe_E_round_20261106/E_evaluation/E1_B','D2':R/'outputs/MoGe_E_round_20261106/evaluation/D2','MoGe-S':R/'outputs/MoGe_E_round_20261106/evaluation/MoGe-S'}
for name in ['NiRNE','NeuralLightRig','Lotus-D','Lotus-G','DSINE','MarigoldNormal']:ROOTS[name]=R/'outputs/external_round_20261106'/name
SEEDS={'E1_B':list(range(42,50)),'D2':list(range(42,50)),'MoGe-S':[42],'NeuralLightRig':[511]}
def read(p):return json.loads(Path(p).read_text())
def asset_values(rows,key,pitch=None):
 grouped=collections.defaultdict(lambda:collections.defaultdict(list));null=0
 for row in rows:
  if pitch is not None and row.get('elevation_bin')!=pitch:continue
  value=row['metrics'].get(key)
  if value is None:null+=1;continue
  assert np.isfinite(value);grouped[row['object']][row.get('elevation_bin')].append(value)
 values={obj:float(np.mean([np.mean(v) for v in per_pitch.values()])) for obj,per_pitch in grouped.items()};return values,null

def ci(values):
 ids=sorted(values);return cluster_percentile_intervals([values[k] for k in ids],ids,draws=10000,seed=20261110)
allrows={};table={};missing=[];refs=[];LEGACY=R/'outputs/P3_object_dino_evaluation_fresh_20261027';PUBLISHED=R/'publish/FNFlow_20261105/results/object_recipe_ladder_20261105/summary.json';published=read(PUBLISHED)
for name,root in ROOTS.items():
 table[name]={};allrows[name]={}
 for panel in PANELS:
  scope='gso' if panel=='gso_primary' else 'real';manifest=read(DATA/(panel+'.json'))['records'];expected={r['id'] for r in manifest};folder=(LEGACY/'per_image_private/primary/NeuralLightRig/511') if name=='NeuralLightRig' and panel=='gso_primary' else root/scope/'metrics'/panel;files=list(folder.glob('*.json'));assert {p.stem for p in files}==expected,(name,panel,'incomplete or extra records');rows=[read(p) for p in files];allrows[name][panel]=rows;entry={'images':len(rows),'assets':len({r['object'] for r in rows}),'inference_seeds_original_protocol':SEEDS.get(name,[42]),'metric_space':'per-image statistic, inference-error averaged as originally recorded; object mean then equal pitch mean forGSO; not pooled-pixel median','metrics':{}}
  for key in FIELDS+BOUND:
   absent=[r['image_key'] for r in rows if key not in r['metrics']]
   if absent:missing.append({'model':name,'panel':panel,'metric':key,'images':len(absent)});continue
   values,null=asset_values(rows,key);entry['metrics'][key]={'mean':float(np.mean(list(values.values()))) if values else None,'cluster_bootstrap':ci(values) if len(values)>1 else None,'eligible_assets':len(values),'undefined_empty_region_images':null}
  mean=entry['metrics']['mean_deg']['mean'];reference=published['evaluations']['NeuralLightRig']['panels']['primary']['equal_pitch_object_macro_mae_deg'] if name=='NeuralLightRig' and panel=='gso_primary' else read(root/scope/'summary.json')['panels'][panel]['equal_pitch_object_macro_MAE_deg'];difference=abs(mean-reference);assert difference<1e-6,(name,panel,difference);entry['original_summary_MAE_deg']=reference;entry['published_MAE_reaggregation_absolute_deviation_deg']=difference;refs.append(difference);entry['by_pitch']={str(p):{'mean_deg':float(np.mean(list(asset_values(rows,'mean_deg',p)[0].values()))),'cluster_bootstrap':ci(asset_values(rows,'mean_deg',p)[0])} for p in [0,15,30,45,60]} if panel=='gso_primary' else {};table[name][panel]=entry;write_json(O/'status.json',{'phase':'cached_original_protocol_aggregation','model':name,'panel':panel,'updated_at':now()})
paired={}
for panel in PANELS:
 paired[panel]={}
 for p in ([None,0,15,30,45,60] if panel=='gso_primary' else [None]):
  a=asset_values(allrows['E1_B'][panel],'mean_deg',p)[0];b=asset_values(allrows['MoGe-S'][panel],'mean_deg',p)[0];assert a.keys()==b.keys();paired[panel]['all' if p is None else str(p)]={'definition':'E1_B minus MoGe-2-S, same object resample indices','bootstrap':ci({k:a[k]-b[k] for k in a})}
supplement={}
for name in ['A','B']:
 seeds=sorted(p.name for p in (LEGACY/'per_image_private/primary'/name).iterdir() if p.is_dir());merged=collections.defaultdict(list)
 for seed in seeds:
  for p in (LEGACY/'per_image_private/primary'/name/seed).glob('*.json'):merged[p.stem].append(read(p))
 expected={r['id'] for r in read(DATA/'gso_primary.json')['records']};assert set(merged)==expected;reduced=[]
 for key,items in merged.items():
  assert len(items)==len(seeds);first=items[0];metrics={k:float(np.mean([x['metrics'][k] for x in items])) if all(x['metrics'][k] is not None for x in items) else None for k in FIELDS+BOUND};reduced.append({**first,'metrics':metrics})
 point=asset_values(reduced,'mean_deg')[0];reference=published['evaluations'][name]['panels']['primary']['equal_pitch_object_macro_mae_deg'];delta=abs(float(np.mean(list(point.values())))-reference);assert delta<1e-6,(name,delta);refs.append(delta);supplement[name]={'inference_seeds_as_original':seeds,'published_MAE_deviation_deg':delta,'regions':{k:{'mean':float(np.mean(list(asset_values(reduced,k)[0].values()))),'eligible_assets':len(asset_values(reduced,k)[0]),'undefined_empty_region_images':asset_values(reduced,k)[1],'cluster_bootstrap':ci(asset_values(reduced,k)[0])} for k in BOUND},'source':'Existing primary fixed-boundary per-image statistics, exact original protocol, not new inference'}
result={'status':'completed_old_protocol_all_requested_statistics','old_protocol_table':table,'paired_E1_B_minus_MoGe_S':paired,'maximum_published_MAE_reaggregation_deviation_deg':max(refs),'missing_cached_fields':missing,'A_B_boundary_status':'completed_from_exact_cached_sufficient_statistics','A_B_boundary_supplement':supplement,'new42_49_main_table_status':'separate protocol, pending stochastic baseline extensions; must not be relabeled as old snapshot reproduction','bootstrap':'10,000 object-cluster resamples, seed20261110; one asset-level value passed tosrc/cluster_bootstrap.py to retain object-equal estimand','authorizations':'user explicitly approved cached sufficient statistics and separate old/new seed tables','test_images_scored':0,'worker_sha256':sha256(__file__),'updated_at':now()};write_json(O/'summary.json',result)
with (O/'old_protocol_table.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['model','panel','MAE_deg','median_macro_deg','below11.25','below22.5','below30','MAE_lo95','MAE_hi95'])
 for name,panels in table.items():
  for panel,e in panels.items():
   m=e['metrics'];interval=m['mean_deg']['cluster_bootstrap']['metrics'][0];writer.writerow([name,panel,*[m[k]['mean'] for k in FIELDS],interval['lower_95'],interval['upper_95']])
lines=['# T1 指标补全：旧协议全部所需统计完成，新种子主表另行计算','','完整mean/median/三个阈值比例已从逐图充分统计恢复；没有从MAE推测像素分布。GSO按物体/视图平均再等权五俯仰角，真实域先物体内平均再物体等权。',f"\n与已完成主表摘要的最大MAE偏差为{max(refs):.3e}°，均小于1e-6°。Bootstrap复用src/cluster_bootstrap.py，10,000次、物体为cluster；先聚合物体，避免十光协议按图数给物体加权。",'\n|模型|GSO MAE|95%物体区间|','|---|---:|---|']
for n in table:
 e=table[n]['gso_primary']['metrics']['mean_deg'];v=e['cluster_bootstrap']['metrics'][0];lines.append(f"|{n}|{e['mean']:.4f}|[{v['lower_95']:.4f}, {v['upper_95']:.4f}]|")
lines+=['','完整五面板/俯仰角/配对区间见summary.json和old_protocol_table.csv。旧外部方法大多seed42、NLR511，FM42–49；经用户确认旧复现表与新的42–49主表分开。','\nA/B和Neural LightRig的主边界带/带外项已从完整逐图记录补齐，无需重推理；未更改mask。待完成新42–49外部随机模型主表扩展。空边界区域显式计数，不当作0或静默删除。未打开测试集，未推送GitHub。'];(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('T1_OLD_PROTOCOL_STATS_COMPLETE',max(refs),flush=True)
