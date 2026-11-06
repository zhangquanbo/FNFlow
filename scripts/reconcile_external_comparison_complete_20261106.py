
"""Reconcile every completed external scope and D2 inference variant; export aggregate artifacts only."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,math,time,statistics
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/external_round_20261106';DATA=ROOT/'data/external_round_20261106';DEST=ROOT/'publish/external_comparison_complete_20261106'
MODELS=['D2','NeuralLightRig','MoGe2','NiRNE','Lotus-D','Lotus-G','DSINE','MarigoldNormal','StableNormal'];REAL=['single_diligent','single_luces','rose10_diligent','rose10_luces']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,data):(DEST/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def macro(rows):
 groups=defaultdict(lambda:defaultdict(list))
 for row in rows:groups[str(row['elevation_bin'])][row['object']].append(row['metrics']['mean_deg'])
 pitches={p:sum(sum(v)/len(v) for v in objects.values())/len(objects) for p,objects in groups.items()}
 return {'by_pitch_MAE_deg':pitches,'equal_pitch_object_macro_MAE_deg':sum(pitches.values())/len(pitches)}
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 versions={};registered={}
 for version,suffix,adapter in [('v1','','external_normal_adapter_20261106.py'),('v2','_v2','external_normal_adapter_v2_20261106.py'),('v3','_v3','external_normal_adapter_20261106.py')]:
  path=ROOT/'provenance'/f'external_comparison{suffix}_20261106.json';reg=read(path);worker=f'scripts/evaluate_external_round_{version}_20261106.py'
  assert sha(ROOT/worker)==reg['worker_sha256'] and sha(ROOT/'src'/adapter)==reg['adapter_sha256']
  signature=sha(path);registered[signature]=reg;versions[version]={'registration_sha256':signature,'worker':worker,'worker_sha256':reg['worker_sha256'],'adapter':'src/'+adapter,'adapter_sha256':reg['adapter_sha256'],'model_specs':reg['models']}
 manifests={k:read(DATA/(k+'.json'))['records'] for k in [*REAL,'gso_primary']}
 results={};checked=0;worst=0.;scopes=0
 for model in MODELS:
  result={'quality_status':'independent_direction_checks_not_passed' if model=='StableNormal' else 'registered_evaluation_completed','panels':{},'source_scope_summary_sha256':{}}
  for scope in (['real'] if model in ['D2','NeuralLightRig'] else ['real','gso']):
   folder=OUT/model/scope;frozen=read(folder/'summary.json');signature=frozen['registration_sha256'];assert frozen['status']=='completed' and signature in registered
   result['source_scope_summary_sha256'][scope]=sha(folder/'summary.json');result.setdefault('measurement_registration_sha256',{})[scope]=signature
   panels=REAL if scope=='real' else ['gso_primary'];scope_count=0
   for panel in panels:
    records=manifests[panel];files=list((folder/'metrics'/panel).glob('*.json'));assert len(files)==len(records) and {x.stem for x in files}=={x['id'] for x in records}
    rows=[]
    for record in records:
     assert '/test/' not in record['gt_path'];row=read(folder/'metrics'/panel/(record['id']+'.json'))
     assert row['model']==model and row['registration_sha256']==signature and row['object']==record['object'] and row['image_key']==record['image_key'] and row['elevation_bin']==record.get('elevation_bin')
     assert row.get('lighting_id')==record.get('lighting_id');assert math.isfinite(row['metrics']['mean_deg']) and row['metrics']['valid_pixels']>0
     assert all(v is None or not isinstance(v,(int,float)) or math.isfinite(v) for v in row['metrics'].values());rows.append(row)
    value=macro(rows);old=frozen['panels'][panel];assert set(value['by_pitch_MAE_deg'])==set(old['by_pitch_MAE_deg'])
    for p,v in value['by_pitch_MAE_deg'].items():difference=abs(v-old['by_pitch_MAE_deg'][p]);assert difference<1e-9;worst=max(worst,difference)
    difference=abs(value['equal_pitch_object_macro_MAE_deg']-old['equal_pitch_object_macro_MAE_deg']);assert difference<1e-9;worst=max(worst,difference)
    result['panels'][panel]={**value,'images':len(records),'objects':len({x['object'] for x in records})};checked+=len(rows);scope_count+=len(rows)
   assert scope_count==frozen['completed_images'];scopes+=1
  results[model]=result;print('RECONCILED',model,flush=True)
 assert checked==15396 and scopes==16
 previous=ROOT/'publish/FNFlow_20261105/results/object_recipe_ladder_20261105/summary.json';baseline=read(previous)
 for model in ['D2','NeuralLightRig']:
  old=baseline['evaluations'][model]['panels']['primary'];results[model]['panels']['gso_primary_reference']={'equal_pitch_object_macro_MAE_deg':old['equal_pitch_object_macro_mae_deg'],'by_pitch_MAE_deg':{k:v['mean_deg'] for k,v in old['by_pitch'].items()},'source':'Previously published frozen recipe-ladder GSO-val evaluation; no new GSO inference for this row.','reference_snapshot_commit':'7b0f10a40eebbdf0a0de555aab0ca563ff835ab4','source_export_summary_sha256':sha(previous)}
 tta_folder=ROOT/'outputs/D2_TTA_20261106';tta=read(tta_folder/'summary.json');assert tta['status']=='completed';tta_checked=0;max_control=0.;tta_result={}
 for panel in ['primary','diligent','luces']:
  manifest=read(ROOT/'outputs/P3_object_dino_evaluation_fresh_20261027/manifests_private'/(panel+'.json'))['records'];tta_result[panel]={}
  for variant in ['single42','flip_TTA42','mean4','flip_TTA_mean4']:
   folder=tta_folder/'per_image_private'/panel/variant;files=list(folder.glob('*.json'));assert len(files)==len(manifest) and {p.stem for p in files}=={r['id'] for r in manifest};rows=[]
   for record in manifest:
    row=read(folder/(record['id']+'.json'));assert row['registration_sha256']==tta['registration_sha256'] and row['image_key']==record['image_key'] and row['object']==record['object'] and row['variant']==variant and math.isfinite(row['metrics']['mean_deg'])
    delta=abs(row['single42_control_delta_deg']);max_control=max(max_control,delta);assert delta<1e-4;rows.append(row)
   value=macro(rows);old=tta['panels'][panel][variant];assert abs(value['equal_pitch_object_macro_MAE_deg']-old['MAE_deg'])<1e-9
   for p,v in value['by_pitch_MAE_deg'].items():assert abs(v-old['by_pitch'][p])<1e-9
   tta_result[panel][variant]={'MAE_deg':value['equal_pitch_object_macro_MAE_deg'],'by_pitch':value['by_pitch_MAE_deg'],'images':len(rows)};tta_checked+=len(rows)
  tta_result[panel]['D2_original8_error_mean_reference']=tta['panels'][panel]['D2_original8_error_mean_reference']
 assert tta_checked==7536
 parity=read(ROOT/'outputs/StableNormal_official_parity_20261106/summary.json');shifted=read(ROOT/'outputs/StableNormal_audit_20261106/summary.json');assert parity['status']=='completed' and parity['adapter_matches_pinned_official_public_RGB_output'] and len(parity['controls'])==4
 failed=[r['control'] for r in parity['controls'] if any(not x['passed'] for x in r['official_RGB8_decoded']['cardinal_checks'].values())];assert failed==[1,3]
 quality={'status':'review_complete_baseline_not_accepted_for_main_table','model':'StableNormal','adapter_official_RGB8_output_identical_controls':4,'adapter_official_max_abs_RGB8_difference':0,'independent_controls':4,'controls_failing_one_or_more_direction_probes':failed,'fixed_matrix_to_common':[-1,-1,1],'parity_source_summary_sha256':sha(ROOT/'outputs/StableNormal_official_parity_20261106/summary.json'),'shifted_sphere_source_summary_sha256':sha(ROOT/'outputs/StableNormal_audit_20261106/summary.json'),'parity_controls':parity['controls'],'root_cause':'Coordinate convention versus shape-prediction reliability remains unresolved. Exact official-entry agreement rules out a difference introduced by the adapter at the tested interface; it does not establish a correct common coordinate frame.','disposition':'Keep all raw benchmark aggregates in summary.json, report separately as diagnostic, exclude from main performance table. Do not select alternate axes using benchmark GT.','benchmark_data_used_for_control_tests':False,'benchmark_scores_changed':False}
 efficiency_old=read(ROOT/'outputs/D2_efficiency_20261106/summary.json');assert efficiency_old['status']=='completed'
 d2_path=ROOT/'outputs/D2_efficiency_v2_20261106/D2.json'
 for _ in range(60):
  if d2_path.exists():break
  time.sleep(2)
 d2=read(d2_path);nlr=next(r for r in efficiency_old['rows'] if r['model']=='NeuralLightRig');assert d2['status']=='completed' and d2['observed_forward_hook_calls_per_inference']==1.0
 for row in [d2,nlr]:
  assert row['warmup']==5 and row['timed']==len(row['timings_seconds'])==30 and '3090' in row['GPU'] and abs(statistics.median(row['timings_seconds'])-row['median_seconds'])<1e-12
 assert d2['input_id']==nlr['input_id'] and d2['parameters_total_unique']==130400259
 efficiency={'status':'completed','rows':[d2,nlr],'NLR_over_D2_median_latency_ratio':nlr['median_seconds']/d2['median_seconds'],'original_efficiency_source_sha256':sha(ROOT/'outputs/D2_efficiency_20261106/summary.json'),'D2_repeated_source_sha256':sha(d2_path),'instrumentation_note':'The original D2 inner-module hook counted0 because the conditioned wrapper executes UNet blocks directly. Original timing records are retained. A D2-only repeat counts the enclosing conditioned forward and observes1 call; no checkpoint/input/precision changes. NLR timing is not repeated. The NLR hook observes150 stage1 UNet calls for75 CFG denoising steps.','scope':d2['scope']}
 source_reg=next(iter(registered.values()));data_freeze=read(ROOT/'provenance/external_data_finalization_20261106.json')
 protocol={'source_versions':versions,'common_frame':source_reg['common_frame'],'input_processing':source_reg['processing'],'real_single_light':source_reg['single_controls'],'real_ten_light':source_reg['ten_light_protocol'],'RNG':source_reg['noise'],'dataset_contracts':data_freeze['panels'],'source_paper_light_selection_reference':'https://arxiv.org/html/2602.09929v5#A3','paper_reproduction_claimed':False,'NLR_RNG_note':'New real panels use independent image-key-derived seed511. Do not merge these numbers with legacy-order seeds511..515 or treat them as an exact reproduction of paper results. NLR GSO in the table is the earlier frozen fresh-stream reference.','D2_seed_note':'One training seed1247. Baseline errors average inference seeds42..49 per image, not vector ensembling. TTA variants are separate and use original one-light real panels, not the ten-light panels.','TTA':read(ROOT/'provenance/D2_TTA_registration_20261106.json'),'efficiency_registration_original':read(ROOT/'provenance/D2_efficiency_registration_20261106.json'),'efficiency_registration_D2_repeat':read(ROOT/'provenance/D2_efficiency_registration_v2_20261106.json'),'test_images_scored':0,'formal_multi_training_seed_comparison_completed':False}
 summary={'status':'completed_measurements_with_one_unverified_baseline','models':results,'completed_new_scopes':16,'main_table_models':8,'diagnostic_only_models':['StableNormal'],'private_metric_rows_reconciled':checked,'test_images_scored':0,'formal_multi_training_seed_comparison_completed':False}
 tta_doc={'status':'completed','training_seed':1247,'panels':tta_result,'registration_sha256':tta['registration_sha256'],'private_metric_rows_reconciled':tta_checked,'max_single42_control_drift_deg':max_control,'test_images_scored':0}
 validation={'status':'passed_for_aggregate_integrity','external_scope_metric_rows_checked':checked,'TTA_metric_rows_checked':tta_checked,'total_private_metric_rows_checked':checked+tta_checked,'completed_new_scopes':scopes,'all_exact_manifest_ID_sets_verified':True,'registration_and_image_key_binding_verified':True,'all_primary_metric_values_finite':True,'max_external_summary_abs_difference_deg':worst,'max_TTA_single42_control_drift_deg':max_control,'efficiency_all30_samples_and_medians_checked':True,'D2_repeated_forward_hook_count':1.0,'StableNormal_direction_acceptance':'not_passed; diagnostic_only; raw scores retained','test_images_scored':0,'reconciler_sha256':sha(Path(__file__))}
 for name,doc in [('summary.json',summary),('protocol.json',protocol),('tta.json',tta_doc),('efficiency.json',efficiency),('quality_review.json',quality),('validation.json',validation)]:write(name,doc)
 lines=['# Completed external measurements, 2026-11-06','','All16 scheduled new measurement scopes, D2 inference variants and RTX3090 efficiency measurements have finished. All22,932 private metric records have been reconciled. StableNormal has not passed independent direction checks and is reported separately as diagnostic. Held-out GSO-test remains sealed. These runs do not constitute multiple independent training seeds.','','## Registered baseline results','','MAE in degrees, lower is better. Real ten-light panels use IDs21..30 as independent monocular inputs, averaging lights within object and then objects equally. The single-light controls use the prior frozen lowest-ID images. Model-specific input defaults, versions and per-image RNG are recorded in [protocol.json](protocol.json); exact reproduction of RoSE paper tables is not claimed.','','| Model | GSO-val five-angle | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |','|---|---:|---:|---:|---:|---:|']
 for name in MODELS:
  if name=='StableNormal':continue
  panels=results[name]['panels'];gso=panels.get('gso_primary',panels.get('gso_primary_reference'));values=[gso['equal_pitch_object_macro_MAE_deg']]+[panels[k]['equal_pitch_object_macro_MAE_deg'] for k in REAL];lines.append('| '+name+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
 lines+=['','D2 uses training seed1247 with mean inference errors over seeds42..49. External pretrained models use inference seed42, and Neural LightRig uses seed511. The D2 and Neural LightRig GSO-val cells are their previously published frozen recipe-ladder references; they were not rerun in the new16-scope matrix. In particular, the Neural LightRig GSO reference uses its earlier fresh-stream RNG protocol, while new real panels use image-key-derived noise. No TTA is included in the baseline table.','','## D2 optional inference variants','','The following real scores use one light per object, separately from the ten-light table above. Each constituent normal prediction is normalized before vector averaging. Flip processing includes the registered RGB/noise reflection and prediction reflection.','','| Variant | GSO-val | DiLiGenT one light | LUCES one light |','|---|---:|---:|---:|']
 for variant in ['single42','flip_TTA42','mean4','flip_TTA_mean4']:
  lines.append('| '+variant+' | '+' | '.join(f"{tta_result[p][variant]['MAE_deg']:.4f}" for p in ['primary','diligent','luces'])+' |')
 lines+=['','The original8-error-mean D2 reference is18.6650deg on GSO-val; flip_TTA_mean4 is18.2916deg. The exact single42 control reproduces its frozen per-image reference with maximum drift0.0000deg. [tta.json](tta.json) contains the full variant and angle breakdown.','','## RTX3090 batch1 efficiency','','Sequential measurements use5 warmups and30 timed samples, synchronizing CUDA each iteration. End-to-end scope includes input PNG reading, preprocessing, model computation, common256 normal conversion and output PNG writing; official Neural LightRig additionally writes its five-column composite. Model loading is excluded. D2 uses BF16, UNet256 and frozen DINOv2-B518; Neural LightRig uses its official512 input and75 denoising steps. The two model environments have their recorded PyTorch versions.','','| Model | Median seconds | P90 seconds | Unique total parameters | Peak allocated GiB |','|---|---:|---:|---:|---:|']
 for row in [d2,nlr]:lines.append('| '+row['model']+' | '+f"{row['median_seconds']:.6f} | {row['p90_seconds']:.6f} | {row['parameters_total_unique']:,} | {row['peak_allocated_bytes']/1024**3:.3f} |")
 lines +=['',f"The measured median-latency ratio is{efficiency['NLR_over_D2_median_latency_ratio']:.1f}. D2 was repeated solely to correct a forward-hook instrumentation issue; the original records remain preserved. [efficiency.json](efficiency.json) contains all timing samples and instrumentation details.",'','## StableNormal diagnostic record','','Raw aggregates are retained below and in summary.json; they are excluded from the main performance table because independent direction checks did not close.','','| GSO-val | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |','|---:|---:|---:|---:|---:|']
 panels=results['StableNormal']['panels'];values=[panels['gso_primary']['equal_pitch_object_macro_MAE_deg']]+[panels[k]['equal_pitch_object_macro_MAE_deg'] for k in REAL];lines.append('| '+' | '.join(f'{v:.4f}' for v in values)+' |')
 lines+=['','On4 independent off-center analytic-sphere controls, the adapter reproduces the pinned official public Predictor output exactly after its official display conversion: maximum8-bit RGB difference0 on every control. The official output fails one or more registered cardinal-direction probes on2 controls. This rules out an adapter/public-entry output difference at the tested interface; it does not determine whether coordinate convention or shape-prediction behavior causes the benchmark scores. The fixed candidate matrix and all raw benchmark records are preserved. No alternate axes were selected using benchmark ground truth. See [quality_review.json](quality_review.json).','','[validation.json](validation.json) records numeric reconciliation. Runtime failures and versioned corrections were retained locally: Marigold CHW-to-HWC layout correction, an early DSINE source-hash startup race, and StableNormal independent-control amendments. No failed images were silently omitted from completed scopes.','','The earlier [recipe ladder and C-D3 boundary supplement](../object_recipe_ladder_20261105/README.md) and [six-model real snapshot](../external_real_20261106/README.md) remain separate historical snapshots.']
 (DEST/'README.md').write_text('\n'.join(lines)+'\n');print('FULL_RECONCILIATION_PASSED',validation['total_private_metric_rows_checked'],efficiency['NLR_over_D2_median_latency_ratio'],flush=True)
if __name__=='__main__':main()
