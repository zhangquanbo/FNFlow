
"""Reconcile completed real-domain model scopes without exporting image-level records."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,math,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/external_round_20261106';DATA=ROOT/'data/external_round_20261106';DEST=ROOT/'publish/external_real_snapshot_20261106';DEST.mkdir(parents=True,exist_ok=True)
NAMES=['D2','DSINE','MoGe2','Lotus-D','Lotus-G','NiRNE'];PANELS=['single_diligent','single_luces','rose10_diligent','rose10_luces']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 reg=read(ROOT/'provenance/external_comparison_20261106.json');signature=sha(ROOT/'provenance/external_comparison_20261106.json');rows=[];checked=0;worst=0.
 for name in NAMES:
  frozen=read(OUT/name/'real/summary.json');assert frozen['status']=='completed' and frozen['completed_images']==264 and frozen['registration_sha256']==signature
  scores={}
  for panel in PANELS:
   records=read(DATA/(panel+'.json'))['records'];files=list((OUT/name/'real/metrics'/panel).glob('*.json'));assert len(files)==len(records) and {p.stem for p in files}=={x['id'] for x in records}
   objects=defaultdict(list)
   for record in records:
    item=read(OUT/name/'real/metrics'/panel/(record['id']+'.json'));assert item['registration_sha256']==signature and item['image_key']==record['image_key'] and item['object']==record['object']
    value=item['metrics']['mean_deg'];assert math.isfinite(value);objects[record['object']].append(value);checked+=1
   score=sum(sum(v)/len(v) for v in objects.values())/len(objects);delta=abs(score-frozen['panels'][panel]['equal_pitch_object_macro_MAE_deg']);assert delta<1e-10;worst=max(worst,delta);scores[panel]=score
  if name=='D2':
   old=read(ROOT/'outputs/D2_train_20261030/evaluation/summary.json')
   for panel,domain in [('single_diligent','diligent'),('single_luces','luces')]:assert abs(scores[panel]-old['panels'][domain]['equal_pitch_object_macro_mae_deg'])<1e-9
  rows.append({'model':name,'MAE_deg':scores,'training_seed':1247 if name=='D2' else 'official pretrained checkpoint','inference_seeds':reg['models'][name]['inference_seeds'],'completed_images':264,'source_summary_sha256':sha(OUT/name/'real/summary.json')})
 result={'status':'completed_model_subset_snapshot','snapshot_date':'2026-11-06','rows':rows,'completed_model_scopes':6,'complete_model_comparison':False,'test_GSO_images_scored':0,'formal_multi_training_seed_comparison_completed':False,'measurement_registration_sha256':signature}
 freeze=read(ROOT/'provenance/external_data_finalization_20261106.json')
 protocol={'common_frame':reg['common_frame'],'model_specs':{k:reg['models'][k] for k in NAMES},'worker_sha256':reg['worker_sha256'],'adapter_sha256':reg['adapter_sha256'],'original_registration_sha256':signature,'panels':{k:freeze['panels'][k] for k in PANELS},'input_processing':reg['processing'],'single_light':reg['single_controls'],'ten_light':reg['ten_light_protocol'],'ten_light_selection_reference':'https://arxiv.org/html/2602.09929v5#A3','paper_comparison_limit':'Light IDs21..30 match RoSE AppendixC. Exact paper reproduction is not claimed: model versions, preprocessing defaults, inference RNG and hardware are independently registered.','RNG':reg['noise'],'aggregation':'Each light is a separate monocular input. Average errors over inference draws per image, lights within each object, objects equally. No multi-image fusion or vector ensembling.','GT_and_masks':'Reuse the same verified native GT/masks for all lighting images and models. No model-output-dependent support.','D2_checkpoint_sha256':'f1cbe7e14db11c4856677731c6294d3a55b4068dcbd2f59fecae68e9fa05d170','D2_config_sha256':sha(ROOT/'configs/D2_actual_20261030.json'),'D2_UNet_input_side':256,'D2_frozen_prior':'DINOv2-B canonical518','real_data_source_verification':{'original_DiLiGenT_control_files_byte_verified_against_alternate_mirror':60,'new_DiLiGenT_images_CRC_verified':100,'LUCES_nested_zip_sha256':'3fad511cb83db13ed107f5bb1068fda195b8a4ca33e2e3727856cf3ac396a072','unchanged_existing_GT':True},'test_images_scored':0}
 validation={'status':'passed','private_metric_rows_reconciled':checked,'model_scopes':6,'panels_per_model':4,'exact_manifest_ID_sets_verified':True,'registration_and_image_key_binding_verified':True,'all_MAE_values_finite':True,'max_abs_summary_difference_deg':worst,'D2_single_light_control_reproduced':True,'test_images_scored':0,'reconciler_sha256':sha(Path(__file__))}
 write(DEST/'summary.json',result);write(DEST/'protocol.json',protocol);write(DEST/'validation.json',validation)
 lines=['# Completed real-domain scopes, 2026-11-06','','This is a snapshot of six fully completed model scopes. The full external comparison is still running; it is not a final ranking. GSO validation results and the remaining models are outside this snapshot. Held-out GSO-test remains sealed.','','MAE in degrees; lower is better. The single-light controls use the same frozen lowest-ID images as the prior recipe snapshot. The ten-light panels use light IDs21..30:100 images/10 DiLiGenT objects and140 images/14 LUCES objects. Every RGB image is processed independently. Light errors are averaged within object, then objects equally.','','| Model | DiLiGenT one light | LUCES one light | DiLiGenT ten lights | LUCES ten lights |','|---|---:|---:|---:|---:|']
 for row in rows:lines.append('| '+row['model']+' | '+' | '.join(f"{row['MAE_deg'][k]:.4f}" for k in PANELS)+' |')
 lines+=['','D2 is one training seed1247, with inference errors averaged across seeds42..49. Each external pretrained model uses one registered inference seed42; Lotus-D is deterministic. These are not multiple independent training seeds. No TTA or normal-vector ensemble is used.','','Light selection follows [RoSE AppendixC](https://arxiv.org/html/2602.09929v5#A3). The registered model versions, input defaults and per-image RNG are our own execution conditions, so these are not claimed as exact reproductions of the paper tables. In particular, they must not be merged with the earlier Neural LightRig legacy-order five-seed results.','','See [summary.json](summary.json), [protocol.json](protocol.json), and [validation.json](validation.json). Reconciliation checks1,584 private metric rows; no image-level records, images, weights or private manifests are exported.','','The prior [recipe result page](../object_recipe_ladder_20261105/README.md) now includes the C/D1/D2/D3 boundary supplement.']
 (DEST/'README.md').write_text('\n'.join(lines)+'\n');print('SNAPSHOT_RECONCILED',checked,worst,flush=True)
if __name__=='__main__':main()
