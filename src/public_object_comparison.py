"""Aggregate-only publication view; never export private records or paths."""
import json
from pathlib import Path
from src.research_io import sha256

METRIC_FIELDS=('model','domain','gt','aggregation','complete','images_done','images_expected','objects_done','mean','median','lt_11_25','lt_22_5','lt_30','boundary_mean','outside_mean','boundary_images')
SEED_FIELDS=('model','domain','complete','seeds','image_macro_mae_by_seed','sd_ddof1')
EFFICIENCY_FIELDS=('median_ms','p90_ms','peak_allocated_bytes','peak_reserved_bytes','parameters','stage1_parameters_including_frozen_components','stage2_parameters','diffusion_steps','underlying_stage1_unet_calls_per_image','unet_forward_calls','precision')

def select_fields(record, fields):
    return {k:record[k] for k in fields if k in record}

def public_summary(state):
    assert state['rows'] and all(r['complete'] for r in state['rows'])
    return {'rows':[select_fields(r,METRIC_FIELDS) for r in state['rows']],
            'seed_variability':[select_fields(r,SEED_FIELDS) for r in state['seed_variability']]}

def build_snapshot(root):
    root=Path(root);report=root/'outputs/object_comparison_report_20261024'
    validation=json.loads((report/'validation.json').read_text());delivery=json.loads((report/'delivery_receipt.json').read_text())
    assert validation['status']=='passed' and set(validation['datasets'])=={'main','diligent','luces'}
    assert delivery['validation_sha256']==sha256(report/'validation.json')
    assert not delivery['training_started'] and not delivery['step3_started']
    result={'status':'completed','training_started':False,'primary_models':['NeuralLightRig','R','V','Xn'],
            'checkpoint_updates':52500,'datasets':{'gso':{'objects':198,'images':792},'arb':{'objects':128,'images':384},'diligent':{'objects':10,'images':10},'luces':{'objects':14,'images':14}},'rows':[],'seed_variability':[],'source_checks':[]}
    for receipt in validation['receipts']:
        name={'main':'object_comparison_20261023','diligent':'object_comparison_diligent_20261023','luces':'object_comparison_luces_20261023'}[receipt['dataset']]
        p=root/'outputs'/name/'summary.json';assert sha256(p)==receipt['summary_sha256']
        state=json.loads(p.read_text());assert state['manifest_sha256']==receipt['manifest_sha256']
        public=public_summary(state);result['rows'].extend(public['rows']);result['seed_variability'].extend(public['seed_variability'])
        result['source_checks'].append(select_fields(receipt,('dataset','manifest_sha256','summary_sha256','case_panels')))
    for row in result['rows']:
        expected=result['datasets'][row['domain']]
        if row['model']!='NeuralLightRig_variability':assert row['images_done']==expected['images'] and row['objects_done']==expected['objects']
    assert len(result['rows'])==validation['aggregate_rows_recomputed']==60
    result['arb_uid_overlap']={'original_assets':198,'excluded_any_lightprop_split':70,'excluded_percent':70/198*100,'train_overlap':65,'eval_overlap':4,'test_overlap':1,'retained_assets':128,'retained_images':384}
    result['paired_object_counts']=validation['paired_object_wins']
    result['inference_seed_sensitivity']=validation['seed_sensitivity']
    result['same_image_subset_seed_sensitivity']=validation['seed_sensitivity_same_subset']
    result['gso_gt_disagreement']=json.loads((report/'gso_gt_disagreement.json').read_text())
    eff=json.loads((root/'outputs/object_efficiency_20261023/results.json').read_text())
    result['efficiency']={k:eff[k] for k in ['device','batch','input_hw','warmups','repetitions']}
    result['efficiency']['models']={m:select_fields(r,EFFICIENCY_FIELDS) for m,r in eff['models'].items()}
    result['verification']={'private_seed_image_records_checked':validation['private_records_checked'],'aggregate_rows_recomputed':validation['aggregate_rows_recomputed'],'sampled_pixel_checks':len(validation['pixel_checks']),'formal_case_panels':len(delivery['case_panels']),'pixel_check_scope':'Two pre-registered fixed cases/domain and four models; independent float64 resampling, normalization, mean, median and strict thresholds. Sampled pixel verification, not all-image pixel recomputation.'}
    paper=json.loads((report/'rose_crosscheck.json').read_text());result['rose_reference']=paper
    result['notes']={
        'aggregation':'Seeds averaged within each image metric, then image/object macro. Median is macro of within-image pixel medians; thresholds are strict pixel percentages, not percentages of objects.',
        'boundary':'GT adjacent normal jump >20 degrees, radius2 dilation and GT-valid intersection; excludes silhouette unless it also meets the angular definition. Empty regions are null with coverage counts.',
        'inference_seeds':'R deterministic; V/Xn image-specific CUDA noise seeds42..49. NLR default511 once per sorted input stream. Pre-registered NLR subset rerun seeds511/512/513; subset511 does not share the full-stream RNG order.',
        'real_protocol':'One preselected lowest-ID light per object, not the RoSE ten-image21..30 protocol. No per-image exposure, gamma, light-intensity division or crop. uint16/65535 mapped to8bit. Official masks. Pot2 excludes73 zero-GT official-mask pixels, before scoring; no LUCES exclusions.',
        'efficiency':'Shared4090,512x512,batch1,5warmups,30timings. OfficialNLR float32; ours bf16 autocast with float32 normalization. Accuracy inference ours256/NLR512. Native output-writing paths differ; not an identical-precision pure-compute comparison.',
        'labels':'GSO main smooth geometry, secondary face and crease45. Arb retains BSDF normal-pass labels and its GT-valid contract. UID exclusion does not prove exhaustive cross-source geometry deduplication.',
        'uncertainty':'Inference seed variation only; one training seed for R/V/Xn; not training-seed uncertainty or confidence intervals.',
        'publication':'Aggregates and code only. Source images, private manifests/records, predictions, example figures, external weights and research interpretations are not distributed.',
    }
    return result

def protocol_config(root):
    root=Path(root);lock=json.loads((root/'outputs/lightrig_calibration_v1/coordinate_lock.json').read_text());assert lock['status']=='closed'
    weights=json.loads((root/'provenance/lightrig_weights_5619cfe.json').read_text())
    base=json.loads((root/'provenance/lightrig_base_intake_partial.json').read_text());assert base['all_required_present']
    component_root=root/'external/base_models';weight_root=root/'external/lightrig_weights_5619cfe'
    return {'version':'object_comparison_20261024','training':False,'input':'single RGB','output':'surface normal',
        'legacy_budget_rules':'The former24-run/review2 rules are archived historical diagnostics and are not unlocked by this evaluation.',
        'source':{'repository':'https://github.com/ZexinHe/Neural-LightRig','commit':'db472e191e7abd3115ced0d5c72d61b75a8412ed'},
        'nlr_weights':{'repository':weights['repo'],'revision':weights['revision'],'components':[{'component':Path(r['path']).relative_to(weight_root).as_posix(),'bytes':r['bytes'],'sha256':r['sha256']} for r in weights['files']]},
        'base_components':[{'component':Path(r['path']).relative_to(component_root).as_posix(),**select_fields(r,('bytes','sha256','metadata_revision','metadata_etag'))} for r in base['verified_existing_files']],
        'nlr_defaults':{'input_size':512,'diffusion_steps':75,'cfg':2.0,'cfg_rescale':0.7,'seed':511},
        'ours':{'model':'FaceNormalUNet_B','parameters':43081539,'checkpoint_updates':52500,'accuracy_size':256,'precision':'bfloat16 autocast, float32 normalization','UNet_calls':1,'inference_seeds':list(range(42,50))},
        'coordinates':{'common':'x-left,y-up,z-toward-viewer','nlr_to_common':lock['nlr_to_common'],'calibration':'Independent non-test analytic sphere/cube and author-referenced normal shader. No test-GT axis search.','real_gt_to_common':[[-1,0,0],[0,1,0],[0,0,1]]},
        'alignment':'Bilinear vector components to native GT resolution, then unit normalization; GT-valid mask shared across models, no prediction-dependent exclusions.',
        'independent_environment':{'python':'3.11','torch':'2.4.1+cu121','torchvision':'0.19.1','numpy':'1.26.4','diffusers':'0.30.3','transformers':'4.44.2','lightning':'2.4.0'},
        'external_access':'Obtain datasets/weights separately with required licenses and permissions; no credentials or third-party binaries distributed.'}

def results_markdown(snapshot):
    def find(d,m):return next(r for r in snapshot['rows'] if r['domain']==d and r['model']==m and r['gt']=='main' and r['aggregation']=='object_macro')
    lines=['# Completed object normal comparison','', 'Completed inference-only evaluation of Neural LightRig and existing52,500-update R/V/Xn checkpoints. No new training. MAE degrees, lower is better; object macro.','', '| Dataset (objects/images) | Neural LightRig | R | V | Xn |','|---|---:|---:|---:|---:|']
    for d,n in snapshot['datasets'].items():lines.append('| '+d+f" ({n['objects']}/{n['images']}) | "+' | '.join(f"{find(d,m)['mean']:.4f}" for m in snapshot['primary_models'])+' |')
    lines+=['','[Full aggregate JSON](summary.json) includes both macro aggregations, main/secondary GT, strict pixel thresholds, boundary coverage, inference-seed sensitivity, paired-object counts, GT disagreement and efficiency. [Pinned protocol and component hashes](../../configs/object_comparison_release_20261024.json).','', '## Measurement conditions','',*['- '+v for v in snapshot['notes'].values()],'', 'RoSE v5 Table1/2 reference: Neural LightRig29.10 degrees on DiLiGenT and20.95 degrees on LUCES, using ten lights21..30 per object. This release uses one lowest-ID light. Source: https://arxiv.org/pdf/2602.09929v5 . Numerical proximity is not proof of exact protocol reproduction.','', '## Local execution','', 'Use an independent Python3.11+/PyTorch2.4+ environment for Neural LightRig. Obtain the pinned upstream source/weights and base components with your own access, and configure the offline paths expected by src/lightrig_adapter.py. The original training environment is separate.','', 'The preparers require local source datasets and registered protocol receipts. They generate private manifests, not a distributed test list. After preparing your own manifests, scripts/evaluate_object_comparison.py uses OBJECT_COMPARISON_MANIFEST and OBJECT_COMPARISON_OUTPUT to select them. scripts/validate_object_comparison_results.py independently checks complete local results. Inference is serialized by the shared GPU lock. These commands do not train models.','', 'The original masks, missing/invalid-GT counts, input range mapping and coordinate transforms are fixed before scoring. RGB compositing differs only as declared: transparent alpha for NLR, white background for ours.','', 'The original archived WSD training commands/results remain historical; this object evaluation does not authorize or start that training queue. External datasets, official example GT (unavailable for the clay vase), weights and sample images are not supplied.','']
    return '\n'.join(lines)
