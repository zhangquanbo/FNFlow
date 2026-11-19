from pathlib import Path
import sys,json,argparse,gc,random,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.icme_rplus_20261110 import make_model
from src.icme_metric_aggregation_20261110 import aggregate
from src.moge_bridge_20261106 import tensor_rgb
from src.noise_protocol import image_noise_seed
from scripts.evaluate_fresh_ab import metrics
from src.gso_boundary_protocol import angular_band
P=argparse.ArgumentParser();P.add_argument('--config',required=True);P.add_argument('--stage',choices=['25000','50000'],required=True);args=P.parse_args();cfgpath=Path(args.config);cfg=json.loads(cfgpath.read_text());args.run=cfg['run'];O=R/cfg['output_root'];reg=json.loads((R/cfg['registration_path']).read_text());assert sha256(cfgpath)==reg['config_sha256'];decision=R/'outputs/T2_zero_noise/default_inference.json';assert decision.exists();default=json.loads(decision.read_text())['default'];checkpoint=O/('diagnostic_25000.pth' if args.stage=='25000' else 'final_50000.pth');checkpoint_sha=sha256(checkpoint);ck=torch.load(str(checkpoint),map_location='cpu',weights_only=False);assert ck['step']==int(args.stage);model=make_model(cfg).eval();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;panelnames=['gso_primary'] if args.stage=='25000' else ['gso_primary','single_diligent','single_luces','rose10_diligent','rose10_luces'];variants=['raw'] if args.stage=='25000' else ['raw','EMA'];results={};DATA=R/'data/external_round_20261106';base=O/'evaluation'/args.stage;signature={'checkpoint_sha256':checkpoint_sha,'config_sha256':sha256(cfgpath),'inference_decision_sha256':sha256(decision),'worker_sha256':sha256(__file__),'aggregation_sha256':sha256(R/'src/icme_metric_aggregation_20261110.py')};visual_selected=R/'outputs/ICME2027_20261110/qualitative_selection_private.json';selected=set(json.loads(visual_selected.read_text())['image_ids']) if visual_selected.exists() else set()
for variant in variants:
 state=dict(ck['model'])
 if variant=='EMA':state.update(ck['ema'])
 model.load_state_dict(state,strict=True);model.features.set_gradient(False);model.eval();results[variant]={}
 for panel in panelnames:
  manifest=DATA/(panel+'.json');records=json.loads(manifest.read_text())['records'];folder=base/variant/panel;rows=[]
  for i,r in enumerate(records):
   dest=folder/'metrics_private'/(r['id']+'.json')
   if dest.exists():row=json.loads(dest.read_text());assert row['signature']==signature;rows.append(row);continue
   assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
   with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
   if 'boundary_primary_angular' not in extra:extra['boundary_primary_angular']=angular_band(gt,valid,20.,2,False)
   rgb=tensor_rgb(Image.open(r['white']).convert('RGB'),256)[None]*2-1;values=[]
   for seed in ([42] if default=='zero' else range(42,50)):
    noise=torch.zeros((1,3,256,256),device='cuda') if default=='zero' else torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(noise,torch.zeros(1,device='cuda'),rgb)
    array=pred.float()[0].permute(1,2,0).cpu().numpy();values.append(metrics(array,gt,valid,extra))
    if args.stage=='50000' and r['id'] in selected:
     p=folder/'qualitative_predictions_private'/(r['id']+'_'+str(seed)+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,normal=array.astype(np.float16))
   reduced={key:float(np.mean([v[key] for v in values])) if all(v[key] is not None for v in values) else None for key in values[0]};row={'object':r['object'],'image_key':r['image_key'],'elevation_bin':r.get('elevation_bin'),'metrics':reduced,'variant':variant,'inference':default,'inference_seeds':list(range(42,50)) if default=='random' else 'deterministic_zero','signature':signature};write_json(dest,row);rows.append(row);write_json(base/'status.json',{'phase':'scoring','run':args.run,'variant':variant,'panel':panel,'images':i+1,'total':len(records),'updated_at':now()})
  assert len(rows)==len(records);results[variant][panel]=aggregate(rows)
summary={'status':'completed','run':args.run,'stage':int(args.stage),'raw_and_EMA':results,'signature':signature,'inference_default':default,'primary_final_weight_variant':'EMA; raw also reported, user receives both before choosing final model','sampler_deviation':cfg['deviation'],'test_images_scored':0,'updated_at':now()};write_json(base/'summary.json',summary);write_json(base/'status.json',summary)
if args.stage=='25000':
 value=results['raw']['gso_primary']['metrics']['mean_deg']['mean'];go=cfg['early_stop_threshold'] is None or value<=cfg['early_stop_threshold'];write_json(O/'gate_result.json',{'continue':go,'raw_GSO_val_MAE_deg':value,'registered_threshold_deg':cfg['early_stop_threshold'],'use':'diagnostic only; fixed50k, no metric-based abandonment or checkpoint selection','inference':default,'signature':signature,'updated_at':now()});write_json(O/'gate_status.json',{'phase':'passed_continue_to_fixed50000' if go else 'abandoned_by_registered_gate','updated_at':now()});(O/'REPORT.md').write_text(f"# {args.run} 25k固定预算诊断\n\nraw GSO-val MAE {value:.6f}°；门槛{cfg['early_stop_threshold']}°；继续={go}。仅用于放弃，不选择检查点。\n\n配置SHA256 `{sha256(cfgpath)}`；检查点SHA256 `{checkpoint_sha}`。\n\n偏差：{cfg['deviation']}\n")
else:
 text=[f'# {args.run} 固定50k结果','',f"推理：{default}；原始权重和EMA同时报告。主候选预登记EMA，不从曲线挑步。",f"\n配置SHA256 `{sha256(cfgpath)}`；检查点SHA256 `{checkpoint_sha}`。",f"\n偏差：{cfg['deviation']}",'','|权重|面板|MAE°|','|---|---|---:|']
 for variant,panels in results.items():
  for panel,v in panels.items():text.append(f"|{variant}|{panel}|{v['metrics']['mean_deg']['mean']:.6f}|")
 text+=['','完整median/阈值比例/边界/物体bootstrap见evaluation/50000/summary.json。效率尚需同卡独占测量，未将训练并发期间的计时当作效率。GSO-test评分0，未推送公开仓库。'];(O/'REPORT.md').write_text('\n'.join(text)+'\n')
print('RPLUS_EVALUATION_COMPLETE',args.run,args.stage,flush=True)
