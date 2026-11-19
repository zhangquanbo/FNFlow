"""Original E1 recipe replicas/S0 final metrics on the frozen five development panels."""
from pathlib import Path
import sys,os,json,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.e_models_20261106 import make_model
from src.moge_bridge_20261106 import tensor_rgb
from src.noise_protocol import image_noise_seed
from src.research_io import write_json,sha256,now
from src.icme_metric_aggregation_20261110 import aggregate
from src.gso_boundary_protocol import angular_band
from scripts.evaluate_fresh_ab import metrics
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['25000','50000'],default='50000');a=p.parse_args();cfgpath=Path(a.config);cfg=json.loads(cfgpath.read_text());checkpoint=R/cfg['checkpoint_relative'];job=checkpoint.parent
if a.stage=='25000':checkpoint=job/'diagnostic_25000.pth'
ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['step']==int(a.stage);model=make_model(cfg).eval();state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);default=json.loads((R/'outputs/T2_zero_noise/default_inference.json').read_text())['default'];signature={'config_sha256':sha256(cfgpath),'checkpoint_sha256':sha256(checkpoint),'worker_sha256':sha256(__file__),'inference_default':default};del ck,state;torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;results={};base=job/('evaluation_'+a.stage);panels=['gso_primary'] if a.stage=='25000' else ['gso_primary','single_diligent','single_luces','rose10_diligent','rose10_luces']
for panel in panels:
 records=json.loads((R/'data/external_round_20261106'/(panel+'.json')).read_text())['records'];rows=[]
 for i,r in enumerate(records):
  dest=base/panel/'metrics_private'/(r['id']+'.json')
  if dest.exists():row=json.loads(dest.read_text());assert row['signature']==signature;rows.append(row);continue
  assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
  with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
  if 'boundary_primary_angular' not in extra:extra['boundary_primary_angular']=angular_band(gt,valid,20.,2,False)
  rgb=tensor_rgb(Image.open(r['white']).convert('RGB'),256)[None]*2-1;values=[]
  for seed in ([42] if default=='zero' else range(42,50)):
   noise=torch.zeros((1,3,256,256),device='cuda') if default=='zero' else torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(noise,torch.zeros(1,device='cuda'),rgb)
   values.append(metrics(out.float()[0].permute(1,2,0).cpu().numpy(),gt,valid,extra))
  reduced={k:float(np.mean([v[k] for v in values])) if all(v[k] is not None for v in values) else None for k in values[0]};row={'object':r['object'],'image_key':r['image_key'],'elevation_bin':r.get('elevation_bin'),'metrics':reduced,'signature':signature};write_json(dest,row);rows.append(row);write_json(base/'status.json',{'phase':'scoring','panel':panel,'images':i+1,'total':len(records),'updated_at':now()})
 results[panel]=aggregate(rows)
summary={'status':'completed','run':cfg['run_label'],'seed':cfg['seed'],'stage':a.stage,'panels':results,'signature':signature,'EMA':False,'distillation':False,'pitch_balanced':False,'original_E1_recipe':True,'test_images_scored':0,'updated_at':now()};write_json(base/'summary.json',summary);write_json(base/'status.json',summary);report=[f"# {cfg['run_label']} {a.stage}结果",'',f"固定预算/原配方，种子{cfg['seed']}，默认推理{default}。没有EMA、蒸馏或俯仰均衡。",f"\n配置SHA256 `{signature['config_sha256']}`；检查点SHA256 `{signature['checkpoint_sha256']}`。",'','|面板|MAE°|','|---|---:|']
for panel,value in results.items():report.append(f"|{panel}|{value['metrics']['mean_deg']['mean']:.6f}|")
report+=['','精确完整配置见actual.json，完整median/阈值比例/边界/bootstrap见evaluation目录。未推送GitHub，GSO-test评分0。'];(job/'REPORT.md').write_text('\n'.join(report)+'\n')
if a.stage=='25000':write_json(job/'gate_result.json',{'continue':True,'raw_GSO_val_MAE_deg':results['gso_primary']['metrics']['mean_deg']['mean'],'registered_abandon_threshold':None,'purpose':'S0 diagnostic only, no user-specified abandon threshold; fixed50k stays final','updated_at':now()})
