from pathlib import Path
import sys,os,json,collections,fcntl,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.moge_bridge_20261106 import load_d2,tensor_rgb
from src.e_models_20261106 import make_model
from scripts.evaluate_fresh_ab import metrics
from src.cluster_bootstrap import cluster_percentile_intervals
OUT=R/'outputs/T2_zero_noise';DATA=R/'data/external_round_20261106';REG=R/'provenance/E_evaluation_registration_20261106.json'
def read(p):return json.loads(Path(p).read_text())
def load(name):
 if name=='D2':return load_d2().eval()
 spec=read(REG)['models'][name];cfg=read(R/spec['config_path']);assert sha256(R/spec['config_path'])==spec['config_sha256'];model=make_model(cfg).eval();path=R/spec['checkpoint_path'];summary=read(path.parent/'summary.json');assert sha256(path)==summary['final_checkpoint_sha256'];ck=torch.load(path,map_location='cpu',weights_only=False);s=model.state_dict();s.update(ck['trainable_model']);model.load_state_dict(s,strict=True);return model

def run(name):
 O=OUT/name/'zero/gso';O.mkdir(parents=True,exist_ok=True);model=load(name);source=R/('outputs/MoGe_E_round_20261106/E_evaluation/E1_B/gso/metrics/gso_primary' if name=='E1_B' else 'outputs/MoGe_E_round_20261106/evaluation/D2/gso/metrics/gso_primary');manifest=DATA/'gso_primary.json';assert sha256(manifest)==read(REG)['manifest_sha256']['gso_primary'];rows=read(manifest)['records'];results=[]
 for i,r in enumerate(rows):
  dest=O/'metrics_private'/(r['id']+'.json')
  if dest.exists():results.append(read(dest));continue
  assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
  with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
  rgb=tensor_rgb(Image.open(r['white']).convert('RGB'),256)[None]*2-1
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model(torch.zeros((1,3,256,256),device='cuda'),torch.zeros(1,device='cuda'),rgb)
  values=metrics(p.float()[0].permute(1,2,0).cpu().numpy(),gt,valid,extra);random=read(source/(r['id']+'.json'));row={'id':r['id'],'object':r['object'],'elevation_bin':r['elevation_bin'],'metrics':values,'random_reference_metrics':random['metrics'],'inference_variant':'zero','forward_calls':1};write_json(dest,row);results.append(row);write_json(O/'status.json',{'phase':'zero_GSO_inference','images':i+1,'total':len(rows),'updated_at':now()})
 assert len(results)==len(rows)==1860;objects=collections.defaultdict(list);by_pitch=collections.defaultdict(list)
 for r in results:objects[r['object']].append(r);by_pitch[r['elevation_bin']].append(r)
 def mean(v,kind='metrics'):return float(np.mean([np.mean([x[kind]['mean_deg'] for x in v if x['object']==obj]) for obj in sorted({x['object'] for x in v})]))
 deltas=[np.mean([r['metrics']['mean_deg']-r['random_reference_metrics']['mean_deg'] for r in objects[obj]]) for obj in sorted(objects)];ci=cluster_percentile_intervals(deltas,sorted(objects),draws=10000,seed=20261110);summary={'status':'completed_zero_GSO_only','model':name,'zero_MAE_deg':mean(results),'random42_49_MAE_deg':mean(results,'random_reference_metrics'),'paired_zero_minus_random':ci,'by_pitch':{str(k):mean(v) for k,v in by_pitch.items()},'test_images_scored':0,'worker_sha256':sha256(__file__),'updated_at':now()};write_json(O/'summary.json',summary);write_json(O/'status.json',summary)
 if name=='E1_B':write_json(OUT/'default_inference.json',{'status':'selected_on_GSO_val_only','default':'zero' if summary['zero_MAE_deg']<=summary['random42_49_MAE_deg'] else 'random','rule':'E1_B GSO-val zero mean <= existing random42-49 mean; other variants not used to choose default','evidence':summary,'complete_T2_status':'remaining real panels and mean4/flip variants pending','updated_at':now()})
 del model;gc.collect();torch.cuda.empty_cache()
with (R/'runtime/external_round_GPU0.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX);torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for name in ['E1_B','D2']:run(name)
