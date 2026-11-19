"""New42-49 baseline table: exact frozen manifests/masks, separate from old snapshot."""
from pathlib import Path
import sys,os,json,random,importlib.util,fcntl,gc
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
from src.icme_metric_aggregation_20261110 import aggregate
from scripts.evaluate_fresh_ab import metrics
from src.gso_boundary_protocol import angular_band
NAME=sys.argv[1];GPU=os.environ['CUDA_VISIBLE_DEVICES'];assert GPU in ['1','3'];assert NAME in ['NiRNE','Lotus-D','Lotus-G','DSINE','MarigoldNormal'];O=R/'outputs/T1_metrics/new42_49'/NAME;O.mkdir(exist_ok=True);OLD=R/'outputs/external_round_20261106'/NAME;DATA=R/'data/external_round_20261106';versions={}
for suffix,adapter in [('', 'src/external_normal_adapter_20261106.py'),('_v2','src/external_normal_adapter_v2_20261106.py'),('_v3','src/external_normal_adapter_20261106.py')]:
 p=R/'provenance'/('external_comparison'+suffix+'_20261106.json');reg=json.loads(p.read_text());assert sha256(R/adapter)==reg['adapter_sha256'];versions[sha256(p)]=(reg,adapter)
summary={};deltas=[]
with (R/f'runtime/external_round_GPU{GPU}.lock').open('a') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 for scope,panels in [('real',['single_diligent','single_luces','rose10_diligent','rose10_luces']),('gso',['gso_primary'])]:
  previous=json.loads((OLD/scope/'summary.json').read_text());registration,adapter=versions[previous['registration_sha256']];spec=registration['models'][NAME];matrix=np.asarray(spec['matrix_to_common'],np.float32);module_spec=importlib.util.spec_from_file_location('pinned_new8_'+scope,R/adapter);module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module);predict=module.load(NAME,O/'scratch_private'/scope);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;signature={'new_worker_sha256':sha256(__file__),'original_registration_sha256':previous['registration_sha256'],'adapter_sha256':sha256(R/adapter),'inference_seeds':list(range(42,50)),'physical_GPU':GPU,'torch':torch.__version__};write_json(O/'registration_private.json',signature)
  for panel in panels:
   mp=DATA/(panel+'.json');records=json.loads(mp.read_text())['records'];rows=[]
   for i,r in enumerate(records):
    dest=O/'metrics_private'/panel/(r['id']+'.json')
    if dest.exists():row=json.loads(dest.read_text());assert row['signature']==signature;rows.append(row);continue
    assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
    with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
    if 'boundary_primary_angular' not in extra:extra['boundary_primary_angular']=angular_band(gt,valid,20.,2,False)
    image=Image.open(r['white']).convert('RGB');values=[]
    for seed in range(42,50):
     derived=image_noise_seed(seed,r['image_key']);random.seed(derived);np.random.seed(derived%(2**32));torch.manual_seed(derived);torch.cuda.manual_seed_all(derived)
     with torch.inference_mode():p=np.asarray(predict(image,r),np.float32)@matrix.T
     assert p.ndim==3 and p.shape[-1]==3 and np.isfinite(p).all();v=metrics(p,gt,valid,extra);values.append(v)
     if seed==42:
      legacy=json.loads((OLD/scope/'metrics'/panel/(r['id']+'.json')).read_text());delta=abs(v['mean_deg']-legacy['metrics']['mean_deg']);deltas.append(delta)
     write_json(O/'per_seed_private'/panel/str(seed)/(r['id']+'.json'),{'object':r['object'],'elevation_bin':r.get('elevation_bin'),'image_key':r['image_key'],'seed':seed,'metrics':v,'signature':signature})
    reduced={key:float(np.mean([v[key] for v in values])) if all(v[key] is not None for v in values) else None for key in values[0]};row={'object':r['object'],'image_key':r['image_key'],'elevation_bin':r.get('elevation_bin'),'metrics':reduced,'signature':signature};write_json(dest,row);rows.append(row);write_json(O/'status.json',{'phase':'inference42_49','model':NAME,'panel':panel,'images':i+1,'total':len(records),'GPU':GPU,'maximum_seed42_control_deviation_deg':max(deltas,default=0),'updated_at':now()})
   assert len(rows)==len(records);summary[panel]=aggregate(rows);write_json(O/'partial_summary.json',{'status':'partial','panels':summary,'updated_at':now()})
  del predict,module;gc.collect();torch.cuda.empty_cache()
 result={'status':'completed_new42_49','model':NAME,'panels':summary,'inference_seeds':list(range(42,50)),'maximum_seed42_control_deviation_deg':max(deltas,default=0),'drift_note':'Original fixed preprocessing/default steps/masks retained; differentGPU kernels and source nondeterminism may change seed42. Full maximum deviation is reported, never hidden or used to change axes/masks.','test_images_scored':0,'worker_sha256':sha256(__file__),'updated_at':now()};write_json(O/'summary.json',result);write_json(O/'status.json',result);(O/'REPORT.md').write_text('# '+NAME+' 新42–49评测\n\n原注册适配器、默认步骤、输入清单和mask保持；与旧协议表分开。八个推理误差等权平均，再物体聚合。\n\n最大seed42控制偏差：'+str(result['maximum_seed42_control_deviation_deg'])+'°。\n\n精确源/配置签名见registration_private.json，每个种子充分统计保留；完整指标见summary.json。测试图像0，未推送GitHub。\n');print(NAME+'_NEW8_COMPLETE',flush=True)
