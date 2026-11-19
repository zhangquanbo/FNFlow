"""T2 exact D2-TTA definitions, all five panels; feature reuse only for accuracy, never timing."""
from pathlib import Path
import sys,os,json,gc,fcntl,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
from src.moge_bridge_20261106 import load_d2,tensor_rgb
from src.e_models_20261106 import make_model
from src.icme_metric_aggregation_20261110 import aggregate
from src.cluster_bootstrap import cluster_percentile_intervals
from src.gso_boundary_protocol import angular_band
from scripts.evaluate_fresh_ab import metrics
O=R/'outputs/T2_zero_noise';DATA=R/'data/external_round_20261106';PANELS=['gso_primary','single_diligent','single_luces','rose10_diligent','rose10_luces'];VARIANTS=['random','zero','mean4','flip_TTA','flip_TTA_mean4'];CALLS={'random':1,'zero':1,'mean4':4,'flip_TTA':2,'flip_TTA_mean4':8}
def read(p):return json.loads(Path(p).read_text())
def unit(x):
 n=torch.linalg.vector_norm(x.float(),dim=0,keepdim=True);assert torch.isfinite(n).all() and (n>1e-8).all();return x.float()/n
def reflection(x):y=x.flip(-1).clone();y[0]*=-1;return y

def load(name):
 if name=='D2':return load_d2().eval(),'configs/D2_actual_20261030.json','outputs/D2_train_20261030/D2_Xn_s1247/last_local.pth'
 spec=read(R/'provenance/E_evaluation_registration_20261106.json')['models'][name];cfg=read(R/spec['config_path']);model=make_model(cfg).eval();ck=torch.load(R/spec['checkpoint_path'],map_location='cpu',weights_only=False);state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);return model,spec['config_path'],spec['checkpoint_path']

def main():
 torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;results={};model_signatures={};max_drift={}
 for name in ['E1_B','D2']:
  model,config,checkpoint=load(name);signature={'config_sha256':sha256(R/config),'checkpoint_sha256':sha256(R/checkpoint),'worker_sha256':sha256(__file__),'precision':'BF16 autocast,FP32 weights','feature_reuse':'Original/flipped RGB each encoded once; fixed-image reuse only in accuracy scoring, parity checked bitwise; not used in efficiency benchmarks','GPU_scope':'shares3090 with training; no timing claim'};model_signatures[name]=signature;results[name]={};max_drift[name]=0.;checked_cache=False;original_feature_forward=model.features.forward
  for panel in PANELS:
   records=read(DATA/(panel+'.json'))['records'];folder=O/'full_variants'/name/panel;allrows={v:[] for v in VARIANTS}
   for i,r in enumerate(records):
    destinations={v:folder/v/(r['id']+'.json') for v in VARIANTS}
    if all(p.exists() for p in destinations.values()):
     for v,p in destinations.items():row=read(p);assert row['signature']==signature;allrows[v].append(row)
     continue
    assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256'];assert sha256(r['white'])==r.get('white_sha256',r.get('ours_rgb_sha256'))
    with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool);extra={k:z[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in z.files}
    if 'boundary_primary_angular' not in extra:extra['boundary_primary_angular']=angular_band(gt,valid,20.,2,False)
    rgb=tensor_rgb(Image.open(r['white']).convert('RGB'),256)[None]*2-1
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):feature=original_feature_forward(rgb);flipped_feature=original_feature_forward(rgb.flip(-1))
    originals=[];flipped=[]
    def forward(noise,flip=False):
     model.features.forward=lambda _:flipped_feature if flip else feature
     try:
      with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(noise,torch.zeros(1,device='cuda'),rgb.flip(-1) if flip else rgb)[0].float()
      return out
     finally:model.features.forward=original_feature_forward
    for seed in range(42,50):
     noise=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])));out=forward(noise);originals.append(out)
     if not checked_cache:
      with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):uncached=model(noise,torch.zeros(1,device='cuda'),rgb)[0].float()
      assert torch.equal(out,uncached),'Frozen feature reuse parity failed';checked_cache=True
     if seed<46:flipped.append(reflection(forward(reflection(noise[0])[None],True)))
    per=[metrics(x.permute(1,2,0).cpu().numpy(),gt,valid,extra) for x in originals];random_metrics={k:float(np.mean([x[k] for x in per])) if all(x[k] is not None for x in per) else None for k in per[0]};scope='gso' if panel=='gso_primary' else 'real';source=R/('outputs/MoGe_E_round_20261106/E_evaluation/E1_B' if name=='E1_B' else 'outputs/MoGe_E_round_20261106/evaluation/D2');reference=read(source/scope/'metrics'/panel/(r['id']+'.json'));delta=abs(random_metrics['mean_deg']-reference['metrics']['mean_deg']);max_drift[name]=max(max_drift[name],delta);assert delta<1e-6,(name,panel,r['id'],delta)
    cached_zero=O/name/'zero/gso/metrics_private'/(r['id']+'.json')
    if panel=='gso_primary' and cached_zero.exists():zero_metrics=read(cached_zero)['metrics']
    else:zero_metrics=metrics(forward(torch.zeros((1,3,256,256),device='cuda')).permute(1,2,0).cpu().numpy(),gt,valid,extra)
    us=[unit(x) for x in originals[:4]];fs=[unit(x) for x in flipped];outputs={'mean4':unit(torch.stack(us).mean(0)),'flip_TTA':unit(us[0]+fs[0]),'flip_TTA_mean4':unit(torch.stack(us+fs).mean(0))};values={'random':random_metrics,'zero':zero_metrics,**{v:metrics(p.permute(1,2,0).cpu().numpy(),gt,valid,extra) for v,p in outputs.items()}}
    for variant,dest in destinations.items():row={'object':r['object'],'image_key':r['image_key'],'elevation_bin':r.get('elevation_bin'),'variant':variant,'metrics':values[variant],'per_output_forward_calls':CALLS[variant],'signature':signature};write_json(dest,row);allrows[variant].append(row)
    write_json(O/'full_variants_status.json',{'phase':'accuracy_only_shared3090','model':name,'panel':panel,'images':i+1,'total':len(records),'maximum_random_control_drift_deg':max_drift[name],'updated_at':now()})
    del originals,flipped,outputs,us,fs,rgb,feature,flipped_feature
   results[name][panel]={v:aggregate(rows) for v,rows in allrows.items()};objects=collections.defaultdict(dict)
   for v in ['random','zero']:
    for row in allrows[v]:objects[row['object']].setdefault(v,[]).append(row['metrics']['mean_deg'])
   ids=sorted(objects);differences=[float(np.mean(objects[x]['zero'])-np.mean(objects[x]['random'])) for x in ids];results[name][panel]['paired_zero_minus_random']=cluster_percentile_intervals(differences,ids,draws=10000,seed=20261110);write_json(O/'partial_full_variants_summary.json',{'status':'partial','models':results,'updated_at':now()})
  del model;gc.collect();torch.cuda.empty_cache()
 result={'status':'completed','models':results,'model_signatures':model_signatures,'maximum_random_control_drift_deg':max_drift,'default_inference':read(O/'default_inference.json')['default'],'selection':'Only prior registered GSO-val zero<=random rule; TTA and real data did not choose default','variant_definitions':'random is42-49 error mean(one forward per stochastic inference);zero deterministic;mean4 vectors42-45;flip_TTA fixed42+reflected42;flip_mean4 vectors42-45 and matched reflections, exactly originalD2TTA definitions','calls_per_output':CALLS,'test_images_scored':0,'updated_at':now()};write_json(O/'summary.json',result);write_json(O/'full_variants_status.json',{'phase':'completed','updated_at':now()});lines=['# T2 完整推理起点与TTA对照','',f"默认保持{result['default_inference']}，只依据预登记GSO-val zero判定。原随机8误差均值的最大复现偏差{max_drift}°。",'','|模型|面板|random|zero|mean4|flip|flip+mean4|','|---|---|---:|---:|---:|---:|---:|']
 for n,panels in results.items():
  for panel,x in panels.items():lines.append('|'+n+'|'+panel+'|'+'|'.join(f"{x[v]['metrics']['mean_deg']['mean']:.5f}" for v in VARIANTS)+'|')
 lines+=['','前向次数：random单个输出1次（评测8次误差均值）、zero1、mean4为4、flip为2、flip_mean4为8。TTA种子沿用旧脚本固定定义，没有挑最佳seed。','\n两个方向分别计算冻结编码器特征并逐bit验证缓存一致，只用于精度评测；未把省略编码器的耗时当作效率。实际GSO边界带复用固定GT字段，真实面板用相同20°/半径2规则，mask保持。配置、检查点及工作脚本SHA见summary.json。GSO-test评分0，未推送GitHub。'];(O/'REPORT.md').write_text('\n'.join(lines)+'\n');print('T2_ALL_VARIANTS_COMPLETE',flush=True)
with (R/'runtime/ICME_T2_aux_GPU0.lock').open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX);main()
