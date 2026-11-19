"""RGB-only MoGe-B conditions for training/heldout Arb; never uses benchmark labels."""
from pathlib import Path
import sys,json,hashlib,os,time,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.moge_bridge_20261106 import load_moge,tensor_rgb,normal_only,common256
from src.research_io import sha256,write_json,now
OUT=ROOT/'outputs/MoGe_E_round_20261106'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--priority-only',action='store_true');args=parser.parse_args()
 cfg=json.loads((ROOT/'configs/E2_B_actual_20261106.json').read_text());contract=cfg['coarse_contract_sha256'];directory=ROOT/cfg['coarse_cache_directory'];directory.mkdir(parents=True,exist_ok=True)
 train=[r for r in json.loads((ROOT/cfg['training_manifest']).read_text())['records'] if r['split']=='train']
 val=[r for r in json.loads((ROOT/cfg['validation_manifest']).read_text())['records'] if r['split']=='val']
 rng=np.random.default_rng(10247);priority=[]
 for _ in range(4*cfg['gradient_accumulation']):
  rng.random()
  for _ in range(cfg['microbatch']):priority.append(train[int(rng.integers(len(train)))]);rng.random()
 unique={r['image_key']:r for r in priority+train+val};ordered=list(unique.values());priority_keys={r['image_key'] for r in priority}
 assert len(unique)==len(train)+len(val)
 if args.priority_only:ordered=list({r['image_key']:r for r in priority}.values())
 folders={};registered_scenes={r['scene'] for r in train+val}
 for name in ['arb_training_raw_v1','arb_scale_c_raw_fresh_20261026']:
  for folder in (ROOT/'data'/name).glob('*/*'):
   if folder.is_dir() and folder.name in registered_scenes:
    if folder.name in folders:raise RuntimeError('Ambiguous native RGB scene path')
    folders[folder.name]=folder
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 model,_=load_moge('B');start=time.perf_counter();done=0;priority_done=set()
 with torch.inference_mode():
  for row in ordered:
   key=hashlib.sha256(row['image_key'].encode()).hexdigest();path=directory/(key+'.npz');receipt=directory/(key+'.json')
   accepted=False
   if path.exists() and receipt.exists():
    d=json.loads(receipt.read_text());accepted=d['contract_sha256']==contract and d['image_key']==row['image_key'] and d['source_rgb_sha256']==row['source_rgb_sha256'] and sha256(path)==d['sha256']
   if not accepted:
    native=folders[row['scene']]/Path(row['image_key']).name;assert sha256(native)==row['source_rgb_sha256']
    with Image.open(native) as raw:
     raw.load();rgba=raw.convert('RGBA');assert rgba.size==(512,512)
     image=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba).convert('RGB')
    prediction=common256(normal_only(model,tensor_rgb(image),use_fp16=True),'MoGe-B').cpu().numpy()
    assert prediction.shape==(256,256,3) and np.isfinite(prediction).all()
    temporary=path.with_suffix('.partial')
    with temporary.open('wb') as f:np.savez_compressed(f,normal=prediction.astype(np.float16),contract_sha256=contract)
    temporary.replace(path);write_json(receipt,{'image_key':row['image_key'],'source_rgb_sha256':row['source_rgb_sha256'],'contract_sha256':contract,'sha256':sha256(path)})
   done+=1
   if row['image_key'] in priority_keys:priority_done.add(row['image_key'])
   if len(priority_done)==len(priority_keys) and not (OUT/'coarse_priority_ready.json').exists():write_json(OUT/'coarse_priority_ready.json',{'status':'completed','contract_sha256':contract,'unique_images':len(priority_keys)})
   if not args.priority_only and (done%10==0 or done==len(ordered)):
    write_json(OUT/'coarse_cache_status.json',{'status':'completed' if done==len(ordered) else 'running','done':done,'total':len(ordered),'contract_sha256':contract,'elapsed_seconds':time.perf_counter()-start,'updated_at':now()})
    print('COARSE_CACHE',done,len(ordered),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:write_json(OUT/'coarse_cache_failure.json',{'status':'failed','error':repr(error),'updated_at':now()});raise
