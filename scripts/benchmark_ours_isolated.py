"""Recheck ours in a fresh process so no LightRig vision encoder stays resident."""
import sys,json,gc,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scripts.benchmark_object_comparison import measure,OUT
from src.models.model_unet import FaceNormalUNet_B
from src.parameterization import one_step_endpoint
from src.noise_protocol import image_noise_seed
from src.research_io import write_json,sha256
from src.gpu_job_lock import gpu_job_lock
def main():
 target=OUT/'results.json';prior=target.read_bytes();backup=OUT/'results_before_ours_isolation.json'
 if not backup.exists():backup.write_bytes(prior)
 result=json.loads(prior);source=ROOT/'data/lightrig_calibration_v1/sphere.png';assert sha256(source)==result['source_sha256']
 cfg=json.loads((ROOT/'configs/parameterization_first_round_v2_review2.json').read_text());assert torch.cuda.memory_allocated()==0
 result['ours_isolation_recheck']={'reason':'Original model-switch loop kept the last obj=vision_encoder reference alive after del a,b. LightRig timings are valid; replace our timings/memory from an independent fresh process.','fresh_process_initial_allocated_bytes':torch.cuda.memory_allocated(),'original_results_sha256':sha256(backup),'script_sha256':sha256(__file__)}
 methods=[sys.argv[sys.argv.index('--method')+1]] if '--method' in sys.argv else ['R','V','Xn']
 for method in methods:
  path=ROOT/cfg['output_root']/f'cooldown_{method}_k50000_s1247/last_local.pth';ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['step']==52500 and ck['signature']['method']==method
  model=FaceNormalUNet_B().eval().cuda();model.load_state_dict(ck['model'],strict=True);del ck
  @torch.inference_mode()
  def run():
   rgba=Image.open(source).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB');rgb=torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
   noise=torch.zeros_like(rgb) if method=='R' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,'calibration/sphere')))
   with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
   n=F.normalize(one_step_endpoint(method,raw.float(),noise),dim=1)[0].permute(1,2,0).cpu().numpy();Image.fromarray(np.rint((n+1)*127.5).clip(0,255).astype(np.uint8)).save(OUT/(method+'_normal_last.png'))
  result['models'][method]=measure(method,run,{'precision':'bfloat16 autocast, float32 normalization','parameters':sum(p.numel() for p in model.parameters()),'unet_forward_calls':1,'checkpoint_sha256':sha256(path),'steps':52500,'end_to_end':'read RGBA + white composite + tensor conversion + model + normalize + CPU RGB PNG write','measurement_process':'fresh, no LightRig models resident'})
  del model;gc.collect();torch.cuda.empty_cache()
  result['models'][method]['post_cleanup_allocated_bytes']=torch.cuda.memory_allocated()
  print('POST_CLEANUP_BYTES',method,torch.cuda.memory_allocated(),flush=True)
 result['isolated_methods']=sorted(set(result.get('isolated_methods',[]))|set(methods))
 write_json(target,result);write_json(OUT/'progress.json',{'status':'completed' if len(result['isolated_methods'])==3 else 'isolation_recheck_in_progress','isolated_methods':result['isolated_methods'],'model_scores_computed':False});print('ISOLATED BENCHMARK COMPLETE',methods,flush=True)
if __name__=='__main__':
 with gpu_job_lock():main()
