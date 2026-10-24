"""Batch-one 512px end-to-end benchmark, no GT or training."""
import sys,time,json,random,gc,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.gpu_job_lock import gpu_job_lock
from src.research_io import write_json,sha256
from src.lightrig_adapter import load_models
from src.models.model_unet import FaceNormalUNet_B
from src.parameterization import one_step_endpoint
from src.noise_protocol import image_noise_seed
OUT=ROOT/'outputs/object_efficiency_20261023';OUT.mkdir(parents=True,exist_ok=True)
def measure(name,run,extra):
 for _ in range(5):run()
 torch.cuda.synchronize();times=[];peaks=[];reserved=[]
 for i in range(30):
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();run();torch.cuda.synchronize()
  times.append((time.perf_counter()-t)*1000);peaks.append(torch.cuda.max_memory_allocated());reserved.append(torch.cuda.max_memory_reserved())
  write_json(OUT/'progress.json',{'model':name,'timed_completed':i+1,'timed_total':30});print(name,i+1,times[-1],flush=True)
 return {'median_ms':float(np.median(times)),'p90_ms':float(np.percentile(times,90)),'peak_allocated_bytes':max(peaks),'peak_reserved_bytes':max(reserved),'times_ms':times,**extra}
def main():
 assert '4090' in torch.cuda.get_device_name(0)
 source=ROOT/'data/lightrig_calibration_v1/sphere.png';result={'device':torch.cuda.get_device_name(0),'batch':1,'input_hw':[512,512],'warmups':5,'repetitions':30,'source_sha256':sha256(source),'amendment_sha256':sha256(ROOT/'docs/object_calibration_intake_amendment_20261023.md'),'models':{}}
 module,a,b=load_models();random.seed(511);np.random.seed(511);torch.manual_seed(511)
 parameters={}
 for name,obj in [('stage1_registered',a),('vae',a.pipeline.vae),('text_encoder',a.pipeline.text_encoder),('vision_encoder',a.pipeline.vision_encoder)]:
  for p in obj.parameters():parameters[p.data_ptr()]=p.numel()
 count=[0]
 hook=a.unet.unet.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1))
 @torch.inference_mode()
 def run_nlr():
  module.inference_impl(a,b,str(source),str(OUT/'nlr_last.png'))
  with Image.open(OUT/'nlr_last.png') as im:im.crop((2048,0,2560,512)).convert('RGB').save(OUT/'nlr_normal_last.png')
 result['models']['NeuralLightRig']=measure('NeuralLightRig',run_nlr,{'precision':'float32 official','diffusion_steps':75,'cfg':2.,'cfg_rescale':.7,'seed':511,'stage1_parameters_including_frozen_components':sum(parameters.values()),'stage2_parameters':sum(p.numel() for p in b.parameters()),'end_to_end':'file read + official inference_impl + concatenated PNG write + extract normal RGB PNG write'})
 result['models']['NeuralLightRig']['underlying_stage1_unet_calls_per_image']=count[0]/35
 # The parameter-count loop leaves its last encoder in obj; release that alias
 # as well so the subsequent models cannot inherit NLR allocated memory.
 write_json(OUT/'results.json',result);hook.remove();del obj,a,b;gc.collect();torch.cuda.empty_cache()
 config=json.loads((ROOT/'configs/parameterization_first_round_v2_review2.json').read_text())
 for method in ['R','V','Xn']:
  path=ROOT/config['output_root']/f'cooldown_{method}_k50000_s1247/last_local.pth';ck=torch.load(path,map_location='cpu',weights_only=False)
  assert ck['step']==52500 and ck['signature']['method']==method
  model=FaceNormalUNet_B().eval().cuda();model.load_state_dict(ck['model'],strict=True);del ck
  @torch.inference_mode()
  def run_ours():
   rgba=Image.open(source).convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB')
   rgb=torch.from_numpy(np.array(im).transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
   noise=torch.zeros_like(rgb) if method=='R' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,'calibration/sphere')))
   with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
   n=F.normalize(one_step_endpoint(method,raw.float(),noise),dim=1)[0].permute(1,2,0).cpu().numpy()
   Image.fromarray(np.rint((n+1)*127.5).clip(0,255).astype(np.uint8)).save(OUT/(method+'_normal_last.png'))
  result['models'][method]=measure(method,run_ours,{'precision':'bfloat16 autocast, float32 normalization','parameters':sum(p.numel() for p in model.parameters()),'unet_forward_calls':1,'checkpoint_sha256':sha256(path),'steps':52500,'end_to_end':'read RGBA + white composite + tensor conversion + model + normalize + CPU RGB PNG write'})
  write_json(OUT/'results.json',result);del model;gc.collect();torch.cuda.empty_cache()
 write_json(OUT/'progress.json',{'status':'completed','model_scores_computed':False});print('COMPLETED',flush=True)
if __name__=='__main__':
 with gpu_job_lock():main()
