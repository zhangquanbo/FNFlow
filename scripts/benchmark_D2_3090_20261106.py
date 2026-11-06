
from pathlib import Path
import sys,os,argparse,time,gc,json,random
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
OUT=ROOT/'outputs/D2_efficiency_20261106';REG=ROOT/'provenance/D2_efficiency_registration_20261106.json'
def read(p):return json.loads(p.read_text())
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['D2','NeuralLightRig'],required=True);name=parser.parse_args().model
 OUT.mkdir(exist_ok=True);reg=read(REG);assert sha256(__file__)==reg['worker_sha256'];assert '3090' in torch.cuda.get_device_name()
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
 manifest=ROOT/'outputs/P3_object_dino_evaluation_fresh_20261027/manifests_private/primary.json';assert sha256(manifest)==reg['manifest_sha256']
 record=min([r for r in read(manifest)['records'] if r['elevation_bin']==30],key=lambda r:r['id']);temp=OUT/'runtime';temp.mkdir(exist_ok=True)
 count=[0]
 def hook(*a):count[0]+=1
 if name=='D2':
  os.environ['FACEFLUX_STAGE_CONFIG']=str(ROOT/'configs/D2_actual_20261030.json')
  from scripts.train_D_stage_20261030 import make_model
  model=make_model().eval();checkpoint=ROOT/'outputs/D2_train_20261030/D2_Xn_s1247/last_local.pth';assert sha256(checkpoint)==reg['checkpoint_sha256']
  state=torch.load(checkpoint,map_location='cpu',weights_only=False);weights=model.state_dict();weights.update(state['trainable_model']);model.load_state_dict(weights,strict=True);del state,weights
  handle=model.unet.register_forward_hook(hook);parameters=list(model.parameters())
  def call():
   rgb=np.asarray(Image.open(record['white']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR));rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
   noise=torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,record['image_key'])))
   with torch.autocast('cuda',dtype=torch.bfloat16):value=model(noise,torch.zeros(1,device='cuda'),rgb)
   normal=torch.nn.functional.normalize(value.float(),dim=1)[0].permute(1,2,0).cpu().numpy()
   Image.fromarray(np.rint((normal+1)*127.5).clip(0,255).astype(np.uint8)).save(temp/'output.png')
 else:
  from src.lightrig_adapter import load_models
  module,m1,m2=load_models();handle=m1.unet.unet.register_forward_hook(hook)
  components=[m1,m2,m1.pipeline.vae,m1.pipeline.text_encoder,m1.pipeline.vision_encoder]
  parameters=list({id(p):p for m in components for p in m.parameters()}.values())
  random.seed(511);np.random.seed(511);torch.manual_seed(511)
  def call():
   module.inference_impl(m1,m2,record['rgba'],str(temp/'concat.png'),input_res=512,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
   with Image.open(temp/'concat.png') as image:value=np.asarray(image.crop((2048,0,2560,512)).convert('RGB')).astype(np.float32)/127.5-1
   value*=np.array([-1,1,1],np.float32)
   value=torch.nn.functional.interpolate(torch.from_numpy(value.copy()).permute(2,0,1)[None],size=(256,256),mode='bilinear',align_corners=False)
   normal=torch.nn.functional.normalize(value,dim=1)[0].permute(1,2,0).numpy()
   Image.fromarray(np.rint((normal+1)*127.5).clip(0,255).astype(np.uint8)).save(temp/'output.png')
 with torch.inference_mode():
  for _ in range(5):call()
  torch.cuda.synchronize();count[0]=0;torch.cuda.reset_peak_memory_stats();times=[]
  for i in range(30):
   torch.cuda.synchronize();start=time.perf_counter();call();torch.cuda.synchronize();times.append(time.perf_counter()-start)
   if (i+1)%5==0:write_json(OUT/'status.json',{'phase':'timing','model':name,'iterations':i+1,'total':30,'updated_at':now()})
 row={'status':'completed','model':name,'UNet_input_side':256 if name=='D2' else 512,'DINO_prior_input_side':518 if name=='D2' else None,'warmup':5,'timed':30,'median_seconds':float(np.median(times)),'p90_seconds':float(np.quantile(times,.9)),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'parameters_total_unique':sum(p.numel() for p in parameters),'parameters_requires_grad':sum(p.numel() for p in parameters if p.requires_grad),'stage1_UNet_forward_calls_per_inference':count[0]/30,'timings_seconds':times,'GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'input_id':record['id'],'registration_sha256':sha256(REG),'updated_at':now(),'scope':reg['scope']}
 write_json(OUT/(name+'.json'),row);handle.remove();print('EFFICIENCY_COMPLETED',name,row['median_seconds'],row['parameters_total_unique'],flush=True)
 if all((OUT/(x+'.json')).exists() for x in ['D2','NeuralLightRig']):write_json(OUT/'summary.json',{'status':'completed','rows':[read(OUT/(x+'.json')) for x in ['D2','NeuralLightRig']],'test_images_scored':0})
if __name__=='__main__':
 try:main()
 except Exception as e:write_json(OUT/'failure.json',{'error':repr(e),'updated_at':now()});raise
