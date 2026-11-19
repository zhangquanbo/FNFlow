"""NLR official75-step graph with GPUresident preparedinputs/noises, noPNG/hosttransfers intiming."""
from pathlib import Path
import os,sys,json,time,statistics,traceback,types,random
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.icme_gpu_exclusivity_20261110 import assert_exclusive
O=R/'outputs/T3_efficiency';P=O/'models/NeuralLightRig_native.json';P.parent.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
class ResidentNoise:
 def __init__(self):self.randn=torch.randn;self.like=torch.randn_like;self.values=[];self.index=0;self.record=True
 def sample(self,which,args,kwargs):
  if self.record:
   value=(self.randn if which=='randn' else self.like)(*args,**kwargs);assert value.is_cuda,'CPUrandomdraw inNLRpuregraph';self.values.append((which,tuple(value.shape),str(value.dtype),value.detach().clone()));return value
  assert self.index<len(self.values),'additionalrandomcall';kind,shape,dtype,value=self.values[self.index];self.index+=1;assert kind==which
  if which=='like':assert tuple(args[0].shape)==shape and str(args[0].dtype)==dtype
  else:
   requested=tuple(args[0]) if len(args)==1 and isinstance(args[0],(tuple,list,torch.Size)) else tuple(args);assert requested==shape,(requested,shape)
  return value.clone()
 def __enter__(self):
  self.index=0;torch.randn=lambda *a,**k:self.sample('randn',a,k);torch.randn_like=lambda *a,**k:self.sample('like',a,k);return self
 def __exit__(self,*unused):torch.randn=self.randn;torch.randn_like=self.like
try:
 if P.exists() and read(P).get('status')=='completed':assert read(P)['worker_sha256']==sha256(__file__);sys.exit(0)
 exclusive=assert_exclusive();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from src.lightrig_adapter import load_models
 module,m1,m2=load_models();pipe=m1.pipeline;pipe.prepare();assert pipe.unet.enable_concats,'Batch4supportmustbereauditedifconcatdisabled';runtime=R/'external/lightrig_offline_runtime';from mld.util import scale_latents
 from torchvision.transforms import v2
 from torchvision.transforms import functional as TF
 record=min([r for r in read(R/'data/external_round_20261106/gso_primary.json')['records'] if r['elevation_bin']==30],key=lambda r:r['id']);rgba=Image.open(record['rgba']).resize((512,512));white=module.replace_bg_preserving_alpha(rgba,255).convert('RGB');black=module.replace_bg_preserving_alpha(rgba,0)
 def seed():random.seed(511);np.random.seed(511);torch.manual_seed(511);torch.cuda.manual_seed_all(511)
 # Originalofficialreference includes itsnativePILquantization at stageboundary.
 seed();referencefile=O/'scratch_private/NLR_reference_concat.png';referencefile.parent.mkdir(exist_ok=True,parents=True)
 with torch.inference_mode():module.inference_impl(m1,m2,record['rgba'],str(referencefile),input_res=512,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
 reference=np.asarray(Image.open(referencefile).crop((2048,0,2560,512)).convert('RGB')).copy()
 device=pipe.vae.device;dtype=pipe.vae.dtype
 image0=pipe.feature_extractor_vae(images=Image.new('RGB',white.size,(0,0,0)),return_tensors='pt').pixel_values.to(device,dtype);image1=pipe.feature_extractor_vae(images=white,return_tensors='pt').pixel_values.to(device,dtype);image2=pipe.feature_extractor_clip(images=white,return_tensors='pt').pixel_values.to(device,dtype)
 concat=torch.from_numpy(np.asarray(white).copy()).permute(2,0,1)[None].float()/255;concat=v2.functional.resize(concat,256,interpolation=3,antialias=True).clamp(0,1);concat=((concat-.5)/.8).to(device,dtype);input_black=TF.to_tensor(black)[None].to(device)
 # Fixedemptytext is modelconstant, cached as forotherdiffusionbaselines. Vision/VAEconditioning staysinsideforward.
 original_encode_prompt=pipe.encode_prompt;positive_text,negative_text=original_encode_prompt('',pipe.device,1,True)
 def cached_encode_prompt(prompt,device,num_images_per_prompt,do_classifier_free_guidance,negative_prompt=None,prompt_embeds=None,negative_prompt_embeds=None,**kw):
  assert num_images_per_prompt==1 and prompt in ['',None] and negative_prompt in ['',None]
  # Outer image pipeline requests constant empty text; inner SD pipeline passes its image-conditioned embedding.
  pos=positive_text if prompt_embeds is None else prompt_embeds
  neg=(negative_text if negative_prompt_embeds is None else negative_prompt_embeds) if do_classifier_free_guidance else None
  return pos,neg
 pipe.encode_prompt=cached_encode_prompt
 original_vae_processor=pipe.feature_extractor_vae;original_clip_processor=pipe.feature_extractor_clip
 def vae_processor(*a,**k):
  im=k.get('images',a[0] if a else None);isblack=im.getbbox() is None;return types.SimpleNamespace(pixel_values=image0 if isblack else image1)
 pipe.feature_extractor_vae=vae_processor;pipe.feature_extractor_clip=lambda *a,**k:types.SimpleNamespace(pixel_values=image2)
 original_concat=pipe.encode_concat_image
 def encode_concat(*unused,**kw):return scale_latents(pipe.vae.encode(concat).latent_dist.sample()*pipe.vae.config.scaling_factor)
 pipe.encode_concat_image=encode_concat
 quantized_unit_lut=(torch.arange(256,dtype=torch.float32)/255).to(device);assert torch.equal(quantized_unit_lut.cpu(),torch.arange(256,dtype=torch.float32)/255)
 thetas=torch.arange(9)*torch.pi/4;phis=torch.tensor([1.,2.,1.,2.,1.,2.,1.,2.,0])*torch.pi/6;illum=torch.stack([thetas,phis],1)[None].to(device)
 def graph():
  refs=pipe(image=white,guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75,output_type='pt').images
  # diffusersPILbridge rounds to8bit; reproduce onGPUbeforestage2, not a differentfloatpipeline.
  refs=quantized_unit_lut[(refs*255).round().clamp(0,255).long()];refs=m2.cut_ref_images(refs);packed=m2.pack_unet_input_sample(input_black,refs);out=super(type(m2),m2).forward(sample=packed,illumination=illum);rm,albedo,normal=out.visualize(alpha=input_black[:,3:]);return normal.clamp(0,1)
 noises=ResidentNoise();seed()
 with torch.inference_mode(),noises:out=graph()
 assert out.shape[1]==4 and torch.equal(out[:,3:],input_black[:,3:]);actual=(out[0,:3].permute(1,2,0)*255).to(torch.uint8).cpu().numpy();delta=int(np.max(np.abs(actual.astype(int)-reference.astype(int))));np.savez_compressed(O/'scratch_private/NLR_graph_parity_diagnostic.npz',actual_RGB=actual,reference_RGB=reference);assert delta==0,('officialNLRparityfailed_8bitmax',delta)
 # Schedulers alreadycontain75residenttimesteps/sigmas; resetcounterswithoutreallocatingtimeinputs.
 schedulers={id(x):x for x in [pipe.scheduler,pipe.unet.val_sched]}
 for scheduler in schedulers.values():
  assert scheduler.num_inference_steps==75
  def cached_timesteps(*a,_scheduler=scheduler,**k):
   steps=k.get('num_inference_steps',a[0] if a else None);assert steps==75;_scheduler._step_index=None;_scheduler._begin_index=None
  scheduler.set_timesteps=cached_timesteps
 noises.record=False
 def forward():
  with noises:y=graph();assert noises.index==len(noises.values);return y
 with torch.inference_mode():
  # Replaycontrol provespreallocatednoiseandtimestepsleavepredictionunchanged.
  repeat=forward();assert torch.equal(out,repeat);del repeat,out
  if '--preflight-only' in sys.argv:
   assert_exclusive();write_json(O/'NLR_preflight_repair_private.json',{'status':'passed_preflight_only','officialPNG_max_abs_uint8_difference':delta,'resident_noise_replay_bitwise_equal':True,'noise_tensors_preloaded':len(noises.values),'worker_sha256':sha256(__file__),'formal_timing_complete':False,'test_images_scored':0,'updated_at':now()});print('NLR_PREFLIGHT_REPAIR_PASSED',flush=True);sys.exit(0)
  for _ in range(5):y=forward();assert torch.isfinite(y).all();del y
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall=[];events=[]
  for i in range(30):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter();a.record();y=forward();b.record();torch.cuda.synchronize();wall.append(time.perf_counter()-start);events.append(a.elapsed_time(b));del y;write_json(O/'status.json',{'phase':'exclusive3090_timing','model':'NeuralLightRig','batch':1,'done':i+1,'total':30,'updated_at':now()})
 row={'status':'completed','batch':1,'warmup':5,'timed':30,'median_ms':statistics.median(wall)*1000,'p90_ms':float(np.quantile(wall,.9))*1000,'CUDA_event_median_ms':statistics.median(events),'images_per_second':1/statistics.median(wall),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds_private':wall};parameters={id(p):p for m in [m1,m2,pipe.vae,pipe.text_encoder,pipe.vision_encoder] for p in m.parameters()};assert_exclusive()
 batch4={'status':'official_batch4_not_supported','batch':4,'reason':'PinnedRefOnlyNoisedUNet.forward asserts sample.shape[0] in[1,2] whenenable_concats; withCFGbatch4requires8. Noarchitecturechangesorsequentialloopfalselyreported asbatch4.','source_sha256':sha256(runtime/'mld/pipeline.py')}
 result={'status':'completed','model':'NeuralLightRig','variant':'native','rows':[row,batch4],'default_steps':75,'CFG':2.,'guidance_rescale':.7,'compute_precision':'FP32 officialstorage/forward, noautocast','parameters_unique_loaded':sum(p.numel() for p in parameters.values()),'parameter_storage_dtypes':{str(d):sum(p.numel() for p in parameters.values() if p.dtype==d) for d in set(p.dtype for p in parameters.values())},'raw_input_side':512,'diffusion_canvas_side':768,'reconstruction_input_side':512,'encoder_tokens':None,'tokens_note':'multi-lightdiffusioncanvas ratherthansingleViTbudget','noise_tensors_preloaded':len(noises.values),'numerical_parity':{'officialPNG_max_abs_uint8_difference':delta,'resident_noise_replay_bitwise_equal':True},'pure_forward_scope':'VAE/visionconditionencode+75officialdiffusionsteps+VAEdecode+nativeGPU8bitstagebridge+reconstruction+normalvisualize. PreparednetworkRGB/CLIPinputs/emptytext/noise/timestepsresident; noPNG/readwrite/hosttransfer. Stagerenderingmaterialsremaininloadednativegraph.','exclusive':exclusive,'TF32':False,'source_sha256':{str(p.relative_to(R)):sha256(p) for p in [runtime/'mld/pipeline.py',runtime/'recon/model.py',R/'src/lightrig_adapter.py']},'weights_provenance':'external/lightrig_weights_5619cfe; pinnedcurrentoldprotocolofficialmodel','weight_sha256':{str(p.relative_to(R)):sha256(p) for p in (R/'external/lightrig_weights_5619cfe').rglob('*') if p.is_file() and p.suffix in ['.pt','.bin','.safetensors','.json']},'worker_sha256':sha256(__file__),'test_images_scored':0,'updated_at':now()};write_json(P,result);print('NLR_NATIVE_FORWARD_COMPLETE',flush=True)
except Exception as e:write_json(P,{'status':'needs_attention','model':'NeuralLightRig','error':repr(e),'traceback':traceback.format_exc(),'worker_sha256':sha256(__file__),'updated_at':now()});raise
