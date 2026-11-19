"""GPU nativeforward baselines with parity to pinnedofficialadapter; noIO/noisecreation in timing."""
from pathlib import Path
import sys,os,json,gc,time,statistics,traceback,importlib
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image
from src.research_io import write_json,sha256,now
from src.icme_gpu_exclusivity_20261110 import assert_exclusive
SMOKE='--parity-smoke' in sys.argv
if SMOKE:
 assert os.environ.get('CUDA_VISIBLE_DEVICES') in ['1','3'], 'smokeusesauthorizedauxiliaryGPUonly'
 def assert_exclusive(*args):return {'mode':'parity_only_no_timing','physical_GPU':os.environ['CUDA_VISIBLE_DEVICES']}
NAME=sys.argv[1];assert NAME in ['DSINE','NiRNE','MarigoldNormal','Lotus-D','Lotus-G'];O=R/'outputs/T3_efficiency';O=O/'parity_smoke_v3_private'/('GPU'+os.environ['CUDA_VISIBLE_DEVICES']) if SMOKE else O;O.mkdir(exist_ok=True,parents=True);P=O/'models'/(NAME+'_native.json');P.parent.mkdir(exist_ok=True);STATUS=O/'status.json'
def read(p):return json.loads(Path(p).read_text())
def measure(fn,batch):
 if SMOKE:
  with torch.inference_mode():out=fn();assert torch.isfinite(out).all();del out
  return {'status':'passed_forward_smoke','batch':batch,'no_latency_measurement':True}
 with torch.inference_mode():
  for _ in range(5):out=fn();assert torch.isfinite(out).all();del out
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();wall=[];events=[]
  for i in range(30):
   a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter();a.record();out=fn();b.record();torch.cuda.synchronize();wall.append(time.perf_counter()-start);events.append(a.elapsed_time(b));del out;write_json(STATUS,{'phase':'exclusive3090_timing','model':NAME,'batch':batch,'done':i+1,'total':30,'updated_at':now()})
 return {'status':'completed','batch':batch,'warmup':5,'timed':30,'median_ms':statistics.median(wall)*1000,'p90_ms':float(np.quantile(wall,.9))*1000,'CUDA_event_median_ms':statistics.median(events),'images_per_second':batch/statistics.median(wall),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'wall_seconds_private':wall}
try:
 if P.exists() and read(P).get('status')=='completed':assert read(P)['worker_sha256']==sha256(__file__);sys.exit(0)
 exclusive=assert_exclusive();torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from src.external_normal_adapter_v2_20261106 import load
 predictor=load(NAME,O/'scratch_private'/NAME);captures=dict(zip(predictor.__code__.co_freevars,[c.cell_contents for c in predictor.__closure__]));root=captures['model'];model=root.model.inner if NAME=='NiRNE' else root;torch.set_num_threads(4)
 mp=R/'data/external_round_20261106/gso_primary.json';record=min([r for r in read(mp)['records'] if r['elevation_bin']==30],key=lambda r:r['id']);assert sha256(record['white'])==record.get('white_sha256',record.get('ours_rgb_sha256'));image=Image.open(record['white']).convert('RGB');rows=[];parity=[];components=[model] if isinstance(model,torch.nn.Module) else [x for x in model.components.values() if isinstance(x,torch.nn.Module)];parameters={id(p):p for c in components for p in c.parameters()};steps=1;precision='FP32';latent_hw=None
 for batch in ([1] if SMOKE else [1,4]):
  # All model inputs/embeddings/noise/timestep tensors are allocated outside timing.
  if NAME=='DSINE':
   from torchvision.transforms import Normalize
   x=torch.from_numpy(np.asarray(image,np.float32).transpose(2,0,1).copy()/255)[None].cuda();h,w=x.shape[-2:];pad=captures['utils'].get_padding(h,w);x=Normalize([.485,.456,.406],[.229,.224,.225])(F.pad(x,pad,value=0.)).repeat(batch,1,1,1);intr=captures['intrins_from_fov'](new_fov=60.,H=h,W=w,device='cuda')[None].repeat(batch,1,1);intr[:,0,2]+=pad[0];intr[:,1,2]+=pad[2]
   def forward():return model(x,intrins=intr)[-1][:,:,pad[2]:pad[2]+h,pad[0]:pad[0]+w].float()
   network_hw=list(x.shape[-2:])
  elif NAME=='NiRNE':
   import inspect
   resize_image=inspect.unwrap(root.__class__.__call__).__globals__['resize_image'];resized=resize_image(image,1024);x,padding,resolution=model.image_processor.preprocess(resized,model.default_processing_resolution,'bilinear',model._execution_device,model.dtype);x=x.repeat(batch,1,1,1);embedding=torch.zeros((1,257,1024),device='cuda',dtype=model.dtype);t=torch.tensor(model.t_start,device='cuda');precision='FP16 storedweights andcompute';network_hw=list(x.shape[-2:]);latent_hw=[v//model.vae_scale_factor for v in network_hw]
   assert model.__class__.__name__=='YOSONormalsPipeline',model.__class__.__name__
   def forward():
    latent=model.vae.encode(x).latent_dist.mode()*model.vae.config.scaling_factor;down,mid=model.controlnet(latent,t,encoder_hidden_states=embedding,conditioning_scale=1.,guess_mode=False,return_dict=False);out=model.unet(latent,t,encoder_hidden_states=embedding,down_block_additional_residuals=down,mid_block_additional_residual=mid,return_dict=False)[0];p=model.decode_prediction(out);p=model.image_processor.unpad_image(p,padding);p=model.image_processor.resize_antialias(p,resolution,'bilinear',is_aa=False);return model.normalize_normals(p)
  elif NAME=='MarigoldNormal':
   module=importlib.import_module(model.__class__.__module__);x=module.pil_to_tensor(image)[None];x=module.resize_max_res(x,max_edge_resolution=model.default_processing_resolution,resample_method=module.get_tv_resample_method('bilinear'));x=(x/255.*2-1).to(device='cuda',dtype=model.dtype).repeat(batch,1,1,1);steps=model.default_denoising_steps;model.scheduler.set_timesteps(steps,device='cuda');timesteps=model.scheduler.timesteps;model.encode_empty_text();embedding=model.empty_text_embed.repeat(batch,1,1);latent_hw=[v//8 for v in x.shape[-2:]];noise=torch.randn((batch,4,*latent_hw),device='cuda',dtype=model.dtype,generator=torch.Generator(device='cuda').manual_seed(42));network_hw=list(x.shape[-2:])
   def forward():
    rgb_latent=model.encode_rgb(x);target=noise
    for t in timesteps:pred=model.unet(torch.cat([rgb_latent,target],1),t,encoder_hidden_states=embedding).sample;target=model.scheduler.step(pred,t,target).prev_sample
    p=model.decode_normals(target).clip(-1,1);return p/torch.linalg.vector_norm(p,dim=1,keepdim=True).clamp(min=1e-6)
  else:
   module=importlib.import_module(model.__class__.__module__);raw=torch.from_numpy(np.asarray(image,np.float32).transpose(2,0,1).copy())[None].cuda()/127.5-1;size=raw.shape[-2:];x=module.resize_max_res(raw,max_edge_resolution=model.default_processing_resolution,resample_method=module.get_tv_resample_method('bilinear')).repeat(batch,1,1,1);embedding=None
   with torch.inference_mode(),torch.autocast('cuda'):embedding=model.encode_prompt('',model._execution_device,num_images_per_prompt=1,do_classifier_free_guidance=None)[0].repeat(batch,1,1)
   task=torch.tensor([[1.,0.]],device='cuda');task=torch.cat([task.sin(),task.cos()],-1).repeat(batch,1);t=torch.tensor(999,device='cuda',dtype=torch.long);precision='FP16 autocast FP32storage';network_hw=list(x.shape[-2:]);latent_hw=[v//model.vae_scale_factor for v in network_hw]
   with torch.inference_mode(),torch.autocast('cuda'):distribution=model.vae.encode(x).latent_dist
   posterior_noise=torch.randn(distribution.mean.shape,device='cuda',dtype=distribution.mean.dtype,generator=torch.Generator(device='cuda').manual_seed(42));noise=torch.randn((batch,model.unet.config.in_channels//2,*latent_hw),device='cuda',dtype=embedding.dtype,generator=torch.Generator(device='cuda').manual_seed(42))*model.scheduler.init_noise_sigma;del distribution
   def forward():
    with torch.autocast('cuda'):
     distribution=model.vae.encode(x).latent_dist;rgb_latent=(distribution.mean+distribution.std*posterior_noise)*model.vae.config.scaling_factor;latent=rgb_latent if NAME=='Lotus-D' else torch.cat([rgb_latent,model.scheduler.scale_model_input(noise,t)],dim=1);p=model.unet(latent,t,encoder_hidden_states=embedding,cross_attention_kwargs=None,return_dict=False,class_labels=task)[0];decoded=model.vae.decode(p/model.vae.config.scaling_factor,return_dict=False)[0];out=model.image_processor.postprocess(decoded,output_type='pt',do_denormalize=[True]*batch);out=module.resize_back(out.float(),size,module.get_tv_resample_method('nearest'));return out*2-1
  try:
   if batch==1:
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    with torch.inference_mode():reference=np.asarray(predictor(image,record),np.float32);out=forward().float()[0].permute(1,2,0).cpu().numpy()
    if NAME=='MarigoldNormal':
     # Officialadapter resizes nativeprediction to512; compare samefinalprediction operation outside timing.
     from torchvision.transforms.functional import resize
     with torch.inference_mode():out=resize(torch.from_numpy(out.copy()).permute(2,0,1)[None].cuda(),image.size[::-1],interpolation=module.get_tv_resample_method('bilinear'),antialias=True)[0].permute(1,2,0).cpu().numpy().clip(-1,1)
    assert reference.shape==out.shape,(reference.shape,out.shape);delta=float(np.max(np.abs(reference-out)));parity.append({'batch':1,'max_abs_native_output_difference':delta,'tolerance':5e-4,'reference':'pinned oldofficialadapter, resetseed42; sameprocessinganddefaultsteps','passed':delta<=5e-4});assert delta<=5e-4,('officialforwardparityfailed',NAME,delta)
   row=measure(forward,batch)
  except torch.cuda.OutOfMemoryError as e:row={'status':'out_of_memory','batch':batch,'error':str(e)}
  except RuntimeError as e:
   if batch==1:raise
   row={'status':'official_batch4_not_supported','batch':batch,'error':repr(e),'traceback':traceback.format_exc()}
  rows.append(row);gc.collect();torch.cuda.empty_cache()
 assert_exclusive();result={'status':'passed_parity_smoke' if SMOKE else 'completed','model':NAME,'variant':'native','rows':rows,'model_class':type(model).__module__+'.'+type(model).__name__,'default_steps':steps,'compute_precision':precision,'encoder_input_hw':network_hw,'encoder_tokens':None,'latent_hw':latent_hw,'tokens_note':'no singleViTencoder tokenbudget in thisofficialpath; spatial latentHW reported instead','parameters_unique_loaded':sum(p.numel() for p in parameters.values()),'numerical_parity':parity,'Lotus_precision_correction':'cachedemptytextcomputedunderoriginalFP16autocast; normconversionpromotestoFP32matchingoriginalNumPybridge','pure_forward_scope':'preprocessedGPUresidentnetworkRGB/embedding/noise/tensors; noPNG/CPUconversion/transfer/noisecreation. IncludesVAE/RGBencode/allofficialdefaultdenoisersteps/decode. Fixedpromptembedding cached, analogous modelconstant. Marigold outputresize excluded; othernativegraph outputnormalization retained.','NiRNE_note':'ExactofficialhubNiRNE entry usesYOSONormalsPipeline withNiRNEweights; retains deterministicimage-latent one-step graph, unusedgaus_noiseoutput omitted.' if NAME=='NiRNE' else None,'official_adapter_sha256':sha256(R/'src/external_normal_adapter_v2_20261106.py'),'source_weight_registration':read(R/'provenance/external_comparison_v2_20261106.json')['models'][NAME],'source_weight_registration_sha256':sha256(R/'provenance/external_comparison_v2_20261106.json'),'exclusive':exclusive,'TF32':False,'torch':torch.__version__,'worker_sha256':sha256(__file__),'test_images_scored':0,'updated_at':now()};write_json(P,result);print('T3_EXTERNAL_COMPLETE',NAME,flush=True)
except Exception as e:write_json(P,{'status':'needs_attention','model':NAME,'error':repr(e),'traceback':traceback.format_exc(),'worker_sha256':sha256(__file__),'updated_at':now()});raise
