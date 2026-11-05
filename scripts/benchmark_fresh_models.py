"""Native configured single-image end-to-end inference on the same RTX3090."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import os,json,time,gc,random
import numpy as np
import torch
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
from scripts.train_object_dino_retrain import make_model
from scripts.evaluate_fresh_ab import OUT,read,assert_registration,prepare
from src.lightrig_adapter import load_models
def main():
    reg=assert_registration();assert '3090' in torch.cuda.get_device_name()
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    record=min([r for r in prepare('primary')['records'] if r['elevation_bin']==30],key=lambda r:r['id'])
    results=[];temp=OUT/'benchmark_private';temp.mkdir(exist_ok=True)
    for name in ['A','B','NeuralLightRig']:
        count=[0]
        def hook(*_):count[0]+=1
        if name in ['A','B']:
            ck=torch.load(reg['checkpoints'][name]['path'],map_location='cpu',weights_only=False)
            model=make_model().eval();state=model.state_dict();state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del state,ck
            handle=model.unet.register_forward_hook(hook);parameters=list(model.parameters())
            def call():
                rgb=np.asarray(Image.open(record['white']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
                noise=torch.zeros_like(rgb) if name=='B' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(42,record['image_key'])))
                with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
                normal=torch.nn.functional.normalize(raw.float(),dim=1)[0].permute(1,2,0).cpu().numpy()
                Image.fromarray(np.rint((normal+1)*127.5).clip(0,255).astype(np.uint8)).save(temp/'output.png')
        else:
            random.seed(511);np.random.seed(511);torch.manual_seed(511)
            module,m1,m2=load_models();handle=m1.unet.unet.register_forward_hook(hook)
            components=[m1,m2,m1.pipeline.vae,m1.pipeline.text_encoder,m1.pipeline.vision_encoder]
            parameters=list({id(p):p for m in components for p in m.parameters()}.values())
            def call():
                module.inference_impl(m1,m2,record['rgba'],str(temp/'concat.png'))
                with Image.open(temp/'concat.png') as image:normal=np.asarray(image.crop((2048,0,2560,512)).convert('RGB')).astype(np.float32)/127.5-1
                normal*=np.array([-1,1,1],np.float32)
                normal=torch.nn.functional.interpolate(torch.from_numpy(normal.copy()).permute(2,0,1)[None],size=(256,256),mode='bilinear',align_corners=False)
                normal=torch.nn.functional.normalize(normal,dim=1)[0].permute(1,2,0).numpy()
                Image.fromarray(np.rint((normal+1)*127.5).clip(0,255).astype(np.uint8)).save(temp/'output.png')
        with torch.inference_mode():
            for _ in range(5):call()
            torch.cuda.synchronize();count[0]=0;torch.cuda.reset_peak_memory_stats();times=[]
            for _ in range(30):
                torch.cuda.synchronize();start=time.perf_counter();call();torch.cuda.synchronize();times.append(time.perf_counter()-start)
        results.append({'model':name,'input_size':512 if name=='NeuralLightRig' else 256,'warmup':5,'timed':30,
          'median_seconds':float(np.median(times)),'p90_seconds':float(np.quantile(times,.9)),
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'parameters_total_unique':sum(p.numel() for p in parameters),'parameters_requires_grad':sum(p.numel() for p in parameters if p.requires_grad),
          'UNet_forward_calls_per_inference':count[0]/30,'timings_seconds':times})
        write_json(OUT/'efficiency.json',{'status':'completed' if len(results)==3 else 'partial','rows':results,'GPU':torch.cuda.get_device_name(),
          'torch':torch.__version__,'cuda':torch.version.cuda,'created_at':now(),'input_id':record['id'],
          'scope':'end-to-end: input file reading, configured preprocessing, official model computation, common256 normal conversion, output PNG writing; official NLR additionally writes its five-column composite. Native256 versus512 inputs disclosed; model loading excluded.'})
        handle.remove();del parameters
        if name in ['A','B']:del model
        else:del module,m1,m2,components
        gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'efficiency_failure.json',{'error':repr(e),'updated_at':now()});raise
