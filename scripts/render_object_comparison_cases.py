"""Fixed pre-score cases, shared GT masks and 0..60 degree error scale."""
import sys,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from PIL import Image,ImageDraw
from src.mesh_normal_gt import gt_angular_gap
from src.research_io import write_json,sha256
from scripts import evaluate_object_comparison as ev
OUT=ev.OUT;MAN=ev.MAN;folder=OUT/'failure_cases_private';folder.mkdir(exist_ok=True)
palette=np.array([[0,0,4],[87,16,110],[188,55,84],[249,142,9],[252,255,164]],float)
def heat(values):
 x=np.clip(values/60,0,1)*4;idx=np.minimum(x.astype(int),3);a=x-idx
 return np.rint(palette[idx]*(1-a[...,None])+palette[idx+1]*a[...,None]).astype(np.uint8)
def panel(array):
 im=Image.fromarray(array);im.thumbnail((256,256));bg=Image.new('RGB',(256,256),'white');bg.paste(im,((256-im.width)//2,(256-im.height)//2));return bg
images=[];missing=[];receipt=[]
for r in [r for r in MAN['records'] if r['id'] in MAN['case_ids']]:
 rows={m:ev.cached(m,511 if m=='NeuralLightRig' else 42,r) for m in ['NeuralLightRig','R','V','Xn']}
 if any(v is None or 'prediction_path' not in v for v in rows.values()):missing.append(r['id']);continue
 with np.load(r['gt_path']) as z:gt=z['normal'].copy();valid=z['valid'].astype(bool)
 gt/=np.maximum(np.linalg.norm(gt,axis=-1,keepdims=True),1e-12);preds={}
 for m,row in rows.items():
  path=Path(row['prediction_path']);assert sha256(path)==row['prediction_sha256']
  if m=='NeuralLightRig':p=np.asarray(Image.open(path)).astype(np.float32)/127.5-1;p=p@np.asarray(ev.COORD['nlr_to_common']).T
  else:
   with np.load(path) as z:p=z['normal'].copy()
  _,preds[m]=ev.metrics(p,r)
 canvas=Image.new('RGB',(1536,592),'white');d=ImageDraw.Draw(canvas)
 d.text((8,4),r['domain']+' / '+r['object']+' / '+r['image_key'],fill='black')
 names=['RGB','GT smooth' if r['domain']=='gso' else 'GT','Neural LightRig seed511','R 52.5k','V 52.5k seed42','Xn 52.5k seed42']
 for col,name in enumerate(names):d.text((col*256+8,24),name,fill='black')
 rgb=np.array(Image.open(r['ours_rgb']).convert('RGB'));canvas.paste(panel(rgb),(0,48));encoded=np.where(valid[...,None],np.rint((gt+1)*127.5).clip(0,255),0).astype(np.uint8);canvas.paste(panel(encoded),(256,48))
 for col,m in enumerate(['NeuralLightRig','R','V','Xn'],2):
  p=preds[m];enc=np.where(valid[...,None],np.rint((p+1)*127.5).clip(0,255),0).astype(np.uint8);canvas.paste(panel(enc),(col*256,48));err=np.zeros(valid.shape);err[valid]=gt_angular_gap(p,gt,valid);colored=np.where(valid[...,None],heat(err),0).astype(np.uint8);canvas.paste(panel(colored),(col*256,320));d.text((col*256+8,307),f'Mean {rows[m]["metrics"]["main"]["mean"]:.2f} deg',fill='black')
 d.text((8,330),'Fixed pre-score selection.',fill='black');d.text((8,348),'Colors: common coordinate frame.',fill='black');d.text((8,366),'Black: GT-invalid region.',fill='black');d.text((264,330),'Angular error (degrees)',fill='black')
 bar=heat(np.tile(np.linspace(0,60,230),(18,1)));canvas.paste(Image.fromarray(bar),(264,360));d.text((264,382),'0                 30                 60',fill='black')
 dest=folder/(r['id']+'.png');canvas.save(dest);images.append(canvas);receipt.append({'id':r['id'],'object':r['object'],'path':str(dest),'sha256':sha256(dest)})
if images:
 atlas=Image.new('RGB',(1536,592*min(3,len(images))),'white')
 for i,im in enumerate(images[:3]):atlas.paste(im,(0,i*592))
 atlas.save(folder/'preview_first3.png')
vase=OUT/'official_vase_no_gt';original=ROOT/'outputs/lightrig_official_examples_20261023/outputs/clay_vase_pred.png'
if all((vase/(m+'.npz')).exists() for m in ['R','V','Xn']) and original.exists():
 canvas=Image.new('RGB',(1280,320),'white');d=ImageDraw.Draw(canvas);d.text((8,4),'Official clay vase: qualitative only; no public GT or angular error map',fill='black')
 inputpath=ROOT/'external/pinned_source/Neural-LightRig-db472e191e7abd3115ced0d5c72d61b75a8412ed/assets/examples/clay_vase.png';rgba=Image.open(inputpath).convert('RGBA');rgb=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB');canvas.paste(panel(np.array(rgb)),(0,48));d.text((8,27),'RGB',fill='black')
 with Image.open(original) as im:p=np.array(im.crop((2048,0,2560,512)).convert('RGB')).astype(np.float32)/127.5-1
 p=p@np.asarray(ev.COORD['nlr_to_common']).T;p/=np.maximum(np.linalg.norm(p,axis=-1,keepdims=True),1e-8);canvas.paste(panel(np.rint((p+1)*127.5).clip(0,255).astype(np.uint8)),(256,48));d.text((264,27),'Neural LightRig',fill='black')
 for col,m in enumerate(['R','V','Xn'],2):
  with np.load(vase/(m+'.npz')) as z:p=z['normal'].copy()
  canvas.paste(panel(np.rint((p+1)*127.5).clip(0,255).astype(np.uint8)),(col*256,48));d.text((col*256+8,27),m+' 52.5k',fill='black')
 canvas.save(vase/'comparison_no_gt.png')
write_json(folder/'receipt_private.json',{'manifest_sha256':ev.MSHA,'case_selection':MAN['case_selection'],'expected':len(MAN['case_ids']),'ready':len(receipt),'missing':missing,'panels':receipt,'scale_degrees':[0,60],'complete':not missing})
print('case panels',len(receipt),'/',len(MAN['case_ids']),flush=True)
