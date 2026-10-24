import json,hashlib,platform
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'outputs/lightrig_official_examples_20261023';paths=sorted((out/'outputs').glob('*.png'))
assert len(paths)==6,len(paths)
canvas=Image.new('RGB',(1024,6*280+40),'white');draw=ImageDraw.Draw(canvas)
for x,t in [(0,'Official input'),(256,'Generated 9 lights'),(512,'Albedo'),(768,'Normal')]:draw.text((x+8,10),t,fill='black')
rows=[]
for i,p in enumerate(paths):
 im=Image.open(p);assert im.size==(2560,512),(p,im.size)
 for j,col in enumerate([0,1,3,4]):
  panel=im.crop((col*512,0,(col+1)*512,512)).convert('RGBA');bg=Image.new('RGBA',panel.size,'white');panel=Image.alpha_composite(bg,panel).convert('RGB');panel.thumbnail((256,256))
  canvas.paste(panel,(j*256,40+i*280));draw.text((8,40+i*280+258),p.stem,fill='black')
 n=np.array(Image.open(out/'normal'/p.name))
 assert n.shape==(512,512,3)
 rows.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'normal_min':int(n.min()),'normal_max':int(n.max())})
preview=ROOT/'figures/lightrig_official_examples_contact.png';canvas.save(preview)
receipt={'status':'six_examples_generated_pending_contact_visual_review','defaults':{'resolution':512,'steps':75,'cfg':2.0,'cfg_rescale':0.7,'seed':511},'source_commit':'db472e191e7abd3115ced0d5c72d61b75a8412ed','weights_revision':'5619cfec5e623ded0701d0b05f26ad5bbf9f0401','model_scores_computed':False,'rows':rows,'preview':str(preview),'preview_sha256':hashlib.sha256(preview.read_bytes()).hexdigest()}
(out/'verification_private.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'images':len(rows),'preview':str(preview)}))