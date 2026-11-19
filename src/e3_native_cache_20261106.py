"""Native Arb RGB and label loading for head-only MoGe-L adaptation."""
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
from src.research_io import sha256
ROOT=Path(__file__).resolve().parents[1]
class NativeCache:
 def __init__(self,records,capacity=16):
  self.records=records;self.capacity=capacity;self.cache=OrderedDict();scenes={r['scene'] for r in records};self.folders={}
  for name in ['arb_training_raw_v1','arb_scale_c_raw_fresh_20261026']:
   for p in (ROOT/'data'/name).glob('*/*'):
    if p.is_dir() and p.name in scenes:
     assert p.name not in self.folders;self.folders[p.name]=p
  assert scenes<=self.folders.keys()
 def get(self,index):
  if index not in self.cache:
   row=self.records[index];path=self.folders[row['scene']]/Path(row['image_key']).name;assert sha256(path)==row['source_rgb_sha256'];assert sha256(row['path'])==row['npz_sha256']
   with Image.open(path) as opened:
    rgba=opened.convert('RGBA');assert rgba.size==(512,512);image=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba).convert('RGB');rgb=np.asarray(image,dtype=np.float32).transpose(2,0,1).copy()/255
   with np.load(row['path'],allow_pickle=False) as data:normal=data['normal'].astype(np.float32);valid=data['valid'].astype(bool)
   assert normal.shape==(256,256,3) and valid.any() and np.isfinite(normal[valid]).all();self.cache[index]=(rgb,normal,valid)
   if len(self.cache)>self.capacity:self.cache.popitem(last=False)
  self.cache.move_to_end(index);return self.cache[index]
