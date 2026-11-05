"""Shared EXR-owned labels and uniform light sampling; lossless256 RGB cache."""
from collections import OrderedDict
import numpy as np
from PIL import Image
class D1ImageCache:
 def __init__(self,records,capacity=128,rng=None):
  if rng is None:raise ValueError('D1 light sampler must use the checkpointed training RNG')
  self.records=records;self.capacity=capacity;self.rng=rng;self.data=OrderedDict()
 def get(self,index):
  row=self.records[index];light=row['rgb'][int(self.rng.integers(len(row['rgb'])))];key=(row['normal_path'],light['path'])
  if key in self.data:self.data.move_to_end(key);return self.data[key]
  with np.load(row['normal_path']) as z:normal=z['normal'].astype(np.float32);valid=z['valid'].astype(bool)
  rgb=np.asarray(Image.open(light['path']).convert('RGB'),dtype=np.float32)/127.5-1
  assert rgb.shape==(256,256,3) and normal.shape==rgb.shape and valid.shape==(256,256) and valid.any()
  item=(rgb.transpose(2,0,1).copy(),normal.transpose(2,0,1).copy(),valid.copy());self.data[key]=item
  if len(self.data)>self.capacity:self.data.popitem(last=False)
  return item
