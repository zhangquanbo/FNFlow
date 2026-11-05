"""Native512 labels and original RGB on the predeclared D3 asset subset."""
from collections import OrderedDict
import os,numpy as np
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
from PIL import Image
from src.object_normals import decode_arb_normal
class D3NativeCache:
 def __init__(self,records,capacity=64,rng=None):
  if rng is None:raise ValueError('Checkpointed training RNG required')
  self.records=records;self.capacity=capacity;self.rng=rng;self.data=OrderedDict()
 def get(self,index):
  row=self.records[index];light=row['rgb'][int(self.rng.integers(len(row['rgb'])))];key=(row['normal_path'],light['path'])
  if key in self.data:self.data.move_to_end(key);return self.data[key]
  raw=cv2.imread(row['normal_path'],cv2.IMREAD_UNCHANGED);assert raw is not None and raw.shape==(512,512,3)
  normal,valid=decode_arb_normal(raw,output_wh=(512,512));assert valid.any()
  rgba=np.asarray(Image.open(light['path']).convert('RGBA'),dtype=np.float32)/255;assert rgba.shape==(512,512,4)
  rgb=(rgba[...,:3]*rgba[...,3:4]+1-rgba[...,3:4])*2-1
  item=(rgb.transpose(2,0,1).copy(),normal.transpose(2,0,1).copy(),valid.copy());self.data[key]=item
  if len(self.data)>self.capacity:self.data.popitem(last=False)
  return item
