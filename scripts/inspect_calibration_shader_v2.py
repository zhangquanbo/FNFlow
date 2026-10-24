from PIL import Image
import numpy as np
from pathlib import Path
r=(Path(__file__).resolve().parents[1] / 'data/lightrig_shader_check_v2')
for n in ['sphere','cube']:
 im=np.array(Image.open(r/(n+'_source_shader.png')).convert('RGBA'));z=np.load(r/(n+'.npz'))
 print(n,'center',im[256,256].tolist(),'analytic',z['normal_gl'][256,256].tolist(),'alpha',im[...,3].min(),im[...,3].max())