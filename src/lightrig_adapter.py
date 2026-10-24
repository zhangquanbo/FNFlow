"""Offline official inference adapter; preserves frozen default computations."""
import sys,os,importlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load_models():
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCH_COMPILE_DISABLE']='1'
 runtime=ROOT/'external/lightrig_offline_runtime';sys.path.insert(0,str(runtime));os.chdir(runtime)
 module=importlib.import_module('inference')
 a=module.prepare_stage1_mld(str(runtime/'mld/configs/infer.yaml'),str(ROOT/'external/lightrig_weights_5619cfe/mld.pt'))
 b=module.prepare_stage2_recon(str(ROOT/'external/lightrig_weights_5619cfe/recon'))
 return module,a,b
