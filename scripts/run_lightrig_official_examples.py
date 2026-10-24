"""Official defaults with documented local paths for retired upstream model IDs."""
import os,sys,json,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TORCH_COMPILE_DISABLE']='1'
sys.dont_write_bytecode=True
source=ROOT/'external/pinned_source/Neural-LightRig-db472e191e7abd3115ced0d5c72d61b75a8412ed'
runtime=ROOT/'external/lightrig_offline_runtime'
# Exact defaults and inference computations retained; only remote location strings change.
shutil.copytree(source,runtime,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','results','ckpt'))
paths={name:str(ROOT/'external/base_models'/mirror.replace('/','__')) for name,mirror in [('stabilityai/stable-diffusion-2-1','sd2-community/stable-diffusion-2-1'),('stabilityai/stable-diffusion-2-1-unclip','sd2-community/stable-diffusion-2-1-unclip'),('sudo-ai/zero123plus-v1.2','sudo-ai/zero123plus-v1.2')]}
changes=[]
for relative in ['mld/pipeline.py','mld/configs/infer.yaml']:
 p=runtime/relative;s=p.read_text()
 for old,new in sorted(paths.items(),key=lambda kv:-len(kv[0])):s=s.replace('"'+old+'"','"'+new+'"').replace("'"+old+"'","'"+new+"'")
 p.write_text(s);changes.append({'path':relative,'original_sha256':hashlib.sha256((source/relative).read_bytes()).hexdigest(),'runtime_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
receipt=ROOT/'provenance/lightrig_offline_adapter.json';receipt.write_text(json.dumps({'paths':paths,'changes':changes,'note':'Mirrors of retired upstream URLs, fixed revisions/hashes in lightrig_base_models.json. No numerical inference changes.'},indent=2))
os.chdir(runtime);sys.path.insert(0,str(runtime))
import inference
inference.snapshot_download=lambda **kwargs:str(ROOT/'external/lightrig_weights_5619cfe')
sys.argv=['inference.py','--img_dir',str(source/'assets/examples'),'--save_dir',str(ROOT/'outputs/lightrig_official_examples_20261023')]
sys.path.insert(0,str(ROOT))
from src.gpu_job_lock import gpu_job_lock
with gpu_job_lock():inference.main()