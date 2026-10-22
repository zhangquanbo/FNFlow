"""Pinned real DINOv2-S weights: forward-only acceptance, no training/evaluation."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,hashlib
import numpy as np
import torch
from src.dino_conditioning import load_local_dino,DINOConditionedUNet,apply_rgb_view
from src.models.model_unet import FaceNormalUNet_B
from src.research_io import ROOT,sha256,write_json,now


def main():
    cfg=json.loads((ROOT/'configs/dinov2_s_compatibility.json').read_text())
    assert sha256(ROOT/cfg['weights'])==cfg['weights_sha256']
    assert sha256(ROOT/'runtime/dinov2/source.zip')==cfg['source_archive_sha256']
    repo=ROOT/cfg['source_directory'];license_text=(repo/'LICENSE').read_text()
    if 'Apache License' not in license_text:raise RuntimeError('Pinned source license mismatch')
    torch.set_num_threads(4);torch.manual_seed(123);device='cuda'
    encoder=load_local_dino(repo,ROOT/cfg['weights'],cfg['architecture'])
    model=DINOConditionedUNet(FaceNormalUNet_B(),encoder,384,14,pad_to_patch=True).to(device).train()
    records=json.loads((ROOT/'data/hypersim_trainval_center_v2/private_manifest.json').read_text())['records']
    row=sorted((r for r in records if r['split']=='train'),key=lambda r:hashlib.sha256(Path(r['path']).name.encode()).hexdigest())[0]
    with np.load(row['path']) as f:rgb=torch.from_numpy(f['rgb'].transpose(2,0,1).copy()).float()[None].to(device)/127.5-1
    tests=[]
    with torch.no_grad():
        for name,view in [('identity',rgb),('flip',apply_rgb_view(rgb,horizontal_flip=True)),
                          ('crop_resize_flip',apply_rgb_view(rgb,(16,16,224,224),True,(256,256)))]:
            feature=model.features(view)
            assert feature.shape==(1,384,19,19) and torch.isfinite(feature).all()
            output=model(torch.zeros_like(view),torch.zeros(1,device=device),view)
            assert output.shape==(1,3,256,256) and torch.isfinite(output).all()
            tests.append({'view':name,'patch_shape':list(feature.shape),'output_shape':list(output.shape),'finite':True})
    assert not model.features.encoder.training and all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    result={'status':'completed','created_at':now(),'architecture':cfg['architecture'],'source_commit':cfg['source_commit'],
        'weights_sha256':cfg['weights_sha256'],'source_archive_sha256':cfg['source_archive_sha256'],
        'license_file_sha256':sha256(repo/'LICENSE'),'license':'Apache-2.0 at pinned upstream source/README',
        'torch':torch.__version__,'encoder_frozen':True,'encoder_eval_after_parent_train':True,
        'padding':cfg['padding'],'conditioning_code_sha256':sha256(ROOT/'src/dino_conditioning.py'),
        'compatibility_config_sha256':sha256(ROOT/'configs/dinov2_s_compatibility.json'),
        'views':tests,'optimizer_updates':0,'benchmark_metrics_computed':False,
        'source_parameters':sum(p.numel() for p in encoder.parameters()),'peak_cuda_bytes':torch.cuda.max_memory_allocated()}
    write_json(ROOT/'outputs/P1_dinov2_symmetric_forward_v2/summary.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
