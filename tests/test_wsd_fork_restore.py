"""A real two-update GPU integration fixture checks parent/RNG isolation."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn


@pytest.mark.skipif(not torch.cuda.is_available(),reason='WSD runner requires CUDA')
def test_cooldown_restores_optimizer_rng_and_never_changes_parent(tmp_path,monkeypatch):
    import scripts.train_wsd_review2 as runner
    class Tiny(nn.Module):
        def __init__(self):super().__init__();self.layer=nn.Conv2d(6,3,1)
        def forward(self,z,t,rgb):return self.layer(torch.cat([z,rgb],1))
    root=tmp_path;out=root/'outputs';(root/'configs').mkdir();out.mkdir()
    rule={'initial_stable_endpoint':30000,'extended_stable_endpoint':50000,'initial_fork_steps':[10000,20000,30000],
          'warmup_updates':100,'cooldown_updates':2,'stable_evaluation_interval':2500}
    (root/'configs/rule.json').write_text(json.dumps(rule))
    cfg={'output_root':'outputs','decision_rule':'configs/rule.json','ray_condition':False,'dino_condition':False,'cfg':False,
        'formal_budget_approved':False,'hypersim_root':'data/hyp','nyu_validation_root':'data/nyu','learning_rate':.0002,
        'weight_decay':.01,'microbatch':2,'gradient_accumulation':1,'gradient_clip_norm':1.,'time_delta':.05,
        'unit_weight':.25,'cosine_weight':.25,'time_bin_edges':[0,.5,.95],'curve_inference_seeds':[42,43]}
    config=root/'configs/test.json';config.write_text(json.dumps(cfg));cfg_hash=hashlib.sha256(config.read_bytes()).hexdigest()
    for folder in ['data/hyp','data/nyu','data/arb_mixed_formal_v1','scripts','src']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    for folder in ['data/hyp','data/nyu','data/arb_mixed_formal_v1']:
        (root/folder/'private_manifest.json').write_text('{}');(root/folder/'summary.json').write_text('{"spatial_protocol":"center_square_v2"}')
    (root/'src/models').mkdir()
    for name in ['scripts/train_parameterization.py','src/wsd_protocol.py','src/parameterization.py','src/models/model_unet.py','src/training_diagnostics.py']:(root/name).write_text('# fixture hash')
    (out/'training_probe_private.json').write_text('{}')
    normal=np.zeros((4,4,3),np.float32);normal[...,2]=1
    sample=root/'sample.npz';np.savez(sample,rgb=np.zeros((4,4,3),np.uint8),normal=normal,valid=np.ones((4,4),bool))
    records=[{'path':str(sample),'split':'train','scene':'fixture'}]
    monkeypatch.setattr(runner,'ROOT',root);monkeypatch.setattr(runner,'FaceNormalUNet_B',Tiny)
    monkeypatch.setattr('src.research_io.ROOT',root)  # Keep the production output guard active inside this fixture.
    monkeypatch.setattr(runner,'load_records',lambda *a:records)
    monkeypatch.setattr(runner,'prepare_probe',lambda cfg:{'hypersim':records,'arb':records})
    monkeypatch.setattr(runner,'evaluate',lambda *a:({'hypersim':{'mean_mae_deg':1.},'arb':{'mean_mae_deg':1.}},{'fixture':[]}))
    torch.manual_seed(4);model=Tiny().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    model(torch.zeros(1,3,4,4,device='cuda'),None,torch.ones(1,3,4,4,device='cuda')).square().mean().backward();optimizer.step()
    parent_dir=out/'stable_Xn_s1247';parent_dir.mkdir();parent=parent_dir/'fork_010000_local.pth'
    generator=torch.Generator(device='cuda').manual_seed(8247)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':10000,'generator':generator.get_state(),
        'numpy_rng':np.random.default_rng(10247).bit_generator.state,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'grad_norms':[],'time_bin_rows':[],'curve':[],
        'signature':{'config_sha256':cfg_hash,'method':'Xn'}}
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    state['signature'].update(rule_sha256=digest(root/'configs/rule.json'),code_sha256=digest(Path(runner.__file__)),
        shared_evaluator_sha256=digest(root/'scripts/train_parameterization.py'),schedule_sha256=digest(root/'src/wsd_protocol.py'),
        target_sha256=digest(root/'src/parameterization.py'),
        model_sha256=digest(root/'src/models/model_unet.py'),diagnostics_sha256=digest(root/'src/training_diagnostics.py'),
        manifests_sha256=[digest(root/p/'private_manifest.json') for p in ['data/hyp','data/arb_mixed_formal_v1','data/nyu']],
        training_probe_sha256=digest(out/'training_probe_private.json'))
    torch.save(state,parent);before=parent.read_bytes()
    monkeypatch.setattr('sys.argv',['fixture','--method','Xn','--phase','cooldown','--fork-step','10000','--config','configs/test.json'])
    runner.main();folder=out/'cooldown_Xn_k10000_s1247';first=torch.load(folder/'last_local.pth',map_location='cpu')
    folder.rename(out/'first_fixture_result')
    runner.main();second=torch.load(folder/'last_local.pth',map_location='cpu')
    assert parent.read_bytes()==before and first['step']==second['step']==10002
    assert first['numpy_rng']==second['numpy_rng']
    assert torch.equal(first['generator'],second['generator'])
    for key in first['model']:torch.testing.assert_close(first['model'][key],second['model'][key],rtol=0,atol=0)
