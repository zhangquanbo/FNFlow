"""Pinned official inference adapters. Coordinate transforms live in the round registration."""
from pathlib import Path
import gc
import json
import os
import sys
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'external_baselines_20261030'

def load(name, scratch):
    family = 'Lotus' if name.startswith('Lotus-') else name
    folder = BASE / family
    source = folder / 'source'
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(source))
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    if family == 'DSINE':
        import geffnet
        original = geffnet.create_model
        def initializer(*args, **kwargs):
            kwargs['pretrained'] = False
            return original(*args, **kwargs)
        geffnet.create_model = initializer
        os.chdir(source / 'projects/dsine')
        import projects
        projects.EXPERIMENT_DIR = str(scratch / 'experiment')
        projects.DATASET_DIR = str(scratch / 'unused')
        import projects.dsine.config as config
        import utils.utils as utils
        from utils.projection import intrins_from_fov
        from models.dsine.v02 import DSINE_v02
        from torchvision.transforms import Normalize
        import torch.nn.functional as F
        old_argv = sys.argv
        sys.argv = ['test_minimal.py', str(source / 'projects/dsine/experiments/exp001_cvpr2024/dsine.txt')]
        args = config.get_args(test=True)
        sys.argv = old_argv
        model = utils.load_checkpoint(str(folder / 'weights/DSINE/dsine.pt'), DSINE_v02(args).cuda()).eval()
        def predict(image, record=None):
            x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255)[None].cuda()
            h, w = x.shape[-2:]
            pad = utils.get_padding(h, w)
            x = Normalize([.485, .456, .406], [.229, .224, .225])(F.pad(x, pad, value=0.))
            intr = intrins_from_fov(new_fov=60., H=h, W=w, device='cuda')[None]
            intr[:, 0, 2] += pad[0]
            intr[:, 1, 2] += pad[2]
            return model(x, intrins=intr)[-1][0, :, pad[2]:pad[2]+h, pad[0]:pad[0]+w].float().permute(1, 2, 0).cpu().numpy()
    elif family == 'MoGe2':
        from moge.model.v2 import MoGeModel
        model = MoGeModel.from_pretrained(folder / 'weights/moge-2-vitl-normal/model.pt').cuda().eval()
        def predict(image, record=None):
            x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy() / 255).cuda()
            return model.infer(x, resolution_level=9, use_fp16=True, apply_mask=False)['normal'].float().cpu().numpy()
    elif family == 'Lotus':
        from pipeline import LotusDPipeline, LotusGPipeline
        variant = 'd' if name == 'Lotus-D' else 'g'
        klass = LotusDPipeline if variant == 'd' else LotusGPipeline
        model = klass.from_pretrained(folder / 'weights' / ('lotus-normal-'+variant+'-v1-1'), torch_dtype=torch.float32).to('cuda')
        model.set_progress_bar_config(disable=True)
        def predict(image, record=None):
            x = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy())[None].cuda() / 127.5 - 1
            embedding = torch.tensor([[1., 0.]], device='cuda')
            embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], -1)
            with torch.autocast('cuda'):
                value = model(rgb_in=x, prompt='', num_inference_steps=1,
                    generator=torch.Generator(device='cuda').manual_seed(torch.initial_seed()),
                    output_type='np', timesteps=[999], task_emb=embedding, processing_res=None,
                    match_input_res=True, resample_method='bilinear').images[0]
            return value * 2 - 1
    elif family == 'MarigoldNormal':
        from marigold import MarigoldNormalsPipeline
        model = MarigoldNormalsPipeline.from_pretrained(folder / 'weights/marigold-normals-v1-1', torch_dtype=torch.float32).to('cuda')
        def predict(image, record=None):
            value = model(image, generator=torch.Generator(device='cuda').manual_seed(torch.initial_seed()), show_progress_bar=False).normals_np
            assert value.ndim == 3 and value.shape[0] == 3
            return np.moveaxis(value, 0, -1)
    elif family in ('StableNormal', 'NiRNE'):
        if family == 'StableNormal':
            dependency = json.loads((folder / 'DINO_L_dependency.json').read_text())
            dino = json.loads((ROOT / 'configs/dinov2_s_compatibility.json').read_text())
            original = torch.hub.load
            def pinned(repo, architecture, *args, **kwargs):
                if repo == 'facebookresearch/dinov2':
                    assert architecture == 'dinov2_vitl14'
                    encoder = original(str(ROOT / dino['source_directory']), architecture, source='local', pretrained=False)
                    encoder.load_state_dict(torch.load(dependency['weights'], map_location='cpu', weights_only=False), strict=True)
                    return encoder
                return original(repo, architecture, *args, **kwargs)
            torch.hub.load = pinned
        model = torch.hub.load(str(source), family, source='local', local_cache_dir=str(folder / 'weights'), device='cuda:0')
        class Capture:
            def __init__(self, inner):
                self.inner = inner
                self.value = None
            def __getattr__(self, key):
                return getattr(self.inner, key)
            def __call__(self, *args, **kwargs):
                value = self.inner(*args, **kwargs)
                self.value = np.asarray(value.prediction[0], dtype=np.float32)
                return value
        capture = Capture(model.model)
        model.model = capture
        def predict(image, record=None):
            model(image, data_type='indoor')
            return capture.value.copy()
    else:
        raise ValueError(name)
    return predict
