"""Private fixed-example R/V preview from immutable, equally budgeted WSD forks.

CPU float32 and CPU-generated image-specific noise are used to leave the training
GPU alone. This is a visual diagnostic, not a replacement for CUDA/bfloat16
evaluation or an aggregate method ranking. No checkpoint or manifest is edited.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.model_unet import FaceNormalUNet_B
from src.noise_protocol import image_noise_seed
from src.parameterization import one_step_endpoint
from src.research_io import ROOT, output_path, write_json, now, sha256


def main():
    torch.set_num_threads(2)
    cfg = json.loads((ROOT / 'configs/parameterization_first_round_v2_review2.json').read_text())
    folders = {'hypersim': ROOT / cfg['hypersim_root'],
               'arb': ROOT / 'data/arb_mixed_formal_v1',
               'nyuv2': ROOT / cfg['nyu_validation_root']}
    chosen = []
    for domain, folder in folders.items():
        records = json.loads((folder / 'private_manifest.json').read_text())['records']
        records = sorted((r for r in records if r['split'] == 'val'),
                         key=lambda r: hashlib.sha256((domain + '/' + Path(r['path']).name).encode()).digest())
        groups = set()
        for record in records:
            if record['scene'] in groups:
                continue
            groups.add(record['scene'])
            with np.load(record['path']) as data:
                sample = {name: data[name].copy() for name in ['rgb', 'normal', 'valid']}
            sample.update(domain=domain, example=len(groups), record=record)
            if not sample['valid'].any():
                raise RuntimeError('Selected validation example has empty mask')
            chosen.append(sample)
            if len(groups) == 2:
                break
        if len(groups) != 2:
            raise RuntimeError('Need two distinct validation groups per domain')

    root = ROOT / cfg['output_root']
    out = output_path(ROOT / 'figures/qualitative_wsd_10k_v1')
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = {}
    for method in ['R', 'V']:
        path = root / f'stable_{method}_s1247/fork_010000_local.pth'
        checkpoints[method] = {'path': str(path), 'sha256': sha256(path)}
    receipt = {'created_at': now(), 'selection': 'fixed SHA256 order; first two distinct validation groups per domain; no error-based selection',
               'scope': 'private visual diagnostic only; not aggregate ranking or official evaluation',
               'device': 'cpu', 'precision': 'float32', 'noise_generator_device': 'cpu',
               'noise_note': 'CPU RNG fields differ from CUDA evaluation fields; image-specific seed derivation is unchanged',
               'inference_base_seed': 42, 'training_seed': 1247, 'updates_each': 10000,
               'schedule': 'stable WSD forks, no cooldown', 'cfg': False, 'ray': False, 'dino': False,
               'checkpoints': checkpoints, 'script_sha256': sha256(__file__),
               'manifests_sha256': {name: sha256(folder / 'private_manifest.json') for name, folder in folders.items()},
               'records': [{'domain': s['domain'], **s['record']} for s in chosen]}
    write_json(out / 'selection_private.json', receipt)
    model = FaceNormalUNet_B().eval()
    predictions = {}
    with torch.inference_mode():
        for method, checkpoint in checkpoints.items():
            state = torch.load(checkpoint['path'], map_location='cpu')
            if state['step'] != 10000 or state['signature']['method'] != method or state['signature']['phase'] != 'stable':
                raise RuntimeError('Wrong immutable checkpoint')
            if state['signature']['condition_dropout'] != 0:
                raise RuntimeError('Preview expects p=0')
            model.load_state_dict(state['model'], strict=True)
            del state
            panels = []
            for sample in chosen:
                rgb = torch.from_numpy(sample['rgb'].transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1
                key = sample['record'].get('image_key', sample['domain'] + '/' + Path(sample['record']['path']).name)
                generator = torch.Generator(device='cpu').manual_seed(image_noise_seed(42, key))
                noise = torch.randn(rgb.shape, generator=generator) if method == 'V' else torch.zeros_like(rgb)
                raw = model(noise, torch.zeros(1), rgb)
                endpoint = one_step_endpoint(method, raw, noise)
                pred = F.normalize(endpoint, dim=1)[0].permute(1, 2, 0).numpy().copy()
                valid = sample['valid'].astype(bool)
                if not np.isfinite(pred[valid]).all() or (np.linalg.norm(pred[valid], axis=-1) < .99).any():
                    raise RuntimeError('Invalid predicted unit direction')
                panels.append(pred)
                print(method, sample['domain'], sample['example'], flush=True)
            predictions[method] = panels
            if sha256(checkpoint['path']) != checkpoint['sha256']:
                raise RuntimeError('Immutable checkpoint changed while previewing')

    fig, axes = plt.subplots(len(chosen), 6, figsize=(14, 14), dpi=120)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')
    for i, sample in enumerate(chosen):
        valid = sample['valid'].astype(bool)
        gt = sample['normal']
        gt = gt / np.maximum(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8)
        normal_panel = lambda n: np.where(valid[..., None], np.clip((n + 1) / 2, 0, 1), 0)
        images = [sample['rgb'], normal_panel(gt), normal_panel(predictions['R'][i]), normal_panel(predictions['V'][i])]
        for j in range(6):
            ax = axes[i, j]
            if j < 4:
                ax.imshow(images[j])
            else:
                pred = predictions[['R', 'V'][j - 4]][i]
                error = np.degrees(np.arccos(np.clip((pred * gt).sum(-1), -1, 1)))
                ax.imshow(np.ma.array(error, mask=~valid), cmap=cmap, vmin=0, vmax=60)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if i == 0:
                ax.set_title(['RGB', 'Label', 'R: 10k stable', 'V: 10k stable', 'R error (0-60 deg)', 'V error (0-60 deg)'][j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sample['domain']}\nexample {sample['example']}", fontsize=10)
    fig.suptitle('CURRENT WSD PREVIEW: equal 10k stable forks | training seed 1247 | p=0\n'
                 'CPU float32, one step, image-specific CPU noise seed 42; visual diagnostic only. Black = invalid GT.', fontsize=11)
    fig.tight_layout(rect=[0, .025, 1, .95], pad=.6)
    bar_axis = fig.add_axes([.72, .015, .23, .008])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap=cmap), cax=bar_axis,
                 orientation='horizontal', ticks=[0, 15, 30, 45, 60])
    target = out / 'R_V_stable_10k_comparison.png'
    fig.savefig(target)
    plt.close(fig)
    write_json(out / 'completion_private.json', {'status': 'completed', 'created_at': now(), 'image': str(target),
               'selection_receipt_sha256': sha256(out / 'selection_private.json'), 'image_sha256': sha256(target)})
    print(target, flush=True)


if __name__ == '__main__':
    main()
