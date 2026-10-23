"""Private latest equal-budget WSD previews on the exact earlier six examples.

CPU float32 / CPU image-specific RNG, identical to the earlier visual preview;
official evaluation remains CUDA bfloat16. Never used for ranking or B selection.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    prior_path = ROOT / 'figures/qualitative_wsd_10k_v1/selection_private.json'
    prior = json.loads(prior_path.read_text())
    domains = {'hypersim': ROOT / cfg['hypersim_root'],
               'arb': ROOT / 'data/arb_mixed_formal_v1',
               'nyuv2': ROOT / cfg['nyu_validation_root']}
    for name, folder in domains.items():
        if sha256(folder / 'private_manifest.json') != prior['manifests_sha256'][name]:
            raise RuntimeError('Validation manifest differs from the earlier preview')
    chosen = []
    group_counts = {key: 0 for key in domains}
    for record in prior['records']:
        with np.load(record['path']) as data:
            sample = {key: data[key].copy() for key in ['rgb', 'normal', 'valid']}
        domain = record['domain']
        group_counts[domain] += 1
        sample.update(record=record, domain=domain, example=group_counts[domain])
        if not sample['valid'].any():
            raise RuntimeError('Empty selected validation mask')
        chosen.append(sample)
    if len(chosen) != 6 or set(group_counts.values()) != {2}:
        raise RuntimeError('Unexpected fixed preview selection')

    created = now()
    tag = created[:19].replace('-', '').replace(':', '').replace('T', '_')
    out = output_path(ROOT / 'figures' / ('qualitative_wsd_latest_' + tag))
    out.mkdir(parents=True, exist_ok=False)
    methods = ['R', 'V', 'Xn']
    checkpoints = {}
    for method in methods:
        folder = ROOT / cfg['output_root'] / f'cooldown_{method}_k50000_s1247'
        summary = json.loads((folder / 'summary.json').read_text())
        if summary['status'] not in ['running', 'completed']:
            raise RuntimeError('Latest source training is not usable')
        path = folder / 'last_local.pth'
        checkpoints[method] = {'path': str(path), 'sha256': sha256(path),
                               'summary_status_at_capture': summary['status']}
    receipt = {'created_at': created, 'scope': 'private visual diagnostic only, no aggregate ranking or budget selection',
               'device': 'cpu', 'precision': 'float32', 'noise_generator_device': 'cpu',
               'noise_note': 'Same CPU seed protocol as previous preview; RNG fields differ from official CUDA evaluation',
               'inference_base_seed': prior['inference_base_seed'], 'training_seed': 1247,
               'updates_each': 52500, 'stable_updates': 50000, 'cooldown_updates': 2500,
               'cfg': False, 'dino': False, 'ray': False, 'condition_dropout': 0,
               'checkpoints': checkpoints, 'prior_selection_sha256': sha256(prior_path),
               'records': prior['records'], 'script_sha256': sha256(__file__)}
    write_json(out / 'selection_private.json', receipt)
    model = FaceNormalUNet_B().eval()
    predictions = {}
    with torch.inference_mode():
        for method, checkpoint in checkpoints.items():
            state = torch.load(checkpoint['path'], map_location='cpu')
            signature = state['signature']
            if state['step'] != 52500 or signature['method'] != method or signature['phase'] != 'cooldown':
                raise RuntimeError('Latest weights do not have the required equal budget')
            if signature['fork_step'] != 50000 or signature['condition_dropout'] != 0:
                raise RuntimeError('Unexpected preview schedule or dropout')
            model.load_state_dict(state['model'], strict=True)
            del state
            panels = []
            for sample in chosen:
                rgb = torch.from_numpy(sample['rgb'].transpose(2, 0, 1).copy()).float()[None] / 127.5 - 1
                record = sample['record']
                key = record.get('image_key', sample['domain'] + '/' + Path(record['path']).name)
                generator = torch.Generator(device='cpu').manual_seed(image_noise_seed(prior['inference_base_seed'], key))
                noise = torch.randn(rgb.shape, generator=generator) if method != 'R' else torch.zeros_like(rgb)
                raw = model(noise, torch.zeros(1), rgb)
                endpoint = one_step_endpoint(method, raw, noise)
                pred = F.normalize(endpoint, dim=1)[0].permute(1, 2, 0).numpy().copy()
                valid = sample['valid'].astype(bool)
                if not np.isfinite(pred[valid]).all() or (np.linalg.norm(pred[valid], axis=-1) < .99).any():
                    raise RuntimeError('Invalid predicted unit normal')
                panels.append(pred)
                print(method, sample['domain'], sample['example'], flush=True)
            predictions[method] = panels
            if sha256(checkpoint['path']) != checkpoint['sha256']:
                raise RuntimeError('Source checkpoint changed during rendering')

    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('black')
    image_paths = {}
    for kind in ['normals', 'errors']:
        fig, axes = plt.subplots(len(chosen), 5, figsize=(12, 14), dpi=130)
        for i, sample in enumerate(chosen):
            valid = sample['valid'].astype(bool)
            gt = sample['normal']
            gt = gt / np.maximum(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8)
            normal_panel = lambda n: np.where(valid[..., None], np.clip((n + 1) / 2, 0, 1), 0)
            for j in range(5):
                ax = axes[i, j]
                if j == 0:
                    ax.imshow(sample['rgb'])
                elif j == 1:
                    ax.imshow(normal_panel(gt))
                elif kind == 'normals':
                    ax.imshow(normal_panel(predictions[methods[j - 2]][i]))
                else:
                    pred = predictions[methods[j - 2]][i]
                    error = np.degrees(np.arccos(np.clip((pred * gt).sum(-1), -1, 1)))
                    ax.imshow(np.ma.array(error, mask=~valid), cmap=cmap, vmin=0, vmax=60)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if i == 0:
                    suffix = '' if kind == 'normals' else ' error'
                    ax.set_title(['RGB', 'Label', 'R' + suffix, 'V' + suffix, 'Xn' + suffix][j], fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"{sample['domain']}\nexample {sample['example']}", fontsize=10)
        fig.suptitle('LATEST WSD: all 50k stable + 2.5k cooldown | seed 1247 | p=0\n'
                     'Same six validation examples as the 10k preview. CPU float32; visual diagnostic only.', fontsize=11)
        fig.tight_layout(rect=[0, .03, 1, .95], pad=.6)
        if kind == 'errors':
            bar_axis = fig.add_axes([.57, .015, .37, .008])
            fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 60), cmap=cmap), cax=bar_axis,
                         orientation='horizontal', ticks=[0, 15, 30, 45, 60])
        else:
            fig.text(.5, .015, 'Black = invalid label. RGB encoding = (unit normal + 1) / 2.', ha='center', fontsize=10)
        target = out / (kind + '.png')
        fig.savefig(target)
        plt.close(fig)
        image_paths[kind] = {'path': str(target), 'sha256': sha256(target)}
    completed = {'status': 'completed', 'created_at': now(), 'directory': str(out), 'images': image_paths,
                 'selection_receipt_sha256': sha256(out / 'selection_private.json'), 'updates_each': 52500}
    write_json(out / 'completion_private.json', completed)
    write_json(ROOT / 'figures/wsd_latest_preview_private.json', completed)
    print(json.dumps(completed), flush=True)


if __name__ == '__main__':
    main()
