"""Final face supplements: paired checkpoints, directional means, cluster intervals."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from scripts.run_diagnostics import Experiment, hwc, aggregate
from scripts.run_imagewise_noise import public_summary
from src.diagnostic_data import prediction_metrics
from src.flow_inference import infer
from src.noise_protocol import image_noise_seed, directional_sensitivity
from src.cluster_bootstrap import cluster_percentile_intervals
from src.research_io import ROOT, BEST, sha256, write_json, compare_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/face_supplements.json')
    parser.add_argument('--run-id')
    parser.add_argument('--subjects', type=int, default=0)
    args = parser.parse_args()
    args.stage = 'd1'
    config = json.loads((ROOT / args.config).read_text())
    extension = {'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__),
        ROOT/'scripts/run_imagewise_noise.py', ROOT/'src/noise_protocol.py', ROOT/'src/cluster_bootstrap.py']},
        'checkpoint_sha256': {name: sha256(BEST/item['checkpoint']) for name,item in config['models'].items()}}
    target = ROOT/'outputs'/(args.run_id or config['run_id'])/'summary.json'
    if target.exists() and json.loads(target.read_text()).get('extension_code') != extension:
        raise RuntimeError('Source changed; use a new run-id')
    experiment = None
    try:
        experiment = Experiment(args)
        experiment.state['extension_code'] = extension
        experiment.state['supplements'] = {}
        experiment.state['rows'] = []
        experiment.save()
        experiment.run_parity()
        modes, seeds = config['modes'], config['inference_seeds']
        columns, labels = [], []
        vectors = {}
        dataset = (BEST/'datasets/FacescapeRender1').resolve()
        for name in config['models']:
            model = experiment.get_model(name)
            buckets = {mode: {seed: [] for seed in seeds} for mode in modes}
            directions = {mode: [] for mode in modes}
            ensembles = {mode: [] for mode in modes}
            for index, record in enumerate(experiment.records):
                key = config['dataset_namespace']+'/'+Path(record['pair']['rgb_path']).resolve().relative_to(dataset).as_posix()
                derived = {str(seed): image_noise_seed(seed,key) for seed in seeds}
                hashes = {kind: sha256(record['pair'][kind+'_path']) for kind in ['rgb','normal']}
                path = experiment.directory/'image_records'/name/(record['id']+'.json')
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['derived_seeds'] != derived or result['input_sha256'] != hashes or result['image_key'] != key:
                        raise RuntimeError('Saved image changed')
                else:
                    result = {'image_key': key, 'derived_seeds': derived, 'input_sha256': hashes,
                              'samples': {}, 'directional': {}, 'ensembles': {}}
                    mask = torch.as_tensor(record['mask'], device=experiment.device)[None]
                    for mode in modes:
                        predictions, items = [], []
                        for seed in seeds:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            output = infer(model,record['condition'],mode=mode,seed=derived[str(seed)],
                                           stats_mask=mask,scale=config['cfg_scale'])
                            torch.cuda.synchronize()
                            elapsed = (time.perf_counter()-start)*1000
                            if output['invalid'][0,0][mask[0]].any():
                                raise RuntimeError('Invalid prediction')
                            normal = hwc(output['normal'])
                            items.append({'id':record['id'],'base_seed':seed,'ms':elapsed,
                                          **prediction_metrics(hwc(output['raw']),normal,record)})
                            predictions.append(normal)
                        sample = np.stack(predictions)
                        mean = sample.astype(np.float64).mean(axis=0)
                        length = np.linalg.norm(mean,axis=-1,keepdims=True)
                        if (length[...,0][record['mask']] < 1e-8).any():
                            raise RuntimeError('Undefined ensemble direction in evaluation mask')
                        mean = (mean/np.maximum(length,1e-8)).astype(np.float32)
                        result['samples'][mode] = items
                        result['directional'][mode] = directional_sensitivity(sample,record['mask'])
                        result['ensembles'][mode] = prediction_metrics(mean,mean,record)
                    write_json(path,result)
                # Exact replay check for the already completed best-checkpoint experiment.
                old = ROOT/'outputs/D3_full_imagewise_noise_v1/image_records'/(record['id']+'.json')
                if name == 'p05_best' and old.exists():
                    prior = json.loads(old.read_text())
                    for mode in modes:
                        for a,b in zip(prior['samples'][mode],result['samples'][mode]):
                            if a['base_seed'] != b['base_seed'] or abs(a['float']['mean']-b['float']['mean']) > 1e-6:
                                raise RuntimeError('Best checkpoint imagewise replay differs')
                for mode in modes:
                    if [item['base_seed'] for item in result['samples'][mode]] != seeds:
                        raise RuntimeError('Incomplete samples')
                    for item in result['samples'][mode]:
                        buckets[mode][item['base_seed']].append(item)
                    directions[mode].append(result['directional'][mode])
                    ensembles[mode].append(result['ensembles'][mode])
                if (index+1)%100 == 0 or index+1 == len(experiment.records):
                    experiment.state['current_task'] = f'{name}: {index+1}/{len(experiment.records)}'
                    experiment.save()
                    print(experiment.state['current_task'],flush=True)
            child = SimpleNamespace(directory=experiment.directory/'models'/name,
                                    config={**config,'reference':name},subjects=args.subjects,
                                    state={'mask_fallbacks':experiment.state['mask_fallbacks']})
            public_summary(child,buckets,directions)
            experiment.state['rows'].extend(child.state['rows'])
            model_result = {'single_sample':child.state['multiseed'],'ensemble8':{}}
            for mode in modes:
                single = np.mean([[i['float']['mean'] for i in buckets[mode][seed]] for seed in seeds],axis=0)
                ensemble = np.array([i['float']['mean'] for i in ensembles[mode]])
                model_result['ensemble8'][mode] = {'float':aggregate(ensembles[mode],'float'),
                    'mean_single_minus_ensemble_deg':float((single-ensemble).mean()),
                    'nfe':8*(1 if mode=='no_cfg' else 2),
                    'interpretation':'finite eight-direction averaging gain; not an isolated irreducible noise cost or guaranteed x-pred target'}
                for tag,value in [('single',single),('ensemble8',ensemble),('averaging_gain',single-ensemble)]:
                    label=f'{name}/{mode}/{tag}'
                    labels.append(label); columns.append(value); vectors[label]=value
            delta = vectors[f'{name}/geom_cfg/single']-vectors[f'{name}/no_cfg/single']
            label=f'{name}/geom_minus_conditional'
            labels.append(label); columns.append(delta); vectors[label]=delta
            experiment.state['supplements'][name]=model_result
            experiment.save()
        difference = vectors['p05_original/geom_minus_conditional']-vectors['p05_best/geom_minus_conditional']
        labels.append('original_minus_best_cfg_delta'); columns.append(difference)
        bootstrap = cluster_percentile_intervals(np.column_stack(columns),
            [r['pair']['subject'] for r in experiment.records],draws=config['bootstrap_resamples'],seed=config['bootstrap_seed'])
        bootstrap['metrics'] = dict(zip(labels,bootstrap['metrics']))
        experiment.state['identity_bootstrap']=bootstrap
        experiment.state['status']='completed'; experiment.state['current_task']='completed'
        write_json(experiment.directory/'aggregate_summary.json',{
            'num_images':len(experiment.records),'num_subjects':experiment.state['num_subjects'],
            'checkpoints':experiment.state['checkpoints'],'supplements':experiment.state['supplements'],
            'identity_bootstrap':bootstrap,'status':'completed','noise_protocol':config['noise_protocol']})
    except BaseException:
        if experiment is not None:
            experiment.state['status']='failed'; experiment.state['error']=traceback.format_exc()
        raise
    finally:
        if experiment is not None:
            experiment.state['best_audit']=compare_best(json.loads((ROOT/'provenance/best_before.json').read_text()))
            experiment.save()
            if not experiment.state['best_audit']['verified_unchanged']:
                raise RuntimeError('Protected reference changed')


if __name__ == '__main__':
    main()
