"""Publishable aggregates for the completed single-training-seed WSD diagnosis.

Source per-image records are read for independent reconciliation, never exported.
This script does not choose a new budget, change a rule, or launch training.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
import statistics
import numpy as np
from src.research_io import ROOT, write_json, sha256, now

DOMAINS = ['hypersim', 'arb', 'nyuv2']
METHODS = ['R', 'V', 'Xn']
FORKS = [10000, 20000, 30000, 50000]


def close(actual, expected):
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise ValueError('Missing value mismatch')
    elif not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError('Aggregate does not reconcile with private inputs')


def mean(values):
    return math.fsum(values) / len(values)


def reconcile_domain(result, details):
    """Use Python scalar arithmetic independently of the evaluator's NumPy helper."""
    if not details or len(details) != result['num_images']:
        raise ValueError('Incomplete per-image evidence')
    count = result['num_inference_replicates']
    if count != 2 or any(len(r['mae_by_seed']) != count for r in details):
        raise ValueError('Unexpected inference replicate count')
    dataset_means = [mean([r['mae_by_seed'][k] for r in details]) for k in range(count)]
    image_sds = [statistics.stdev(r['mae_by_seed']) for r in details]
    close(mean(dataset_means), result['mean_mae_deg'])
    close(statistics.stdev(dataset_means), result['dataset_mae_sd_ddof1_deg'])
    close(mean(image_sds), result['mean_image_mae_sd_ddof1_deg'])
    close(mean([r['pixel_angular_dispersion_deg'] for r in details]), result['mean_pixel_angular_dispersion_deg'])
    bands = [r['boundary_mae_by_seed'] for r in details if r['boundary_mae_by_seed'] is not None]
    if len(bands) != result['boundary_images']:
        raise ValueError('Boundary denominator mismatch')
    close(mean([mean(row) for row in bands]) if bands else None, result['boundary_mae_deg'])
    return {'images': len(details), 'inference_replicates': count, 'boundary_images': len(bands), 'reconciled': True}


def load(path):
    return json.loads(path.read_text())


def main():
    cfg_path = ROOT / 'configs/parameterization_first_round_v2_review2.json'
    cfg = load(cfg_path)
    root = ROOT / cfg['output_root']
    rule_path = ROOT / cfg['decision_rule']
    decision = load(root / 'budget_decision.json')
    queue = load(root / 'queue_status.json')
    if queue['status'] != 'awaiting_convergence_review' or not queue['formal_grid_locked']:
        raise RuntimeError('Expected completed, formally locked diagnostic queue')
    flags = load(ROOT / 'outputs/P1_arb_materials_review4/validation_material_index_private.json')['assets']
    rows, checks, curves, signatures = [], [], {}, []
    plans = [('stable', method, None) for method in METHODS]
    plans += [('cooldown', method, step) for step in FORKS for method in METHODS]
    plans += [('clip_diagnostic', 'Xv', None)]
    for phase, method, fork in plans:
        name = f'cooldown_{method}_k{fork}_s1247' if fork else f'{phase}_{method}_s1247'
        folder = root / name
        summary_path = folder / 'summary.json'
        result = load(summary_path)
        signature = load(folder / 'signature_private.json')
        if result['status'] != 'completed' or result['phase'] != phase or result['method'] != method:
            raise RuntimeError('Missing completed planned result: ' + name)
        expected_step = fork + 2500 if fork else 50000 if phase == 'stable' else 2500
        if result['last_update'] != expected_step or result['inference_seeds'] != [42, 43]:
            raise RuntimeError('Unexpected completed budget or inference seeds')
        if signature['config_sha256'] != sha256(cfg_path) or signature['rule_sha256'] != sha256(rule_path):
            raise RuntimeError('Registered source configuration changed')
        signatures.append(signature)
        details = load(folder / 'per_image_private.json')
        checked = {domain: reconcile_domain(result['validation'][domain], details[domain]) for domain in DOMAINS}
        sensitivity = result['validation']['arb']['without_blend_or_transmission_sensitivity']
        usable_assets = {r['group'] for r in details['arb']}
        retained = [r for r in details['arb'] if not flags[r['group']]['flagged']]
        remaining_assets = {r['group'] for r in retained}
        if len(retained) != sensitivity['remaining_images'] or len(remaining_assets) != sensitivity['remaining_assets']:
            raise RuntimeError('Material subset denominator mismatch')
        if len(usable_assets) != sensitivity['full_usable_assets'] or len(usable_assets - remaining_assets) != sensitivity['excluded_flagged_assets']:
            raise RuntimeError('Material exclusion denominator mismatch')
        close(mean([mean(r['mae_by_seed']) for r in retained]), sensitivity['metrics']['mean_mae_deg'])
        checks.append({'run': name, 'domains': checked, 'material_subset_reconciled': True})
        rows.append({'run': name, 'phase': phase, 'method': method, 'stable_fork_step': fork,
                     'total_updates': result['last_update'], 'summary_sha256': sha256(summary_path),
                     'validation': result['validation'], 'training_probe': result['training_probe'],
                     'clip_fraction': result['clip_fraction'], 'gradient_updates_recorded': result['gradient_updates_recorded'],
                     'time_loss_bins': result['time_loss_bins'], 'peak_cuda_bytes': result['peak_cuda_bytes']})
        if phase == 'stable':
            curve = load(folder / 'learning_curve.json')
            if curve['status'] != 'completed' or [p['update'] for p in curve['points']] != list(range(2500, 50001, 2500)):
                raise RuntimeError('Incomplete stable learning curve')
            curves[method] = curve['points']
    shared_keys = ['config_sha256', 'rule_sha256', 'code_sha256', 'shared_evaluator_sha256', 'schedule_sha256',
                   'target_sha256', 'model_sha256', 'diagnostics_sha256', 'manifests_sha256', 'training_probe_sha256',
                   'data_quality_approval_sha256', 'validation_material_sidecar_sha256', 'material_protocol_sha256',
                   'training_seed', 'condition_dropout']
    for key in shared_keys:
        if any(sig[key] != signatures[0][key] for sig in signatures):
            raise RuntimeError('Nonmatching controlled-run signature: ' + key)
    clipping = {}
    for method in [*METHODS, 'Xv']:
        folder = root / (f'clip_diagnostic_{method}_s1247' if method == 'Xv' else f'stable_{method}_s1247')
        with (folder / 'training_log.jsonl').open() as stream:
            window = [r for line in stream if (r := json.loads(line))['update'] <= 2500]
        if [r['update'] for r in window] != list(range(1, 2501)):
            raise RuntimeError('Incomplete matched clipping window')
        norms = [r['gradient_norm_before_clip'] for r in window]
        clipping[method] = {'updates': 2500, 'clip_threshold': cfg['gradient_clip_norm'],
                            'clipped_fraction': mean([int(r['gradient_clipped']) for r in window]),
                            'gradient_norm_mean': mean(norms), 'gradient_norm_median': statistics.median(norms),
                            'gradient_norm_p95': float(np.percentile(norms, 95)), 'gradient_norm_maximum': max(norms)}
    cooled = {(row['method'], row['stable_fork_step']): row for row in rows if row['phase'] == 'cooldown'}
    rule = load(rule_path)
    evidence = []
    for method in METHODS:
        raw = {p['update']: p['validation'] for p in curves[method]}
        for domain in DOMAINS:
            value = lambda fork: cooled[method, fork]['validation'][domain]['mean_mae_deg']
            evidence.append({'method': method, 'domain': domain,
                             'stable_27500_to_30000_gain_deg': raw[27500][domain]['mean_mae_deg'] - raw[30000][domain]['mean_mae_deg'],
                             'cooled_22500_to_32500_gain_deg': value(20000) - value(30000),
                             'stable_47500_to_50000_gain_deg': raw[47500][domain]['mean_mae_deg'] - raw[50000][domain]['mean_mae_deg'],
                             'cooled_32500_to_52500_gain_deg': value(30000) - value(50000)})
    extension = any(row[key] > rule['extension_threshold_deg'] + rule['numerical_epsilon_deg'] for row in evidence
                    for key in ['stable_27500_to_30000_gain_deg', 'cooled_22500_to_32500_gain_deg'])
    still_improving = any(row[key] > rule['extension_threshold_deg'] + rule['numerical_epsilon_deg'] for row in evidence
                          for key in ['stable_47500_to_50000_gain_deg', 'cooled_32500_to_52500_gain_deg'])
    if not extension or not still_improving or decision['status'] != 'inconclusive_still_improving':
        raise RuntimeError('Budget decision does not independently reconcile with actual curves')
    summary = {'status': 'completed', 'created_at': now(), 'scope': 'single-training-seed convergence and clipping diagnosis',
               'formal_grid_completed': False, 'formal_grid_locked': True, 'training_seeds': [1247],
               'condition_dropout': 0, 'cfg': False, 'dino': False, 'ray': False, 'spatial_protocol': cfg['spatial_protocol'],
               'curve_inference_seeds': [42, 43], 'noise_protocol': 'sha256_image_noise_v1; independent field per image and inference replicate',
               'metric': 'macro-image masked mean angular error, degrees; lower is better',
               'uncertainty_scope': 'Two inference replicates only; dataset and within-image SDs are not training-seed variance or confidence intervals',
               'completed_run_count': len(rows), 'stable_curve_points_each': 20,
               'config_sha256': sha256(cfg_path), 'rule_sha256': sha256(rule_path),
               'shared_signature': {key: signatures[0][key] for key in shared_keys},
               'budget_decision': decision, 'budget_evidence': evidence,
               'matched_first_2500_clipping': clipping, 'rows': rows,
               'validation_reconciliation': {'all_planned_results_checked': True, 'checked_run_count': len(checks),
                                             'checked_domain_count': len(checks) * len(DOMAINS), 'checks': checks},
               'training_seed_variance_estimated': False, 'gso_model_scores_computed': False}
    out = ROOT / 'outputs/P2_wsd_convergence_review2'
    write_json(out / 'summary.json', summary)
    write_json(out / 'stable_learning_curves.json', {'status': 'completed', 'training_seed': 1247, 'points': curves})
    lines = ['# Completed WSD convergence diagnostics', '',
             'Single training seed 1247, condition dropout p=0, 256px center-square data; no DINO, ray input or CFG.', '',
             '**Diagnostic training is complete. The formal 24-run comparison is not complete and remains locked.**', '',
             'Each row below uses a separate 2,500-update linear cooldown restored from a stable checkpoint. Two paired inference replicates (42, 43), image-specific noise, macro-image masked MAE in degrees; lower is better.', '',
             '| Total updates | Method | Hypersim (700) | Arb (594 views / 198 assets) | NYUv2 (795) |',
             '|---:|---|---:|---:|---:|']
    for fork in FORKS:
        for method in METHODS:
            score = cooled[method, fork]['validation']
            lines.append(f'| {fork+2500} | {method} | ' + ' | '.join(f"{score[d]['mean_mae_deg']:.4f}" for d in DOMAINS) + ' |')
    lines += ['', '## Final 52,500-update metrics', '',
              '| Method | Domain | MAE | Boundary MAE | Mean within-image MAE SD | Mean pixel angular dispersion |',
              '|---|---|---:|---:|---:|---:|']
    for method in METHODS:
        for domain in DOMAINS:
            score = cooled[method, 50000]['validation'][domain]
            lines.append(f'| {method} | {domain} | ' + ' | '.join(f'{score[k]:.4f}' for k in
                         ['mean_mae_deg', 'boundary_mae_deg', 'mean_image_mae_sd_ddof1_deg', 'mean_pixel_angular_dispersion_deg']) + ' |')
    lines += ['', 'R has no inference noise; its within-image noise metrics are zero by construction. Two-replicate statistics are coarse diagnostics, not calibrated uncertainty, training-seed variation or confidence intervals.', '',
              '## Same-prediction Arb material sensitivity', '',
              '| Method | Full 198 usable assets / 594 views | Remaining 170 assets / 510 views |', '|---|---:|---:|']
    for method in METHODS:
        score = cooled[method, 50000]['validation']['arb']
        sub = score['without_blend_or_transmission_sensitivity']
        lines.append(f"| {method} | {score['mean_mae_deg']:.4f} | {sub['metrics']['mean_mae_deg']:.4f} |")
    lines += ['', '28 flagged usable validation assets are excluded in the sensitivity subset; the 200-asset source plan has 30 flagged assets, including two without usable near-unit-normal labels. Main metrics and budget decisions retain the original full usable set. Metadata flags do not establish actual rendered transparency or causal label mismatch.', '',
              '## Matched first-2,500-update clipping window', '',
              '| Method | Clipped updates (%) | Median preclip L2 norm | P95 preclip L2 norm |', '|---|---:|---:|---:|']
    for method, clip in clipping.items():
        lines.append(f"| {method} | {100*clip['clipped_fraction']:.2f} | {clip['gradient_norm_median']:.4f} | {clip['gradient_norm_p95']:.4f} |")
    lines += ['', 'All four methods use the unchanged clipping threshold 1.0. Xv has only the 2,500-update clipping diagnostic; it is not a fourth completed long-run method.', '',
              '## Registered budget status', '',
              '`inconclusive_still_improving`: at least one registered method/domain improves by more than 0.3 degrees between stable 47.5k and 50k or between cooled 32.5k and 52.5k. No new common budget is approved, no automatic extension beyond the registered cap is enabled, and the formal grid remains locked.', '',
              'All 16 completed source summaries, their 48 domain metrics, material subset counts/means, controlled signatures and actual budget-trigger values were reconciled. Private per-image inputs are not distributed. Training probes and full stable curves are in the JSON files. GSO preparation summaries describe rendering readiness, not model performance.', '',
              '[Aggregate JSON](summary.json) · [Stable learning curves](stable_learning_curves.json)', '']
    (out / 'cooled_results.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
