"""Recover omitted region metrics from frozen private inference records."""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / 'outputs/P3_object_dino_evaluation_fresh_20261027'
OUT = ROOT / 'outputs/boundary_supplement_20261106'
FIELDS = ['mean_deg', 'normal_face_mae_deg', 'normal_crease45_diagnostic_mae_deg',
          'boundary_primary_angular_mae_deg', 'boundary_primary_angular_outside_mae_deg',
          'boundary_crease45_angular_plus_silhouette_mae_deg',
          'boundary_crease45_angular_plus_silhouette_outside_mae_deg']
REGIONS = ['boundary_primary_angular', 'boundary_crease45_angular_plus_silhouette']

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    result = {'status': 'completed', 'models': {}, 'private_rows_checked': 0,
              'method': 'Average errors over eight inference seeds per image, views within object, objects within pitch, then pitches equally. Absent region labels and empty GT regions remain null and have separate coverage counts.',
              'region_rule': 'Frozen GT normal jump>20deg, radius2 dilation, valid-mask intersection. Separate crease45 angular-plus-silhouette region. No output-dependent masks.',
              'inference_rerun': False, 'test_images_scored': 0}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for model, run in [('C', 'P3_C_train_20261029'), ('D1', 'D1_train_20261030'),
                           ('D2', 'D2_train_20261030'), ('D3', 'D3_train_20261030')]:
            evaluation = ROOT / 'outputs' / run / 'evaluation'
            original = read(evaluation / 'summary.json')
            result['models'][model] = {'panels': {}, 'source_summary_sha256': sha(evaluation / 'summary.json')}
            for panel, saved in original['panels'].items():
                records = read(AB / 'manifests_private' / (panel + '.json'))['records']
                def image_metrics(record):
                    rows = [read(evaluation / 'per_image_private' / panel / str(seed) / (record['id'] + '.json')) for seed in range(42, 50)]
                    assert all(x['checkpoint_sha256'] == original['checkpoint_sha256'] for x in rows)
                    metrics = [x['metrics'] for x in rows]
                    values = {}
                    for key in FIELDS:
                        numbers = [x.get(key) for x in metrics]
                        assert all(x is None for x in numbers) or all(isinstance(x, (int, float)) and math.isfinite(x) for x in numbers)
                        values[key] = None if numbers[0] is None else sum(numbers) / len(numbers)
                    counts = {key: metrics[0].get(key + '_pixels') for key in REGIONS}
                    assert all(all(x.get(key + '_pixels') == value for x in metrics) for key, value in counts.items())
                    return str(record['elevation_bin']), record['object'], values, counts
                grouped = defaultdict(list)
                for pitch, obj, values, counts in pool.map(image_metrics, records):
                    grouped[pitch].append((obj, values, counts))
                pitches = {}
                for pitch, images in grouped.items():
                    aggregate = {}
                    for field in FIELDS:
                        objects = defaultdict(list)
                        for obj, values, _ in images:
                            if values[field] is not None:
                                objects[obj].append(values[field])
                        aggregate[field] = sum(sum(v) / len(v) for v in objects.values()) / len(objects) if objects else None
                    assert abs(aggregate['mean_deg'] - saved['by_pitch'][pitch]['mean_deg']) < 1e-9
                    coverage = {}
                    for region in REGIONS:
                        pixels = [x[2][region] for x in images]
                        coverage[region] = {'images_with_nonempty_GT_region': sum(x is not None and x > 0 for x in pixels),
                                            'images_with_empty_GT_region': sum(x == 0 for x in pixels),
                                            'images_without_GT_region_label': sum(x is None for x in pixels),
                                            'GT_pixels_sum_over_images_once': sum(x for x in pixels if x is not None) if any(x is not None for x in pixels) else None}
                    pitches[pitch] = {**aggregate, 'images': len(images), 'coverage': coverage}
                equal_pitch = {key: sum(v[key] for v in pitches.values()) / len(pitches) if all(v[key] is not None for v in pitches.values()) else None for key in FIELDS}
                assert abs(equal_pitch['mean_deg'] - saved['equal_pitch_object_macro_mae_deg']) < 1e-9
                result['models'][model]['panels'][panel] = {'by_pitch': pitches, 'equal_pitch_object_macro': equal_pitch}
                result['private_rows_checked'] += len(records) * 8
            print('BOUNDARIES', model, result['models'][model]['panels']['primary']['equal_pitch_object_macro'], flush=True)
    result['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    temporary = OUT / 'summary.json.partial'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(OUT / 'summary.json')
    print('BOUNDARY_SUPPLEMENT_COMPLETE', result['private_rows_checked'], flush=True)

if __name__ == '__main__':
    main()
