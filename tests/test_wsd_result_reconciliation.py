import math
import pytest
from scripts.summarize_wsd_results import reconcile_domain, close


def evidence():
    details = [
        {'mae_by_seed': [1., 3.], 'boundary_mae_by_seed': [2., 4.], 'pixel_angular_dispersion_deg': 1.},
        {'mae_by_seed': [5., 7.], 'boundary_mae_by_seed': None, 'pixel_angular_dispersion_deg': 3.},
    ]
    result = {'num_images': 2, 'num_inference_replicates': 2, 'mean_mae_deg': 4.,
              'dataset_mae_sd_ddof1_deg': math.sqrt(2), 'mean_image_mae_sd_ddof1_deg': math.sqrt(2),
              'mean_pixel_angular_dispersion_deg': 2., 'boundary_images': 1, 'boundary_mae_deg': 3.}
    return result, details


def test_reconcile_uses_image_macro_and_nonempty_boundary_denominator():
    result, details = evidence()
    assert reconcile_domain(result, details) == {'images': 2, 'inference_replicates': 2, 'boundary_images': 1, 'reconciled': True}


def test_reconcile_rejects_population_sd_instead_of_sample_sd():
    result, details = evidence()
    result['dataset_mae_sd_ddof1_deg'] = 1.
    with pytest.raises(ValueError):
        reconcile_domain(result, details)


def test_reconcile_rejects_silently_dropped_image():
    result, details = evidence()
    with pytest.raises(ValueError, match='Incomplete'):
        reconcile_domain(result, details[:1])


def test_reconcile_keeps_empty_boundary_missing_not_zero():
    result, details = evidence()
    for row in details:
        row['boundary_mae_by_seed'] = None
    result['boundary_images'] = 0
    result['boundary_mae_deg'] = None
    assert reconcile_domain(result, details)['boundary_images'] == 0
    with pytest.raises(ValueError, match='Missing'):
        close(None, 0.)
