import numpy as np
import pytest
from src.cluster_bootstrap import cluster_percentile_intervals


def test_cluster_resampling_preserves_unequal_cluster_sizes_and_pairing():
    x = np.array([0., 0., 9.])
    result = cluster_percentile_intervals(np.column_stack([x, x + 3]), ['a','a','b'], draws=1000, seed=7)
    assert result['metrics'][0]['estimate'] == 3  # Macro images, not mean of identity means (4.5).
    assert result['metrics'][0]['lower_95'] == 0
    assert result['metrics'][0]['upper_95'] == 9
    for key in ['estimate','lower_95','upper_95']:
        assert result['metrics'][1][key] - result['metrics'][0][key] == pytest.approx(3)


def test_cluster_bootstrap_is_reproducible_and_rejects_single_cluster():
    a = cluster_percentile_intervals([1., 2.], ['a','b'], draws=200, seed=5)
    b = cluster_percentile_intervals([1., 2.], ['a','b'], draws=200, seed=5)
    assert a == b
    with pytest.raises(ValueError):
        cluster_percentile_intervals([1., 2.], ['a','a'])
