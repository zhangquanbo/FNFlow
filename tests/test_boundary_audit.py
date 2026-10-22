import numpy as np
from src.boundary_audit import audit_mask


def test_rejected_silhouette_is_counted_even_when_strict_gt_erases_edge():
    decoded=np.zeros((16,16,3),np.float32);decoded[4:12,4:12,2]=1
    decoded[4:12,4,2]=.5  # coverage-like antialias row rejected by length threshold
    alpha=np.zeros((16,16),np.float32);alpha[4:12,4:12]=1
    row=audit_mask(decoded,alpha)['0.01']
    assert row['unit_rejected_foreground_pixels']==8
    assert row['strict_gt_before_valid_intersection']['unit_rejected_in_band']==0
    assert row['relaxed_direction_plus_silhouette']['unit_rejected_in_band']==8


def test_nonfinite_and_zero_are_explicit_and_never_validated():
    decoded=np.zeros((8,8,3),np.float32);decoded[...,2]=1
    decoded[2,2]=np.nan;decoded[3,3]=0
    row=audit_mask(decoded,np.ones((8,8),np.float32))['0.5']
    assert row['nonfinite_foreground_pixels']==1
    assert row['zero_foreground_pixels']==1
    assert row['unit_rejected_foreground_pixels']==1
