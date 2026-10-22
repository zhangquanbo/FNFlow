import numpy as np
from src.object_normals import decode_arb_normal

def test_exr_background_and_nonunit_mixtures_are_not_foreground():
    native=np.array([[[.5,.5,.5],[.5,.5,1.],[.5,.5,.8],[np.nan,.5,1.]]],np.float32)
    normal,valid=decode_arb_normal(native,(4,1))
    np.testing.assert_array_equal(valid,[[False,True,False,False]])
    np.testing.assert_allclose(normal[0,1],[0,0,1])
    assert np.isfinite(normal).all()

def test_native_channel_transform_is_fixed_and_preserves_direction():
    normal,valid=decode_arb_normal(np.array([[[1.,.5,.5]]],np.float32),(1,1))
    np.testing.assert_allclose(normal[0,0],[-1,0,0]);assert valid.all()
