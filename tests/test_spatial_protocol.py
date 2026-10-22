import numpy as np
from src.spatial_protocol import center_square, crop_camera
from src.data_contract import resize_intrinsics_pixel_centers


def test_crop_and_resize_reprojects_native_pixels_without_anisotropic_scaling():
    image=np.arange(480*640).reshape(480,640)
    cropped,(left,top,side)=center_square(image)
    assert (left,top,side)==(80,0,480)
    assert cropped[0,0]==image[0,80] and cropped[-1,-1]==image[-1,559]
    matrix=np.array([[600.,3.,319.5],[0.,590.,239.5],[.01,0.,1.]])
    ray=np.array([.15,-.2,1.])
    native=matrix@ray;native=native[:2]/native[2]
    transformed=resize_intrinsics_pixel_centers(crop_camera(matrix,left,top),(side,side),(256,256))@ray
    expected=(native-[left,top]+.5)*(256/side)-.5
    np.testing.assert_allclose(transformed[:2]/transformed[2],expected)


def test_square_crop_is_identity_and_odd_dimensions_are_explicit():
    array=np.zeros((5,8,3))
    result,box=center_square(array)
    assert box==(1,0,5) and result.shape==(5,5,3)
    result,box=center_square(result)
    assert box==(0,0,5)
