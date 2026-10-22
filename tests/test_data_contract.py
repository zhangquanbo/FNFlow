import numpy as np
from src.data_contract import opengl_projection_to_intrinsics,opengl_projection_to_pixel_matrix,resize_intrinsics_pixel_centers,hypersim_normals_for_training
import pytest


def test_projection_matches_opengl_after_basis_and_pixel_center_transform():
    p=np.array([[2.,.1,.2,0],[.05,3.,-.1,0],[0,0,-1,-2],[0,0,-1,0]])
    point=np.array([.2,-.3,2.])
    gl=np.r_[point*np.array([1,-1,-1]),1]
    clip=p@gl; ndc=clip[:2]/clip[3]
    expected=np.array([(ndc[0]+1)*320-.5,(1-ndc[1])*240-.5])
    k=opengl_projection_to_intrinsics(p,640,480)
    projected=k@point
    assert np.allclose(projected[:2]/projected[2],expected)


def test_resize_intrinsics_uses_same_half_pixel_affine_as_image_resize():
    k=np.array([[500,0,319.5],[0,500,239.5],[0,0,1.]])
    new=resize_intrinsics_pixel_centers(k,(640,480),(256,256))
    point=np.array([.1,.2,1.])
    old=k@point; expected=(old[:2]+.5)*np.array([256/640,256/480])-.5
    assert np.allclose((new@point)[:2],expected)


def test_tilted_central_camera_is_preserved_without_fake_pinhole_intrinsics():
    p=np.array([[2.,0,.2,0],[0,3.,-.1,0],[0,0,-1,-2],[0,.15,-1,0]])
    point=np.array([.2,-.3,2.]);gl=np.r_[point*np.array([1,-1,-1]),1]
    clip=p@gl;ndc=clip[:2]/clip[3]
    expected=np.array([(ndc[0]+1)*320-.5,(1-ndc[1])*240-.5])
    matrix=opengl_projection_to_pixel_matrix(p,640,480)
    projected=matrix@point
    assert np.allclose(projected[:2]/projected[2],expected)
    with pytest.raises(ValueError):
        opengl_projection_to_intrinsics(p,640,480)


def test_nearest_normals_do_not_average_across_surface_or_invalid_boundary():
    x=np.array([[[1.,0,0],[0,0,1],[np.nan,0,0],[0,0,0]]])
    normal,valid=hypersim_normals_for_training(x,(4,1))
    assert np.array_equal(valid,[[True,True,False,False]])
    assert np.allclose(normal[0,0],[-1,0,0])
    assert np.allclose(normal[0,1],[0,0,1])
    assert np.isfinite(normal).all() and np.all(normal[~valid]==0)
