import numpy as np
from src.mesh_normal_gt import geometry_vertex_normals,interpolated_geometry_normals,gt_angular_gap,dual_gt_mae


def test_planar_triangle_has_zero_gap_and_smooth_normal_is_barycentric():
    vertices=np.array([[0.,0,0],[1,0,0],[0,1,0]])
    triangles=np.array([[0,1,2]]);normals=geometry_vertex_normals(vertices,triangles)
    smooth=interpolated_geometry_normals(vertices,triangles,normals,np.array([0]),np.array([[.2,.3,0]]))
    np.testing.assert_allclose(smooth,[[0,0,1]])
    np.testing.assert_allclose(gt_angular_gap(np.array([[[0.,0,1]]]),smooth.reshape(1,1,3),np.ones((1,1),bool)),0,atol=1e-6)


def test_adjacent_bent_triangles_produce_continuous_smooth_normals_at_shared_edge():
    vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1]])
    triangles=np.array([[0,1,2],[0,3,1]]);normals=geometry_vertex_normals(vertices,triangles)
    hit=np.array([[.5,0,0],[.5,0,0]])
    result=interpolated_geometry_normals(vertices,triangles,normals,np.array([0,1]),hit)
    np.testing.assert_allclose(result[0],result[1]);np.testing.assert_allclose(result[0],[0,1/np.sqrt(2),1/np.sqrt(2)])


def test_dual_gt_gap_is_not_a_lower_bound_on_prediction_error():
    def n(degrees):return np.array([[[np.sin(np.deg2rad(degrees)),0,np.cos(np.deg2rad(degrees))]]])
    metrics=dual_gt_mae(n(15),n(30),n(0),np.ones((1,1),bool))
    np.testing.assert_allclose(metrics['main_smooth_mae_deg'],15,atol=1e-6)
    np.testing.assert_allclose(metrics['secondary_face_mae_deg'],15,atol=1e-6)
    np.testing.assert_allclose(metrics['gt_definition_gap_mae_deg'],30,atol=1e-6)
