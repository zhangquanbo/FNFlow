import numpy as np
from scripts.check_hypersim_anomalous_directions import compare


def test_euclidean_distance_backprojection_recovers_signed_tilted_plane_normal():
    h,w=64,80;v,u=np.mgrid[:h,:w];k=np.array([[70,0,39.5],[0,70,31.5],[0,0,1.]])
    rays=np.stack([(u-39.5)/70,(v-31.5)/70,np.ones_like(u)],-1);rays/=np.linalg.norm(rays,axis=-1,keepdims=True)
    gl=rays*np.array([1,-1,-1]);normal=np.array([.2,-.1,1.]);normal/=np.linalg.norm(normal)
    depth=-3/(gl@normal)
    raw=np.broadcast_to(normal*98.8,(h,w,3)).copy()
    rule={'max_neighbor_relative_depth_jump':.1,'minimum_stable_pixels':100,
          'retain_if_median_angle_at_most_deg':10.,'retain_if_p90_angle_at_most_deg':30.}
    result,*_=compare(depth,raw,k,rule)
    assert result['disposition']=='retain' and result['p90_deg']<1e-4
    reversed_result,*_=compare(depth,-raw,k,rule)
    assert reversed_result['disposition']=='exclude_direction_inconsistent' and reversed_result['median_deg']>179.99
