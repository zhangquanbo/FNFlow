import numpy as np
import pytest
import torch
from src.mesh_normal_gt import crease_corner_normals,interpolated_geometry_normals
from src.gso_boundary_protocol import angular_band,gt_boundary_report,aggregate_boundary_reports
from src.material_protocol import material_flags,validation_material_sensitivity
from scripts.train_parameterization import boundary_mask


def test_crease45_preserves_right_angle_and_smooths_gentle_edge():
    triangles=np.array([[0,1,2],[0,3,1]]);hits=np.array([[.5,0,0],[.5,0,0]])
    for degrees in [90,30]:
        a=np.deg2rad(degrees);vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-np.cos(a),np.sin(a)]])
        normals=crease_corner_normals(vertices,triangles,45)
        result=interpolated_geometry_normals(vertices,triangles,normals,np.array([0,1]),hits)
        if degrees==90:
            np.testing.assert_allclose(result,[[0,0,1],[0,1,0]],atol=1e-12)
        else:
            np.testing.assert_allclose(result[0],result[1],atol=1e-12)
            np.testing.assert_allclose(result[0],[0,np.sin(a/2),np.cos(a/2)],atol=1e-12)


def test_nonmanifold_common_edge_is_not_smoothed():
    vertices=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,.2],[0,-1,.3]])
    triangles=np.array([[0,1,2],[0,3,1],[0,4,1]])
    normals=crease_corner_normals(vertices,triangles,45)
    expected=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
    expected/=np.linalg.norm(expected,axis=-1,keepdims=True)
    np.testing.assert_allclose(normals[:,0],expected,atol=1e-12)


def test_primary_boundary_is_identical_to_existing_metric_and_empty_regions_are_explicit():
    rng=np.random.default_rng(5);n=rng.normal(size=(12,15,3));n/=np.linalg.norm(n,axis=-1,keepdims=True)
    mask=rng.random((12,15))>.1
    reference=boundary_mask(torch.from_numpy(n.transpose(2,0,1)[None]),torch.from_numpy(mask[None]))[0].numpy()
    np.testing.assert_array_equal(angular_band(n,mask),reference)
    flat=np.zeros((16,16,3));flat[...,2]=1;mask=np.zeros((16,16),bool);mask[1:15,1:15]=True
    crease=flat.copy();crease[:,8:]=[0,1,0]
    report,bands,_=gt_boundary_report(crease,flat,crease,mask)
    aggregate=aggregate_boundary_reports([{'scene':'fixture','boundary_gt_report':report}])
    assert aggregate['primary_angular']['comparisons']['face_vs_smooth']['inside']['pixel_pooled_mean_deg'] is None
    assert aggregate['primary_angular']['comparisons']['face_vs_smooth']['inside']['images_without_region']==1
    sharp=aggregate['crease45_angular_plus_silhouette']
    assert sharp['inside_pixels']+sharp['outside_pixels']==mask.sum() and sharp['inside_pixels']>0


def test_material_flag_is_metadata_only_and_includes_transmission_with_opaque_alpha():
    assert not material_flags({'materials':[{}]})['flagged']
    assert material_flags({'materials':[{'alphaMode':'BLEND'}]})['flagged']
    flag=material_flags({'materials':[{'alphaMode':'OPAQUE','extensions':{'KHR_materials_transmission':{'transmissionFactor':.1}}}]})
    assert flag['flagged'] and not flag['referenced_material_flagged']
    with pytest.raises(ValueError):material_flags({'materials':[{'alphaMode':'UNKNOWN'}]})


def test_validation_material_sensitivity_reuses_same_errors_and_requires_known_assets():
    rows=[{'scene':'a'},{'scene':'a'},{'scene':'b'},{'scene':'c'}]
    errors=[[50.,52.],[40.,42.],[1.,3.],[3.,5.]];before=np.array(errors)
    flags={'a':{'flagged':True},'b':{'flagged':False},'c':{'flagged':False}}
    result=validation_material_sensitivity(rows,errors,[[5,5],None,[1,1],[3,3]],[1,2,3,4],flags)
    assert result['remaining_assets']==2 and result['remaining_images']==2 and result['excluded_flagged_assets']==1
    assert result['metrics']['mean_mae_deg']==3. and result['metrics']['boundary_mae_deg']==2.
    np.testing.assert_array_equal(errors,before)
    with pytest.raises(RuntimeError):validation_material_sensitivity(rows,errors,[None]*4,[0]*4,{})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Production evaluator uses CUDA')
def test_extra_validation_stratum_does_not_change_primary_score_or_add_model_calls(tmp_path):
    from scripts.train_parameterization import evaluate
    class Fixed(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,z,t,rgb):
            self.calls+=1;out=torch.zeros_like(rgb);out[:,2]=1;return out
    records=[]
    for uid,vector in [('a',[1,0,0]),('b',[0,0,1])]:
        path=tmp_path/(uid+'.npz');normal=np.broadcast_to(vector,(256,256,3)).astype('float32')
        np.savez(path,rgb=np.zeros((256,256,3),np.uint8),normal=normal,valid=np.ones((256,256),bool))
        records.append({'path':str(path),'scene':uid,'split':'val','image_key':'fixture/'+uid})
    model=Fixed().cuda()
    old,_=evaluate(model,'R',{'arb':records},[42,43],0,'cuda');calls=model.calls
    new,_=evaluate(model,'R',{'arb':records},[42,43],0,'cuda',{'a':{'flagged':True},'b':{'flagged':False}})
    assert model.calls-calls==calls==1
    sensitivity=new['arb'].pop('without_blend_or_transmission_sensitivity')
    assert new==old and sensitivity['metrics']['mean_mae_deg']==0.
