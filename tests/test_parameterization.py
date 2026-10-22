import torch
from src.parameterization import parameterization_loss,one_step_endpoint

def test_xv_matches_velocity_change_of_variables_in_value_and_gradient():
    torch.manual_seed(7)
    gt=torch.randn(2,3,2,2);noise=torch.randn_like(gt);times=torch.tensor([.2,.8])
    state=(1-times[:,None,None,None])*noise+times[:,None,None,None]*gt
    endpoint=torch.randn_like(gt,requires_grad=True);valid=torch.ones(2,2,2,dtype=torch.bool)
    loss_x,_=parameterization_loss('Xv',endpoint,state,gt,noise,times,valid)
    velocity=(endpoint-state)/(1-times[:,None,None,None])
    loss_v,_=parameterization_loss('V',velocity,state,gt,noise,times,valid)
    torch.testing.assert_close(loss_x,loss_v)
    gx=torch.autograd.grad(loss_x,endpoint,retain_graph=True)[0]
    gv=torch.autograd.grad(loss_v,endpoint)[0]
    torch.testing.assert_close(gx,gv)

def test_macro_image_mask_and_endpoint_are_analytical():
    gt=torch.zeros(2,3,1,2);pred=torch.tensor([1.,3.])[:,None,None,None].expand_as(gt)
    valid=torch.tensor([[[True,False]],[[True,True]]])
    loss,_=parameterization_loss('R',pred,pred,gt,gt,torch.zeros(2),valid,0,0)
    torch.testing.assert_close(loss,torch.tensor(5.))
    torch.testing.assert_close(one_step_endpoint('V',pred,gt),pred)
    assert torch.isfinite(torch.autograd.functional.jacobian(lambda x:parameterization_loss('R',x,x,gt,gt,torch.zeros(2),valid)[0],torch.zeros_like(gt))).all()

def test_boundary_metric_ignores_invalid_holes_on_a_constant_plane():
    from scripts.train_parameterization import boundary_mask
    gt=torch.zeros(1,3,8,8);gt[:,2]=1
    valid=torch.ones(1,8,8,dtype=torch.bool);valid[:,3,3]=False
    gt[:,:,3,3]=0
    assert not boundary_mask(gt,valid).any()
    gt[:,2,:,4:]=0;gt[:,0,:,4:]=1
    band=boundary_mask(gt,valid)
    assert band.any() and not band[:,3,3].any()
