"""Four predeclared targets on the same linear FM path and image-macro mask."""
import torch
import torch.nn.functional as F

def parameterization_loss(mode,prediction,state,target,noise,times,valid,unit_weight=.25,cosine_weight=.25):
    if mode not in {'R','V','Xv','Xn'}:raise ValueError('Unknown target parameterization')
    t=times.float().view(-1,1,1,1)
    p=prediction.float();gt=target.float();mask=valid.bool()
    if mask.ndim==4:mask=mask[:,0]
    counts=mask.sum((1,2))
    if (counts==0).any():raise ValueError('Empty sample mask')
    if not torch.isfinite(p.permute(0,2,3,1)[mask]).all():raise ValueError('Nonfinite valid output')
    if mode=='V':
        error=p-(gt-noise.float());endpoint=state.float()+(1-t)*p
    else:
        endpoint=p;error=p-gt
        if mode=='Xv':
            if (t>=1).any():raise ValueError('Xv requires t < 1')
            error=error/(1-t)
    error=torch.where(mask[:,None],error,0)
    mse=(error.square().sum((1,2,3))/(3*counts)).mean()
    unit=(((endpoint.norm(dim=1)-1).square()*mask).sum((1,2))/counts).mean()
    cosine=((1-F.cosine_similarity(endpoint,gt,dim=1,eps=1e-4))*mask).sum((1,2))/counts
    total=mse+unit_weight*unit+cosine_weight*cosine.mean()
    return total,{'mse':float(mse.detach()),'unit':float(unit.detach()),'cosine':float(cosine.mean().detach())}

def one_step_endpoint(mode,output,noise):
    return noise+output if mode=='V' else output
