import pytest
import torch
from src.training_diagnostics import TimeLossBins
from src.parameterization import parameterization_loss


def test_time_bins_weight_images_and_reconstruct_objective():
    gt=torch.zeros(2,3,1,1);prediction=torch.ones_like(gt)
    loss,info=parameterization_loss('Xn',prediction,prediction,gt,gt,torch.tensor([.1,.8]),torch.ones(2,1,1,dtype=torch.bool),return_per_image=True)
    bins=TimeLossBins([0,.5,.95]);bins.add([.1,.8],info['per_image'])
    rows=bins.summary()
    assert [x['samples'] for x in rows]==[1,1]
    assert sum(x['loss'] for x in rows)/2==pytest.approx(float(loss))
    with pytest.raises(ValueError):bins.add([.95],info['per_image'])


def test_formal_budget_gate_cannot_be_bypassed_by_step_override():
    from scripts.train_parameterization import require_formal_budget
    for cfg in [{},{'formal_steps':5000},{'formal_steps':None,'formal_budget_approved':True}]:
        with pytest.raises(RuntimeError):require_formal_budget(cfg,'formal')
    require_formal_budget({},'convergence')
    require_formal_budget({'formal_steps':30000,'formal_budget_approved':True},'formal')


def test_preregistered_continuation_holds_original_cosine_floor():
    from src.training_diagnostics import learning_rate_factor
    assert learning_rate_factor(30000,30000)==pytest.approx(.1)
    assert learning_rate_factor(30001,30000)==pytest.approx(.1)
    assert learning_rate_factor(50000,30000)==pytest.approx(.1)
    assert learning_rate_factor(15000,30000)==pytest.approx(.55)


def test_interrupted_continuation_cannot_be_relabelled_as_initial_endpoint():
    from src.training_diagnostics import resume_segment_action
    assert resume_segment_action(35000,30000,'convergence',None,[{'update':30000}])=='skip_initial'
    assert resume_segment_action(30000,30000,'convergence',None,[{'update':30000}])=='run'
    assert resume_segment_action(35000,50000,'convergence',50000,[{'update':30000}])=='run'
    with pytest.raises(RuntimeError):resume_segment_action(35000,30000,'formal',None,[])
