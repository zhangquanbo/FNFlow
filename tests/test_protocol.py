import numpy as np
import pytest
import torch
from src.flow_inference import infer
from src.losses import geometry_loss, velocity_loss
from src.normal_geometry import resize_normals, mirror_normals, crop_resize_intrinsics
from src.research_io import output_path, BEST


class ConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, z, t, cond):
        self.calls += 1
        return torch.ones_like(z)


class ConditionalEndpoint(torch.nn.Module):
    def forward(self, z, t, cond):
        sign = torch.where(cond.mean((1, 2, 3)) > 0, 1.0, -1.0)
        endpoint = torch.zeros_like(z)
        endpoint[:, 0] = sign[:, None, None]
        return (endpoint - z) / (1 - t[:, None, None, None])


@pytest.mark.parametrize("steps", [1, 2, 4, 8])
def test_euler_integrates_exactly_one_unit(steps):
    model = ConstantVelocity()
    condition = torch.zeros(1, 3, 2, 2)
    initial = torch.randn(condition.shape, generator=torch.Generator().manual_seed(7))
    result = infer(model, condition, steps=steps, mode="no_cfg", seed=7, amp=False)
    assert torch.allclose(result["raw"], initial + 1, atol=1e-6)
    assert model.calls == steps


def test_local_seed_is_reproducible_without_resetting_global_rng():
    condition = torch.zeros(1, 3, 2, 2)
    state = torch.random.get_rng_state().clone()
    a = infer(ConstantVelocity(), condition, mode="no_cfg", seed=5)
    b = infer(ConstantVelocity(), condition, mode="no_cfg", seed=5)
    c = infer(ConstantVelocity(), condition, mode="no_cfg", seed=6)
    assert torch.equal(a["raw"], b["raw"])
    assert not torch.equal(a["raw"], c["raw"])
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.allclose(a["normal"].norm(dim=1), torch.ones(1, 2, 2))


def test_geom_cfg_opposing_endpoints_uses_unconditional_branch():
    result = infer(ConditionalEndpoint(), torch.ones(1, 3, 2, 2),
                   initial_noise=torch.zeros(1, 3, 2, 2), amp=False)
    assert torch.allclose(result["normal"][:, 0], -torch.ones(1, 2, 2))
    assert result["trace"][0]["zero_fraction"] == 1
    assert result["nfe"] == 2


def test_fixed_cfg_scale_one_equals_conditional_endpoint():
    condition = torch.ones(1, 3, 2, 2)
    plain = infer(ConditionalEndpoint(), condition, mode="no_cfg", amp=False)
    fixed = infer(ConditionalEndpoint(), condition, mode="fixed_cfg", scale=1, amp=False)
    assert torch.allclose(plain["raw"], fixed["raw"], atol=1e-6)


def test_invalid_nan_pixels_do_not_contribute_to_loss_or_gradient():
    pred = torch.tensor([[[[2., float("nan")]], [[0., float("nan")]], [[0., float("nan")]]]],
                        requires_grad=True)
    gt = torch.tensor([[[[1., float("nan")]], [[0., float("nan")]], [[0., float("nan")]]]])
    loss, info = geometry_loss(pred, gt, torch.tensor([[[True, False]]]))
    assert info["valid_pixels"] == 1
    assert info["unit"] == pytest.approx(1)
    assert float(loss) == pytest.approx(1 / 3 + 0.25)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.equal(pred.grad[..., 1], torch.zeros_like(pred.grad[..., 1]))


def test_all_invalid_batch_is_explicitly_skipped():
    shape = (1, 3, 2, 2)
    loss, info = velocity_loss(torch.zeros(shape), torch.zeros(shape), torch.zeros(shape),
                               torch.zeros(shape), torch.zeros(1), torch.zeros(1, 2, 2))
    assert loss is None and info["skip_batch"]


def test_normal_resize_does_not_mix_invalid_nan_values():
    normal = np.zeros((4, 4, 3), dtype=np.float32)
    normal[..., 2] = 1
    normal[0, 0] = np.nan
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    resized, mask = resize_normals(normal, valid, (8, 8))
    assert np.isfinite(resized).all()
    assert np.allclose(np.linalg.norm(resized[mask], axis=-1), 1)
    assert np.all(resized[~mask] == 0)


def test_mirror_twice_recovers_geometry():
    normal = np.arange(36, dtype=np.float32).reshape(3, 4, 3)
    mask = np.ones((3, 4), dtype=bool)
    mirrored, mirrored_mask = mirror_normals(normal, mask)
    recovered, recovered_mask = mirror_normals(mirrored, mirrored_mask)
    assert np.array_equal(normal, recovered)
    assert np.array_equal(mask, recovered_mask)
    assert np.array_equal(mirrored[..., 0], -normal[:, ::-1, 0])


def test_crop_resize_preserves_continuous_camera_ray():
    k = np.array([[500., 0, 320.], [0, 500., 240.], [0, 0, 1.]])
    new_k = crop_resize_intrinsics(k, (20, 30, 600, 400), (300, 200))
    point = np.array([120., 100., 1.])
    new_point = np.array([(120 - 20) * .5, (100 - 30) * .5, 1.])
    assert np.allclose(np.linalg.solve(k, point), np.linalg.solve(new_k, new_point))


def test_best_is_never_an_output_target():
    with pytest.raises(ValueError):
        output_path(BEST / "outputs/forbidden.json")
