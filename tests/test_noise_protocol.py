import numpy as np
import pytest
import torch
from src.noise_protocol import image_noise_seed, mae_seed_statistics, directional_sensitivity
from src.flow_inference import infer


def test_seed_mapping_is_order_independent_and_changes_with_image_and_replicate():
    # Fixture independently checked against .NET SHA256; changing it requires a protocol version.
    assert image_noise_seed(42, "rgb/a.png") == 1427854216745095732
    keys = ["rgb/a.png", "rgb/b.png", "图像/c.png"]
    forward = {key: image_noise_seed(42, key) for key in keys}
    reverse = {key: image_noise_seed(42, key) for key in reversed(keys)}
    assert forward == reverse
    assert len(set(forward.values())) == len(keys)
    assert image_noise_seed(42, keys[0]) != image_noise_seed(43, keys[0])
    assert all(0 <= value < 2 ** 63 for value in forward.values())


@pytest.mark.parametrize("key", ["/absolute.png", "../image.png", "rgb\\image.png", "a//b.png"])
def test_noncanonical_image_keys_are_rejected(key):
    with pytest.raises(ValueError):
        image_noise_seed(42, key)


def test_image_noise_pairs_across_modes_without_touching_global_rng():
    class ZeroVelocity(torch.nn.Module):
        def forward(self, z, t, condition):
            return torch.zeros_like(z)
    state = torch.random.get_rng_state().clone()
    condition = torch.zeros(1, 3, 2, 2)
    seed = image_noise_seed(42, "rgb/a.png")
    a = infer(ZeroVelocity(), condition, mode="no_cfg", seed=seed, amp=False)
    b = infer(ZeroVelocity(), condition, mode="fixed_cfg", seed=seed, amp=False)
    c = infer(ZeroVelocity(), condition, mode="no_cfg",
              seed=image_noise_seed(42, "rgb/b.png"), amp=False)
    assert torch.equal(a["raw"], b["raw"])
    assert not torch.equal(a["raw"], c["raw"])
    assert torch.equal(state, torch.random.get_rng_state())


def test_dataset_sd_can_vanish_while_image_sensitivity_stays_constant():
    shared = np.array([[9., 11.]] * 4)
    independent_balanced = np.array([[9., 11.], [11., 9.], [9., 11.], [11., 9.]])
    a, b = mae_seed_statistics(shared), mae_seed_statistics(independent_balanced)
    assert a["mean_mae_deg"] == b["mean_mae_deg"] == 10
    assert a["dataset_mae_sd_ddof1_deg"] == pytest.approx(np.sqrt(2))
    assert b["dataset_mae_sd_ddof1_deg"] == 0
    assert a["mean_image_mae_sd_ddof1_deg"] == b["mean_image_mae_sd_ddof1_deg"]


def test_directional_sensitivity_has_known_orthogonal_geometry():
    samples = np.array([[[[1., 0., 0.]]], [[[0., 1., 0.]]]])
    result = directional_sensitivity(samples, np.ones((1, 1), dtype=bool))
    assert result["mean_pixel_angular_dispersion_deg"] == pytest.approx(45)
    assert result["mean_pixel_directional_variance_1_minus_R"] == pytest.approx(1 - 1 / np.sqrt(2))


def test_opposite_directions_are_explicitly_undefined_not_silently_omitted():
    samples = np.array([[[[1., 0., 0.]]], [[[-1., 0., 0.]]]])
    result = directional_sensitivity(samples, np.ones((1, 1), dtype=bool))
    assert result["undefined_mean_direction_pixels"] == 1
    assert result["mean_pixel_angular_dispersion_deg"] is None
    assert result["mean_pixel_directional_variance_1_minus_R"] == 1
