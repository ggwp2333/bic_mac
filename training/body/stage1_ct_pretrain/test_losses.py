import unittest

import torch

from losses import (
    blended_umap_mae_loss,
    tissue_class_map,
    tissue_weight_map,
    tissue_weighted_umap_mae_loss,
    umap_mae_loss,
    validate_tissue_loss_config,
)


DEFAULT_CONFIG = {
    "enabled": True,
    "blend": 0.5,
    "lung_air_max_hu": -500,
    "bone_min_hu": 300,
    "lung_air_weight": 1.5,
    "soft_tissue_weight": 1.0,
    "bone_weight": 2.0,
    "interface_multiplier": 1.5,
}


def normalized_hu(values, device="cpu"):
    return (torch.as_tensor(values, dtype=torch.float32, device=device) + 1000) / 3000


class TissueWeightedLossTests(unittest.TestCase):
    def test_threshold_assignments_are_exact(self):
        target = normalized_hu([-501, -500, 299, 300]).reshape(1, 1, 1, 1, 4)
        classes = tissue_class_map(target, -500, 300)
        self.assertEqual(classes.flatten().tolist(), [0, 1, 1, 2])

    def test_bone_error_contributes_more_than_equal_soft_tissue_error(self):
        target = normalized_hu([0, 500]).reshape(1, 1, 1, 1, 2)
        mask = torch.ones_like(target, dtype=torch.bool)
        config = {**DEFAULT_CONFIG, "interface_multiplier": 1.0}
        mu_error = 0.001

        soft_error = target.clone()
        soft_error[..., 0] += mu_error / 0.288
        bone_error = target.clone()
        bone_error[..., 1] += mu_error / 0.153

        plain_soft = umap_mae_loss(soft_error, target, mask)
        plain_bone = umap_mae_loss(bone_error, target, mask)
        weighted_soft = tissue_weighted_umap_mae_loss(
            soft_error, target, mask, config
        )
        weighted_bone = tissue_weighted_umap_mae_loss(
            bone_error, target, mask, config
        )

        self.assertTrue(torch.allclose(plain_soft, plain_bone, atol=1e-7))
        self.assertGreater(weighted_bone.item(), weighted_soft.item())

    def test_interface_error_gets_additional_multiplier(self):
        target = normalized_hu(torch.zeros(7, 7, 7)).reshape(1, 1, 7, 7, 7)
        target[..., 3, 3, 3] = normalized_hu(500)
        mask = torch.ones_like(target, dtype=torch.bool)
        delta = 0.001 / 0.288

        interface_error = target.clone()
        interface_error[..., 3, 3, 4] += delta
        interior_error = target.clone()
        interior_error[..., 0, 0, 0] += delta

        interface_loss = tissue_weighted_umap_mae_loss(
            interface_error, target, mask, DEFAULT_CONFIG
        )
        interior_loss = tissue_weighted_umap_mae_loss(
            interior_error, target, mask, DEFAULT_CONFIG
        )
        self.assertAlmostEqual(
            interface_loss.item() / interior_loss.item(), 1.5, places=4
        )

    def test_changes_outside_prediction_mask_have_zero_effect(self):
        target_a = normalized_hu(torch.zeros(3, 3, 3)).reshape(1, 1, 3, 3, 3)
        target_b = normalized_hu(torch.full((3, 3, 3), 1000)).reshape(
            1, 1, 3, 3, 3
        )
        target_b[..., 1, 1, 1] = target_a[..., 1, 1, 1]
        mask = torch.zeros_like(target_a, dtype=torch.bool)
        mask[..., 1, 1, 1] = True

        pred_a = target_a.clone()
        pred_b = normalized_hu(torch.full((3, 3, 3), -1000)).reshape(
            1, 1, 3, 3, 3
        )
        pred_a[..., 1, 1, 1] += 0.01
        pred_b[..., 1, 1, 1] = pred_a[..., 1, 1, 1]

        loss_a = tissue_weighted_umap_mae_loss(
            pred_a, target_a, mask, DEFAULT_CONFIG
        )
        loss_b = tissue_weighted_umap_mae_loss(
            pred_b, target_b, mask, DEFAULT_CONFIG
        )
        self.assertTrue(torch.equal(loss_a, loss_b))

    def test_unit_weights_equal_plain_mu_mae(self):
        torch.manual_seed(7)
        target = torch.rand(1, 1, 5, 5, 5)
        pred = torch.rand_like(target)
        mask = torch.rand_like(target) > 0.25
        config = {
            **DEFAULT_CONFIG,
            "lung_air_weight": 1.0,
            "soft_tissue_weight": 1.0,
            "bone_weight": 1.0,
            "interface_multiplier": 1.0,
        }
        plain = umap_mae_loss(pred, target, mask)
        weighted = tissue_weighted_umap_mae_loss(pred, target, mask, config)
        self.assertTrue(torch.equal(plain, weighted))

    def test_mixed_precision_backward_is_finite(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        target = torch.rand(1, 1, 6, 6, 6, device=device)
        mask = torch.ones_like(target, dtype=torch.bool)
        pred = torch.rand_like(target, requires_grad=True)

        with torch.autocast(device_type=device, dtype=dtype):
            total, plain, weighted = blended_umap_mae_loss(
                pred, target, mask, DEFAULT_CONFIG
            )
        total.backward()

        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(plain))
        self.assertTrue(torch.isfinite(weighted))
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_weight_map_labels_bone_and_interfaces(self):
        target = normalized_hu(torch.zeros(5, 5, 5)).reshape(1, 1, 5, 5, 5)
        target[..., 2, 2, 2] = normalized_hu(500)
        weights, interfaces = tissue_weight_map(target, DEFAULT_CONFIG)
        self.assertEqual(weights[..., 2, 2, 2].item(), 3.0)
        self.assertTrue(interfaces[..., 2, 2, 1].item())
        self.assertEqual(weights[..., 0, 0, 0].item(), 1.0)

    def test_invalid_configuration_is_rejected(self):
        invalid_cases = (
            {**DEFAULT_CONFIG, "blend": -0.1},
            {**DEFAULT_CONFIG, "blend": 1.1},
            {**DEFAULT_CONFIG, "lung_air_max_hu": 300},
            {**DEFAULT_CONFIG, "bone_weight": 0},
        )
        for config in invalid_cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_tissue_loss_config(config)


if __name__ == "__main__":
    unittest.main()
