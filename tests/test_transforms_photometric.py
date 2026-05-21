"""Unit tests for the v0.2.0 photometric transform."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_augment.transforms.photometric import MultiplicativeIlluminationScaling


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestMultiplicativeIlluminationScaling:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube(channels=16)
        out_cube, out_mask = MultiplicativeIlluminationScaling(prob=1.0)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert torch.equal(out_mask, mask), "mask must pass through untouched"

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, _ = MultiplicativeIlluminationScaling(prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)

    def test_prob_one_produces_change(self, make_cube):
        cube, _ = make_cube()
        out, _ = MultiplicativeIlluminationScaling(prob=1.0)(cube, None, _rng(seed=5))
        assert not torch.equal(out, cube)

    def test_gain_in_range(self, make_cube):
        """Output ratio out/cube must stay inside the gain range (modulo float jitter)."""
        cube, _ = make_cube(batch_size=4, channels=20)
        # Avoid division-by-zero by lifting cube above 0.
        cube = cube + 1.0
        gain_range = (0.8, 1.2)
        out, _ = MultiplicativeIlluminationScaling(gain_range=gain_range, smoothness=4, prob=1.0)(
            cube, None, _rng(0)
        )
        ratio = (out / cube).flatten()
        # Allow a tiny float-precision tolerance.
        assert ratio.min().item() >= gain_range[0] - 1e-5
        assert ratio.max().item() <= gain_range[1] + 1e-5

    def test_gain_curve_smooth(self, make_cube):
        """With smoothness=2 (linear ramp), neighbouring bands should change by tiny
        increments — assert max absolute first-difference is bounded by (hi - lo) / C."""
        B, C = 1, 32
        cube = torch.ones((B, 4, 4, C), dtype=torch.float32)
        out, _ = MultiplicativeIlluminationScaling(gain_range=(0.5, 1.5), smoothness=2, prob=1.0)(
            cube, None, _rng(0)
        )
        gain = out[0, 0, 0, :]  # (C,) — input is all ones, so output == gain curve
        diffs = (gain[1:] - gain[:-1]).abs()
        # Linear ramp: each step ≈ (g_C - g_0) / (C - 1) ≤ 1.0 / (C - 1) ≈ 0.032.
        assert diffs.max().item() < 0.05

    def test_determinism(self, make_cube):
        cube, _ = make_cube(channels=10)
        t = MultiplicativeIlluminationScaling(prob=0.7)
        o1 = t(cube, None, _rng(seed=3))
        o2 = t(cube, None, _rng(seed=3))
        assert torch.equal(o1[0], o2[0])

    def test_per_sample_independence(self, make_cube):
        cube, _ = make_cube(batch_size=32, channels=10)
        out, _ = MultiplicativeIlluminationScaling(prob=0.5)(cube, None, _rng(seed=0))
        equal_orig = (out == cube).flatten(1).all(dim=1)
        assert equal_orig.any() and (~equal_orig).any()

    def test_bad_gain_range_rejected(self):
        with pytest.raises(ValueError, match="gain_range must be"):
            MultiplicativeIlluminationScaling(gain_range=(0.0, 1.0))  # lo == 0
        with pytest.raises(ValueError, match="gain_range must be"):
            MultiplicativeIlluminationScaling(gain_range=(1.5, 0.5))  # lo > hi

    def test_bad_smoothness_rejected(self):
        with pytest.raises(ValueError, match="smoothness must be >= 2"):
            MultiplicativeIlluminationScaling(smoothness=1)
