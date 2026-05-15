"""Unit tests for the v0.2.0 spectral transforms.

Each transform is checked for:
* shape + dtype preservation,
* ``prob=0`` identity,
* ``prob=1`` produces a measurable change,
* per-sample independence,
* mask passthrough (spectral transforms must not move pixels),
* determinism under a fixed seed.
"""

from __future__ import annotations

import torch

from cuvis_ai_augment.transforms.spectral import GaussianBandNoise, RandomBandDropout


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ============================================================== GaussianBandNoise


class TestGaussianBandNoise:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube(channels=12)
        out_cube, out_mask = GaussianBandNoise(sigma=0.05, prob=1.0)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert out_mask is not None
        assert torch.equal(out_mask, mask), "mask must pass through untouched"

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = GaussianBandNoise(sigma=0.1, prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)
        assert torch.equal(out_mask, mask)

    def test_sigma_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, _ = GaussianBandNoise(sigma=0.0, prob=1.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)

    def test_prob_one_produces_change(self, make_cube):
        cube, mask = make_cube()
        out_cube, _ = GaussianBandNoise(sigma=0.1, prob=1.0)(cube, mask, _rng(seed=42))
        assert not torch.equal(out_cube, cube)
        # All pixels should differ when sigma > 0.
        assert (out_cube != cube).float().mean().item() > 0.99

    def test_noise_magnitude_matches_sigma(self, make_cube):
        """Per-pixel residual std should be ≈ sigma (large batch for stability)."""
        cube, _ = make_cube(batch_size=4, height=32, width=32, channels=16)
        sigma = 0.05
        out_cube, _ = GaussianBandNoise(sigma=sigma, prob=1.0)(cube, None, _rng(0))
        residual = (out_cube - cube).std().item()
        assert 0.04 < residual < 0.06, f"residual std {residual:.4f} far from {sigma}"

    def test_per_band_scale_increases_in_high_var_band(self):
        """With per_band_scale=True, a band with larger std should get larger noise."""
        # Construct a cube where band 0 has std≈1.0 and band 1 has std≈0.01.
        B, H, W = 2, 32, 32
        g = _rng(0)
        cube = torch.zeros((B, H, W, 2), dtype=torch.float32)
        cube[..., 0] = torch.randn((B, H, W), generator=g)  # std ≈ 1
        cube[..., 1] = torch.randn((B, H, W), generator=g) * 0.01  # std ≈ 0.01
        out, _ = GaussianBandNoise(sigma=0.1, per_band_scale=True, prob=1.0)(
            cube, None, _rng(seed=99)
        )
        std_band_0 = (out[..., 0] - cube[..., 0]).std().item()
        std_band_1 = (out[..., 1] - cube[..., 1]).std().item()
        assert std_band_0 > 10 * std_band_1, (
            f"per_band_scale didn't scale: band0 noise std={std_band_0:.4f}, "
            f"band1 noise std={std_band_1:.4f}"
        )

    def test_determinism(self, make_cube):
        cube, mask = make_cube()
        t = GaussianBandNoise(sigma=0.05, prob=0.5)
        out1 = t(cube, mask, _rng(seed=7))
        out2 = t(cube, mask, _rng(seed=7))
        assert torch.equal(out1[0], out2[0])

    def test_per_sample_independence(self, make_cube):
        """At prob=0.5 over a large batch, some samples are noised, others identical."""
        cube, _ = make_cube(batch_size=64, channels=8)
        out, _ = GaussianBandNoise(sigma=0.1, prob=0.5)(cube, None, _rng(seed=0))
        equal_orig = (out == cube).flatten(1).all(dim=1)
        # At least some samples touched, at least some untouched.
        assert equal_orig.any() and (~equal_orig).any()

    def test_negative_sigma_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="sigma must be non-negative"):
            GaussianBandNoise(sigma=-0.1)


# ============================================================== RandomBandDropout


class TestRandomBandDropout:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube(channels=20)
        out_cube, out_mask = RandomBandDropout(drop_fraction=0.2, prob=1.0)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert torch.equal(out_mask, mask), "mask must pass through untouched"

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, _ = RandomBandDropout(drop_fraction=0.5, prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)

    def test_drop_fraction_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, _ = RandomBandDropout(drop_fraction=0.0, prob=1.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)

    def test_correct_number_of_bands_dropped(self, make_cube):
        """With drop_fraction=0.25 on C=20, expect exactly 5 zero bands per sample."""
        cube, _ = make_cube(batch_size=4, channels=20)
        # Ensure no band is already all-zero by adding 1.
        cube = cube + 1.0
        out, _ = RandomBandDropout(drop_fraction=0.25, prob=1.0)(cube, None, _rng(0))
        # A band is "dropped" iff every spatial pixel is exactly 0.
        for b in range(out.shape[0]):
            dropped = (out[b].abs().sum(dim=(0, 1)) == 0).sum().item()
            assert dropped == 5, f"sample {b}: expected 5 dropped, got {dropped}"

    def test_mask_unchanged(self, make_cube):
        cube, mask = make_cube(channels=10)
        marker_val = 99
        mask[0, 0, 0] = marker_val
        _, out_mask = RandomBandDropout(drop_fraction=0.5, prob=1.0)(cube, mask, _rng())
        assert out_mask is not None
        assert int(out_mask[0, 0, 0].item()) == marker_val

    def test_determinism(self, make_cube):
        cube, _ = make_cube(channels=10)
        t = RandomBandDropout(drop_fraction=0.3, prob=0.7)
        o1 = t(cube, None, _rng(seed=11))
        o2 = t(cube, None, _rng(seed=11))
        assert torch.equal(o1[0], o2[0])

    def test_different_samples_drop_different_bands(self, make_cube):
        """Per-sample independence — two samples in the same call shouldn't drop the
        same band set with high probability."""
        cube, _ = make_cube(batch_size=2, channels=10)
        cube = cube + 1.0
        out, _ = RandomBandDropout(drop_fraction=0.3, prob=1.0)(cube, None, _rng(0))
        dropped_b0 = (out[0].abs().sum(dim=(0, 1)) == 0).nonzero(as_tuple=True)[0].tolist()
        dropped_b1 = (out[1].abs().sum(dim=(0, 1)) == 0).nonzero(as_tuple=True)[0].tolist()
        assert dropped_b0 != dropped_b1, "Both samples dropped the same bands."

    def test_bad_drop_fraction_rejected(self):
        import pytest

        with pytest.raises(ValueError, match=r"drop_fraction must be in \[0, 1\]"):
            RandomBandDropout(drop_fraction=1.5)
        with pytest.raises(ValueError, match=r"drop_fraction must be in \[0, 1\]"):
            RandomBandDropout(drop_fraction=-0.1)
