"""Unit tests for the four V1 spatial transforms.

Tested directly as plain Python objects (no Node wrapping). Each transform is checked for:
* shape + dtype preservation,
* `prob=0` identity,
* `prob=1` matches a deterministic reference op,
* per-sample independence,
* paired mask correctness via marker pixels.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_augment.transforms.spatial import (
    Random90Rotate,
    RandomHorizontalFlip,
    RandomSpatialCrop,
    RandomVerticalFlip,
)


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestRandomHorizontalFlip:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = RandomHorizontalFlip(prob=0.5)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert out_mask is not None
        assert out_mask.shape == mask.shape
        assert out_mask.dtype == mask.dtype

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = RandomHorizontalFlip(prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)
        assert torch.equal(out_mask, mask)

    def test_prob_one_matches_torch_flip(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = RandomHorizontalFlip(prob=1.0)(cube, mask, _rng())
        assert torch.equal(out_cube, torch.flip(cube, dims=(2,)))
        assert torch.equal(out_mask, torch.flip(mask, dims=(2,)))

    def test_determinism(self, make_cube):
        cube, mask = make_cube()
        t = RandomHorizontalFlip(prob=0.5)
        out1 = t(cube, mask, _rng(seed=123))
        out2 = t(cube, mask, _rng(seed=123))
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_paired_mask_via_marker(self, make_cube):
        cube, mask = make_cube(batch_size=1, height=8, width=8, channels=4)
        cube[0, 2, 6, :] = 999.0
        mask[0, 2, 6] = 7
        out_cube, out_mask = RandomHorizontalFlip(prob=1.0)(cube, mask, _rng())
        # Width axis (W=8) is flipped → original (2, 6) lands at (2, 1).
        assert out_cube[0, 2, 1, 0].item() == pytest.approx(999.0)
        assert out_mask[0, 2, 1].item() == 7

    def test_per_sample_independence(self, make_cube):
        # Large batch; at prob=0.5 we should see a mix of flipped vs not.
        cube, mask = make_cube(batch_size=64)
        out_cube, _ = RandomHorizontalFlip(prob=0.5)(cube, mask, _rng(seed=0))
        flipped = torch.flip(cube, dims=(2,))
        # Per-sample equality with original vs flipped.
        equal_orig = (out_cube == cube).flatten(1).all(dim=1)
        equal_flip = (out_cube == flipped).flatten(1).all(dim=1)
        # Every sample is either original or flipped (no garbage).
        assert (equal_orig | equal_flip).all()
        # At least some of each — independence check.
        assert equal_orig.any() and equal_flip.any()


class TestRandomVerticalFlip:
    def test_prob_one_matches_torch_flip(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = RandomVerticalFlip(prob=1.0)(cube, mask, _rng())
        assert torch.equal(out_cube, torch.flip(cube, dims=(1,)))
        assert torch.equal(out_mask, torch.flip(mask, dims=(1,)))

    def test_paired_mask_via_marker(self, make_cube):
        cube, mask = make_cube(batch_size=1, height=8, width=8, channels=4)
        cube[0, 1, 3, :] = 999.0
        mask[0, 1, 3] = 5
        out_cube, out_mask = RandomVerticalFlip(prob=1.0)(cube, mask, _rng())
        # Height axis (H=8) is flipped → original (1, 3) lands at (6, 3).
        assert out_cube[0, 6, 3, 0].item() == pytest.approx(999.0)
        assert out_mask[0, 6, 3].item() == 5

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = RandomVerticalFlip(prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)
        assert torch.equal(out_mask, mask)


class TestRandom90Rotate:
    def test_shape_dtype_preserved_square(self, make_cube):
        cube, mask = make_cube(height=8, width=8)
        out_cube, out_mask = Random90Rotate(prob=0.5)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert out_mask.dtype == mask.dtype

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube(height=8, width=8)
        out_cube, out_mask = Random90Rotate(prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)
        assert torch.equal(out_mask, mask)

    def test_paired_mask_via_marker(self, make_cube):
        # Single-sample batch; force the sample to be rotated by choosing prob=1 and a
        # seed that yields a specific k. Run, then verify the marker tracks.
        cube, mask = make_cube(batch_size=1, height=8, width=8, channels=4)
        cube[0, 2, 5, :] = 999.0
        mask[0, 2, 5] = 3
        rng = _rng(seed=0)
        out_cube, out_mask = Random90Rotate(prob=1.0)(cube, mask, rng)
        # Find the marker in the output.
        match = (out_cube[0, :, :, 0] == 999.0).nonzero(as_tuple=False)
        assert match.shape == (1, 2), "Marker pixel should appear exactly once."
        h_out, w_out = int(match[0, 0]), int(match[0, 1])
        assert int(out_mask[0, h_out, w_out].item()) == 3

    def test_non_square_with_odd_k_raises(self, make_cube):
        cube, mask = make_cube(height=8, width=16)
        with pytest.raises(ValueError, match="square spatial dims"):
            Random90Rotate(prob=1.0)(cube, mask, _rng(seed=0))

    def test_non_square_with_even_k_works(self, make_cube):
        """k=2 (180°) preserves H/W layout, so it must work on non-square cubes —
        the square requirement only applies when k is odd. This pins the negative
        assertion so a future tightening of the check fails the build."""
        # Force k=2 by directly patching torch.randint via a fixed-output rng.
        # We can't easily force k from outside, so instead we verify many seeds
        # against a non-square cube and confirm every successful run that survives
        # produces output of the same shape as input — i.e. either k=2 succeeds
        # silently or the implementation raised because the seed picked k∈{1,3}.
        cube, mask = make_cube(height=8, width=16)
        successes = 0
        for seed in range(20):
            try:
                out_cube, out_mask = Random90Rotate(prob=1.0)(cube, mask, _rng(seed=seed))
            except ValueError:
                continue
            assert out_cube.shape == cube.shape
            assert out_mask.shape == mask.shape
            successes += 1
        # k is uniform over {1, 2, 3}; over 20 seeds we'd expect ~6-7 even k draws.
        assert successes >= 1, "Expected at least one even-k rotation to succeed."


class TestRandomSpatialCrop:
    def test_output_shape(self, make_cube):
        cube, mask = make_cube(height=32, width=32, channels=4)
        t = RandomSpatialCrop(size=(16, 12), prob=1.0)
        out_cube, out_mask = t(cube, mask, _rng())
        assert out_cube.shape == (cube.shape[0], 16, 12, 4)
        assert out_mask.shape == (cube.shape[0], 16, 12)

    def test_prob_zero_centre_crop(self, make_cube):
        cube, mask = make_cube(batch_size=1, height=8, width=8, channels=4)
        # max_top = max_left = 4; centre = 2.
        t = RandomSpatialCrop(size=(4, 4), prob=0.0)
        out_cube, out_mask = t(cube, mask, _rng())
        assert torch.equal(out_cube, cube[:, 2:6, 2:6, :])
        assert torch.equal(out_mask, mask[:, 2:6, 2:6])

    def test_paired_mask_window(self, make_cube):
        cube, mask = make_cube(batch_size=4, height=32, width=32, channels=4)
        # Place markers across the batch.
        for b in range(4):
            cube[b, 10, 10, :] = float(100 + b)
            mask[b, 10, 10] = 100 + b
        out_cube, out_mask = RandomSpatialCrop(size=(16, 16), prob=1.0)(cube, mask, _rng(7))
        for b in range(4):
            # If the marker survived the crop, both cube and mask must agree.
            match = (out_cube[b, :, :, 0] == float(100 + b)).nonzero(as_tuple=False)
            if match.numel() > 0:
                h_out, w_out = int(match[0, 0]), int(match[0, 1])
                assert int(out_mask[b, h_out, w_out].item()) == 100 + b

    def test_crop_larger_than_cube_raises(self, make_cube):
        cube, mask = make_cube(height=8, width=8)
        with pytest.raises(ValueError, match="exceeds cube spatial dims"):
            RandomSpatialCrop(size=(16, 16))(cube, mask, _rng())

    def test_bad_size_param_raises(self):
        # Wrong arity.
        with pytest.raises(ValueError, match="size must be a pair"):
            RandomSpatialCrop(size=[16])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="size must be a pair"):
            RandomSpatialCrop(size=[16, 16, 16])  # type: ignore[arg-type]
        # Non-positive entry.
        with pytest.raises(ValueError, match="size must be a pair"):
            RandomSpatialCrop(size=(0, 16))
        with pytest.raises(ValueError, match="size must be a pair"):
            RandomSpatialCrop(size=(16, -1))

    def test_crop_equal_to_cube_dims(self, make_cube):
        """When size == cube spatial dims, max_top and max_left are 0 — the
        zero-margin branch in RandomSpatialCrop. Output must equal input regardless
        of prob (no offset to randomise)."""
        cube, mask = make_cube(batch_size=2, height=8, width=8, channels=4)
        for prob in (0.0, 1.0):
            t = RandomSpatialCrop(size=(8, 8), prob=prob)
            out_cube, out_mask = t(cube, mask, _rng())
            assert torch.equal(out_cube, cube)
            assert torch.equal(out_mask, mask)


class TestStatisticalApply:
    """Over many calls with prob=0.5, the fraction applied should be ~0.5."""

    def test_horizontal_flip_application_rate(self, make_cube):
        cube, _ = make_cube(batch_size=1)
        flipped_ref = torch.flip(cube, dims=(2,))
        n_trials = 1000
        n_applied = 0
        for s in range(n_trials):
            out, _ = RandomHorizontalFlip(prob=0.5)(cube, None, _rng(seed=s))
            if torch.equal(out, flipped_ref):
                n_applied += 1
        fraction = n_applied / n_trials
        assert 0.45 <= fraction <= 0.55, f"Application rate {fraction:.3f} out of bounds."
