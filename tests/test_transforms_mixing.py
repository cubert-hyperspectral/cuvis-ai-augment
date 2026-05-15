"""Unit tests for the v0.2.0 mixing/erasing transform."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_augment.transforms.mixing import Cutout


def _rng(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestCutout:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube(height=16, width=16, channels=4)
        out_cube, out_mask = Cutout(patch_size=(4, 4), prob=1.0)(cube, mask, _rng())
        assert out_cube.shape == cube.shape
        assert out_cube.dtype == cube.dtype
        assert out_mask is not None
        assert out_mask.shape == mask.shape
        assert out_mask.dtype == mask.dtype

    def test_prob_zero_identity(self, make_cube):
        cube, mask = make_cube()
        out_cube, out_mask = Cutout(patch_size=(4, 4), prob=0.0)(cube, mask, _rng())
        assert torch.equal(out_cube, cube)
        assert torch.equal(out_mask, mask)

    def test_int_patch_size(self, make_cube):
        cube, mask = make_cube(height=16, width=16, channels=4)
        out_cube, _ = Cutout(patch_size=4, prob=1.0)(cube, mask, _rng())
        # Total zero pixels per sample = 4 * 4 = 16 spatial * 4 channels = 64.
        # (cube is rand-initialised > 0, so any zero is from the patch.)
        zeros_per_sample = (out_cube == 0.0).sum(dim=(1, 2, 3))
        assert all(z.item() == 64 for z in zeros_per_sample)

    def test_patch_zeroed_in_cube_and_mask(self, make_cube):
        """Marker-pixel check: cube zero region and mask fill region must align."""
        cube, mask = make_cube(batch_size=1, height=16, width=16, channels=4)
        cube = cube + 1.0  # ensure no preexisting zeros
        mask = mask + 10  # ensure mask_fill_value=0 stands out
        out_cube, out_mask = Cutout(
            patch_size=(5, 5), cube_fill_value=0.0, mask_fill_value=0, prob=1.0
        )(cube, mask, _rng(seed=7))
        # Sum over channels gives a (H, W) zero-detector for the cube.
        cube_zero_mask = out_cube[0].sum(dim=-1) == 0.0
        mask_zero_mask = out_mask[0] == 0
        # The two regions must occupy the same rectangle.
        assert torch.equal(cube_zero_mask, mask_zero_mask)
        # ...and the rectangle must be 5×5.
        assert int(cube_zero_mask.sum().item()) == 25

    def test_custom_mask_fill_value(self, make_cube):
        cube, mask = make_cube(batch_size=1, channels=4)
        # Use a distinct ignore label.
        mask = mask + 10
        _, out_mask = Cutout(patch_size=(3, 3), mask_fill_value=-1, prob=1.0)(
            cube, mask, _rng(seed=0)
        )
        assert out_mask is not None
        # At least one pixel of the cutout must carry the ignore label.
        assert (out_mask == -1).any()

    def test_per_sample_independence(self, make_cube):
        """At prob=0.5, some samples in the batch should be touched, others identical."""
        cube, mask = make_cube(batch_size=32, height=16, width=16, channels=4)
        out_cube, out_mask = Cutout(patch_size=(4, 4), prob=0.5)(cube, mask, _rng(seed=0))
        equal_orig = (out_cube == cube).flatten(1).all(dim=1)
        assert equal_orig.any() and (~equal_orig).any()
        # Mask must move in lockstep — if the cube was untouched, the mask must be too.
        for b in range(32):
            if bool(equal_orig[b]):
                assert torch.equal(out_mask[b], mask[b])

    def test_determinism(self, make_cube):
        cube, mask = make_cube()
        t = Cutout(patch_size=(4, 4), prob=0.7)
        o1 = t(cube, mask, _rng(seed=11))
        o2 = t(cube, mask, _rng(seed=11))
        assert torch.equal(o1[0], o2[0])
        assert torch.equal(o1[1], o2[1])

    def test_patch_larger_than_cube_raises(self, make_cube):
        cube, mask = make_cube(height=8, width=8)
        with pytest.raises(ValueError, match="exceeds cube spatial dims"):
            Cutout(patch_size=(16, 16), prob=1.0)(cube, mask, _rng())

    def test_bad_patch_size_rejected(self):
        with pytest.raises(ValueError, match="patch_size must be int or a pair"):
            Cutout(patch_size=[4, 4, 4])  # wrong arity
        with pytest.raises(ValueError, match="patch_size entries must be positive"):
            Cutout(patch_size=(0, 4))
        with pytest.raises(ValueError, match="patch_size entries must be positive"):
            Cutout(patch_size=(4, -1))

    def test_mask_none_works(self, make_cube):
        cube, _ = make_cube()
        out_cube, out_mask = Cutout(patch_size=(4, 4), prob=1.0)(cube, None, _rng())
        assert out_cube.shape == cube.shape
        assert out_mask is None

    def test_patch_exact_size_of_cube(self, make_cube):
        """patch_size == cube spatial dims must work and zero everything."""
        cube, mask = make_cube(height=8, width=8, channels=4)
        out_cube, out_mask = Cutout(patch_size=(8, 8), prob=1.0)(cube, mask, _rng())
        assert (out_cube == 0.0).all()
        assert (out_mask == 0).all()
