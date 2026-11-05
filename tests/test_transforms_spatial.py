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
    RandomForegroundBiasedCrop,
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


class TestRandomForegroundBiasedCrop:
    def test_shape_dtype_preserved(self, make_cube):
        cube, mask = make_cube(height=16, width=16)
        out_cube, out_mask = RandomForegroundBiasedCrop(size=(8, 8))(cube, mask, _rng())
        assert out_cube.shape == (2, 8, 8, 8)
        assert out_cube.dtype == cube.dtype
        assert out_mask is not None
        assert out_mask.shape == (2, 8, 8)
        assert out_mask.dtype == mask.dtype

    def test_forced_last_sample_contains_foreground(self, make_cube):
        # Single fg pixel in a far corner: a centre crop misses it, the forced
        # sample's window is centered on it (clamped) and must contain it.
        cube, mask = make_cube(batch_size=3, height=16, width=16)
        mask.zero_()
        mask[:, 0, 0] = 5
        cube[:, 0, 0, :] = 999.0
        t = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=0.33, prob=0.0)
        out_cube, out_mask = t(cube, mask, _rng())
        # nnU-Net rule: 3 - round(3 * 0.67) == 1 -> exactly the last sample is forced.
        assert (out_mask[2] == 5).any()
        assert out_cube[2, 0, 0, 0].item() == pytest.approx(999.0)  # paired alignment
        assert not (out_mask[0] == 5).any()  # centre crops miss the corner
        assert not (out_mask[1] == 5).any()

    def test_forced_count_matches_nnunet_rounding(self, make_cube):
        # The forced count is nnU-Net's B - round(B * (1 - fg_percent)), NOT round(B * fg_percent):
        # at B=5, fg_percent=0.1 that is 5 - round(4.5) == 1 (banker's round -> 4), so the last
        # sample is forced -- whereas round(5 * 0.1) rounds to 0 and would force none.
        cube, mask = make_cube(batch_size=5, height=16, width=16)
        mask.zero_()
        mask[:, 0, 0] = 5
        t = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=0.1, prob=0.0)
        _, out_mask = t(cube, mask, _rng())
        assert (out_mask[4] == 5).any()  # last sample forced -> centred on the corner fg
        assert not any((out_mask[i] == 5).any() for i in range(4))  # first 4 are centre crops

    def test_class_balanced_center_choice(self, make_cube):
        # Rare class must be sampled as often as the common one (class-uniform,
        # not pixel-uniform): one pixel of class 2 vs many of class 1.
        cube, mask = make_cube(batch_size=1, height=16, width=16)
        mask.zero_()
        mask[0, :8, :] = 1
        mask[0, 15, 15] = 2
        t = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=1.0)
        hits = 0
        for seed in range(64):
            _, out_mask = t(cube, mask, _rng(seed))
            hits += bool((out_mask[0] == 2).any())
        assert hits > 8  # pixel-uniform would give ~1/129 * 64 < 1 expected hit

    def test_empty_mask_matches_random_spatial_crop(self, make_cube):
        # No foreground anywhere -> identical RNG stream and output as the
        # uniform crop, seed for seed.
        cube, mask = make_cube(batch_size=4, height=16, width=16)
        mask.zero_()
        a = RandomForegroundBiasedCrop(size=(8, 8), fg_percent=0.5)(cube, mask, _rng(7))
        b = RandomSpatialCrop(size=(8, 8))(cube, mask, _rng(7))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_fg_percent_zero_matches_random_spatial_crop(self, make_cube):
        cube, mask = make_cube(batch_size=4, height=16, width=16)
        a = RandomForegroundBiasedCrop(size=(8, 8), fg_percent=0.0)(cube, mask, _rng(3))
        b = RandomSpatialCrop(size=(8, 8))(cube, mask, _rng(3))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_determinism(self, make_cube):
        cube, mask = make_cube(batch_size=3)
        t = RandomForegroundBiasedCrop(size=(8, 8), fg_percent=0.33)
        out1 = t(cube, mask, _rng(seed=123))
        out2 = t(cube, mask, _rng(seed=123))
        assert torch.equal(out1[0], out2[0])
        assert torch.equal(out1[1], out2[1])

    def test_mask_none_falls_back_uniform(self, make_cube):
        cube, _ = make_cube(with_mask=False)
        out_cube, out_mask = RandomForegroundBiasedCrop(size=(8, 8), fg_percent=1.0)(
            cube, None, _rng()
        )
        assert out_cube.shape == (2, 8, 8, 8)
        assert out_mask is None

    def test_validation_errors(self, make_cube):
        cube, mask = make_cube(height=8, width=8)
        with pytest.raises(ValueError, match="exceeds"):
            RandomForegroundBiasedCrop(size=(16, 16))(cube, mask, _rng())
        with pytest.raises(ValueError, match="fg_percent"):
            RandomForegroundBiasedCrop(size=(4, 4), fg_percent=1.5)
        with pytest.raises(ValueError, match="size"):
            RandomForegroundBiasedCrop(size=(0, 4))

    def test_fg_labels_restricts_eligible_classes(self, make_cube):
        # Label 1 covers a large region but is declared background via
        # fg_labels=[2]; the forced sample must center on the single class-2
        # pixel in the far corner, not on any class-1 pixel.
        cube, mask = make_cube(batch_size=2, height=16, width=16)
        mask.zero_()
        mask[:, :8, :] = 1
        mask[:, 15, 15] = 2
        t = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=1.0, fg_labels=[2])
        for seed in range(8):
            _, out_mask = t(cube, mask, _rng(seed))
            assert (out_mask == 2).any(dim=-1).any(dim=-1).all()

    def test_fg_labels_all_background_falls_back(self, make_cube):
        # Every present label excluded by fg_labels -> identical to the
        # uniform crop, RNG stream included.
        cube, mask = make_cube(batch_size=4, height=16, width=16)
        mask.fill_(1)
        a = RandomForegroundBiasedCrop(size=(8, 8), fg_percent=1.0, fg_labels=[7])(
            cube, mask, _rng(5)
        )
        b = RandomSpatialCrop(size=(8, 8))(cube, mask, _rng(5))
        assert torch.equal(a[0], b[0])
        assert torch.equal(a[1], b[1])

    def test_probabilistic_batch_one(self, make_cube):
        # Deterministic mode at B=1: 1 - round(1 * 0.67) == 0 -> never forced.
        # Probabilistic mode with fg_percent=1.0 forces every sample even at B=1.
        cube, mask = make_cube(batch_size=1, height=16, width=16)
        mask.zero_()
        mask[0, 0, 0] = 3
        det = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=0.33, prob=0.0)
        _, det_mask = det(cube, mask, _rng())
        assert not (det_mask == 3).any()  # centre crop, no forcing at B=1
        prob_t = RandomForegroundBiasedCrop(
            size=(4, 4), fg_percent=1.0, prob=0.0, probabilistic=True
        )
        _, prob_mask = prob_t(cube, mask, _rng())
        assert (prob_mask == 3).any()

    def test_probabilistic_mixes_forced_and_uniform(self, make_cube):
        # fg_percent=0.5 probabilistic over a large batch: some crops contain
        # the corner fg pixel (forced), some do not (uniform on a 16x16 with a
        # 4x4 window rarely hits the corner).
        cube, mask = make_cube(batch_size=64, height=16, width=16)
        mask.zero_()
        mask[:, 0, 0] = 1
        t = RandomForegroundBiasedCrop(size=(4, 4), fg_percent=0.5, prob=1.0, probabilistic=True)
        _, out_mask = t(cube, mask, _rng(0))
        contains = (out_mask == 1).any(dim=-1).any(dim=-1)
        assert contains.any() and (~contains).any()

    def test_fg_labels_validation(self):
        with pytest.raises(ValueError, match="fg_labels"):
            RandomForegroundBiasedCrop(size=(4, 4), fg_labels=[])


# ---------------------------------------------------------------- samples_per_input
def _frames(batch_size: int = 2, height: int = 48, width: int = 64, channels: int = 3):
    """Input ``b`` is filled with the value ``b``, so a crop shows which input it came from."""
    return (
        torch.arange(batch_size, dtype=torch.float32)
        .view(batch_size, 1, 1, 1)
        .expand(batch_size, height, width, channels)
        .clone()
    )


def _one_fg_pixel_per_frame():
    mask = torch.zeros(2, 48, 64, dtype=torch.int32)
    mask[0, 5, 7] = 3
    mask[1, 40, 55] = 3
    return mask


def _golden_inputs():
    """Cube values encode ``b * 10000 + h * 100 + w``: a crop's first pixel names its window."""
    b = torch.arange(3).view(3, 1, 1, 1)
    h = torch.arange(20).view(1, 20, 1, 1)
    w = torch.arange(24).view(1, 1, 24, 1)
    cube = (b * 10000 + h * 100 + w).float().expand(3, 20, 24, 2).clone()
    mask = torch.zeros(3, 20, 24, dtype=torch.int32)
    mask[0, 2, 3] = 1
    mask[0, 15, 20] = 2
    mask[1, 10, 1] = 1
    return cube, mask  # input 2 has no foreground


_GOLDEN_CONFIGS = [
    {"size": (6, 8), "fg_percent": 0.33, "prob": 1.0},
    {"size": (6, 8), "fg_percent": 0.5, "prob": 0.5, "probabilistic": True},
    {"size": (6, 8), "fg_percent": 1.0, "prob": 0.0, "fg_labels": [2]},
    {"size": (6, 8), "fg_percent": 0.66, "prob": 0.7},
]
# Recorded with the transform before samples_per_input existed:
# (config, seed, with mask, first pixel of every crop, the RNG's next draw after the call).
_GOLDEN = [
    (0, 0, True, [1416, 10902, 20800], 0.4900934100151062),
    (0, 0, False, [1416, 10902, 20800], 0.4900934100151062),
    (0, 1, True, [1008, 11414, 20907], 0.3971373438835144),
    (0, 1, False, [1008, 11414, 20907], 0.3971373438835144),
    (0, 2, True, [305, 11208, 21103], 0.44253265857696533),
    (0, 2, False, [305, 11208, 21103], 0.44253265857696533),
    (1, 0, True, [16, 10700, 20906], 0.2938884496688843),
    (1, 0, False, [16, 10708, 20906], 0.022325754165649414),
    (1, 1, True, [0, 11316, 21305], 0.46354562044143677),
    (1, 1, False, [708, 11316, 21305], 0.6826140880584717),
    (1, 2, True, [1216, 10815, 20708], 0.5333684086799622),
    (1, 2, False, [708, 10815, 20708], 0.39005082845687866),
    (2, 0, True, [1216, 10708, 20708], 0.455627977848053),
    (2, 0, False, [708, 10708, 20708], 0.4900934100151062),
    (2, 1, True, [1216, 10708, 20708], 0.5695084929466248),
    (2, 1, False, [708, 10708, 20708], 0.3971373438835144),
    (2, 2, True, [1216, 10708, 20708], 0.6141571998596191),
    (2, 2, False, [708, 10708, 20708], 0.44253265857696533),
    (3, 0, True, [16, 10700, 20906], 0.40171730518341064),
    (3, 0, False, [16, 10708, 20906], 0.6323062777519226),
    (3, 1, True, [708, 10700, 21305], 0.524665892124176),
    (3, 1, False, [708, 11316, 21305], 0.4387779235839844),
    (3, 2, True, [914, 10700, 20900], 0.5332289338111877),
    (3, 2, False, [914, 10815, 20900], 0.05732768774032593),
]


class TestRandomForegroundBiasedCropSamplesPerInput:
    def test_shapes_and_each_crop_comes_from_its_input(self):
        t = RandomForegroundBiasedCrop(size=(16, 16), samples_per_input=4, probabilistic=True)
        out_cube, out_mask = t(_frames(), _one_fg_pixel_per_frame(), _rng())
        assert out_cube.shape == (8, 16, 16, 3)
        assert out_mask is not None and out_mask.shape == (8, 16, 16)
        for j in range(8):
            assert torch.all(out_cube[j] == j // 4)  # the crops of one input sit next to each other

    @pytest.mark.parametrize("row", _GOLDEN)
    def test_one_sample_per_input_keeps_the_recorded_crops_and_rng_stream(self, row):
        config, seed, with_mask, corners, next_draw = row
        cube, mask = _golden_inputs()
        g = _rng(seed)
        t = RandomForegroundBiasedCrop(**_GOLDEN_CONFIGS[config], samples_per_input=1)
        out_cube, _ = t(cube, mask if with_mask else None, g)
        assert [int(v) for v in out_cube[:, 0, 0, 0].tolist()] == corners
        assert float(torch.rand(1, generator=g).item()) == next_draw

    def test_every_crop_draws_its_own_offset(self):
        cube = torch.zeros(1, 64, 64, 1)
        cube[0, :, :, 0] = torch.arange(64 * 64, dtype=torch.float32).view(64, 64)
        t = RandomForegroundBiasedCrop(size=(8, 8), samples_per_input=6, fg_percent=0.0)
        out_cube, _ = t(cube, None, _rng(1))
        corners = {float(out_cube[j, 0, 0, 0]) for j in range(6)}
        assert len(corners) > 1

    def test_forced_crops_contain_their_inputs_foreground(self):
        t = RandomForegroundBiasedCrop(
            size=(16, 16), samples_per_input=4, fg_percent=1.0, probabilistic=True
        )
        _, out_mask = t(_frames(), _one_fg_pixel_per_frame(), _rng())
        assert out_mask is not None
        assert all(bool((out_mask[j] == 3).any()) for j in range(8))

    def test_deterministic_forcing_applies_per_input(self):
        # prob=0 makes the unforced crops centre crops, which miss both pixels; the last half
        # of every input's crops is forced onto that input's foreground.
        t = RandomForegroundBiasedCrop(size=(16, 16), samples_per_input=4, fg_percent=0.5, prob=0.0)
        _, out_mask = t(_frames(), _one_fg_pixel_per_frame(), _rng())
        assert out_mask is not None
        hits = [bool((out_mask[j] == 3).any()) for j in range(8)]
        assert hits == [False, False, True, True] * 2

    def test_without_a_mask(self):
        t = RandomForegroundBiasedCrop(size=(16, 16), samples_per_input=4, probabilistic=True)
        out_cube, out_mask = t(_frames(), None, _rng())
        assert out_cube.shape == (8, 16, 16, 3) and out_mask is None

    @pytest.mark.parametrize("bad", [0, -2, 1.5])
    def test_rejects_bad_samples_per_input(self, bad):
        with pytest.raises(ValueError, match="samples_per_input"):
            RandomForegroundBiasedCrop(size=(16, 16), samples_per_input=bad)
