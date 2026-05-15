"""Spectral augmentation transforms for hyperspectral cubes.

The spectral axis is the *defining* feature of hyperspectral imaging — these
augmentations perturb it directly. Both transforms here are mask-passthrough
(they don't move pixels, only modify spectral values), so the paired-mask
contract is preserved by construction.

References
----------
* Nalepa, Myller, Kawulok — "Hyperspectral Data Augmentation"
  IEEE TGRS / arXiv:1903.05580 (2019). https://arxiv.org/abs/1903.05580
* Ahmad et al. — "A Comprehensive Survey for HSI Classification…"
  arXiv:2404.14955 (2024). https://arxiv.org/abs/2404.14955
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai_augment.transforms.base import Transform, register


@register("GaussianBandNoise")
class GaussianBandNoise(Transform):
    """Add per-band Gaussian noise ``N(0, σ²)`` to the cube.

    The noise is drawn independently for every pixel and band. When
    ``per_band_scale=True``, the per-band standard deviation of the input cube
    is used to scale ``sigma`` — useful when bands have very different dynamic
    ranges (the common case for hyperspectral cubes that haven't been
    band-normalised). Otherwise ``sigma`` is the absolute noise standard
    deviation.

    The mask is returned untouched: this transform doesn't move pixels.

    Parameters
    ----------
    sigma : float
        Noise standard deviation (interpreted as absolute when
        ``per_band_scale=False``, otherwise as a multiplier on the per-band std).
    per_band_scale : bool, default False
        If True, scale ``sigma`` by the cube's per-band standard deviation,
        evaluated across the batch + spatial dims. This makes the noise
        magnitude follow the band's natural variation.
    prob : float, default 0.5
        Probability of application per sample.
    """

    def __init__(
        self,
        sigma: float = 0.01,
        per_band_scale: bool = False,
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        if float(sigma) < 0.0:
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        self.sigma = float(sigma)
        self.per_band_scale = bool(per_band_scale)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # wavelength-agnostic — same Gaussian per band
        self._validate_shapes(cube, mask)
        if self.sigma == 0.0:
            return cube, mask

        B = cube.shape[0]
        device = cube.device
        apply = self._draw_apply_mask(B, rng, device)
        if not apply.any():
            return cube, mask

        # Always draw a full-shape noise tensor so the RNG state advances
        # deterministically regardless of which samples are masked in.
        noise = torch.randn(cube.shape, generator=rng, dtype=cube.dtype) * self.sigma
        noise = noise.to(device=device)

        if self.per_band_scale:
            # Per-band std across (B, H, W) — keep dim C, broadcast back.
            band_std = cube.flatten(0, 2).std(dim=0, unbiased=False)  # (C,)
            noise = noise * band_std.view(1, 1, 1, -1)

        apply_b = apply.view(-1, 1, 1, 1)
        cube_out = torch.where(apply_b, cube + noise, cube)
        return cube_out, mask


@register("RandomBandDropout")
class RandomBandDropout(Transform):
    """Zero out a random subset of bands per sample.

    Acts as a spectral regulariser — encourages downstream models to use the
    full spectrum rather than over-relying on a few bands. The mask is
    returned untouched.

    Parameters
    ----------
    drop_fraction : float, default 0.1
        Fraction of bands to drop on each application. ``round(drop_fraction * C)``
        bands are zeroed, drawn uniformly without replacement, independently per
        sample. Must be in ``[0, 1]``.
    prob : float, default 0.5
        Probability of application per sample.
    """

    def __init__(
        self,
        drop_fraction: float = 0.1,
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        if not 0.0 <= float(drop_fraction) <= 1.0:
            raise ValueError(f"drop_fraction must be in [0, 1], got {drop_fraction!r}")
        self.drop_fraction = float(drop_fraction)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # wavelength-agnostic — uniform draw over band indices
        self._validate_shapes(cube, mask)
        B, _, _, C = cube.shape
        device = cube.device

        n_drop = int(round(self.drop_fraction * C))
        if n_drop == 0 or self.drop_fraction == 0.0:
            return cube, mask

        apply = self._draw_apply_mask(B, rng, device)
        if not apply.any():
            return cube, mask

        # Per-sample band-keep mask: shape (B, C) bool, True where the band is kept.
        # torch.topk on randperm-like draws gives us "pick n_drop indices per sample".
        # Use torch.argsort(rand) — vectorised over the batch.
        rand_scores = torch.rand((B, C), generator=rng).to(device=device)
        drop_indices = rand_scores.argsort(dim=1)[:, :n_drop]  # (B, n_drop)
        keep = torch.ones((B, C), dtype=cube.dtype, device=device)
        keep.scatter_(dim=1, index=drop_indices, value=0.0)  # zero the dropped bands

        # Only apply for samples where apply==True; others keep their full spectrum.
        apply_b = apply.view(-1, 1)
        keep = torch.where(apply_b, keep, torch.ones_like(keep))

        cube_out = cube * keep.view(B, 1, 1, C)
        return cube_out, mask
