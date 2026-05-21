"""Photometric augmentation transforms for hyperspectral cubes.

Photometric augmentations modify pixel *intensities* — not positions and not the
spectral axis structure. The key HSI-specific concern is that naive per-band
brightness/contrast jitter (the RGB recipe) is physically implausible after
calibration: it breaks the spectral signature that downstream models rely on.

The transform in this module uses a **smooth random gain curve** across the
spectral axis — a low-frequency multiplicative perturbation that preserves
spectral *ratios* between nearby bands while modelling slow illumination drift
(lamp temperature, exposure variation, atmospheric scattering surrogate).

References
----------
* Roddan et al. — "Calibration-Jitter"
  Healthcare Technology Letters (2024).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11665780/
  Our transform is a simplified surrogate: it doesn't need the white-reference
  cube (``W(λ)``) but achieves the same training effect — model robustness to
  illumination variation across the spectrum.
* Nalepa et al. 2019 §III-A — intensity scaling as HSI augmentation.
  https://arxiv.org/abs/1903.05580
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from cuvis_ai_augment.transforms.base import Transform, register


@register("MultiplicativeIlluminationScaling")
class MultiplicativeIlluminationScaling(Transform):
    """Multiply each band by a smooth random gain curve ``g(λ) ∈ [gain_lo, gain_hi]``.

    The curve is built by drawing ``smoothness`` uniform anchors in the gain range
    and linearly interpolating across ``C`` bands. Lower ``smoothness`` → smoother
    curve (closer to a uniform scale); higher ``smoothness`` → more variation
    across the spectrum.

    The mask is returned untouched — this is a per-band intensity transform, not
    a spatial one.

    Parameters
    ----------
    gain_range : tuple[float, float], default (0.85, 1.15)
        ``(lo, hi)`` range for the gain curve. ``lo`` must be > 0.
    smoothness : int, default 4
        Number of anchor points along the spectral axis. Linear interpolation fills
        the curve between anchors. ``smoothness=2`` ≈ linear ramp; large values
        (≥ C) approach independent per-band noise.
    prob : float, default 0.5
        Probability of application per sample.
    """

    def __init__(
        self,
        gain_range: tuple[float, float] | list[float] = (0.85, 1.15),
        smoothness: int = 4,
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        gr = tuple(float(x) for x in gain_range)
        if len(gr) != 2 or gr[0] <= 0.0 or gr[1] < gr[0]:
            raise ValueError(f"gain_range must be (lo, hi) with 0 < lo <= hi, got {gain_range!r}")
        self.gain_range: tuple[float, float] = (gr[0], gr[1])
        if int(smoothness) < 2:
            raise ValueError(f"smoothness must be >= 2, got {smoothness!r}")
        self.smoothness = int(smoothness)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        # v0.2.0: wavelength-aware variant deferred — uses anchor-index spacing, not nm.
        del wavelengths
        self._validate_shapes(cube, mask)
        B, _, _, C = cube.shape
        device = cube.device

        apply = self._draw_apply_mask(B, rng, device)
        if not apply.any():
            return cube, mask

        lo, hi = self.gain_range
        n_anchors = min(self.smoothness, C)
        # Draw (B, 1, n_anchors) on CPU then move; F.interpolate expects (N, C, L).
        anchors = torch.rand((B, 1, n_anchors), generator=rng) * (hi - lo) + lo  # (B, 1, n_anchors)
        anchors = anchors.to(device=device)
        # Linear interp to C bands → (B, 1, C).
        gain_curve = F.interpolate(anchors, size=C, mode="linear", align_corners=True)
        gain_curve = gain_curve.squeeze(1)  # (B, C)

        # Don't apply for samples where apply==False — set their curve to 1.
        apply_b = apply.view(-1, 1)
        gain_curve = torch.where(apply_b, gain_curve, torch.ones_like(gain_curve))

        cube_out = cube * gain_curve.view(B, 1, 1, C).to(cube.dtype)
        return cube_out, mask
