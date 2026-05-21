"""Mixing / erasing augmentation transforms for hyperspectral cubes.

The v0.2.0 entry here is :class:`Cutout` — zero out a rectangular spatial patch
in both the cube and the mask. This is the simplest of the cut/mix family and
is well-validated on HSI classification (Haut et al. 2019, "Cutout-based
Spatial Data Augmentation for Hyperspectral Classification").

CutMix and MixUp variants are deferred to v0.2.1+; see the project plan.

References
----------
* DeVries & Taylor — "Improved Regularization of CNNs with Cutout"
  arXiv:1708.04552 (2017). https://arxiv.org/abs/1708.04552
* Haut et al. — "Hyperspectral Image Classification Using Random Occlusion Data
  Augmentation". IEEE GRSL (2019). The HSI-specific adaptation of Cutout.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai_augment.transforms.base import Transform, register


@register("Cutout")
class Cutout(Transform):
    """Erase a rectangular patch in both cube and mask at a per-sample random location.

    For each sample where the transform is applied, a single rectangular patch of
    shape ``patch_size`` is placed at a uniformly-random top-left corner such that
    the patch fits entirely inside the spatial dims. Cube pixels in the patch are
    set to ``cube_fill_value``; mask pixels in the patch are set to
    ``mask_fill_value`` (defaults to 0, the typical "background" / "ignore"
    label).

    Parameters
    ----------
    patch_size : tuple[int, int] or int
        ``(H_patch, W_patch)`` of the erasure rectangle. A single int means a
        square patch. Both must be ``> 0`` and ``<= cube spatial dim``.
    cube_fill_value : float, default 0.0
        Value written into the cube within the patch.
    mask_fill_value : int, default 0
        Value written into the mask within the patch. Use the ignore-label of
        your downstream loss here.
    prob : float, default 0.5
        Probability of application per sample.
    """

    def __init__(
        self,
        patch_size: tuple[int, int] | list[int] | int,
        cube_fill_value: float = 0.0,
        mask_fill_value: int = 0,
        prob: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        if isinstance(patch_size, int):
            ps = (int(patch_size), int(patch_size))
        else:
            ps_t = tuple(int(x) for x in patch_size)
            if len(ps_t) != 2:
                raise ValueError(f"patch_size must be int or a pair of ints, got {patch_size!r}")
            ps = (ps_t[0], ps_t[1])
        if ps[0] <= 0 or ps[1] <= 0:
            raise ValueError(f"patch_size entries must be positive, got {patch_size!r}")
        self.patch_size: tuple[int, int] = ps
        self.cube_fill_value = float(cube_fill_value)
        self.mask_fill_value = int(mask_fill_value)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
        wavelengths: list[float] | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        del wavelengths  # wavelength-agnostic spatial erasure
        self._validate_shapes(cube, mask)
        B, H, W, _ = cube.shape
        H_p, W_p = self.patch_size
        if H_p > H or W_p > W:
            raise ValueError(f"patch_size ({H_p}, {W_p}) exceeds cube spatial dims (H={H}, W={W}).")

        device = cube.device
        apply = self._draw_apply_mask(B, rng, device)
        if not apply.any():
            return cube, mask

        max_top = H - H_p
        max_left = W - W_p
        # Always draw rng so state advances deterministically across B.
        if max_top > 0:
            top = torch.randint(0, max_top + 1, (B,), generator=rng).to(device=device)
        else:
            top = torch.zeros(B, dtype=torch.long, device=device)
        if max_left > 0:
            left = torch.randint(0, max_left + 1, (B,), generator=rng).to(device=device)
        else:
            left = torch.zeros(B, dtype=torch.long, device=device)

        cube_out = cube.clone()
        mask_out = mask.clone() if mask is not None else None
        for b in range(B):
            if not bool(apply[b].item()):
                continue
            t = int(top[b].item())
            le = int(left[b].item())
            cube_out[b, t : t + H_p, le : le + W_p, :] = self.cube_fill_value
            if mask_out is not None:
                mask_out[b, t : t + H_p, le : le + W_p] = self.mask_fill_value
        return cube_out, mask_out
