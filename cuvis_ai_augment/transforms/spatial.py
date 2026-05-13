"""Spatial augmentation transforms for hyperspectral cubes.

Operates on cubes shaped ``(B, H, W, C)`` torch.float32 and masks shaped ``(B, H, W)``.
Each transform:

* draws an independent per-sample decision from the shared RNG;
* applies the same spatial operation to the cube and the mask (when provided) so the
  pair stays aligned (the marker pixel in ``cube[b, h, w, :]`` and the label in
  ``mask[b, h, w]`` end up at the same ``(h, w)`` after augmentation).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai_augment.transforms.base import Transform, register


def _apply_flip_per_sample(
    tensor: Tensor,
    apply: Tensor,
    dim: int,
) -> Tensor:
    """Flip ``tensor`` along ``dim`` only for samples where ``apply[b]`` is True."""
    if not apply.any():
        return tensor
    if apply.all():
        return torch.flip(tensor, dims=(dim,))
    flipped = torch.flip(tensor, dims=(dim,))
    # Broadcast apply (B,) over the remaining dims via where().
    view_shape = [-1] + [1] * (tensor.ndim - 1)
    apply_b = apply.view(view_shape)
    return torch.where(apply_b, flipped, tensor)


@register("RandomHorizontalFlip")
class RandomHorizontalFlip(Transform):
    """Flip the width axis (dim 2) with probability ``prob`` per sample.

    Cube shape: ``(B, H, W, C)`` → width axis is ``dim=2``.
    Mask shape: ``(B, H, W)`` → width axis is also ``dim=2``.
    """

    def __init__(self, prob: float = 0.5, **kwargs: Any) -> None:
        super().__init__(prob=prob, **kwargs)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_shapes(cube, mask)
        apply = self._draw_apply_mask(cube.shape[0], rng, cube.device)
        cube_out = _apply_flip_per_sample(cube, apply, dim=2)
        mask_out = _apply_flip_per_sample(mask, apply, dim=2) if mask is not None else None
        return cube_out, mask_out


@register("RandomVerticalFlip")
class RandomVerticalFlip(Transform):
    """Flip the height axis (dim 1) with probability ``prob`` per sample.

    Cube shape: ``(B, H, W, C)`` → height axis is ``dim=1``.
    Mask shape: ``(B, H, W)`` → height axis is also ``dim=1``.
    """

    def __init__(self, prob: float = 0.5, **kwargs: Any) -> None:
        super().__init__(prob=prob, **kwargs)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_shapes(cube, mask)
        apply = self._draw_apply_mask(cube.shape[0], rng, cube.device)
        cube_out = _apply_flip_per_sample(cube, apply, dim=1)
        mask_out = _apply_flip_per_sample(mask, apply, dim=1) if mask is not None else None
        return cube_out, mask_out


@register("Random90Rotate")
class Random90Rotate(Transform):
    """Rotate by a uniform random multiple of 90° per sample.

    With probability ``prob`` the sample is rotated; otherwise it passes through
    unchanged. When rotated, ``k`` ∈ {1, 2, 3} is drawn uniformly (k=0 == identity is
    handled by the apply mask, so we only sample 1/2/3 here).

    Cube ``(B, H, W, C)`` rotates on dims ``(1, 2)`` — H and W swap when k is odd.
    Mask ``(B, H, W)`` rotates on the same dims.

    Output shape can therefore vary per-sample when k is odd vs even. Implementation
    requires same k across the batch when the post-rotation height/width differ from
    pre. We resolve this by drawing one k per *group of samples that share an output
    shape* — concretely: draw k uniformly per sample, then for samples whose k is odd
    we route through a separate transpose path. To keep the output shape uniform across
    the batch, **this transform requires square spatial dims (H == W)** when applied
    with k ∈ {1, 3}. Non-square cubes still work for k ∈ {0, 2}.
    """

    def __init__(self, prob: float = 0.5, **kwargs: Any) -> None:
        super().__init__(prob=prob, **kwargs)

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_shapes(cube, mask)
        B, H, W, _ = cube.shape
        device = cube.device

        apply = self._draw_apply_mask(B, rng, device)

        # k=0 for samples where apply is False; k ∈ {1, 2, 3} uniformly for the rest.
        k_active = torch.randint(low=1, high=4, size=(B,), generator=rng).to(device=device)
        k = torch.where(apply, k_active, torch.zeros_like(k_active))

        # If any sample has odd k, the spatial dims swap. Enforce H == W in that case so
        # the batch can be stacked back into one (B, H, W, C) tensor.
        odd_k_present = bool(((k % 2) == 1).any())
        if odd_k_present and H != W:
            raise ValueError(
                f"Random90Rotate with k in {{1, 3}} requires square spatial dims "
                f"(H == W); got (H={H}, W={W}). Either crop to square first or "
                f"restrict k to even rotations (not currently supported in v0.1.0)."
            )

        cube_out = cube.clone()
        mask_out = mask.clone() if mask is not None else None
        # Apply per-sample. B is typically small (1–32), so a Python loop is fine.
        for b in range(B):
            kb = int(k[b].item())
            if kb == 0:
                continue
            cube_out[b] = torch.rot90(cube[b], k=kb, dims=(0, 1))
            if mask_out is not None:
                mask_out[b] = torch.rot90(mask_out[b], k=kb, dims=(0, 1))
        return cube_out, mask_out


@register("RandomSpatialCrop")
class RandomSpatialCrop(Transform):
    """Crop to a fixed output size ``(H_out, W_out)`` at a per-sample random offset.

    Unlike the flip/rotate transforms, ``prob`` here is the probability of taking a
    *random* crop; with probability ``1 - prob`` a deterministic centre crop is used
    instead. (A "no crop" option would change output shape so we always crop.)

    Parameters
    ----------
    size : tuple[int, int]
        Output ``(H_out, W_out)``. Must satisfy ``H_out <= H`` and ``W_out <= W``.
    prob : float
        Probability of random offset; otherwise centre crop.
    """

    def __init__(
        self,
        size: tuple[int, int] | list[int],
        prob: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(prob=prob, **kwargs)
        size_t = tuple(int(x) for x in size)
        if len(size_t) != 2 or any(s <= 0 for s in size_t):
            raise ValueError(f"size must be a pair of positive ints, got {size!r}")
        self.size: tuple[int, int] = (size_t[0], size_t[1])

    def __call__(
        self,
        cube: Tensor,
        mask: Tensor | None,
        rng: torch.Generator,
    ) -> tuple[Tensor, Tensor | None]:
        self._validate_shapes(cube, mask)
        B, H, W, C = cube.shape
        H_out, W_out = self.size

        if H_out > H or W_out > W:
            raise ValueError(
                f"Crop size ({H_out}, {W_out}) exceeds cube spatial dims (H={H}, W={W})."
            )

        device = cube.device
        random_apply = self._draw_apply_mask(B, rng, device)

        max_top = H - H_out
        max_left = W - W_out

        # Random offsets for samples in random_apply; centre otherwise.
        if max_top > 0:
            rand_top = torch.randint(low=0, high=max_top + 1, size=(B,), generator=rng).to(
                device=device
            )
        else:
            rand_top = torch.zeros(B, dtype=torch.long, device=device)
        if max_left > 0:
            rand_left = torch.randint(low=0, high=max_left + 1, size=(B,), generator=rng).to(
                device=device
            )
        else:
            rand_left = torch.zeros(B, dtype=torch.long, device=device)

        centre_top = torch.full((B,), max_top // 2, dtype=torch.long, device=device)
        centre_left = torch.full((B,), max_left // 2, dtype=torch.long, device=device)

        top = torch.where(random_apply, rand_top, centre_top)
        left = torch.where(random_apply, rand_left, centre_left)

        cube_out = torch.empty((B, H_out, W_out, C), dtype=cube.dtype, device=device)
        mask_out = (
            torch.empty((B, H_out, W_out), dtype=mask.dtype, device=device)
            if mask is not None
            else None
        )
        for b in range(B):
            t = int(top[b].item())
            l = int(left[b].item())  # noqa: E741 — `l` is fine for index here
            cube_out[b] = cube[b, t : t + H_out, l : l + W_out, :]
            if mask is not None and mask_out is not None:
                mask_out[b] = mask[b, t : t + H_out, l : l + W_out]
        return cube_out, mask_out
