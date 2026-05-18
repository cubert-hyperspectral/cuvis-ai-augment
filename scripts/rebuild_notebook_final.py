"""
Rebuild lentils_augmentation.ipynb as a clean tutorial notebook.
Run from the repo root:  python scripts/rebuild_notebook_final.py
"""

from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path("notebooks/use_cases/lentils_augmentation.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ---------------------------------------------------------------------------
# Load the existing notebook to carry over cells we keep verbatim
# ---------------------------------------------------------------------------
with open(NB_PATH, encoding="utf-8") as f:
    old_nb = json.load(f)

old = old_nb["cells"]


def _src(idx: int) -> str:
    s = old[idx]["source"]
    return "".join(s) if isinstance(s, list) else s


# ---------------------------------------------------------------------------
# Build the new cell list
# ---------------------------------------------------------------------------
cells = []

# ── Cell 0 · Title ──────────────────────────────────────────────────────────
cells.append(md("""\
# Hyperspectral data augmentation with `cuvis-ai-augment`

End-to-end tutorial for the **eight transforms across four families** shipped in
[`cuvis-ai-augment`](https://github.com/cubert-hyperspectral/cuvis-ai-augment) v0.2.0.

| Family | Transforms |
|---|---|
| **Spatial** | `RandomHorizontalFlip`, `RandomVerticalFlip`, `Random90Rotate`, `RandomSpatialCrop` |
| **Spectral** | `GaussianBandNoise`, `RandomBandDropout` |
| **Photometric** | `MultiplicativeIlluminationScaling` |
| **Mixing / erasing** | `Cutout` |

We use a real lentils hyperspectral cube (334 × 334 × 61 bands, loaded in calibrated \
reflectance via the CUVIS SDK) and walk through each transform with **before/after \
three-band panels** — low, mid, and high band indices side by side. \
Showing all three bands proves that spatial transforms apply identically across every \
channel, and that spectral/photometric transforms perturb the spectral dimension \
without touching spatial structure.

All transforms operate on `(B, H, W, C)` float32 cubes and `(B, H, W)` int32 masks — \
the same random decision drives cube and mask in lockstep.\
"""))

# ── Cell 1 · Prerequisites ───────────────────────────────────────────────────
cells.append(md(_src(1)))

# ── Cell 2 · §1 Setup header ────────────────────────────────────────────────
cells.append(md("## 1 · Setup"))

# ── Cell 3 · Colab bootstrap ────────────────────────────────────────────────
cells.append(code(_src(3)))

# ── Cell 4 · Imports ────────────────────────────────────────────────────────
cells.append(code(_src(4)))

# ── Cell 5 · §2 Load the lentils cube header ────────────────────────────────
cells.append(md("""\
## 2 · Load the lentils cube

The cube is loaded in **calibrated reflectance** using the CUVIS SDK:
`ProcessingMode.Reflectance` applies `(raw − dark) / (white − dark)` using the dark
and white reference frames embedded in the `.cu3s` session file.
The SDK returns values scaled ×10 000; dividing by 10 000 gives physical reflectance
where 1.0 equals the white reference panel. Values above 1.0 (specular highlights) are
preserved.

The notebook tries three data sources in order so it runs anywhere:

1. **HuggingFace** — pulls `Auto_003+01.cu3s` from the public
   `cubert-gmbh/XMR_Demo_Industrial_Foreign_Object_Detection_Lentils` dataset
   (cached to `~/.cache/huggingface/` after the first download).
2. **Local path** — edit `LOCAL_CANDIDATES` to point at a local `.cu3s` file.
3. **Synthetic fallback** — a small random cube so the notebook still runs
   without any data files (geometry checks still pass; spectral plots are noise).\
"""))

# ── Cell 6 · Cube loading code ──────────────────────────────────────────────
cells.append(code(_src(6)))

# ── Cell 7 · §3 Helpers header ──────────────────────────────────────────────
cells.append(md("""\
## 3 · Three-band visualisation helpers

Hyperspectral cubes have 61 bands — far too many to show at once. \
The helpers below pick three representative band indices (low / mid / high, \
at 15 %, 50 %, and 85 % of the spectral range) and display them as separate \
grayscale panels with the mask overlaid in red.

Using **three independent band panels** rather than one false-colour composite \
catches axis-swap bugs: if a transform accidentally applied different geometry \
to band 5 vs band 30, the two panels would disagree — impossible to miss.\
"""))

# ── Cell 8 · Helper functions ────────────────────────────────────────────────
# Keep only the clean helpers; drop show_augmentation_detail
helpers_src = _src(8)  # _pick_3bands, to_uint8_band, overlay_mask, show_before_after_3bands, run_single_transform
cells.append(code(helpers_src))

# ── Cell 9 · §3 Original cube ────────────────────────────────────────────────
cells.append(md("""\
### Original cube — three bands with mask overlay

Reference panel. The mask (red overlay) marks foreign-object pixels. \
Keep this in mind when comparing the augmented outputs below.\
"""))

# ── Cell 10 · Original cube plot ────────────────────────────────────────────
cells.append(code(_src(11)))

# ── §4–§7  Spatial transforms (keep existing cells verbatim) ────────────────

# Cell 11 · §4 RandomHorizontalFlip md
cells.append(md("""\
## 4 · `RandomHorizontalFlip` (spatial)

Mirrors the **width axis** left ↔ right. \
The disc in the mask moves to the horizontally mirrored position; \
all 61 band panels agree on the flip direction.\
"""))
cells.append(code(_src(13)))  # Cell 12

# Cell 13 · §5 RandomVerticalFlip
cells.append(md("""\
## 5 · `RandomVerticalFlip` (spatial)

Mirrors the **height axis** top ↔ bottom. \
The same vertical flip appears identically in every band.\
"""))
cells.append(code(_src(15)))  # Cell 14

# Cell 15 · §6 Random90Rotate
cells.append(md("""\
## 6 · `Random90Rotate` (spatial)

Rotates by a random multiple of 90° (k ∈ {1, 2, 3}) drawn per sample. \
Three seeds below show k=1, k=2, and k=3 so you can verify all three cases.\
"""))
cells.append(code(_src(17)))  # Cell 16

# Cell 17 · §7 RandomSpatialCrop
cells.append(md("""\
## 7 · `RandomSpatialCrop` (spatial)

Crops to a fixed `(H_out, W_out)` at a per-sample random offset, \
returning a smaller cube with the mask aligned. \
The crop below uses half the original spatial dimensions.\
"""))
cells.append(code(_src(19)))  # Cell 18

# ── §8 GaussianBandNoise ─────────────────────────────────────────────────────
cells.append(md("""\
## 8 · `GaussianBandNoise` (spectral)

Adds independent Gaussian noise `N(0, σ²)` to every `(pixel, band)` location.
With `per_band_scale=True` the noise amplitude is proportional to each band's own
standard deviation, matching real per-band shot noise where bands with higher
natural variation also have higher noise floor.

The spatial structure is completely untouched — the same pixel pattern is still there,
just with its spectral signature jittered. The model can no longer memorise exact
reflectance values and is forced to learn spectral *shape* rather than magnitude.

> **Note:** sigma=0.05 is a realistic production value. On this cube (reflectance,
> per-band std ≈ 0.07–0.24) it adds ~5 % of per-band std — intentionally subtle.
> The right panel below verifies numerically that the correct noise level was applied.

> Reference: Nalepa et al. 2019 (TGRS); Ahmad et al. 2024 §4.2.\
"""))

cells.append(code("""\
import numpy as np

out_gn = run_single_transform(
    {"type": "GaussianBandNoise", "sigma": 0.05, "per_band_scale": True, "prob": 1.0},
    title="GaussianBandNoise (sigma=0.05, per_band_scale=True)",
)

# ── Spectral insight: profile + noise verification ───────────────────────────
py, px = H // 2, W // 2
spec_b  = cube[0, py, px, :].cpu().float().numpy()
spec_a  = out_gn["cube"][0, py, px, :].cpu().float().numpy()
xs      = np.arange(C)

noise       = (out_gn["cube"] - cube)[0].float()
pb_std      = cube[0].float().std(dim=(0, 1)).cpu().numpy()
expect_std  = 0.05 * pb_std
measured_std = noise.std(dim=(0, 1)).cpu().numpy()
ratio = measured_std.mean() / expect_std.mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5))

ax1.plot(xs, spec_b, color="steelblue", lw=1.8, label="original")
ax1.plot(xs, spec_a, color="tomato",    lw=1.2, label="augmented", alpha=0.9)
ax1.set_xlabel("band index")
ax1.set_ylabel("reflectance")
ax1.set_title(f"Spectral profile at pixel ({py},{px}): per-band random wiggles")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.bar(xs, expect_std, alpha=0.7, color="steelblue",
        label="expected = 0.05 × band std")
ax2.plot(xs, measured_std, "o-", color="tomato", ms=2.5, lw=1.0,
         label="measured std(after − before)")
ax2.set_xlabel("band index")
ax2.set_ylabel("noise std")
ax2.set_title(f"Noise level per band — measured/expected ratio: {ratio:.4f}  (target 1.0)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ── §9 RandomBandDropout ─────────────────────────────────────────────────────
cells.append(md("""\
## 9 · `RandomBandDropout` (spectral)

Zeros a random `drop_fraction` of bands per sample. With 61 bands and
`drop_fraction=0.15`, roughly nine bands go to zero each call.
Every sample in the batch draws its own independent set of dropped bands.

Acts as spectral dropout: forces the model to produce correct output from any
arbitrary band subset — exactly the situation when some wavelengths hit
water-absorption windows, when a sensor channel fails, or when bands are
discarded during preprocessing.

In the three-band panel below, any column whose band index was dropped shows
a completely black "after" image. The bar chart identifies the exact dropped
band indices for this run.

> Reference: Ahmad et al. 2024 §4.3.\
"""))

cells.append(code("""\
import numpy as np

out_bd = run_single_transform(
    {"type": "RandomBandDropout", "drop_fraction": 0.15, "prob": 1.0},
    title="RandomBandDropout (drop_fraction=0.15) — black panels = dropped bands",
)

# ── Which bands were zeroed? ─────────────────────────────────────────────────
mean_b   = cube[0].mean(dim=(0, 1)).cpu().float().numpy()
mean_a   = out_bd["cube"][0].mean(dim=(0, 1)).cpu().float().numpy()
dropped  = (mean_a == 0) & (mean_b > 0)
n_drop   = int(dropped.sum())
xs       = np.arange(C)
colors   = ["tomato" if d else "steelblue" for d in dropped]

fig, ax = plt.subplots(figsize=(13, 2.8))
ax.bar(xs, mean_b, alpha=0.3, color="steelblue", label="original mean reflectance")
ax.bar(xs, mean_a, color=colors, alpha=0.85,
       label=f"after dropout  ({n_drop} bands zeroed in red)")
ax.set_xlabel("band index")
ax.set_ylabel("mean reflectance")
ax.set_title(
    f"Per-band mean reflectance — {n_drop} of {C} bands zeroed "
    f"(drop_fraction=0.15 × {C} ≈ {round(0.15 * C)})"
)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ── §10 MultiplicativeIlluminationScaling ────────────────────────────────────
cells.append(md("""\
## 10 · `MultiplicativeIlluminationScaling` (photometric)

Multiplies the whole cube by a smooth random gain curve `g(λ) ∈ gain_range`.
A small number of random anchor values (controlled by `smoothness`) are sampled and
upsampled to `C` bands via linear interpolation, so adjacent bands scale together
rather than independently — matching the physics of lamp-temperature drift and
illumination-angle variation, which are spectrally correlated processes.

The spatial content is completely unchanged; only the radiometric amplitude shifts.
The right panel below recovers the applied gain curve from the spatial means,
showing the characteristic smooth shape.

> Reference: Roddan et al. 2024 (Calibration-Jitter, simplified); Nalepa 2019 §III-A.\
"""))

cells.append(code("""\
import numpy as np

out_il = run_single_transform(
    {
        "type": "MultiplicativeIlluminationScaling",
        "gain_range": [0.75, 1.25],
        "smoothness": 4,
        "prob": 1.0,
    },
    title="MultiplicativeIlluminationScaling (gain=[0.75, 1.25], smoothness=4)",
)

# ── Spectral insight: profile + gain curve ───────────────────────────────────
py, px  = H // 2, W // 2
spec_b  = cube[0, py, px, :].cpu().float().numpy()
spec_a  = out_il["cube"][0, py, px, :].cpu().float().numpy()
xs      = np.arange(C)

# Recover gain curve from spatial means (robust to scene variation)
mean_b  = cube[0].float().mean(dim=(0, 1)).cpu().numpy().clip(1e-6)
mean_a  = out_il["cube"][0].float().mean(dim=(0, 1)).cpu().numpy()
gain    = mean_a / mean_b

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.5))

ax1.plot(xs, spec_b, color="steelblue", lw=1.8, label="original")
ax1.plot(xs, spec_a, color="tomato",    lw=1.2, label="augmented", alpha=0.9)
ax1.set_xlabel("band index")
ax1.set_ylabel("reflectance")
ax1.set_title("Spectral profile: same shape, smoothly scaled amplitude")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(xs, gain, color="darkorange", lw=2.0, label="estimated gain g(λ)")
ax2.fill_between(xs, gain, 1.0, alpha=0.2, color="orange")
ax2.axhline(1.0, color="grey", lw=1, linestyle="--", label="gain = 1 (no change)")
ax2.set_xlabel("band index")
ax2.set_ylabel("gain")
ax2.set_title("Smooth gain curve recovered from spatial means")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
"""))

# ── §11 Cutout ────────────────────────────────────────────────────────────────
cells.append(md("""\
## 11 · `Cutout` (mixing / erasing)

Zeros a random rectangular patch in the spatial dimensions simultaneously across
all spectral bands, and sets the corresponding mask pixels to `mask_fill_value`
(default `0`). Patch position is drawn independently per sample.

The black rectangle appears at the same location in all three band panels — the
same spatial region is erased regardless of wavelength. The mask overlay vanishes
inside the patch, signalling to the loss function that those labels should be ignored.

Forces the model to reason from partial evidence: a foreign-object detector trained
only on fully-visible lentils will be surprised when a seed is partially occluded.
Cutout makes partial observations routine.

> Reference: DeVries & Taylor 2017; Haut et al. 2019 (HSI adaptation).\
"""))

cells.append(code("""\
patch_h = min(cube.shape[1], cube.shape[2]) // 4
_ = run_single_transform(
    {"type": "Cutout", "patch_size": [patch_h, patch_h], "mask_fill_value": 0, "prob": 1.0},
    title=f"Cutout (patch {patch_h}×{patch_h} px) — black rectangle same position in all bands",
)
"""))

# ── §12 Full AugmentationCompose ─────────────────────────────────────────────
cells.append(md("""\
## 12 · Full `AugmentationCompose`

All seven concrete transforms chained — one from each family — with `prob < 1.0` so
not every transform fires on every call. Five different seeds illustrate how the
stochastic pipeline varies across training steps while remaining fully reproducible.

```yaml
transforms:
  - {type: RandomHorizontalFlip,           prob: 0.5}
  - {type: RandomVerticalFlip,             prob: 0.5}
  - {type: Random90Rotate,                 prob: 0.5}
  - {type: GaussianBandNoise,  sigma: 0.03, prob: 0.5}
  - {type: MultiplicativeIlluminationScaling, prob: 0.5}
  - {type: Cutout, patch_size: [16, 16],   prob: 0.5}
  - {type: RandomSpatialCrop, size: [out_side, out_side], prob: 1.0}
```\
"""))

cells.append(code(_src(29)))  # Full compose code cell

# ── §13 Sanity checks ────────────────────────────────────────────────────────
cells.append(md(_src(30)))

cells.append(code(_src(31)))

# ── §14 Takeaways ────────────────────────────────────────────────────────────
cells.append(md("""\
## 13 · Takeaways

- **`execution_stages={TRAIN}`** — `AugmentationCompose` is a no-op outside training.
  No need to remove it from your pipeline YAML for validation or inference runs.
- **Seed once at the Node** — `seed:` on `AugmentationCompose` drives the shared
  `torch.Generator` that every transform draws from. Reproducible runs need only this
  seed plus the data-loader seed.
- **Three-band sanity is cheap** — whenever you add a new transform, render the 2×3
  panel. If band 0 and band 30 disagree on what was applied, there is an axis bug.
- **Spectral transforms leave geometry intact** — `GaussianBandNoise`,
  `RandomBandDropout`, and `MultiplicativeIlluminationScaling` never change pixel
  positions, so the mask always passes through unchanged.
- **Reflectance loading matters** — loading the cube in calibrated reflectance
  (`ProcessingMode.Reflectance`, then / 10 000) gives physically meaningful per-band
  stds (0.07–0.24 here) compared to raw max-normalisation (0.01–0.13). Augmentation
  hyperparameters like `sigma` mean something concrete: *fraction of per-band std*.
- **`Random90Rotate` wants square cubes** — when k is odd the spatial dimensions swap.
  Centre-crop or pad before this transform if your sensor output is not square.
- **Mask fill values** — `Cutout`'s `mask_fill_value` should match the ignore-label
  used in your downstream loss (e.g. `255` for cross-entropy ignore or `-1` for some
  segmentation frameworks).
"""))

# ---------------------------------------------------------------------------
# Write the new notebook
# ---------------------------------------------------------------------------
new_nb = {
    "nbformat": old_nb["nbformat"],
    "nbformat_minor": old_nb.get("nbformat_minor", 5),
    "metadata": old_nb["metadata"],
    "cells": cells,
}

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(new_nb, f, ensure_ascii=False, indent=1)

print(f"Written {NB_PATH}  —  {len(cells)} cells")
for i, c in enumerate(cells):
    src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    print(f"  [{i:2d}] {c['cell_type'][:4]}  {src[:80].replace(chr(10),' | ')}")
