"""
Patch the lentils notebook to add rich before/after visualisations
for the four v0.2.0 transforms: zoomed crops, difference images,
spectral profiles, and "what to look for" captions.

Run from the repo root:
    python scripts/update_notebook_v0_2_0_viz.py
"""

import json
import sys
from pathlib import Path

NB_PATH = Path("notebooks/use_cases/lentils_augmentation.ipynb")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


# ---------------------------------------------------------------------------
# New helper cell — inserted after cell 8 (the existing helpers)
# ---------------------------------------------------------------------------
HELPER_CELL = code_cell(
    '''\
def show_augmentation_detail(
    cube_before: torch.Tensor,
    mask_before: torch.Tensor | None,
    cube_after: torch.Tensor,
    mask_after: torch.Tensor | None,
    title: str,
    what_to_look_for: str,
    zoom_frac: float = 0.30,
    sample_idx: int = 0,
    amplify_diff: int = 8,
    spectral_pixel: tuple[int, int] | None = None,
    bands: list[int] | None = None,
) -> None:
    """Rich 4-row visualisation for subtle augmentations.

    Row 0  original at 3 bands (full image) — yellow box = zoom region
    Row 1  augmented at 3 bands (full image) — same yellow box
    Row 2  zoom crop: before | after | diff × amplify_diff (per band col)
    Row 3  spectral profile at one pixel: before (blue) vs after (red)

    ``what_to_look_for`` is printed as a subtitle so you always know
    what the visual change should be.
    """
    if bands is None:
        bands = _pick_3bands(cube_before.shape[-1])

    b = sample_idx
    _, H, W, C = cube_before.shape

    # zoom box centred on image
    zh = max(32, int(H * zoom_frac))
    zw = max(32, int(W * zoom_frac))
    r0 = H // 2 - zh // 2
    c0 = W // 2 - zw // 2
    r1, c1 = r0 + zh, c0 + zw

    if spectral_pixel is None:
        spectral_pixel = (H // 2, W // 2)
    py, px = spectral_pixel

    fig = plt.figure(figsize=(14, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[3, 3, 3, 2.8],
                          hspace=0.5, wspace=0.22)

    for col, band in enumerate(bands):
        img_b_raw = to_uint8_band(cube_before[b], band)
        img_a_raw = to_uint8_band(cube_after[b], band)

        if mask_before is not None:
            img_b = overlay_mask(img_b_raw.copy(), mask_before[b])
        else:
            img_b = img_b_raw.copy()

        if mask_after is not None:
            img_a = overlay_mask(img_a_raw.copy(), mask_after[b])
        else:
            img_a = img_a_raw.copy()

        # ---- Row 0: original full ----
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img_b)
        rect = plt.Rectangle((c0, r0), zw, zh, linewidth=1.8,
                              edgecolor="yellow", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"original · band {band}", fontsize=9)
        ax.axis("off")
        if col == 1:
            ax.set_title(f"original · band {band}  [yellow box = zoom region]",
                         fontsize=9)

        # ---- Row 1: augmented full ----
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(img_a)
        rect2 = plt.Rectangle((c0, r0), zw, zh, linewidth=1.8,
                               edgecolor="yellow", facecolor="none")
        ax.add_patch(rect2)
        ax.set_title(f"augmented · band {band}", fontsize=9)
        ax.axis("off")

        # ---- Row 2: zoom crop — before / after / diff ----
        # Use 3 sub-columns per band
        inner = gs[2, col].subgridspec(1, 3, wspace=0.08)

        crop_b = img_b_raw[r0:r1, c0:c1]
        crop_a = img_a_raw[r0:r1, c0:c1]

        ax_b = fig.add_subplot(inner[0])
        ax_b.imshow(crop_b)
        ax_b.set_title("before", fontsize=7.5)
        ax_b.axis("off")

        ax_a = fig.add_subplot(inner[1])
        ax_a.imshow(crop_a)
        ax_a.set_title("after", fontsize=7.5)
        ax_a.axis("off")

        # difference: centred at 0.5, amplified
        diff_f = (cube_after[b, r0:r1, c0:c1, band].cpu().float().numpy()
                  - cube_before[b, r0:r1, c0:c1, band].cpu().float().numpy())
        diff_vis = np.clip(diff_f * amplify_diff + 0.5, 0.0, 1.0)
        diff_rgb = np.stack([diff_vis, diff_vis, diff_vis], axis=-1)

        ax_d = fig.add_subplot(inner[2])
        ax_d.imshow(diff_rgb, vmin=0, vmax=1)
        ax_d.set_title(f"diff ×{amplify_diff}", fontsize=7.5)
        ax_d.axis("off")

        if col == 1:
            ax_b.set_title("before (zoom)", fontsize=7.5)

    # ---- Row 3: spectral profile ----
    ax_sp = fig.add_subplot(gs[3, :])
    spec_b = cube_before[b, py, px, :].cpu().float().numpy()
    spec_a = cube_after[b, py, px, :].cpu().float().numpy()
    xs = np.arange(C)
    ax_sp.plot(xs, spec_b, color="steelblue", lw=1.8, label="before")
    ax_sp.plot(xs, spec_a, color="tomato", lw=1.8, label="after")
    ax_sp.fill_between(xs, spec_b, spec_a, alpha=0.25,
                       color="orange", label="difference")
    ax_sp.set_xlabel("band index", fontsize=9)
    ax_sp.set_ylabel("reflectance (raw)", fontsize=9)
    ax_sp.set_title(
        f"spectral profile at pixel ({py}, {px})", fontsize=9)
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.tick_params(labelsize=8)

    fig.suptitle(
        f"{title}\\n"
        + r"$\\bf{What\\ to\\ look\\ for:}$\\ " + what_to_look_for,
        fontsize=11, y=1.02,
    )
    plt.show()
'''
)

# ---------------------------------------------------------------------------
# Updated markdown + code cells for each transform section
# ---------------------------------------------------------------------------

MD_GAUSSIAN = markdown_cell(
    """\
## 8 · `GaussianBandNoise` (spectral)

**What it does:** Adds independent Gaussian noise to each spectral band —
`N(0, σ²)` per band, with an optional `per_band_scale=True` mode that
scales σ by the measured std of each band so noisier bands (typically
mid-spectrum) get proportionally more noise.

**Why HSI needs this:** Spectral sensors have per-band shot noise.
Training without it causes models to over-fit to the noise-free lab
distribution and fail on real sensor data.

**What to look for in the panels below:**
- *Spatial (full image):* the images look nearly identical at this zoom level — that's expected!
  Spectral noise is too subtle to see in single-band thumbnails.
- *Zoomed crop / diff ×8:* faint grain visible in the difference panel —
  each pixel independently jittered.
- *Spectral profile:* the after-curve (red) is a noisy copy of the
  before-curve (blue) — random up/down wiggles per band.
  With `per_band_scale`, the wiggle amplitude tracks each band's natural variation.

> Reference: Nalepa et al. 2019; Ahmad et al. 2024 §4.2.
"""
)

CODE_GAUSSIAN = code_cell(
    """\
out_gaussian = run_single_transform(
    {"type": "GaussianBandNoise", "sigma": 0.05, "per_band_scale": True, "prob": 1.0},
    title="GaussianBandNoise — quick overview (2×3 grid)",
)

show_augmentation_detail(
    cube, mask, out_gaussian["cube"], out_gaussian.get("mask"),
    title="GaussianBandNoise (sigma=0.05, per_band_scale=True, prob=1.0)",
    what_to_look_for=(
        "Red spectral profile has random per-band wiggles vs the smooth blue original. "
        "Spatial images look almost identical — the noise lives in the spectral dimension."
    ),
    amplify_diff=8,
)
"""
)

MD_DROPOUT = markdown_cell(
    """\
## 9 · `RandomBandDropout` (spectral)

**What it does:** Zeros out a random subset of bands per sample — with
`drop_fraction=0.15` on a 61-band cube, ~9 bands go to zero.
Each sample in the batch gets a *different* random set of dropped bands.

**Why HSI needs this:** Acts as a spectral dropout regulariser — forces
the classifier/detector to learn from any arbitrary subset of bands,
which is exactly what happens when some wavelengths are saturated,
water-absorbed, or sensor-dead.

**What to look for in the panels below:**
- *Spatial (full image):* if the visualised band was **dropped**, the
  after-image is completely black (all zeros). If it wasn't dropped,
  before = after. Watch which of the three columns go black.
- *Zoomed crop / diff ×8:* dropped bands are uniformly bright in the
  diff panel (constant +0.5 after centering) — non-dropped bands show
  zero diff (grey).
- *Spectral profile:* discrete vertical drops to zero at the dropped
  band indices — sharp spikes downward, everything else unchanged.

> Reference: Ahmad et al. 2024 §4.3.
"""
)

CODE_DROPOUT = code_cell(
    """\
out_dropout = run_single_transform(
    {"type": "RandomBandDropout", "drop_fraction": 0.15, "prob": 1.0},
    title="RandomBandDropout — quick overview (2×3 grid)",
)

show_augmentation_detail(
    cube, mask, out_dropout["cube"], out_dropout.get("mask"),
    title="RandomBandDropout (drop_fraction=0.15, prob=1.0)",
    what_to_look_for=(
        "Spectral profile (red) has sharp spikes to 0 at the dropped band indices. "
        "A spatial panel goes fully black if that band was one of the ~9 dropped bands."
    ),
    amplify_diff=8,
)

# ---- extra: show which bands were dropped for this sample ----
mean_before = cube[0].mean(dim=(0, 1)).cpu().float().numpy()   # (C,)
mean_after  = out_dropout["cube"][0].mean(dim=(0, 1)).cpu().float().numpy()
dropped_mask = (mean_after == 0) & (mean_before != 0)
dropped_idx  = dropped_mask.nonzero()[0]

fig2, ax2 = plt.subplots(figsize=(12, 2.5))
bar_colors = ["tomato" if d else "steelblue" for d in dropped_mask]
ax2.bar(range(C), mean_after, color=bar_colors, alpha=0.85, label="after (dropped=red)")
ax2.bar(range(C), mean_before, color="steelblue", alpha=0.25, label="before")
ax2.set_xlabel("band index", fontsize=9)
ax2.set_ylabel("mean reflectance", fontsize=9)
ax2.set_title(
    f"Per-band mean — {int(dropped_mask.sum())} bands dropped (red bars): {list(dropped_idx)}",
    fontsize=10,
)
ax2.legend(fontsize=8)
ax2.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

MD_ILLUM = markdown_cell(
    """\
## 10 · `MultiplicativeIlluminationScaling` (photometric)

**What it does:** Multiplies the entire cube by a *smooth* random gain
curve `g(λ)` drawn from `gain_range`. Internally, a handful of random
anchor values are sampled and then upsampled to `C` bands with linear
interpolation — this prevents hard per-band jumps that would look
unrealistically noisy.

**Why HSI needs this:** Real hyperspectral acquisition is affected by
lamp-temperature drift, illumination angle, and integration-time
variation. All of these appear as a smooth multiplicative distortion
across wavelengths — not random per-band flicker.

**What to look for in the panels below:**
- *Spatial images:* Overall brightness changes slightly — the image
  may look a touch darker or lighter. The *spatial structure* is
  identical; only the radiometric scale shifts.
- *Zoomed crop:* subtle brightness difference between before/after;
  diff ×8 shows a soft uniform grey shift (same across the crop).
- *Spectral profile:* the red (after) curve is a smoothly
  *scaled version* of the blue (before) curve — it follows the same
  shape but with a gentle, slowly-varying gain multiplier.
  Compare the two lines: they diverge more at some wavelength regions
  (where the gain peaked) and converge at others.

> Reference: Roddan et al. 2024 (Calibration-Jitter, simplified);
> Nalepa 2019 §III-A.
"""
)

CODE_ILLUM = code_cell(
    """\
out_illum = run_single_transform(
    {
        "type": "MultiplicativeIlluminationScaling",
        "gain_range": [0.75, 1.25],
        "smoothness": 4,
        "prob": 1.0,
    },
    title="MultiplicativeIlluminationScaling — quick overview (2×3 grid)",
)

show_augmentation_detail(
    cube, mask, out_illum["cube"], out_illum.get("mask"),
    title="MultiplicativeIlluminationScaling (gain=[0.75, 1.25], smoothness=4, prob=1.0)",
    what_to_look_for=(
        "Spectral profile (red) follows the same shape as the original (blue) "
        "but is smoothly scaled up or down — no per-band spikes. "
        "The diff panel shows a soft, uniform brightness shift across the crop."
    ),
    amplify_diff=8,
)

# ---- extra: show the gain curve that was applied ----
# gain = after / before  (avoid /0 with a small epsilon)
gain_per_band = (
    out_illum["cube"][0].float().mean(dim=(0, 1))
    / cube[0].float().mean(dim=(0, 1)).clamp(min=1e-6)
).cpu().numpy()

fig3, ax3 = plt.subplots(figsize=(12, 2.5))
ax3.plot(range(C), gain_per_band, color="darkorange", lw=2, label="estimated gain g(λ)")
ax3.axhline(1.0, color="grey", lw=1, linestyle="--", label="gain = 1 (no change)")
ax3.fill_between(range(C), gain_per_band, 1.0, alpha=0.2, color="orange")
ax3.set_xlabel("band index", fontsize=9)
ax3.set_ylabel("gain", fontsize=9)
ax3.set_title("Smooth gain curve applied to this sample", fontsize=10)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

MD_CUTOUT = markdown_cell(
    """\
## 11 · `Cutout` (mixing / erasing)

**What it does:** Zeros a random rectangular patch in the spatial
dimensions of every band simultaneously, and sets the corresponding
mask pixels to `mask_fill_value` (default `0`). The patch position is
drawn independently per sample.

**Why HSI needs this:** Forces the model to learn from *partial*
observations — if a foreign object is partially occluded or the camera
view clips the scene edge, the model still needs to fire confidently.
Also regularises spatial features, similar to Dropout but spatially
structured.

**What to look for in the panels below:**
- *Spatial images:* a black rectangle appears in the same location
  in all three band columns (same patch = same spatial location
  across the full spectrum). The mask overlay disappears inside the
  patch (labels set to ignore).
- *Zoomed crop:* if the zoom region overlaps the patch, you'll see
  a hard black rectangle edge — very clean and geometric.
- *Spectral profile:* if the sampled pixel falls inside the patch,
  the red line is flat at zero across all bands. If outside, before = after.

> Reference: DeVries & Taylor 2017 (Cutout); Haut et al. 2019 (HSI adaptation).
"""
)

CODE_CUTOUT = code_cell(
    """\
patch_h = min(cube.shape[1], cube.shape[2]) // 4
out_cutout = run_single_transform(
    {"type": "Cutout", "patch_size": [patch_h, patch_h], "prob": 1.0},
    title="Cutout — quick overview (2×3 grid)",
)

# Locate the patch to centre the zoom on it
diff_spatial = (
    (out_cutout["cube"][0] - cube[0]).abs().mean(dim=-1).cpu().numpy()
)  # (H, W)
if diff_spatial.max() > 0:
    # find centroid of the zeroed-out region
    ys, xs_idx = (diff_spatial > 0).nonzero()
    cy = int(ys.mean()) if len(ys) else cube.shape[1] // 2
    cx = int(xs_idx.mean()) if len(xs_idx) else cube.shape[2] // 2
    spectral_px = (cy, cx)
    # pick a pixel just outside the patch for the spectral profile
    # (inside will be all zeros — less interesting)
    sp_row = max(0, cy - patch_h // 2 - 5)
    sp_col = cx
    spectral_px_outside = (sp_row, sp_col)
else:
    spectral_px_outside = (cube.shape[1] // 2, cube.shape[2] // 2)

show_augmentation_detail(
    cube, mask, out_cutout["cube"], out_cutout.get("mask"),
    title=f"Cutout (patch_size=[{patch_h}, {patch_h}], prob=1.0)",
    what_to_look_for=(
        "A black rectangle appears at the same spatial location in every band column. "
        "The mask overlay is removed inside the patch (labels set to 0). "
        "Spectral profile shows flat zero if the sampled pixel is inside the erased region."
    ),
    zoom_frac=0.45,
    spectral_pixel=spectral_px_outside,
    amplify_diff=3,
)
"""
)


# ---------------------------------------------------------------------------
# Apply patches to the notebook
# ---------------------------------------------------------------------------
def main() -> None:
    with open(NB_PATH, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # 1. Insert the new helper cell after cell 8 (index 8)
    #    The helper cell becomes index 9; everything after shifts by +1.
    cells.insert(9, HELPER_CELL)

    # After insert: old indices shift +1 for everything > 8
    # Old cell 19 (GaussianBandNoise markdown) → now 20
    # Old cell 20 (GaussianBandNoise code)     → now 21
    # Old cell 21 (RandomBandDropout markdown) → now 22
    # Old cell 22 (RandomBandDropout code)     → now 23
    # Old cell 23 (Illumination markdown)      → now 24
    # Old cell 24 (Illumination code)          → now 25
    # Old cell 25 (Cutout markdown)            → now 26
    # Old cell 26 (Cutout code)                → now 27

    replacements = {
        20: MD_GAUSSIAN,
        21: CODE_GAUSSIAN,
        22: MD_DROPOUT,
        23: CODE_DROPOUT,
        24: MD_ILLUM,
        25: CODE_ILLUM,
        26: MD_CUTOUT,
        27: CODE_CUTOUT,
    }

    for idx, new_cell in replacements.items():
        cells[idx] = new_cell

    # Clear all outputs and execution counts (notebook will be re-run)
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    nb["cells"] = cells

    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Patched {NB_PATH} — total cells: {len(cells)}")


if __name__ == "__main__":
    main()
