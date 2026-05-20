"""
preprocess_v2.py
==================
Global BRISC 2025 preprocessing script.

This script keeps the preprocessing procedure from the original
StyleGAN2-ADA notebook (Cell 5) but makes it usable for any dataset split:
training, test, or a generic image tree.

Pipeline (mirrors Cell 5 exactly):
    1. Load as greyscale float32
    2. Skull-strip mask via Otsu threshold + corner flood-fill fallback chain
    3. Percentile-clip + intensity normalise using brain-region pixels only
    4. Zero background at full resolution BEFORE resize
    5. Skull-frame normalise: crop to brain bbox (+ 5 % margin) then resize to
       IMAGE_SIZE x IMAGE_SIZE via LANCZOS; scale-adaptive unsharp sharpening
    6. Re-apply mask + clip threshold of 5 to kill LANCZOS fringe
    7. Save as PNG (lossless) preserving the original class subfolder structure

Usage:
    python preprocess_v2.py
    python preprocess_v2.py --dataset-type test --force
    python preprocess_v2.py --dataset-type train --force
    python preprocess_v2.py --input-dir path/to/images --output-dir path/to/out

Default test output mirrors the input class structure:
    test/glioma/img_gl_ax_001.jpg  ->  brisc2025_test_preprocessed/glioma/img_gl_ax_001.png

Default training output uses class-plane folders for StyleGAN2-ADA:
    train/glioma/img_gl_ax_001.jpg -> brisc2025_preprocessed/glioma_ax/img_gl_ax_001.png
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)


# ---------------------------------------------------------------------------
# Image size - must match the value used in the notebook (IMAGE_SIZE = 128)
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE  = 128
DEFAULT_WORKERS     = max(1, (os.cpu_count() or 2) - 1)
VALID_EXTS          = {".jpg", ".jpeg", ".png"}
CLASSES             = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLANES              = ["ax", "co", "sa"]
PLANE_CODES         = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
TUMOR_CODES         = {
    "gl": "glioma",
    "me": "meningioma",
    "pi": "pituitary",
    "no": "no_tumor",
    "nt": "no_tumor",
}


# =============================================================================
# PREPROCESSING FUNCTIONS (verbatim from notebook Cell 5)
# =============================================================================

def _flood_fill_background(closed_binary):
    """Flood-fill from padded border to find external background."""
    import numpy as np
    from skimage.segmentation import flood_fill as sk_flood_fill
    padded = np.pad(closed_binary, 2, mode="constant", constant_values=0)
    bg_map = (padded == 0).astype(np.uint8)
    filled = sk_flood_fill(bg_map, (0, 0), 2)
    return (filled == 2)[2:-2, 2:-2]


def _try_mask(img_array, thresh_fraction, close_iters):
    """Single skull-strip attempt. Returns (mask, coverage) or (None, 0.0)."""
    import numpy as np
    from scipy import ndimage
    from scipy.ndimage import label, binary_closing
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    try:
        thresh = threshold_otsu(img_array) * thresh_fraction
    except Exception:
        thresh = 20.0

    binary = (img_array > thresh).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)
    closed = binary_closing(
        binary, structure=ndimage.iterate_structure(struct, close_iters)
    ).astype(np.uint8)

    external_bg = _flood_fill_background(closed)
    brain_mask  = (~external_bg).astype(np.uint8)

    lbl, n = label(brain_mask)
    if n == 0:
        return None, 0.0

    sizes      = ndimage.sum(brain_mask, lbl, range(1, n + 1))
    brain_mask = (lbl == int(np.argmax(sizes) + 1)).astype(np.uint8)
    brain_mask = remove_small_objects(
        brain_mask.astype(bool), min_size=500
    ).astype(np.uint8)
    brain_mask = binary_closing(
        brain_mask, structure=ndimage.iterate_structure(struct, 3)
    ).astype(np.uint8)
    return brain_mask, float(brain_mask.mean())


def compute_skull_strip_mask(img_array):
    """
    Robust skull-strip via Otsu + border flood-fill with fallback chain.

    Tries progressively tighter Otsu fractions until coverage lands in
    the plausible brain window (15 %-85 % of FOV). Returns the first
    acceptable mask, or the closest-to-50 %-coverage result as fallback.

      Attempt 1: Otsu x 0.50 + closing 5  (conservative, keeps dark bone)
      Attempt 2: Otsu x 0.75 + closing 5
      Attempt 3: Otsu x 1.00 + closing 7  (standard)
      Attempt 4: Otsu x 1.25 + closing 7
      Attempt 5: Otsu x 1.50 + closing 5  (aggressive)
    """
    import numpy as np
    h, w = img_array.shape
    attempts = [(0.50, 5), (0.75, 5), (1.00, 7), (1.25, 7), (1.50, 5)]
    best_mask, best_dist = None, 1.0
    for thresh_frac, close_iters in attempts:
        mask, coverage = _try_mask(img_array, thresh_frac, close_iters)
        if mask is None:
            continue
        if 0.15 <= coverage <= 0.85:
            return mask
        dist = abs(coverage - 0.50)
        if dist < best_dist:
            best_mask, best_dist = mask, dist
    return best_mask if best_mask is not None else np.ones((h, w), dtype=np.uint8)


def normalize_intensity_with_mask(img_array, mask):
    """
    Percentile-clip and normalise intensity using only brain-region pixels.
    Background zeroing is intentionally deferred until AFTER resize to prevent
    LANCZOS from reintroducing non-zero boundary fringe values.
    """
    import numpy as np
    brain_pixels = img_array[mask > 0]
    if len(brain_pixels) == 0:
        # Empty mask fallback - note: ptp() removed in NumPy 2.0
        rng = img_array.max() - img_array.min()
        return ((img_array - img_array.min()) / (rng + 1e-8) * 255).astype(np.uint8)
    p1  = np.percentile(brain_pixels, 1)
    p99 = np.percentile(brain_pixels, 99)
    normalized = np.clip(img_array, p1, p99)
    normalized = (normalized - p1) / (p99 - p1 + 1e-8)
    return (normalized * 255).astype(np.uint8)


def normalize_skull_frame(img_array, mask, target_size, margin_frac=0.05):
    """
    Crop tightly to the skull bounding box then resize to target_size x target_size.

    Removes FOV / acquisition-framing variation so every output image has the
    brain filling a consistent proportion of the frame. Handles image resize
    (LANCZOS) and mask resize (NEAREST) together.

    Includes scale-adaptive unsharp sharpening: no sharpening for images where
    the skull already fills the FOV (scale_factor < 1.2); linearly increasing
    strength up to scale_factor = 3.0 for heavy crops.
    """
    import numpy as np
    from PIL import Image, ImageFilter

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        # Empty mask - fall back to full-image resize
        img_out  = np.array(Image.fromarray(img_array).resize(
                       (target_size, target_size), Image.LANCZOS))
        mask_out = np.array(Image.fromarray(mask).resize(
                       (target_size, target_size), Image.NEAREST))
        return img_out, mask_out

    rmin, rmax = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    cmin, cmax = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

    h, w      = img_array.shape
    bbox_span = max(rmax - rmin, cmax - cmin)
    pad       = max(int(bbox_span * margin_frac), 4)
    r0 = max(0, rmin - pad);  r1 = min(h, rmax + pad + 1)
    c0 = max(0, cmin - pad);  c1 = min(w, cmax + pad + 1)

    img_crop  = img_array[r0:r1, c0:c1]
    mask_crop = mask[r0:r1, c0:c1]

    img_out  = np.array(Image.fromarray(img_crop).resize(
                   (target_size, target_size), Image.LANCZOS))
    mask_out = np.array(Image.fromarray(mask_crop).resize(
                   (target_size, target_size), Image.NEAREST))

    # Scale-adaptive sharpening
    scale_factor = target_size / max(bbox_span, 1)
    if scale_factor > 1.2:
        t         = min((scale_factor - 1.2) / (3.0 - 1.2), 1.0)
        radius    = 1.0 + t * 1.5
        percent   = int(80 + t * 120)
        threshold = max(1, int(4 - t * 3))
        img_out = np.array(
            Image.fromarray(img_out).filter(
                ImageFilter.UnsharpMask(
                    radius=radius, percent=percent, threshold=threshold
                )
            )
        )
        img_out[mask_out == 0] = 0   # re-zero border after filter bleed

    return img_out, mask_out


def preprocess_array(raw_uint8_rgb, image_size: int = DEFAULT_IMAGE_SIZE):
    """Apply the shared Cell 5 preprocessing pipeline to an RGB/greyscale array.

    Returns a single-channel uint8 array at image_size x image_size.
    """
    import numpy as np
    from PIL import Image

    # 1. Load as greyscale float32
    img = Image.fromarray(raw_uint8_rgb).convert("L")
    arr = np.array(img, dtype=np.float32)

    # 2. Skull-strip mask
    mask = compute_skull_strip_mask(arr)

    # 3. Intensity normalise (background NOT zeroed yet)
    arr_normalized = normalize_intensity_with_mask(arr, mask)

    # 4. Zero background at full resolution BEFORE resize
    arr_normalized[mask == 0] = 0

    # 5. Skull-frame normalise + resize + sharpening
    img_final_arr, mask_resized = normalize_skull_frame(
        arr_normalized, mask, image_size
    )

    # 6. Re-apply mask + clip threshold of 5 to kill LANCZOS fringe
    img_final_arr[mask_resized == 0] = 0
    img_final_arr[img_final_arr < 5] = 0
    return img_final_arr.astype(np.uint8)


def preprocess_image(img_path: Path, dst_path: Path, image_size: int):
    """
    Apply the full Cell 5 pipeline to a single image and save as PNG.
    Returns (True, None) on success or (False, error_string) on failure.
    """
    import numpy as np
    from PIL import Image

    try:
        img = Image.open(img_path).convert("RGB")
        img_final_arr = preprocess_array(np.array(img, dtype=np.uint8), image_size)

        # 7. Save as PNG
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_final_arr).save(dst_path)
        return True, None

    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Multiprocessing worker (top-level so it is picklable on Windows spawn)
# ---------------------------------------------------------------------------
def _worker(args_tuple):
    src_str, dst_str, image_size = args_tuple
    ok, err = preprocess_image(Path(src_str), Path(dst_str), image_size)
    return ok, err, src_str, dst_str


def _plane_from_filename(path: Path):
    tokens = path.stem.split("_")
    for token in tokens:
        if token in PLANES:
            return token
    for plane in PLANES:
        if f"_{plane}_" in path.stem:
            return plane
    return None


def _class_from_path(path: Path, input_dir: Path):
    try:
        rel_parts = path.relative_to(input_dir).parts
    except ValueError:
        rel_parts = path.parts
    for part in rel_parts:
        if part in CLASSES:
            return part

    tokens = path.stem.split("_")
    for token in tokens:
        if token in TUMOR_CODES:
            return TUMOR_CODES[token]
    return path.parent.name if path.parent.name else "images"


def _resolve_defaults(args):
    project_root = args.project_root.resolve()
    dataset_type = args.dataset_type

    input_dir = args.input_dir or args.test_dir
    if input_dir is None:
        if dataset_type == "train":
            input_dir = project_root / "brisc2025" / "classification_task" / "train"
        elif dataset_type == "test":
            input_dir = project_root / "brisc2025" / "classification_task" / "test"
        else:
            raise SystemExit("--input-dir is required when --dataset-type generic.")

    output_dir = args.output_dir
    if output_dir is None:
        if dataset_type == "train":
            output_dir = project_root / "brisc2025_preprocessed"
        elif dataset_type == "test":
            output_dir = project_root / "brisc2025_test_preprocessed"
        else:
            output_dir = project_root / f"{Path(input_dir).name}_preprocessed"

    layout = args.layout
    if layout == "auto":
        layout = "class-plane" if dataset_type == "train" else "preserve"

    comparison_dir = args.comparison_dir
    if comparison_dir is None:
        comparison_dir = project_root / "figures" / f"preprocessing_{dataset_type}"

    return project_root, Path(input_dir).resolve(), Path(output_dir).resolve(), layout, Path(comparison_dir).resolve()


def _destination_for_image(img_path: Path, input_dir: Path, output_dir: Path, layout: str):
    if layout == "preserve":
        rel = img_path.relative_to(input_dir)
        return output_dir / rel.parent / f"{img_path.stem}.png", None

    if layout == "class-plane":
        cls = _class_from_path(img_path, input_dir)
        plane = _plane_from_filename(img_path)
        if cls not in CLASSES:
            return None, f"could not infer BRISC class from {img_path}"
        if plane not in PLANES:
            return None, f"could not infer BRISC plane from {img_path.name}"
        return output_dir / f"{cls}_{plane}" / f"{img_path.stem}.png", None

    raise ValueError(f"Unknown layout: {layout}")


def _collect_image_pairs(input_dir: Path, output_dir: Path, layout: str):
    pairs = []
    skipped = []
    for img_path in sorted(input_dir.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTS:
            continue
        dst_path, reason = _destination_for_image(img_path, input_dir, output_dir, layout)
        if reason:
            skipped.append((img_path, reason))
            continue
        pairs.append((img_path, dst_path))
    return pairs, skipped


def _make_comparison_figure(pairs, input_dir: Path, output_dir: Path,
                            comparison_dir: Path, per_group: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.unicode_minus": False})
    from PIL import Image

    available = [(src, dst) for src, dst in pairs if dst.exists()]
    if not available:
        print("  [comparison] no processed image pairs available yet.")
        return []

    groups = {}
    for src, dst in available:
        cls = _class_from_path(src, input_dir)
        plane = _plane_from_filename(src)
        key = f"{cls}_{plane}" if plane else cls
        groups.setdefault(key, []).append((src, dst))

    selected = []
    for key in sorted(groups):
        selected.extend(groups[key][:max(1, per_group)])

    comparison_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, max(2.4, 2.2 * n_rows)))
    if n_rows == 1:
        axes = [axes]

    for row_i, (src, dst) in enumerate(selected):
        raw = Image.open(src).convert("L")
        proc = Image.open(dst).convert("L")
        cls = _class_from_path(src, input_dir)
        plane = _plane_from_filename(src)
        label = f"{cls} {PLANE_CODES.get(plane, plane or '')}".strip()

        axes[row_i][0].imshow(raw, cmap="gray")
        axes[row_i][0].set_title(f"Raw\n{label}", fontsize=9)
        axes[row_i][0].axis("off")

        axes[row_i][1].imshow(proc, cmap="gray")
        axes[row_i][1].set_title("Preprocessed", fontsize=9)
        axes[row_i][1].axis("off")

    fig.suptitle("BRISC MRI preprocessing: raw vs preprocessed", fontweight="bold")
    plt.tight_layout()
    png_path = comparison_dir / "preprocessing_comparison.png"
    pdf_path = comparison_dir / "preprocessing_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote comparison figure: {png_path}")
    print(f"  Wrote comparison figure: {pdf_path}")
    return [png_path, pdf_path]


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocess BRISC 2025 images with the notebook-faithful pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-type",
        choices=("test", "train", "generic"),
        default="test",
        help="Default paths and output layout preset.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Root image directory. Overrides the dataset-type input default.",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Backward-compatible alias for --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write preprocessed PNGs. Defaults depend on --dataset-type.",
    )
    parser.add_argument(
        "--layout",
        choices=("auto", "preserve", "class-plane"),
        default="auto",
        help="'preserve' mirrors the input tree; 'class-plane' writes folders "
             "like glioma_ax for StyleGAN2-ADA.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root directory.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output image side length (square).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Parallel worker processes.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process images even if the output file already exists.",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Do not generate raw-vs-preprocessed comparison figures.",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Where to save preprocessing_comparison.{png,pdf}.",
    )
    parser.add_argument(
        "--comparison-per-group",
        type=int,
        default=1,
        help="Number of raw/processed pairs to show per class or class-plane group.",
    )
    args = parser.parse_args(argv)

    project_root, input_dir, output_dir, layout, comparison_dir = _resolve_defaults(args)

    print("=" * 70)
    print("BRISC 2025 IMAGE PREPROCESSING")
    print("=" * 70)
    print(f"  Dataset : {args.dataset_type}")
    print(f"  Input   : {input_dir}")
    print(f"  Output  : {output_dir}")
    print(f"  Layout  : {layout}")
    print(f"  Size    : {args.image_size} x {args.image_size}")
    print(f"  Workers : {args.workers}")
    print(f"  Force   : {args.force}")
    if not args.no_comparison:
        print(f"  Compare : {comparison_dir}")
    print()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Collect all images --------------------------------------------------
    pairs, invalid = _collect_image_pairs(input_dir, output_dir, layout)
    tasks = []   # (src_path, dst_path)
    skipped = 0

    for img_path, dst_path in pairs:
        if not args.force and dst_path.exists():
            skipped += 1
            continue
        tasks.append((img_path, dst_path))

    print(f"  Found {len(pairs)} valid image(s) - {len(tasks)} to process, {skipped} already done.")
    if invalid:
        print(f"  Skipped {len(invalid)} image(s) that did not match the requested layout.")
        for src, reason in invalid[:5]:
            print(f"    {src}: {reason}")

    if not tasks:
        print("\nNothing to do. Pass --force to reprocess existing outputs.")
        if not args.no_comparison:
            _make_comparison_figure(
                pairs, input_dir, output_dir, comparison_dir,
                args.comparison_per_group,
            )
        return

    print()

    # -- Process -------------------------------------------------------------
    t0      = time.time()
    done    = 0
    errors  = []

    worker_args = [(str(src), str(dst), args.image_size) for src, dst in tasks]

    if args.workers <= 1:
        for wa in worker_args:
            ok, err, src_str, _dst_str = _worker(wa)
            done += 1
            if not ok:
                errors.append((src_str, err))
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  {done:>4}/{len(tasks)}  ({rate:.1f} img/s  eta {eta:.1f}s)", flush=True)
    else:
        from multiprocessing import get_context
        ctx = get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for ok, err, src_str, _dst_str in pool.imap_unordered(_worker, worker_args, chunksize=4):
                done += 1
                if not ok:
                    errors.append((src_str, err))
                if done % 50 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    rate    = done / elapsed
                    eta     = (len(tasks) - done) / rate if rate > 0 else 0
                    print(f"  {done:>4}/{len(tasks)}  ({rate:.1f} img/s  eta {eta:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print("COMPLETE")
    print(f"  Processed : {done - len(errors)} / {len(tasks)}")
    print(f"  Errors    : {len(errors)}")
    print(f"  Elapsed   : {elapsed:.1f}s  ({done / elapsed:.1f} img/s)")
    print(f"  Output    : {output_dir}")
    print("=" * 70)

    if errors:
        print("\nFailed images:")
        for src, err in errors:
            print(f"  {src}: {err}")

    if errors:
        raise SystemExit(1)

    if not args.no_comparison:
        _make_comparison_figure(
            pairs, input_dir, output_dir, comparison_dir,
            args.comparison_per_group,
        )


if __name__ == "__main__":
    main()
