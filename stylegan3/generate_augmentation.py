"""
StyleGAN3 Augmentation Image Generator
=======================================
Generates 1,000 synthetic images per class-plane combination from trained
StyleGAN3 .pkl checkpoints.

Output filenames match BRISC 2025 naming convention:
  brisc2025_synth_{INDEX:05d}_{CLASS_CODE}_t1.png
  e.g. brisc2025_synth_00001_gl_ax_t1.png

Quality filter: discards only genuinely blank/degenerate frames (mean < 10).
BRISC 2025 contains legitimate dark high-contrast acquisitions that must be
preserved — the filter only catches true generation failures, not dark scans.

Usage:
  python generate_augmentation.py \
      --pkl_dir  ./checkpoints \
      --out_dir  ./brisc2025_synthetic \
      --n        1000 \
      --trunc    0.75 \
      --seed     42
"""

import os
import sys
import argparse
import pickle
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

# Maps (class_code, plane_code) → pkl filename stem expected in pkl_dir
# Edit these to match your actual checkpoint filenames
MODEL_MAP = {
    ("gl",  "ax"): "glioma_ax",
    ("gl",  "co"): "glioma_co",
    ("gl",  "sa"): "glioma_sa",
    ("me",  "ax"): "meningioma_ax",
    ("me",  "co"): "meningioma_co",
    ("me",  "sa"): "meningioma_sa",
    ("pi",  "ax"): "pituitary_ax",
    ("pi",  "co"): "pituitary_co",
    ("pi",  "sa"): "pituitary_sa",
    ("nt",  "ax"): "no_tumor_ax",
    ("nt",  "co"): "no_tumor_co",
    ("nt",  "sa"): "no_tumor_sa",
}

TARGET_SIZE   = (128, 128)   # must match training image dimensions
TARGET_MODE   = "RGB"        # training images are RGB

# Quality filter: only rejects genuinely blank/degenerate outputs.
# BRISC 2025 contains legitimate dark high-contrast acquisitions (mean ~20–35)
# so the lower threshold is set conservatively to catch true generation failures
# (near-black featureless frames) without discarding valid dark-acquisition scans.
QUALITY_MIN_MEAN  = 10       # below this = blank/degenerate frame, not a real acquisition
QUALITY_MAX_MEAN  = 240      # above this = blown-out artifact (extremely rare)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_generator(pkl_path: str, device: torch.device):
    """Load G_ema from a StyleGAN3 pickle checkpoint."""
    print(f"  Loading {Path(pkl_path).name} ...", end=" ", flush=True)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    G = data["G_ema"].eval().to(device)
    print(f"OK  (res={G.img_resolution}, z_dim={G.z_dim})")
    return G


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """
    Convert StyleGAN3 output tensor [1, C, H, W] float32 in [-1, 1]
    to a PIL Image in uint8 [0, 255].
    """
    img = img_tensor[0]                          # [C, H, W]
    img = (img * 127.5 + 128).clamp(0, 255)      # [-1,1] → [0,255]
    img = img.permute(1, 2, 0).cpu().numpy()     # [H, W, C]
    img = img.astype(np.uint8)

    if img.shape[2] == 1:
        # Grayscale output — replicate to RGB
        img = np.repeat(img, 3, axis=2)

    return Image.fromarray(img, mode="RGB")


def passes_quality_filter(pil_img: Image.Image) -> bool:
    """
    Returns True if the image is within acceptable contrast range.
    Converts to grayscale for the mean check so RGB channel imbalances
    don't affect the threshold.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    mean_val = gray.mean()
    return QUALITY_MIN_MEAN <= mean_val <= QUALITY_MAX_MEAN


def generate_for_model(
    G,
    class_code: str,
    plane_code: str,
    out_dir: Path,
    n: int,
    trunc: float,
    seed: int,
    device: torch.device,
):
    """
    Generate n quality-filtered images for one class-plane model.
    Saves to out_dir / f"brisc2025_synth_{idx:05d}_{class_code}_{plane_code}_t1.png"
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    saved       = 0
    attempted   = 0
    rejected    = 0
    global_idx  = 1          # filename index, starts at 1

    print(f"\n  Generating {n} images → {out_dir.name}/")
    print(f"  truncation_psi={trunc}  seed={seed}")

    with torch.no_grad():
        while saved < n:
            # Sample random latent
            z = torch.randn(1, G.z_dim, generator=rng, device=device)
            c = None          # unconditional model

            # Generate
            img_tensor = G(z, c, truncation_psi=trunc, noise_mode="const")
            pil_img    = tensor_to_pil(img_tensor)
            attempted += 1

            # Quality filter
            if not passes_quality_filter(pil_img):
                rejected += 1
                continue

            # Resize to match training image dimensions
            if pil_img.size != TARGET_SIZE:
                pil_img = pil_img.resize(
                    TARGET_SIZE,
                    resample=Image.LANCZOS
                )

            # Ensure correct colour mode
            if pil_img.mode != TARGET_MODE:
                pil_img = pil_img.convert(TARGET_MODE)

            # Save with BRISC-style identifier
            # Format: brisc2025_synth_{index:05d}_{class}_{plane}_t1.png
            filename = f"brisc2025_synth_{global_idx:05d}_{class_code}_{plane_code}_t1.png"
            pil_img.save(out_dir / filename, format="PNG")

            saved      += 1
            global_idx += 1

            if saved % 100 == 0 or saved == n:
                reject_pct = 100 * rejected / attempted if attempted else 0
                print(f"    {saved:4d}/{n}  "
                      f"attempted={attempted}  "
                      f"rejected={rejected} ({reject_pct:.1f}%)")

    print(f"  ✅ Done. {saved} images saved, {rejected} rejected by quality filter.")
    return saved, rejected


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate StyleGAN3 augmentation images")
    parser.add_argument("--pkl_dir", required=True,
                        help="Directory containing .pkl checkpoint files")
    parser.add_argument("--out_dir", required=True,
                        help="Root output directory for synthetic images")
    parser.add_argument("--n", type=int, default=1000,
                        help="Number of images to generate per model (default: 1000)")
    parser.add_argument("--trunc", type=float, default=0.75,
                        help="Truncation psi: 0.7=quality, 1.0=diversity (default: 0.75)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (each model gets seed + offset)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Limit to specific models e.g. --models gl_ax me_co "
                             "(default: all 12)")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pkl_dir  = Path(args.pkl_dir)
    out_dir  = Path(args.out_dir)

    print("=" * 65)
    print("StyleGAN3 Augmentation Generator — BRISC 2025")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  PKL dir     : {pkl_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"  N per model : {args.n}")
    print(f"  Truncation  : {args.trunc}")
    print(f"  Base seed   : {args.seed}")
    print(f"  Target size : {TARGET_SIZE[0]}×{TARGET_SIZE[1]} {TARGET_MODE}")
    print(f"  Quality     : mean pixel ∈ [{QUALITY_MIN_MEAN}, {QUALITY_MAX_MEAN}]")

    # Build list of (class_code, plane_code) to process
    if args.models:
        targets = []
        for m in args.models:
            parts = m.split("_")
            if len(parts) == 2:
                targets.append((parts[0], parts[1]))
            else:
                print(f"  ⚠ Unrecognised model spec '{m}', skipping")
    else:
        targets = list(MODEL_MAP.keys())

    print(f"\n  Models to process ({len(targets)}):")
    for cls, pln in targets:
        stem = MODEL_MAP.get((cls, pln), f"{cls}_{pln}")
        print(f"    {cls}_{pln}  →  {stem}")

    # ── Process each model ────────────────────────────────────────────────────
    summary = []
    for offset, (class_code, plane_code) in enumerate(targets):
        stem    = MODEL_MAP[(class_code, plane_code)]
        # Find the pkl — accept exact match or any pkl containing the stem
        matches = sorted(pkl_dir.glob(f"*{stem}*.pkl"))
        if not matches:
            print(f"\n  ❌ No pkl found for {stem} in {pkl_dir} — skipping")
            summary.append((f"{class_code}_{plane_code}", 0, 0, "PKL NOT FOUND"))
            continue

        pkl_path = matches[-1]   # use latest if multiple checkpoints exist
        model_out = out_dir / f"{class_code}_{plane_code}"

        print(f"\n{'─'*65}")
        print(f"  Model  : {class_code}_{plane_code}  ({pkl_path.name})")

        try:
            G = load_generator(str(pkl_path), device)
            saved, rejected = generate_for_model(
                G           = G,
                class_code  = class_code,
                plane_code  = plane_code,
                out_dir     = model_out,
                n           = args.n,
                trunc       = args.trunc,
                seed        = args.seed + offset * 1000,
                device      = device,
            )
            summary.append((f"{class_code}_{plane_code}", saved, rejected, "OK"))
        except Exception as e:
            print(f"  ❌ Error: {e}")
            summary.append((f"{class_code}_{plane_code}", 0, 0, str(e)))
        finally:
            # Free VRAM before loading next model
            del G
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("GENERATION SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<14} {'Saved':>6} {'Rejected':>9} {'Status'}")
    print(f"  {'-'*14} {'-'*6} {'-'*9} {'-'*20}")
    total_saved = 0
    for model, saved, rejected, status in summary:
        print(f"  {model:<14} {saved:>6} {rejected:>9}   {status}")
        total_saved += saved
    print(f"\n  Total images generated: {total_saved}")
    print(f"  Output root           : {out_dir.resolve()}")
    print("\n✅ Generation complete.")


if __name__ == "__main__":
    main()
