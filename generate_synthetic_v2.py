"""
generate_synthetic_v2.py
========================
v2 of generate_synthetic.py - defaults tuned for a more diverse synthetic
manifold for the 1:2 augmentation experiment, plus minor robustness fixes.

CHANGES vs generate_synthetic.py
--------------------------------
  * --truncation default      : 0.85 -> 1.00
        Removes the 15% pull toward the latent-space mean. Trades a small
        amount of per-image quality for a wider span across the GAN's
        learned manifold. Critical when the training dataset's effective
        diversity is bounded by patient-level repetition (low GAN recall
        is a *dataset* artifact, not generator mode collapse).
  * --noise-mode default      : 'const' -> 'random'
        Per-sample stochastic noise is sampled fresh for each forward
        pass, so two nearby latents cannot share fine-grain details.
  * --output-subdir default   : 'synthetic' -> 'synthetic_v2'
        Keeps v1 outputs intact for comparison; both generators can coexist
        on disk under stylegan3_results/<combo>/.
  * Quality gate ordering     : the structural check (mean_l < 30 OR
        nonzero_frac < 0.08) now runs IN ADDITION TO the optional
        Mahalanobis filter, not as an either/or. Catches dark/skull-only
        artifacts whose pool3 features happen to land near the real
        centroid (rare but possible - never silently keep them).
  * Latent batch sampling     : single torch.Generator per while-iteration,
        re-seeded per row, instead of allocating B Generators per batch.
        Functionally equivalent (row j still uses seed
        args.seed + attempt + j) but materially faster at B=256.

UNCHANGED
---------
  Preprocessing pipeline (skull-strip, intensity normalize, <11->0 cutoff)
  is byte-identical to v1. --hash-threshold default stays at 6 per the
  project decision. Per-combo checkpoint selection (FID + precision +
  recall consensus) is unchanged.

USAGE
-----
    python generate_synthetic_v2.py --combo glioma_ax --ratio 2.0
        # Default ratio for the v2 pool. classify_rf_v2.py then derives both
        # the 1:1 and 1:2 splits from this pool by lexical-order slicing,
        # so a single generation run feeds both experiments.

OUTPUT
------
    stylegan3_results/<combo>/synthetic_v2/
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import sys
import tempfile
import time
import warnings
from multiprocessing import Pool, get_context
from pathlib import Path
from typing import Optional

# scikit-image 0.26 deprecated the `min_size` keyword on
# `remove_small_objects` in favour of `max_size` (with off-by-one semantics).
# We deliberately keep the original `min_size=500` call so preprocessing stays
# byte-identical with the notebook cell. Suppressing the warning at module
# level so it doesn't fire once per spawned worker process.
warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)

# StyleGAN2-ADA pickles need the `stylegan3/` repo on sys.path BEFORE
# torch tries to unpickle them (custom modules live there).
def _add_stylegan3_to_path(project_root: Path) -> None:
    sg = str(project_root / "stylegan3")
    if sg not in sys.path:
        sys.path.insert(0, sg)


def _setup_windows_env():
    """Configure environment variables for PyTorch CUDA extensions on Windows.
    Mirrors the setup in classify_rf.py and the optimized notebook to fix plugin
    compilation on Windows (path-length limits, extension caching, etc.)."""
    if os.name != 'nt':
        return

    ext_dir = os.path.join(tempfile.gettempdir(), 'torch_ext')
    os.environ['TORCH_EXTENSIONS_DIR'] = ext_dir
    if not os.path.exists(ext_dir):
        try:
            os.makedirs(ext_dir, exist_ok=True)
        except OSError:
            pass

    os.environ['PYTHONWARNINGS'] = 'ignore'


# =============================================================================
# CONSTANTS - match the notebook cell verbatim
# =============================================================================
FID_TIERS = [
    (0,    30,   "EXTRAORDINARY"),
    (30,   50,   "EXCELLENT"),
    (50,   75,   "GOOD"),
    (75,   1e9,  "FAIR"),
]
TIER_ORDER = ["EXTRAORDINARY", "EXCELLENT", "GOOD", "FAIR"]

PRECISION_WEIGHT     = 0.5
RECALL_WEIGHT        = 0.5
SCORE_TIE_THRESHOLD  = 0.005

REAL_N_BRIEFING = {
    "glioma_ax":      394, "glioma_co":      430, "glioma_sa":      323,
    "meningioma_ax":  423, "meningioma_co":  426, "meningioma_sa":  480,
    "pituitary_ax":   426, "pituitary_co":   510, "pituitary_sa":   521,
    "no_tumor_ax":    352, "no_tumor_co":    310, "no_tumor_sa":    405,
}

VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def count_real_images(real_data_dir: Path, combo: str) -> int:
    """Count preprocessed real images for a (class x plane) combo."""
    combo_dir = real_data_dir / combo
    if not combo_dir.is_dir():
        raise FileNotFoundError(
            f"Real-image directory not found: {combo_dir}\n"
            f"Pass --real-data-dir to point at the preprocessed-real root, "
            f"or pass --real-n N to override the auto-count."
        )
    return sum(
        1 for p in combo_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    )


def _parse_test_filename(fname: str):
    parts = Path(fname).stem.split("_")
    if len(parts) < 6:
        return None, None
    cls_short, plane = parts[3], parts[4]
    cls_map = {"gl": "glioma", "me": "meningioma",
               "no": "no_tumor", "nt": "no_tumor",
               "pi": "pituitary"}
    return cls_map.get(cls_short), plane


def _load_reference_hashes(real_data_dir: Path, test_data_dir: Path, combo: str):
    """Load pHashes for real train and held-out test images in the same combo."""
    from PIL import Image
    import imagehash

    cls, plane = combo.rsplit("_", 1)
    refs = []

    def _add_file(fp: Path, source: str):
        try:
            with Image.open(fp) as im:
                refs.append((imagehash.phash(im.convert("RGB")), source, str(fp)))
        except Exception as exc:
            warnings.warn(
                f"Could not hash reference image {fp}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    real_combo_dir = real_data_dir / combo
    if real_combo_dir.is_dir():
        for fp in sorted(real_combo_dir.iterdir()):
            if fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS:
                _add_file(fp, "real_train")
    else:
        warnings.warn(
            f"Reference dedup: real combo directory missing: {real_combo_dir}",
            RuntimeWarning,
            stacklevel=2,
        )

    test_cls_dir = test_data_dir / cls
    if test_cls_dir.is_dir():
        for fp in sorted(test_cls_dir.iterdir()):
            if not (fp.is_file() and fp.suffix.lower() in VALID_IMAGE_EXTS):
                continue
            cls_from_name, plane_from_name = _parse_test_filename(fp.name)
            if cls_from_name == cls and plane_from_name == plane:
                _add_file(fp, "heldout_test")
    else:
        warnings.warn(
            f"Reference dedup: test class directory missing: {test_cls_dir}",
            RuntimeWarning,
            stacklevel=2,
        )

    return refs


CLS_MAP = {"glioma": "gl", "meningioma": "me", "pituitary": "pi", "no_tumor": "nt"}


# =============================================================================
# PREPROCESSING - delegated to preprocess_v2.py for exact pipeline parity.
# =============================================================================
def _apply_preprocessing_array(raw_uint8_rgb):
    """Run the shared preprocessing on a uint8 HxWx3 RGB numpy array.
    Returns: (preprocessed_pil_RGB, preprocessed_pil_L_mean, phash_object)
    """
    import imagehash
    import numpy as np
    from PIL import Image
    from preprocess_v2 import preprocess_array

    normalized = preprocess_array(raw_uint8_rgb)
    rgb        = np.stack([normalized, normalized, normalized], axis=-1)
    pil_pp     = Image.fromarray(rgb.astype(np.uint8), mode='RGB')

    mean_l       = float(np.array(pil_pp.convert('L')).mean())
    nonzero_frac = float((np.array(pil_pp.convert('L')) > 0).mean())
    phash        = imagehash.phash(pil_pp)
    return pil_pp, mean_l, nonzero_frac, phash


def _worker(raw_uint8_rgb):
    """Multiprocessing worker entrypoint - pure function, picklable."""
    pil_pp, mean_l, nonzero_frac, phash = _apply_preprocessing_array(raw_uint8_rgb)
    return pil_pp, mean_l, nonzero_frac, str(phash)


# =============================================================================
# MAHALANOBIS QUALITY FILTER - optional online screening
# =============================================================================
def _load_inception(weights_path: Path, device):
    """Load FID-style InceptionV3 TorchScript model."""
    import torch
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"InceptionV3 weights not found: {weights_path}")
    return torch.jit.load(str(weights_path), map_location=device).eval()


def _fit_combo_mahal_model(X_real_combo):
    """PCA-reduced LedoitWolf model on pool3 real features.

    v2 note: dimension lifted from 50 to 200 to retain finer texture
    information. With ~400 real samples per combo and LedoitWolf shrinkage,
    200 dims stays well-conditioned (n/d ~ 2). The 50-d model captured
    only global structure, which is exactly the axis along which a
    mode-collapsed-or-redundant generator looks most "valid."
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    n_samples = X_real_combo.shape[0]
    n_components = min(n_samples - 1, 200)

    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    X_reduced = pca.fit_transform(X_real_combo)

    lw = LedoitWolf(assume_centered=False)
    lw.fit(X_reduced)

    mean_  = X_reduced.mean(axis=0)
    prec_  = lw.get_precision()
    n_feat = n_components
    return mean_, prec_, n_feat, pca


def _compute_mahal_batch(results, inception_model, device, mean_, prec_, pca):
    """Run InceptionV3 on a batch of preprocessed images; return m^2 distances."""
    import torch
    import numpy as np
    from PIL import Image

    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    arrays = []
    for pil_pp, _, _, _ in results:
        arr = np.array(pil_pp.resize((299, 299), bilinear), dtype=np.uint8)
        arrays.append(arr)

    batch = np.stack(arrays, axis=0)
    t = torch.from_numpy(batch).to(device)
    t = t.permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        feats = inception_model(t, return_features=True)
    X = feats.cpu().numpy().astype(np.float32)

    X_reduced = pca.transform(X)
    diff = X_reduced - mean_
    m2   = np.einsum("ij,jk,ik->i", diff, prec_, diff)
    return m2


# =============================================================================
# CHECKPOINT SELECTION - verbatim from v1
# =============================================================================
def _get_fid_tier(fid):
    for lo, hi, name in FID_TIERS:
        if lo <= fid < hi:
            return name, TIER_ORDER.index(name)
    return "FAIR", 3


def _resolve_unique_checkpoint(all_pkls, checkpoint_name: str, *, preferred_dir: Path | None = None):
    if preferred_dir is not None:
        candidate = preferred_dir / checkpoint_name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"{checkpoint_name} was selected from metrics in {preferred_dir}, "
            f"but that checkpoint file is missing."
        )

    matches = [p for p in all_pkls if p.name == checkpoint_name]
    if not matches:
        raise FileNotFoundError(f"{checkpoint_name} not found under the model directory.")
    if len(matches) > 1:
        joined = "\n  ".join(str(p) for p in matches)
        raise RuntimeError(
            f"Ambiguous checkpoint name {checkpoint_name!r}; found it in multiple "
            f"StyleGAN run directories:\n  {joined}\n"
            "Use --checkpoint optimal so the metrics file identifies the run, or "
            "remove/archive stale duplicate run directories."
        )
    return matches[0]


def resolve_checkpoint(model_dir: Path, combo: str, spec):
    all_pkls = sorted(model_dir.glob("**/network-snapshot-*.pkl"))
    if not all_pkls:
        raise FileNotFoundError(f"No checkpoints found under {model_dir}")

    if spec == "final":
        return _resolve_unique_checkpoint(all_pkls, "network-snapshot-001000.pkl"), 1000, None

    if isinstance(spec, int):
        target = f"network-snapshot-{spec:06d}.pkl"
        return _resolve_unique_checkpoint(all_pkls, target), spec, None

    if spec != "optimal":
        raise ValueError(f"--checkpoint must be 'optimal', 'final', or an int kimg. Got: {spec!r}")

    json_candidates = list(model_dir.glob(f"**/checkpoint_metrics_{combo}.json"))
    if not json_candidates:
        raise FileNotFoundError(
            f"checkpoint_metrics_{combo}.json not found under {model_dir}."
        )
    json_candidates.sort(key=lambda p: p.parent.name)
    metrics_path = json_candidates[-1]
    with open(metrics_path) as f:
        metrics = json.load(f)

    candidates = [(int(k), v) for k, v in metrics.items() if int(k) > 0]
    candidates.sort(key=lambda x: x[0])
    for k, v in candidates:
        v["tier"], v["tier_rank"] = _get_fid_tier(v["fid"])

    best_tier_rank = min(v["tier_rank"] for _, v in candidates)
    best_tier      = TIER_ORDER[best_tier_rank]
    plateau        = [(k, v) for k, v in candidates if v["tier_rank"] == best_tier_rank]

    print(f"\n  {'-'*62}")
    print(f"  CHECKPOINT SELECTION AUDIT - {combo}")
    print(f"  {'-'*62}")
    print(f"  Metrics JSON       : {metrics_path}")
    print(f"  Best FID tier      : {best_tier}  "
          f"(FID bounds: {FID_TIERS[best_tier_rank][0]}-{FID_TIERS[best_tier_rank][1]})")
    print(f"  Plateau size       : {len(plateau)} checkpoint(s) in tier")
    print(f"  Weights            : P={PRECISION_WEIGHT}, R={RECALL_WEIGHT}  |  "
          f"Tie threshold: {SCORE_TIE_THRESHOLD}")
    print()

    if len(plateau) == 1:
        best_kimg, best_entry = plateau[0]
        decision = "FID unambiguous - single tier member"
    else:
        scored = sorted(
            [(k, v, PRECISION_WEIGHT * v["precision"] + RECALL_WEIGHT * v["recall"])
             for k, v in plateau],
            key=lambda x: x[2], reverse=True,
        )
        print(f"  {'kimg':>6}  {'FID':>8}  {'Prec':>7}  {'Rec':>7}  {'Score':>7}")
        print(f"  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")

        top_k, top_v, top_score = scored[0]
        sec_k, sec_v, sec_score = scored[1]
        gap = top_score - sec_score

        if gap <= SCORE_TIE_THRESHOLD:
            if sec_v["recall"] > top_v["recall"]:
                best_kimg, best_entry = sec_k, sec_v
                decision = f"composite tie (gap={gap:.4f}) -> recall tiebreak -> rank-2 wins"
            else:
                best_kimg, best_entry = top_k, top_v
                decision = f"composite tie (gap={gap:.4f}) -> recall tiebreak -> rank-1 holds"
        else:
            best_kimg, best_entry = top_k, top_v
            decision = f"composite score decisive (gap={gap:.4f})"

        for k, v, score in scored:
            marker = "  <- selected" if k == best_kimg else ""
            print(f"  {k:>6}  {v['fid']:>8.4f}  "
                  f"{v['precision']:>7.4f}  {v['recall']:>7.4f}  "
                  f"{score:>7.4f}{marker}")
        print()
        print(f"  Decision           : {decision}")

    print()
    print(f"  [OK] Selected        : {best_entry['checkpoint']}  "
          f"(kimg={best_kimg}, FID={best_entry['fid']:.4f}, "
          f"P={best_entry['precision']:.4f}, R={best_entry['recall']:.4f})")
    print(f"  {'-'*62}\n")

    checkpoint = _resolve_unique_checkpoint(
        all_pkls,
        best_entry["checkpoint"],
        preferred_dir=metrics_path.parent,
    )
    return checkpoint, best_kimg, metrics_path


# =============================================================================
# MAIN
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate synthetic MRI images for a single (class x plane) combo (v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--combo", required=True, choices=sorted(REAL_N_BRIEFING.keys()),
                   help="Class x plane combination to generate for.")
    p.add_argument("--project-root", type=Path,
                   default=Path(__file__).resolve().parent,
                   help="Project root containing stylegan3/, stylegan3_results/, etc.")
    p.add_argument("--target", type=int, default=None,
                   help="Total real+synthetic images per combo. Synthetic count is "
                        "(target - actual real count). Ignored when --ratio is set.")
    p.add_argument("--ratio", type=float, default=2.0,
                   help="Generate (ratio x real_n) synthetic images per combo. "
                        "Default 2.0 in v2 so a single generation feeds both the "
                        "1:1 and 1:2 augmentation experiments via lexical-order "
                        "slicing in classify_rf_v2.py.")
    p.add_argument("--real-data-dir", type=Path, default=None,
                   help="Root directory of preprocessed REAL images.")
    p.add_argument("--test-data-dir", type=Path, default=None,
                   help="Root directory of preprocessed held-out TEST images. "
                        "Used for synthetic near-duplicate auditing.")
    p.add_argument("--real-n", type=int, default=None,
                   help="Manual override for the real-image count.")
    p.add_argument("--checkpoint", default="optimal",
                   help="'optimal' (3-stage), 'final', or an integer kimg.")
    p.add_argument("--truncation", type=float, default=1.0,
                   help="Truncation psi. v2 default is 1.0 (was 0.85 in v1) to "
                        "use the GAN's full latent range - needed because the "
                        "training set's effective diversity is bounded by "
                        "patient-level repetition, not generator capacity.")
    p.add_argument("--seed", type=int, default=42,
                   help="Starting seed; each row uses seed + attempt + j.")
    p.add_argument("--noise-mode", default="random",
                   choices=("const", "random", "none"),
                   help="StyleGAN2 noise mode. v2 default is 'random' (was "
                        "'const' in v1) so two close latents do not share "
                        "fine-grain stochastic detail.")
    p.add_argument("--hash-threshold", type=int, default=6,
                   help="pHash Hamming distance - preserved at 6 per project decision.")
    p.add_argument("--reference-hash-threshold", type=int, default=None,
                   help="pHash Hamming distance used to reject generated images "
                        "near real train/test references. Defaults to --hash-threshold.")
    p.add_argument("--skip-reference-dedup", action="store_true",
                   help="Disable real/test near-duplicate rejection; use only for "
                        "diagnostic runs, not publication results.")
    p.add_argument("--batch-size", type=int, default=256,
                   help="GPU forward-pass batch size.")
    p.add_argument("--max-attempts", type=int, default=None,
                   help="Hard cap on generated candidates before failing clearly.")
    p.add_argument("--max-attempt-multiplier", type=float, default=50.0,
                   help="Default max attempts is this multiplier times the target "
                        "synthetic count when --max-attempts is not set.")
    p.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)),
                   help="Preprocessing worker processes.")
    p.add_argument("--output-subdir", default="synthetic_v2",
                   help="Subdirectory name under stylegan3_results/<combo>/. "
                        "Default 'synthetic_v2' so v1 outputs in 'synthetic/' "
                        "are not overwritten.")
    p.add_argument("--device", default="cuda",
                   help="Torch device.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve paths/checkpoint, print the plan, then exit.")
    p.add_argument("--classify-out-dir", type=Path, default=None,
                   help="classify_rf_v2.py output directory. Enables online "
                        "Mahalanobis quality filtering using the same real "
                        "feature distribution the offline filter uses.")
    p.add_argument("--inception-weights", type=Path, default=None,
                   help="InceptionV3 TorchScript weights for online filtering.")
    p.add_argument("--mahal-threshold-pct", type=float, default=97.5,
                   help="Empirical percentile of real m^2 used as the rejection "
                        "cutoff. v2 uses an empirical threshold (the (pct)th "
                        "percentile of the *real* m^2 distribution) instead of "
                        "the chi^2 approximation, which is calibration-free.")
    return p.parse_args(argv)


def _coerce_checkpoint(spec: str):
    if spec in ("optimal", "final"):
        return spec
    try:
        return int(spec)
    except ValueError:
        raise SystemExit(f"--checkpoint must be 'optimal', 'final', or an integer; got {spec!r}")


def main(argv=None):
    _setup_windows_env()
    args = parse_args(argv)
    args.checkpoint = _coerce_checkpoint(args.checkpoint)

    project_root = args.project_root.resolve()
    if not project_root.exists():
        raise SystemExit(f"--project-root not found: {project_root}")
    _add_stylegan3_to_path(project_root)

    results_dir   = project_root / "stylegan3_results"
    model_dir     = results_dir / args.combo
    real_data_dir = args.real_data_dir or (project_root / "brisc2025_preprocessed")
    test_data_dir = args.test_data_dir or (project_root / "brisc2025_test_preprocessed")

    if args.real_n is not None:
        real_n      = args.real_n
        real_source = f"manual (--real-n {real_n})"
    else:
        real_n      = count_real_images(real_data_dir, args.combo)
        real_source = f"auto-counted from {real_data_dir / args.combo}"

    expected = REAL_N_BRIEFING.get(args.combo)
    if expected is not None and real_n != expected:
        print(f"  WARNING:  Real count mismatch - found {real_n}, briefing says {expected}.")

    if args.ratio is not None:
        if args.ratio <= 0:
            raise SystemExit(f"--ratio must be > 0; got {args.ratio}")
        n_generate     = int(round(args.ratio * real_n))
        n_source       = f"ratio-driven ({args.ratio} x {real_n} real)"
        target_display = f"{real_n} + {n_generate} = {real_n + n_generate} (--ratio {args.ratio})"
    else:
        if args.target is None:
            raise SystemExit(
                "Either --target or --ratio must be specified."
            )
        n_generate     = args.target - real_n
        n_source       = "target-driven (--target - real_n)"
        target_display = f"{real_n} + {n_generate} = {args.target} (--target {args.target})"

    if n_generate <= 0:
        raise SystemExit(
            f"Synthetic count <= 0 for {args.combo} (real_n={real_n})."
        )

    reference_hash_threshold = (
        args.hash_threshold if args.reference_hash_threshold is None
        else args.reference_hash_threshold
    )
    if args.hash_threshold < 0 or reference_hash_threshold < 0:
        raise SystemExit("Hash thresholds must be >= 0.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0.")
    if args.max_attempts is not None:
        if args.max_attempts < n_generate:
            raise SystemExit("--max-attempts must be at least the requested synthetic count.")
        max_attempts = args.max_attempts
    else:
        if args.max_attempt_multiplier < 1.0:
            raise SystemExit("--max-attempt-multiplier must be >= 1.0.")
        max_attempts = int(round(n_generate * args.max_attempt_multiplier))

    print(f"{'='*70}")
    print(f"SYNTHETIC IMAGE GENERATION (v2): {args.combo}")
    print(f"  Project root         : {project_root}")
    print(f"  Real-image directory : {real_data_dir / args.combo}")
    print(f"  Real dataset size    : {real_n}  ({real_source})")
    print(f"  Synthetic to generate: {n_generate}  [{n_source}]")
    print(f"  Real + synthetic     : {target_display}")
    print(f"  Truncation           : {args.truncation}  (v2 default 1.0)")
    print(f"  Noise mode           : {args.noise_mode}  (v2 default random)")
    print(f"  pHash threshold      : {args.hash_threshold}  (preserved at 6)")
    print(f"  Reference pHash gate : "
          f"{'disabled' if args.skip_reference_dedup else reference_hash_threshold}")
    print(f"  Max attempts         : {max_attempts}")
    print(f"  GPU batch size       : {args.batch_size}")
    print(f"  CPU workers          : {args.workers}")
    print(f"{'='*70}")

    checkpoint, kimg_used, metrics_path = resolve_checkpoint(model_dir, args.combo, args.checkpoint)
    out_dir = model_dir / args.output_subdir

    if not args.dry_run and out_dir.is_dir():
        for f in out_dir.glob("*.png"):
            try:
                f.unlink()
            except OSError:
                pass

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Checkpoint  : {checkpoint.name}  ({kimg_used} kimgs)")
    print(f"  Output      : {out_dir}")
    print(f"  Start seed  : {args.seed}  (increments per attempt)")
    print(f"{'-'*70}")

    if args.dry_run:
        print("\n[--dry-run] stopping before model load and generation.\n")
        return

    import torch
    import numpy as np
    import imagehash

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory cleared: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device
                          else "cpu")
    print(f"\nLoading generator on {device}...", end=" ", flush=True)
    with open(checkpoint, "rb") as f:
        G = pickle.load(f)["G_ema"].eval().to(device)
    print("OK")

    parts      = args.combo.split("_")
    if args.combo.startswith("no_tumor"):
        cls_code, plane_code = "nt", parts[2]
    else:
        cls_code   = CLS_MAP.get(parts[0], parts[0][:2])
        plane_code = parts[1]

    if args.skip_reference_dedup:
        reference_hashes = []
    else:
        reference_hashes = _load_reference_hashes(real_data_dir, test_data_dir, args.combo)
    print(f"  Reference hashes : {len(reference_hashes)} "
          f"({'disabled' if args.skip_reference_dedup else 'real/test combo images'})")

    # -- Optional online Mahalanobis filter ------------------------------------
    mahal_filter    = None
    inception_model = None

    if args.classify_out_dir is not None:
        classify_out = Path(args.classify_out_dir).resolve()
        feat_X_path  = classify_out / "features" / "train_real_X.npy"
        manif_path   = classify_out / "manifests" / "train_real.csv"
        inc_weights  = args.inception_weights or (project_root / "inception-2015-12-05.pt")

        _mahal_ok = True
        if not feat_X_path.exists():
            print(f"  WARNING:  Mahalanobis filter disabled - {feat_X_path} not found.")
            _mahal_ok = False
        elif not manif_path.exists():
            print(f"  WARNING:  Mahalanobis filter disabled - {manif_path} not found.")
            _mahal_ok = False
        elif not Path(inc_weights).is_file():
            print(f"  WARNING:  Mahalanobis filter disabled - InceptionV3 weights not found.")
            _mahal_ok = False

        if _mahal_ok:
            import pandas as pd

            cls_full, _ = args.combo.rsplit("_", 1)
            df_real      = pd.read_csv(manif_path)
            X_real_all   = np.load(feat_X_path)
            combo_mask   = (
                (df_real["class"] == cls_full) & (df_real["plane"] == plane_code)
            ).to_numpy()
            X_real_combo = X_real_all[combo_mask]
            n_real_feat  = int(combo_mask.sum())

            if n_real_feat < 2:
                print(f"  WARNING:  Mahalanobis filter disabled - only {n_real_feat} real "
                      f"feature row(s) found for {args.combo} (need >= 2).")
            else:
                mean_, prec_, n_feat, pca = _fit_combo_mahal_model(X_real_combo)
                # v2 empirical threshold: (pct)th percentile of REAL m^2, not
                # chi^2.ppf. Self-calibrating; no normality assumption.
                X_real_reduced = pca.transform(X_real_combo)
                diff_real      = X_real_reduced - mean_
                m2_real        = np.einsum("ij,jk,ik->i", diff_real, prec_, diff_real)
                threshold      = float(np.percentile(m2_real, args.mahal_threshold_pct))

                mahal_filter    = {"mean_": mean_, "prec_": prec_,
                                   "threshold": threshold, "pca": pca}
                inception_model = _load_inception(inc_weights, device)
                print(f"  Mahalanobis filter : ENABLED (v2 empirical threshold)")
                print(f"    Real combo rows  : {n_real_feat}")
                print(f"    Feature dim      : {n_feat}")
                print(f"    Threshold (m^2)   : {threshold:.2f}  "
                      f"[{args.mahal_threshold_pct}th percentile of real m^2]")

    # -- Worker pool (spawn so it works on Windows) ----------------------------
    ctx = get_context("spawn")
    pool: Optional[Pool] = None
    if args.workers > 1:
        pool = ctx.Pool(processes=args.workers)
        print(f"Spawned worker pool: {args.workers} processes")

    print(f"\nGenerating {n_generate} distinct images...")
    saved_hashes  = []
    saved         = 0
    rejected_qual = 0
    rejected_dup  = 0
    rejected_refdup = 0
    attempt       = 0
    t0            = time.time()
    last_log_save = 0

    def _write_generation_audit(completed: bool, failure_reason: str | None = None):
        audit = {
            "combo":                    args.combo,
            "completed":                completed,
            "failure_reason":           failure_reason,
            "target_synthetic":         n_generate,
            "saved":                    saved,
            "real_n":                   real_n,
            "attempts":                 attempt,
            "max_attempts":             max_attempts,
            "quality_rejects":          rejected_qual,
            "synthetic_duplicate_rejects": rejected_dup,
            "reference_duplicate_rejects": rejected_refdup,
            "synthetic_hash_threshold": args.hash_threshold,
            "reference_hash_threshold": reference_hash_threshold,
            "reference_hash_count":     len(reference_hashes),
            "reference_dedup_enabled":  not args.skip_reference_dedup,
            "truncation":               args.truncation,
            "noise_mode":               args.noise_mode,
            "checkpoint":               checkpoint.name,
            "checkpoint_path":          str(checkpoint),
            "metrics_path":             str(metrics_path) if metrics_path is not None else None,
            "kimg":                     kimg_used,
        }
        with open(out_dir / f"generation_audit_v2_{args.combo}.json", "w") as f:
            json.dump(audit, f, indent=2)

    # v2: single Generator object reused across batches and rows. Re-seeding
    # is cheap; allocating is not.
    rng = torch.Generator(device=device)

    try:
        with torch.no_grad():
            while saved < n_generate:
                if attempt >= max_attempts:
                    reason = (
                        f"Reached max_attempts={max_attempts} before saving "
                        f"{n_generate} images (saved={saved})."
                    )
                    _write_generation_audit(False, reason)
                    raise SystemExit(reason)

                B = min(args.batch_size, max_attempts - attempt)

                z_rows = []
                for j in range(B):
                    rng.manual_seed(args.seed + attempt + j)
                    z_rows.append(torch.randn(G.z_dim, generator=rng, device=device))
                z = torch.stack(z_rows, dim=0)
                attempt += B

                imgs = G(z, None,
                         truncation_psi=args.truncation,
                         noise_mode=args.noise_mode)
                imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs_np = imgs.permute(0, 2, 3, 1).cpu().numpy()

                if pool is not None:
                    results = pool.map(_worker, list(imgs_np))
                else:
                    results = [_worker(arr) for arr in imgs_np]

                if mahal_filter is not None:
                    m2_batch = _compute_mahal_batch(
                        results, inception_model, device,
                        mahal_filter["mean_"], mahal_filter["prec_"], mahal_filter["pca"],
                    )
                else:
                    m2_batch = None

                # Serial dedup + save
                for idx, (pil_pp, mean_l, nonzero_frac, phash_str) in enumerate(results):
                    if saved >= n_generate:
                        break
                    # v2 fix: structural quality check ALWAYS runs, even when
                    # the Mahalanobis filter is active. The two checks are
                    # complementary, not exclusive - pool3 features can map
                    # near-black images onto the real centroid in degenerate
                    # cases.
                    if mean_l < 30 or nonzero_frac < 0.08:
                        rejected_qual += 1
                        continue
                    if m2_batch is not None and m2_batch[idx] > mahal_filter["threshold"]:
                        rejected_qual += 1
                        continue
                    new_hash = imagehash.hex_to_hash(phash_str)
                    if any(abs(new_hash - h) <= args.hash_threshold for h in saved_hashes):
                        rejected_dup += 1
                        continue
                    ref_match = None
                    for ref_hash, ref_source, ref_path in reference_hashes:
                        dist = abs(new_hash - ref_hash)
                        if dist <= reference_hash_threshold:
                            ref_match = (ref_source, ref_path, dist)
                            break
                    if ref_match is not None:
                        rejected_refdup += 1
                        continue
                    saved_hashes.append(new_hash)
                    filename = f"brisc2025_synth_{saved+1:05d}_{cls_code}_{plane_code}_t1.png"
                    pil_pp.save(out_dir / filename)
                    saved += 1

                if saved // 100 > last_log_save // 100 or saved == n_generate:
                    last_log_save = saved
                    elapsed  = time.time() - t0
                    rate     = saved / elapsed if elapsed > 0 else 0.0
                    dup_pct  = 100 * rejected_dup  / max(attempt, 1)
                    qual_pct = 100 * rejected_qual / max(attempt, 1)
                    ref_pct  = 100 * rejected_refdup / max(attempt, 1)
                    eta_s    = (n_generate - saved) / rate if rate > 0 else 0
                    print(
                        f"  {saved:5d}/{n_generate}  "
                        f"attempts={attempt}  "
                        f"dup={rejected_dup} ({dup_pct:.1f}%)  "
                        f"refdup={rejected_refdup} ({ref_pct:.1f}%)  "
                        f"qual={rejected_qual} ({qual_pct:.1f}%)  "
                        f"rate={rate:.1f}/s  eta={eta_s/60:.1f}m"
                    )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"COMPLETE (v2)")
    print(f"  Saved            : {saved} synthetic images")
    print(f"  Real images      : {real_n}")
    print(f"  Total for combo  : {saved + real_n}  [{n_source}]")
    print(f"  Total attempts   : {attempt}")
    print(f"  Quality rejects  : {rejected_qual}  ({100*rejected_qual/max(attempt,1):.1f}%)")
    print(f"  Dup rejects      : {rejected_dup}  ({100*rejected_dup/max(attempt,1):.1f}%)")
    print(f"  Ref dup rejects  : {rejected_refdup}  ({100*rejected_refdup/max(attempt,1):.1f}%)")
    print(f"  Elapsed          : {elapsed/60:.2f} min  ({saved/max(elapsed,1e-9):.1f} img/s)")
    print(f"  Output           : {out_dir}")
    print(f"{'='*70}")
    _write_generation_audit(True)

    del G
    if inception_model is not None:
        del inception_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
