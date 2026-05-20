"""
classify_rf_v2.py
==============
v2 of classify_rf.py - adds native 1:1 and 1:2 augmentation experiments,
wired to generate_synthetic_v2.py for the synth pool and to
filter_synthetic_v2.py for the diversity-aware filtered manifests.

EXPERIMENTAL MATRIX
-------------------
For each ratio R in --ratios (default 1.0 2.0):

    baseline             real-only, run once (independent of R)
    augmented_r{R}       real + R*real_n synthetic (no filter)
    filtered_r{R}        real + R*real_n synthetic from the v2 filter
                          (Mahalanobis cutoff + farthest-point sampling)

Same 10 seeds across all five experiments per classifier, so paired
t-tests are valid. Stats and the publication figure compare every
augmented/filtered variant against baseline.

WHY V2 EXISTS
-------------
The 1:1 v1 experiment converged 51% faster than baseline but did not
move accuracy. v2 explores whether (a) a more diverse synthetic pool
(generation parameters tuned in generate_synthetic_v2.py) and
(b) a coverage-driven filter (filter_synthetic_v2.py) raise accuracy at
1:2 - i.e., whether the bottleneck was distribution coverage rather
than count.

CHANGES vs classify_rf.py
----------------------
  * --ratios <R> [<R> ...]    : list, default [1.0 2.0]. Drives manifest,
                                feature-cache and result naming.
  * --synth-subdir            : default 'synthetic_v2' (was 'synthetic').
                                Reads from stylegan3_results/<combo>/<subdir>/.
  * --out-dir                 : default <project-root>/outputs_v2 (keeps v1
                                outputs intact).
  * Manifest naming           : train_augmented_r{R}.csv  (was train_augmented.csv)
  * Feature naming            : train_augmented_r{R}_X.npy etc.
  * Result naming             : augmented_r{R}_seed{i}.json,
                                filtered_r{R}_seed{i}.json
  * RF hyperparameters        : max_features='log2' (was sqrt),
                                min_samples_leaf=2 (was 1).
                                Both reduce per-tree overfitting on the
                                larger augmented training set.
  * Cache safety              : the filtered cache hash now requires the
                                companion augmented cache hash to match,
                                so a stale filtered run cannot silently
                                consume features that were re-extracted
                                after the filter ran.
  * New stages                : 'generate' (subprocess loop over the 12
                                combos) and 'filter' (calls filter_v2).
                                'all' does NOT include 'generate' because
                                it's heavy GPU work; run it explicitly.

USAGE
-----
    # Step 1 - heavy GPU pass, run once. Generates synthetic_v2/ with 2*real_n
    # images per combo so 1:1 and 1:2 share a single pool.
    python classify_rf_v2.py --stage generate

    # Step 2 - everything else (manifests -> features -> baseline -> augmented
    # -> filter -> filtered -> stats -> figure)
    python classify_rf_v2.py

    # Or step-by-step:
    python classify_rf_v2.py --stage features
    python classify_rf_v2.py --stage filter      # writes train_augmented_r{R}_filtered.*
    python classify_rf_v2.py --stage filtered

OUTPUTS  (under <out-dir>, default outputs_v2/)
-------
    manifests/{train_real, train_augmented_r{R}, train_augmented_r{R}_filtered}.csv
    features/{...}_{X,y}.npy
    results/{baseline, augmented_r{R}, filtered_r{R}}_seed{0..9}.json
    results/{baseline, augmented_r{R}, filtered_r{R}}_summary.json
    results/stats_v2.json
    figures/classifier_comparison_v2.{svg,pdf}
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)


# =============================================================================
# CONSTANTS
# =============================================================================
CLASSES         = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX    = {c: i for i, c in enumerate(CLASSES)}
PLANES          = ["ax", "co", "sa"]
COMBOS          = [f"{c}_{p}" for c in CLASSES for p in PLANES]
SEEDS           = list(range(10))
RF_N_ESTIMATORS = 500
RF_TREE_CHECKPOINTS = (25, 50, 100, 200, 300, 400, 500)

# v2 RF hyperparameters: tighter regularization to stop individual trees
# from chasing synthetic noise on the larger augmented training set.
RF_MAX_FEATURES     = "log2"
RF_MIN_SAMPLES_LEAF = 2

DEFAULT_RATIOS = [1.0, 2.0]

REAL_N_BRIEFING = {
    "glioma_ax":      394, "glioma_co":      430, "glioma_sa":      323,
    "meningioma_ax":  423, "meningioma_co":  426, "meningioma_sa":  480,
    "pituitary_ax":   426, "pituitary_co":   510, "pituitary_sa":   521,
    "no_tumor_ax":    352, "no_tumor_co":    310, "no_tumor_sa":    405,
}
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Color palette: baseline + N augmented + N filtered.
BASE_COLOR  = "#4C72B0"
AUG_COLORS  = ["#DD8452", "#E89A65"]   # r1.0, r2.0
FILT_COLORS = ["#2CA02C", "#5DBE5D"]   # r1.0, r2.0


def _bilinear_filter():
    from PIL import Image
    return getattr(Image, "Resampling", Image).BILINEAR


def _setup_windows_env():
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


def _ratio_tag(R: float) -> str:
    """Filename-safe ratio tag using the ":g" float format.

    Examples: 1.0 -> 'r1', 2.0 -> 'r2', 1.5 -> 'r1.5', 0.5 -> 'r0.5'.
    Whole-number ratios drop the trailing '.0' - manifests are written as
    train_augmented_r1.csv / train_augmented_r2.csv, NOT r1.0 / r2.0.
    """
    return f"r{R:g}"


def _filter_pool_ratio(args) -> float:
    if args.filter_pool_ratio is not None:
        return float(args.filter_pool_ratio)
    return float(max(args.ratios) * args.filter_pool_multiplier)


def _manifest_ratios(args):
    ratios = list(args.ratios)
    pool_R = _filter_pool_ratio(args)
    seen = {_ratio_tag(R) for R in ratios}
    if _ratio_tag(pool_R) not in seen:
        ratios.append(pool_R)
    return ratios


# =============================================================================
# PREPROCESSING - shared through preprocess_v2.py for parity across real,
# synthetic, RF feature extraction, and CNN fallback loading.
# =============================================================================
def _import_preprocessing_libs():
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from scipy.ndimage import label, binary_closing
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import flood_fill as sk_flood_fill
    return (np, Image, ndimage, label, binary_closing,
            threshold_otsu, remove_small_objects, sk_flood_fill)


def _flood_fill_background(closed_binary, np, sk_flood_fill):
    padded = np.pad(closed_binary, 2, mode='constant', constant_values=0)
    bg_map = (padded == 0).astype(np.uint8)
    filled = sk_flood_fill(bg_map, (0, 0), 2)
    return (filled == 2)[2:-2, 2:-2]


def _try_mask(img_array, thresh_fraction, close_iters, libs):
    np, _, ndimage, label, binary_closing, threshold_otsu, \
        remove_small_objects, sk_flood_fill = libs
    h, w = img_array.shape
    try:
        thresh = threshold_otsu(img_array) * thresh_fraction
    except Exception:
        thresh = 20.0
    binary = (img_array > thresh).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)
    closed = binary_closing(
        binary, structure=ndimage.iterate_structure(struct, close_iters)
    ).astype(np.uint8)
    external_bg = _flood_fill_background(closed, np, sk_flood_fill)
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


def _compute_skull_strip_mask(img_array, libs):
    np = libs[0]
    h, w = img_array.shape
    attempts = [(0.50, 5), (0.75, 5), (1.00, 7), (1.25, 7), (1.50, 5)]
    best_mask, best_dist = None, 1.0
    for thresh_frac, close_iters in attempts:
        mask, coverage = _try_mask(img_array, thresh_frac, close_iters, libs)
        if mask is None:
            continue
        if 0.15 <= coverage <= 0.85:
            return mask
        dist = abs(coverage - 0.50)
        if dist < best_dist:
            best_mask, best_dist = mask, dist
    return best_mask if best_mask is not None else np.ones((h, w), dtype=np.uint8)


def _normalize_intensity_with_mask(img_array, mask, np):
    brain_pixels = img_array[mask > 0]
    if len(brain_pixels) == 0:
        return ((img_array - img_array.min()) /
                ((img_array.max() - img_array.min()) + 1e-8) * 255).astype(np.uint8)
    p1, p99    = np.percentile(brain_pixels, 1), np.percentile(brain_pixels, 99)
    normalized = np.clip(img_array, p1, p99)
    normalized = ((normalized - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
    return normalized


def _preprocess_pil(pil_img):
    import numpy as np
    from preprocess_v2 import preprocess_array

    normalized = preprocess_array(np.array(pil_img.convert("RGB"), dtype=np.uint8))
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return rgb.astype(np.uint8)


IMG_SIZE = 128
INCEPTION_INPUT = 299


def _worker_preprocess_path(args_tuple):
    from PIL import Image
    import numpy as np
    filepath, project_root_str = args_tuple
    fp = _resolve_manifest_path(filepath, Path(project_root_str))
    pil = Image.open(fp)
    arr = _preprocess_pil(pil)
    pil_pp = Image.fromarray(arr).resize(
        (INCEPTION_INPUT, INCEPTION_INPUT), _bilinear_filter()
    )
    return np.array(pil_pp, dtype=np.uint8)


# =============================================================================
# STAGE A - DATASET ASSEMBLY (per-ratio manifests)
# =============================================================================
def parse_test_filename(fname: str):
    base = Path(fname).stem.split("_")
    if len(base) < 6:
        return None, None
    cls_short, plane = base[3], base[4]
    cls_map = {"gl": "glioma", "me": "meningioma",
               "no": "no_tumor", "pi": "pituitary",
               "nt": "no_tumor"}
    cls = cls_map.get(cls_short)
    if plane not in PLANES:
        plane = None
    return cls, plane


def stage_a_manifests(args, out_dir: Path):
    print(f"\n{'='*72}")
    print("STAGE A - DATASET ASSEMBLY (v2)")
    print(f"{'='*72}")
    manif_dir = out_dir / "manifests"
    manif_dir.mkdir(parents=True, exist_ok=True)

    real_root  = Path(args.real_data_dir)
    synth_root = Path(args.synth_data_dir)
    test_root  = Path(args.test_data_dir)
    project_root = args.project_root

    def _split_combo(combo):
        cls, plane = combo.rsplit("_", 1)
        return cls, plane

    def _portable(fp):
        try:
            return str(fp.resolve().relative_to(project_root))
        except (ValueError, OSError):
            return str(fp.resolve())

    # -- Real -----------------------------------------------------------------
    real_per_combo = {}     # combo -> list[(path, cls, plane, "real")]
    rows_real = []
    print(f"\n[real]  scanning {real_root}")
    for combo in COMBOS:
        cdir = real_root / combo
        cls, plane = _split_combo(combo)
        if not cdir.is_dir():
            print(f"  WARNING:  missing real dir: {cdir}")
            real_per_combo[combo] = []
            continue
        files = sorted(p for p in cdir.iterdir()
                       if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS)
        rows = [(_portable(fp), cls, plane, "real") for fp in files]
        real_per_combo[combo] = rows
        rows_real.extend(rows)
        expected = REAL_N_BRIEFING.get(combo)
        flag = "" if expected is None or len(files) == expected else f"  (briefing: {expected})"
        print(f"    {combo:<14}  {len(files):>5} images{flag}")

    # -- Synthetic pool (lexical-ordered so r1.0 subset r2.0) ----------------------
    synth_per_combo = {}    # combo -> list[(path, cls, plane, "synthetic")] sorted
    print(f"\n[synth] scanning {synth_root}  (subdir: {args.synth_subdir})")
    for combo in COMBOS:
        cdir = synth_root / combo / args.synth_subdir
        cls, plane = _split_combo(combo)
        if not cdir.is_dir():
            print(f"  WARNING:  missing synth dir: {cdir}  (run generate_synthetic_v2.py for {combo})")
            synth_per_combo[combo] = []
            continue
        files = sorted(p for p in cdir.iterdir()
                       if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS)
        rows = [(_portable(fp), cls, plane, "synthetic") for fp in files]
        synth_per_combo[combo] = rows
        print(f"    {combo:<14}  {len(files):>5} images (pool)")

    # -- Test -----------------------------------------------------------------
    rows_test = []
    print(f"\n[test]  scanning {test_root}")
    for cls in CLASSES:
        cdir = test_root / cls
        if not cdir.is_dir():
            print(f"  WARNING:  missing test dir: {cdir}")
            continue
        files = sorted(p for p in cdir.iterdir()
                       if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS)
        n_per_plane = {"ax": 0, "co": 0, "sa": 0, "?": 0}
        for fp in files:
            cls_fname, plane = parse_test_filename(fp.name)
            if cls_fname is None:
                continue
            n_per_plane[plane or "?"] += 1
            rows_test.append((_portable(fp), cls, plane or "?", "real"))
        plane_str = "  ".join(f"{p}={n}" for p, n in n_per_plane.items() if n > 0)
        print(f"    {cls:<14}  {len(files):>5} images   ({plane_str})")

    if not args.keep_train_test_duplicates:
        import hashlib
        import numpy as np
        from PIL import Image

        audit_dir = out_dir / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)

        def _row_pixels(row):
            fp = _resolve_manifest_path(row[0], project_root)
            with Image.open(fp) as im:
                im = im.convert("RGB")
                if im.size != (IMG_SIZE, IMG_SIZE):
                    im = im.resize((IMG_SIZE, IMG_SIZE), _bilinear_filter())
                return np.asarray(im, dtype=np.uint8)

        def _pixel_hash(arr):
            h = hashlib.sha256()
            h.update(str(arr.shape).encode("ascii"))
            h.update(b"\0")
            h.update(str(arr.dtype).encode("ascii"))
            h.update(b"\0")
            h.update(arr.tobytes())
            return h.hexdigest()

        print("\n[dedup] removing train-real rows that are pixel-identical to test images")
        test_hash_to_rows = {}
        for row in rows_test:
            arr = _row_pixels(row)
            test_hash_to_rows.setdefault(_pixel_hash(arr), []).append((row, arr))

        removed_pair_rows = []
        removed_train_rows = []
        removed_train_keys = set()
        cleaned_real_per_combo = {}
        cleaned_rows_real = []
        for combo, rows in real_per_combo.items():
            kept = []
            for row in rows:
                arr = _row_pixels(row)
                h = _pixel_hash(arr)
                exact_test_matches = [
                    test_row for test_row, test_arr in test_hash_to_rows.get(h, [])
                    if np.array_equal(arr, test_arr)
                ]
                if exact_test_matches:
                    if row[0] not in removed_train_keys:
                        removed_train_keys.add(row[0])
                        removed_train_rows.append({
                            "pixel_sha256": h,
                            "pixel_shape": "x".join(str(v) for v in arr.shape),
                            "pixel_dtype": str(arr.dtype),
                            "train_filepath": row[0],
                            "train_class": row[1],
                            "train_plane": row[2],
                        })
                    for test_row in exact_test_matches:
                        removed_pair_rows.append({
                            "pixel_sha256": h,
                            "pixel_shape": "x".join(str(v) for v in arr.shape),
                            "pixel_dtype": str(arr.dtype),
                            "train_filepath": row[0],
                            "train_class": row[1],
                            "train_plane": row[2],
                            "test_filepath": test_row[0],
                            "test_class": test_row[1],
                            "test_plane": test_row[2],
                        })
                    continue
                kept.append(row)
            cleaned_real_per_combo[combo] = kept
            cleaned_rows_real.extend(kept)

        if removed_train_rows:
            clusters = []
            by_hash = {}
            for pair in removed_pair_rows:
                by_hash.setdefault(pair["pixel_sha256"], []).append(pair)
            for pixel_hash, pairs in sorted(by_hash.items()):
                train_paths = sorted({p["train_filepath"] for p in pairs})
                test_paths = sorted({p["test_filepath"] for p in pairs})
                clusters.append({
                    "pixel_sha256": pixel_hash,
                    "pixel_shape": pairs[0]["pixel_shape"],
                    "pixel_dtype": pairs[0]["pixel_dtype"],
                    "unique_train_rows": len(train_paths),
                    "unique_test_rows": len(test_paths),
                    "pairwise_links": len(pairs),
                    "train_filepaths": " | ".join(train_paths),
                    "test_filepaths": " | ".join(test_paths),
                })

            audit_path = audit_dir / "train_rows_removed_due_to_test_pixel_overlap_v2.csv"
            with open(audit_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(removed_train_rows[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(removed_train_rows)

            pairs_path = audit_dir / "train_test_pixel_overlap_pairs_removed_v2.csv"
            with open(pairs_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(removed_pair_rows[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(removed_pair_rows)

            clusters_path = audit_dir / "train_test_pixel_overlap_clusters_removed_v2.csv"
            with open(clusters_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = list(clusters[0].keys())
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(clusters)

            before = len(rows_real)
            rows_real = cleaned_rows_real
            real_per_combo = cleaned_real_per_combo
            after = len(rows_real)
            print(f"    removed train rows : {before - after}")
            print(f"    unique test rows   : {len({p['test_filepath'] for p in removed_pair_rows})}")
            print(f"    pixel identities   : {len(clusters)}")
            print(f"    pairwise links     : {len(removed_pair_rows)}")
            print(f"    train rows kept    : {after}")
            print(f"    train audit CSV    : {audit_path}")
            print(f"    pair audit CSV     : {pairs_path}")
            print(f"    cluster audit CSV  : {clusters_path}")
        else:
            print("    no pixel-identical train/test duplicates found")

    if not any(synth_per_combo.values()):
        msg = (
            f"\n  *** NO SYNTHETIC IMAGES FOUND under "
            f"{synth_root}/<combo>/{args.synth_subdir}/. ***\n"
            "  Run:  python classify_rf_v2.py --stage generate\n"
            "  Or:   python generate_synthetic_v2.py --combo <combo> "
            "--ratio <pool_ratio>  (per combo)\n"
        )
        if args.stage == "all":
            raise SystemExit(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    if not rows_real:
        raise SystemExit(
            f"No preprocessed training images found under {real_root}. "
            "Run preprocess_v2.py --dataset-type train first."
        )
    if not rows_test:
        raise SystemExit(
            f"No preprocessed test images found under {test_root}. "
            "Run preprocess_v2.py --dataset-type test first."
        )

    # -- Write per-ratio augmented manifests ----------------------------------
    # No shuffling: keep deterministic order (real-then-synth, sorted within
    # each combo) so derived caches and slicing stay legible.
    def _write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "class", "plane", "source"])
            w.writerows(rows)

    _write(manif_dir / "train_real.csv", rows_real)
    _write(manif_dir / "test.csv",       rows_test)

    out_paths = {
        "train_real": manif_dir / "train_real.csv",
        "test":       manif_dir / "test.csv",
    }

    print(f"\n{'-'*72}\nPer-ratio augmented manifests\n{'-'*72}")
    for R in _manifest_ratios(args):
        rows_aug = []
        deficits = []
        for combo in COMBOS:
            real_rows = real_per_combo.get(combo, [])
            synth_rows = synth_per_combo.get(combo, [])
            target_synth = int(round(R * len(real_rows)))
            if len(synth_rows) < target_synth:
                deficits.append((combo, len(synth_rows), target_synth))
                kept_synth = synth_rows
            else:
                kept_synth = synth_rows[:target_synth]
            rows_aug.extend(real_rows)
            rows_aug.extend(kept_synth)

        tag = _ratio_tag(R)
        path = manif_dir / f"train_augmented_{tag}.csv"
        _write(path, rows_aug)
        out_paths[f"train_augmented_{tag}"] = path

        n_real_total  = sum(len(rs) for rs in real_per_combo.values())
        n_synth_total = sum(min(int(round(R * len(real_per_combo[c]))),
                                len(synth_per_combo.get(c, [])))
                            for c in COMBOS)
        print(f"  {tag}:  real={n_real_total}  synth={n_synth_total}  "
              f"(target ratio {R:g})  -> {path.name}")
        for combo, have, want in deficits:
            print(f"    WARNING:  {combo}: pool has {have} synth but ratio {R:g} wants {want}")

    print(f"\n  Manifests written to {manif_dir}")
    return out_paths


# =============================================================================
# STAGE B - InceptionV3 feature extraction
# =============================================================================
def _manifest_data_hash(path: Path, project_root: Path) -> str:
    """Hash the manifest plus the bytes of every referenced image.

    Synthetic generation reuses stable filenames, so a manifest-only hash can
    stay unchanged while image contents are replaced. This content-aware hash
    makes feature caches invalidate when the PNG/JPEG bytes change.
    """
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("filepath", "")
            h.update(b"\0filepath\0")
            h.update(rel.encode("utf-8", errors="surrogateescape"))
            fp = _resolve_manifest_path(rel, project_root)
            if not fp.is_file():
                raise FileNotFoundError(
                    f"Manifest {path} references missing image: {fp}"
                )
            with open(fp, "rb") as img_f:
                for chunk in iter(lambda: img_f.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]


def _file_md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _require_feature_manifest_hash(feat_dir: Path, name: str,
                                   manifest_path: Path,
                                   project_root: Path) -> str:
    expected = _manifest_data_hash(manifest_path, project_root)
    hash_path = feat_dir / f"{name}.manifest_hash"
    if not hash_path.exists():
        raise RuntimeError(
            f"[{name}] missing manifest hash file: {hash_path}. "
            f"Re-run the feature/filter stage that creates this cache."
        )
    observed = hash_path.read_text().strip()
    if observed != expected:
        raise RuntimeError(
            f"[{name}] cached features do not match the current manifest. "
            f"expected={expected}, found={observed}. Rebuild the cache."
        )
    return expected


def _validate_filtered_lineage(out_dir: Path, tag: str, project_root: Path) -> None:
    feat_dir = out_dir / "features"
    manif_dir = out_dir / "manifests"
    name = f"train_augmented_{tag}_filtered"
    filt_csv = manif_dir / f"{name}.csv"
    lineage_path = feat_dir / f"{name}.lineage.json"

    filtered_hash = _require_feature_manifest_hash(feat_dir, name, filt_csv, project_root)
    if not lineage_path.exists():
        raise RuntimeError(
            f"[filtered_{tag}] missing lineage file: {lineage_path}. "
            f"Re-run --stage filter with the updated filter_synthetic_v2.py."
        )

    lineage = json.loads(lineage_path.read_text())
    checks = [
        ("filtered_manifest_hash", filtered_hash),
        ("real_manifest_hash", _manifest_data_hash(manif_dir / "train_real.csv", project_root)),
        ("real_feature_hash", _file_md5(feat_dir / "train_real_X.npy")),
        ("real_label_hash", _file_md5(feat_dir / "train_real_y.npy")),
    ]

    pool_name = lineage.get("pool_name")
    if not pool_name:
        raise RuntimeError(
            f"[filtered_{tag}] lineage file does not name the parent pool. "
            f"Re-run --stage filter."
        )
    checks.extend([
        ("pool_manifest_hash", _manifest_data_hash(manif_dir / f"{pool_name}.csv", project_root)),
        ("pool_feature_hash", _file_md5(feat_dir / f"{pool_name}_X.npy")),
        ("pool_label_hash", _file_md5(feat_dir / f"{pool_name}_y.npy")),
    ])

    mismatches = [
        f"{key}: lineage={lineage.get(key)} current={current}"
        for key, current in checks
        if lineage.get(key) != current
    ]
    if mismatches:
        raise RuntimeError(
            f"[filtered_{tag}] filtered feature lineage is stale:\n  "
            + "\n  ".join(mismatches)
            + "\nRe-run classify_rf_v2.py --stage features --force, then --stage filter."
        )


def _load_inception(weights_path: Path, device):
    import torch
    if not weights_path.is_file():
        raise FileNotFoundError(f"InceptionV3 weights not found: {weights_path}")
    return torch.jit.load(str(weights_path), map_location=device).eval()


def _extract_features_for_arrays(arrays_iter, model, device, batch_size, log_label):
    import torch
    import numpy as np
    feats_chunks = []
    buf = []
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for arr in arrays_iter:
            buf.append(arr)
            if len(buf) >= batch_size:
                feats_chunks.append(_flush_inception(buf, model, device))
                n_done += len(buf)
                rate = n_done / max(time.time() - t0, 1e-9)
                print(f"    [{log_label}] {n_done:>6} processed   ({rate:.1f} img/s)")
                buf = []
        if buf:
            feats_chunks.append(_flush_inception(buf, model, device))
            n_done += len(buf)
    if not feats_chunks:
        return np.zeros((0, 2048), dtype=np.float32)
    return np.concatenate(feats_chunks, axis=0)


def _flush_inception(arrays, model, device):
    import torch
    import numpy as np
    batch = np.stack(arrays, axis=0)
    t = torch.from_numpy(batch).to(device)
    t = t.permute(0, 3, 1, 2).contiguous()
    feats = model(t, return_features=True)
    return feats.detach().cpu().numpy().astype(np.float32)


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _read_real_or_synth(path: str, project_root: Path):
    from PIL import Image
    import numpy as np
    fp = _resolve_manifest_path(path, project_root)
    pil = Image.open(fp).convert("RGB").resize(
        (INCEPTION_INPUT, INCEPTION_INPUT), _bilinear_filter()
    )
    return np.array(pil, dtype=np.uint8)


def stage_b_features(args, out_dir: Path, manifests: dict):
    print(f"\n{'='*72}")
    print("STAGE B - InceptionV3 pool3 feature extraction (v2)")
    print(f"{'='*72}")
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    missing_manifests = [p for p in manifests.values() if not p.exists()]
    if missing_manifests:
        raise SystemExit(
            "Missing prerequisite manifest(s); run "
            "classify_rf_v2.py --stage manifests first:\n"
            + "\n".join(f"  {p}" for p in missing_manifests)
        )

    import torch
    import numpy as np
    import pandas as pd

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    if device.type == "cuda":
        print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    print(f"  Device           : {device}")
    print(f"  Inception weights: {args.inception_weights}")
    model = _load_inception(Path(args.inception_weights), device)

    # All splits use the same loader; test images are preprocessed on disk.
    splits = [(name, manifests[name], False) for name in manifests]

    for name, manif_path, needs_preproc in splits:
        x_path    = feat_dir / f"{name}_X.npy"
        y_path    = feat_dir / f"{name}_y.npy"
        hash_path = feat_dir / f"{name}.manifest_hash"
        current_hash = _manifest_data_hash(manif_path, args.project_root)
        cache_valid = (
            x_path.exists() and y_path.exists() and hash_path.exists()
            and hash_path.read_text().strip() == current_hash
        )
        if cache_valid and not args.force:
            X = np.load(x_path); y = np.load(y_path)
            print(f"\n  [{name}] cached (manifest unchanged): X={X.shape} y={y.shape}")
            continue
        if x_path.exists() and not cache_valid and not args.force:
            print(f"\n  [{name}] manifest changed - rebuilding features")

        df = pd.read_csv(manif_path)
        labels = df["class"].map(CLASS_TO_IDX).to_numpy(dtype=np.int64)
        paths  = df["filepath"].tolist()
        print(f"\n  [{name}] {len(paths)} images")

        if needs_preproc:
            ctx = get_context("spawn")
            project_root_str = str(args.project_root)
            worker_args = [(p, project_root_str) for p in paths]
            with ctx.Pool(processes=args.workers) as pool:
                arrays = []
                t0 = time.time()
                for i, arr in enumerate(pool.imap(_worker_preprocess_path, worker_args,
                                                   chunksize=8), start=1):
                    arrays.append(arr)
                    if i % 50 == 0 or i == len(paths):
                        rate = i / max(time.time() - t0, 1e-9)
                        print(f"    [{name}] preprocessed {i:>5}/{len(paths)}   ({rate:.1f}/s)")
            arrays_iter = iter(arrays)
        else:
            arrays_iter = (_read_real_or_synth(p, args.project_root) for p in paths)

        X = _extract_features_for_arrays(
            arrays_iter, model, device, args.batch_size, log_label=name
        )
        if X.shape[0] != len(labels):
            raise RuntimeError(
                f"[{name}] feature/label count mismatch: X={X.shape[0]} y={len(labels)}"
            )
        if not np.isfinite(X).all():
            raise RuntimeError(f"[{name}] features contain NaN/Inf")
        np.save(x_path, X)
        np.save(y_path, labels)
        hash_path.write_text(current_hash)
        print(f"    [{name}] wrote {x_path.name}  shape={X.shape}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# STAGES - RF EXPERIMENTS
# =============================================================================
def _train_eval_rf(X_train, y_train, X_test, y_test, seed):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    checkpoints = sorted({
        int(n) for n in RF_TREE_CHECKPOINTS
        if 1 <= int(n) <= int(RF_N_ESTIMATORS)
    } | {int(RF_N_ESTIMATORS)})
    rf = RandomForestClassifier(
        n_estimators=checkpoints[0],
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=seed,
        bootstrap=True,
        oob_score=True,
        warm_start=True,
    )

    training_history = []
    y_pred = None
    for n_estimators in checkpoints:
        rf.set_params(n_estimators=n_estimators)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*Some inputs do not have OOB scores.*",
                category=UserWarning,
            )
            rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        per_class_f1 = f1_score(y_test, y_pred, labels=list(range(len(CLASSES))),
                                average=None, zero_division=0).tolist()
        training_history.append({
            "source": "rf_tree_growth",
            "seed": int(seed),
            "n_estimators": int(n_estimators),
            "oob_score": float(getattr(rf, "oob_score_", np.nan)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "test_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            **{f"f1_{CLASSES[i]}": float(per_class_f1[i]) for i in range(len(CLASSES))},
        })

    if y_pred is None:
        raise RuntimeError("RF training produced no predictions.")
    y_pred = rf.predict(X_test)
    per_class_f1 = f1_score(y_test, y_pred, labels=list(range(len(CLASSES))),
                            average=None, zero_division=0).tolist()
    return {
        "seed":         seed,
        "accuracy":     float(accuracy_score(y_test, y_pred)),
        "macro_f1":     float(f1_score(y_test, y_pred, average="macro",   zero_division=0)),
        "weighted_f1":  float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": {CLASSES[i]: per_class_f1[i] for i in range(len(CLASSES))},
        "confusion":    confusion_matrix(y_test, y_pred,
                                         labels=list(range(len(CLASSES)))).tolist(),
        "y_true":       y_test.astype(int).tolist(),
        "y_pred":       y_pred.astype(int).tolist(),
        "training_history": training_history,
    }


def _write_history_table(path_base: Path, rows: list[dict]):
    if not rows:
        return

    jsonl_path = path_base.with_suffix(".jsonl")
    csv_path = path_base.with_suffix(".csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_history(res_dir: Path, label: str, seed: int, history: list[dict]) -> list[dict]:
    if not history:
        return []

    hist_dir = res_dir / "training_history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in history:
        saved = dict(row)
        saved.setdefault("experiment", label)
        saved.setdefault("seed", int(seed))
        rows.append(saved)

    _write_history_table(hist_dir / f"{label}_seed{seed}_history", rows)
    return rows


def _summarize_seeds(per_seed):
    import numpy as np
    summary = {"n_seeds": len(per_seed)}
    for k in ("accuracy", "macro_f1", "weighted_f1"):
        vals = np.array([s[k] for s in per_seed], dtype=np.float64)
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    pc = {}
    for cls in CLASSES:
        vals = np.array([s["per_class_f1"][cls] for s in per_seed], dtype=np.float64)
        pc[cls] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    summary["per_class_f1"] = pc
    cms = np.array([s["confusion"] for s in per_seed], dtype=np.float64)
    summary["confusion_mean"] = cms.mean(axis=0).tolist()
    return summary


def _run_experiment(label, X_train, y_train, X_test, y_test, out_dir):
    print(f"\n{'='*72}")
    print(f"EXPERIMENT - {label.upper()}")
    print(f"{'='*72}")
    print(f"  Train shape: X={X_train.shape}  y={y_train.shape}  classes={list(map(int, _bincount(y_train)))}")
    print(f"  Test  shape: X={X_test.shape}   y={y_test.shape}    classes={list(map(int, _bincount(y_test)))}")
    print(f"  RF: n_estimators={RF_N_ESTIMATORS}, max_features={RF_MAX_FEATURES}, "
          f"min_samples_leaf={RF_MIN_SAMPLES_LEAF}, n_jobs=-1, seeds={SEEDS}")

    results = []
    all_history = []
    res_dir = out_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        t0 = time.time()
        r = _train_eval_rf(X_train, y_train, X_test, y_test, seed)
        elapsed = time.time() - t0
        for row in r.get("training_history", []):
            row.setdefault("experiment", label)
        all_history.extend(_write_training_history(
            res_dir, label, seed, r.get("training_history", [])
        ))
        results.append(r)
        print(f"\n  seed={seed:>2}  ({elapsed:5.1f}s)")
        print(f"    accuracy     : {r['accuracy']:.4f}")
        print(f"    macro_f1     : {r['macro_f1']:.4f}")
        print(f"    weighted_f1  : {r['weighted_f1']:.4f}")
        for cls in CLASSES:
            print(f"    f1[{cls:<10}]: {r['per_class_f1'][cls]:.4f}")
        with open(res_dir / f"{label}_seed{seed}.json", "w") as f:
            json.dump(r, f, indent=2)

    if all_history:
        _write_history_table(res_dir / "training_history" / f"{label}_history", all_history)

    summary = _summarize_seeds(results)
    with open(res_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  --- Summary across {len(SEEDS)} seeds -----------------------------")
    for k in ("accuracy", "macro_f1", "weighted_f1"):
        s = summary[k]
        print(f"    {k:<13}: {s['mean']:.4f} +/- {s['std']:.4f}")
    for cls in CLASSES:
        s = summary["per_class_f1"][cls]
        print(f"    f1[{cls:<10}]: {s['mean']:.4f} +/- {s['std']:.4f}")
    return results, summary


def _bincount(y):
    import numpy as np
    return np.bincount(y, minlength=len(CLASSES))


def _load_features(out_dir: Path, names):
    import numpy as np
    feat = out_dir / "features"
    return {
        name: (np.load(feat / f"{name}_X.npy"),
               np.load(feat / f"{name}_y.npy"))
        for name in names
    }


def stage_c_baseline(args, out_dir: Path):
    feats = _load_features(out_dir, ["train_real", "test"])
    X_tr, y_tr = feats["train_real"]
    X_te, y_te = feats["test"]
    return _run_experiment("baseline", X_tr, y_tr, X_te, y_te, out_dir)


def stage_d_augmented(args, out_dir: Path):
    """Run the augmented experiment for every ratio in args.ratios."""
    for R in args.ratios:
        tag = _ratio_tag(R)
        feats = _load_features(out_dir, [f"train_augmented_{tag}", "test"])
        X_tr, y_tr = feats[f"train_augmented_{tag}"]
        X_te, y_te = feats["test"]
        _run_experiment(f"augmented_{tag}", X_tr, y_tr, X_te, y_te, out_dir)


def stage_d_filtered(args, out_dir: Path):
    """Run the filtered experiment for every ratio in args.ratios.

    v2 cache safety: refuse to run if the filtered manifest hash and the
    augmented manifest hash from which it was derived are out of sync -
    that would mean filter_synthetic_v2 has not been re-run since features
    were last extracted, and the filtered features point at stale rows.
    """
    feat_dir = out_dir / "features"
    for R in args.ratios:
        tag    = _ratio_tag(R)
        x_path = feat_dir / f"train_augmented_{tag}_filtered_X.npy"
        y_path = feat_dir / f"train_augmented_{tag}_filtered_y.npy"
        if not x_path.exists() or not y_path.exists():
            print(
                f"\n  [filtered_{tag}] filtered features missing - "
                f"run --stage filter (or filter_synthetic_v2.py) first.\n"
                f"    expected: {x_path.name}, {y_path.name}"
            )
            continue

        # Cache-sync check: filtered features must match their own manifest
        # and the parent real/pool feature caches recorded by the filter.
        import numpy as np
        import pandas as pd
        filt_csv = out_dir / "manifests" / f"train_augmented_{tag}_filtered.csv"
        if filt_csv.exists():
            _validate_filtered_lineage(out_dir, tag, args.project_root)
            df = pd.read_csv(filt_csv)
            if len(df) != np.load(x_path).shape[0]:
                raise RuntimeError(
                    f"[filtered_{tag}] manifest/feature row count mismatch: "
                    f"manifest={len(df)} vs features={np.load(x_path).shape[0]}. "
                    f"Re-run --stage filter."
                )
        else:
            raise RuntimeError(
                f"[filtered_{tag}] filtered manifest missing: {filt_csv}. "
                f"Re-run --stage filter."
            )

        feats = _load_features(out_dir, ["test"])
        X_tr  = np.load(x_path)
        y_tr  = np.load(y_path)
        X_te, y_te = feats["test"]
        _run_experiment(f"filtered_{tag}", X_tr, y_tr, X_te, y_te, out_dir)


# =============================================================================
# STAGE - GENERATE (subprocess loop)
# =============================================================================
def stage_generate(args, out_dir: Path):
    """Invoke generate_synthetic_v2.py for each combo at the filter pool ratio.

    A single overgenerated pool covers the requested augmented ratios and
    leaves replacement candidates for filtered ratios after rejection.
    """
    print(f"\n{'='*72}")
    print("STAGE GENERATE - running generate_synthetic_v2.py per combo")
    print(f"{'='*72}")
    pool_ratio = _filter_pool_ratio(args)
    script = args.project_root / "generate_synthetic_v2.py"
    if not script.exists():
        raise SystemExit(f"generate_synthetic_v2.py not found at {script}")

    for combo in COMBOS:
        cmd = [
            sys.executable, str(script),
            "--combo",         combo,
            "--ratio",         str(pool_ratio),
            "--project-root",  str(args.project_root),
            "--real-data-dir", str(args.real_data_dir),
            "--test-data-dir", str(args.test_data_dir),
            "--device",        args.device,
        ]
        print(f"\n  > {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"generate_synthetic_v2.py failed for {combo} (rc={rc})")

    print(f"\n{'-'*72}\nGenerate stage complete.")


# =============================================================================
# STAGE - FILTER (subprocess to filter_synthetic_v2.py)
# =============================================================================
def stage_filter(args, out_dir: Path):
    print(f"\n{'='*72}")
    print("STAGE FILTER - running filter_synthetic_v2.py")
    print(f"{'='*72}")
    script = args.project_root / "filter_synthetic_v2.py"
    if not script.exists():
        raise SystemExit(f"filter_synthetic_v2.py not found at {script}")

    cmd = [
        sys.executable, str(script),
        "--project-root", str(args.project_root),
        "--out-dir",      str(out_dir),
        "--threshold-pct", str(args.mahal_threshold_pct),
        "--pool-ratio",    str(_filter_pool_ratio(args)),
        "--ratios",       *[str(R) for R in args.ratios],
    ]
    if args.no_filter_figures:
        cmd.append("--no-figures")
    print(f"  > {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit(f"filter_synthetic_v2.py failed (rc={rc})")
    print(f"\n{'-'*72}\nFilter stage complete.")


# =============================================================================
# STAGE E - STATS
# =============================================================================
def _metric_from_predictions(y_true, y_pred, metric):
    from sklearn.metrics import accuracy_score, f1_score

    labels = list(range(len(CLASSES)))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, labels=labels,
                              average="weighted", zero_division=0))
    if metric.startswith("f1["):
        cls = metric[3:-1]
        per_class = f1_score(y_true, y_pred, labels=labels,
                             average=None, zero_division=0)
        return float(per_class[CLASSES.index(cls)])
    raise KeyError(metric)


def _prediction_matrix(seed_results, label):
    import numpy as np

    y_true_ref = None
    preds = []
    for i, result in enumerate(seed_results):
        if "y_true" not in result or "y_pred" not in result:
            raise RuntimeError(
                f"[{label}] seed result {i} lacks per-case predictions. "
                f"Re-run baseline/augmented/filtered with the updated script "
                f"before running --stage stats."
            )
        y_true = np.asarray(result["y_true"], dtype=np.int64)
        y_pred = np.asarray(result["y_pred"], dtype=np.int64)
        if y_true_ref is None:
            y_true_ref = y_true
        elif not np.array_equal(y_true_ref, y_true):
            raise RuntimeError(
                f"[{label}] y_true differs across seeds; paired case-level "
                f"inference is not valid."
            )
        if len(y_pred) != len(y_true_ref):
            raise RuntimeError(f"[{label}] y_pred length mismatch in seed {i}.")
        preds.append(y_pred)
    return y_true_ref, np.stack(preds, axis=0)


def _mean_metric_over_seeds(y_true, pred_matrix, metric):
    import numpy as np

    vals = [
        _metric_from_predictions(y_true, pred_matrix[i], metric)
        for i in range(pred_matrix.shape[0])
    ]
    return float(np.mean(vals))


def _stats_metrics():
    return ["accuracy", "macro_f1", "weighted_f1"] + [f"f1[{c}]" for c in CLASSES]


def _case_resampling_seed(base_seed: int, comparison_idx: int, metric_idx: int) -> int:
    return int(base_seed) + 1_000_003 * int(comparison_idx) + 10_007 * int(metric_idx + 1)


def _paired_case_resampling_metric_worker(payload):
    import numpy as np

    label_a, metric, y_true, pred_a, pred_b, n_iter, rng_seed = payload
    n_cases = len(y_true)
    rng = np.random.default_rng(rng_seed)

    mean_a = _mean_metric_over_seeds(y_true, pred_a, metric)
    mean_b = _mean_metric_over_seeds(y_true, pred_b, metric)
    observed_delta = mean_a - mean_b

    boot_delta = np.empty(n_iter, dtype=np.float64)
    perm_delta = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n_cases, size=n_cases)
        boot_delta[i] = (
            _mean_metric_over_seeds(y_true[idx], pred_a[:, idx], metric)
            - _mean_metric_over_seeds(y_true[idx], pred_b[:, idx], metric)
        )

        swap = rng.random(n_cases) < 0.5
        perm_a = np.where(swap[None, :], pred_b, pred_a)
        perm_b = np.where(swap[None, :], pred_a, pred_b)
        perm_delta[i] = (
            _mean_metric_over_seeds(y_true, perm_a, metric)
            - _mean_metric_over_seeds(y_true, perm_b, metric)
        )

    ci_lo, ci_hi = np.percentile(boot_delta, [2.5, 97.5])
    p_val = (
        np.count_nonzero(np.abs(perm_delta) >= abs(observed_delta)) + 1
    ) / (n_iter + 1)

    return label_a, metric, {
        "mean_a":              mean_a,
        "mean_b":              mean_b,
        "mean_delta":          observed_delta,
        "ci95_low":            float(ci_lo),
        "ci95_high":           float(ci_hi),
        "p_value":             float(p_val),
        "p_value_raw":         float(p_val),
        "n_cases":             n_cases,
        "n_seeds":             int(pred_a.shape[0]),
        "inference_unit":      "test_case",
        "p_value_method":      "paired case-level permutation, two-sided",
        "ci_method":           "paired case-level bootstrap",
    }


def _build_paired_case_resampling_tasks(a_seeds, b_seeds, label_a, label_b,
                                        n_iter: int, rng_seed: int,
                                        comparison_idx: int):
    import numpy as np

    y_true_a, pred_a = _prediction_matrix(a_seeds, label_a)
    y_true_b, pred_b = _prediction_matrix(b_seeds, label_b)
    if not np.array_equal(y_true_a, y_true_b):
        raise RuntimeError(
            f"{label_a} and {label_b} do not share the same ordered test cases."
        )

    tasks = []
    for metric_idx, metric in enumerate(_stats_metrics()):
        tasks.append((
            label_a,
            metric,
            y_true_a,
            pred_a,
            pred_b,
            int(n_iter),
            _case_resampling_seed(rng_seed, comparison_idx, metric_idx),
        ))
    return tasks


def _run_paired_case_resampling_tasks(tasks, n_workers: int):
    if not tasks:
        return []

    worker_count = min(max(1, int(n_workers)), len(tasks))
    if worker_count == 1:
        return [_paired_case_resampling_metric_worker(task) for task in tasks]

    try:
        ctx = get_context("spawn")
        with ctx.Pool(processes=worker_count) as pool:
            return pool.map(_paired_case_resampling_metric_worker, tasks)
    except OSError as exc:
        import warnings
        from concurrent.futures import ThreadPoolExecutor

        warnings.warn(
            f"Process-based stats workers unavailable ({exc}); "
            f"falling back to thread workers.",
            RuntimeWarning,
        )
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(_paired_case_resampling_metric_worker, tasks))


def _paired_case_resampling(a_seeds, b_seeds, label_a, label_b,
                            n_iter: int, rng_seed: int, n_workers: int = 1):
    tasks = _build_paired_case_resampling_tasks(
        a_seeds, b_seeds, label_a, label_b,
        n_iter=n_iter,
        rng_seed=rng_seed,
        comparison_idx=0,
    )
    results = _run_paired_case_resampling_tasks(tasks, n_workers)
    out = {metric: result for _, metric, result in results}
    return {metric: out[metric] for metric in _stats_metrics() if metric in out}


def _holm_adjust_in_place(all_stats, alpha: float):
    import numpy as np

    records = []
    for label, metrics in all_stats.items():
        for metric, result in metrics.items():
            records.append((label, metric, result))
    if not records:
        return

    pvals = np.array([r[2]["p_value_raw"] for r in records], dtype=np.float64)
    order = np.argsort(pvals)
    m_total = len(records)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = min(1.0, (m_total - rank) * float(pvals[idx]))
        running = max(running, adjusted)
        result = records[idx][2]
        result["p_value_holm"] = float(running)
        result["significant_at_alpha_after_holm"] = bool(running <= alpha)
        result["alpha"] = alpha
        result["multiplicity_correction"] = (
            f"Holm across {m_total} comparisons in this stats run"
        )


def stage_e_stats(args, out_dir: Path):
    print(f"\n{'='*72}")
    print("STAGE E - paired case-level resampling vs baseline")
    print(f"{'='*72}")
    print(f"  Resamples        : {args.resample_iters}")
    print(f"  Workers          : {args.stats_workers}")
    print("  P-values         : two-sided paired permutation over test cases")
    print("  CIs              : paired bootstrap over test cases")
    print("  Multiplicity     : Holm correction across all comparisons")
    res_dir = out_dir / "results"

    base_seeds = []
    for s in SEEDS:
        with open(res_dir / f"baseline_seed{s}.json") as f: base_seeds.append(json.load(f))

    labels_to_compare = [f"augmented_{_ratio_tag(R)}" for R in args.ratios] \
                      + [f"filtered_{_ratio_tag(R)}"  for R in args.ratios]
    all_stats = {}
    comparison_order = []
    tasks = []
    for comparison_idx, label in enumerate(labels_to_compare):
        seed_files = [res_dir / f"{label}_seed{s}.json" for s in SEEDS]
        if not all(p.exists() for p in seed_files):
            print(f"\n  [{label}] missing seed files - skipping comparison.")
            continue

        comp_seeds = [json.loads(p.read_text()) for p in seed_files]
        print(f"\n{'-'*72}")
        print(f"  Comparison: {label} vs baseline")
        comparison_order.append(label)
        all_stats[label] = {}
        tasks.extend(_build_paired_case_resampling_tasks(
            comp_seeds, base_seeds, label, "baseline",
            n_iter=args.resample_iters,
            rng_seed=args.resample_seed,
            comparison_idx=comparison_idx,
        ))

    if tasks:
        worker_count = min(max(1, int(args.stats_workers)), len(tasks))
        print(f"\n  Running {len(tasks)} resampling jobs on {worker_count} worker(s).")
        results = _run_paired_case_resampling_tasks(tasks, args.stats_workers)
        for label, metric, result in results:
            all_stats[label][metric] = result

    for label in comparison_order:
        print(f"\n{'-'*72}")
        print(f"  Results: {label} vs baseline")
        for m in _stats_metrics():
            if m not in all_stats[label]:
                continue
            d = all_stats[label][m]
            print(f"    {m:<18} Delta={d['mean_delta']:+.4f}  "
                  f"[{d['ci95_low']:+.4f},{d['ci95_high']:+.4f}]  "
                  f"p_raw={d['p_value_raw']:.4g}")

    _holm_adjust_in_place(all_stats, args.stats_alpha)
    for label, metrics in all_stats.items():
        print(f"\n  Holm-adjusted p-values: {label}")
        for m, d in metrics.items():
            flag = "" if d["p_value_holm"] <= args.stats_alpha else "ns"
            print(f"    {m:<18} p_holm={d['p_value_holm']:.4g}  {flag}")

    out_path = res_dir / "stats_v2.json"
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Wrote {out_path}")
    return all_stats


# =============================================================================
# STAGE F - FIGURE
# =============================================================================
def _stars(p_val: float) -> str:
    if p_val <= 0.001: return "***"
    if p_val <= 0.01:  return "**"
    if p_val <= 0.05:  return "*"
    return "ns"


def _save_figure_with_lock_fallback(fig, out_path: Path, **savefig_kwargs) -> Path:
    try:
        fig.savefig(out_path, **savefig_kwargs)
        return out_path
    except PermissionError as exc:
        alt_path = out_path.with_name(
            f"{out_path.stem}_locked_{time.strftime('%Y%m%d_%H%M%S')}{out_path.suffix}"
        )
        print(
            f"  WARNING: could not overwrite {out_path.name} ({exc}). "
            f"Saving this run as {alt_path.name} instead."
        )
        fig.savefig(alt_path, **savefig_kwargs)
        return alt_path


def stage_f_figure(args, out_dir: Path):
    print(f"\n{'='*72}")
    print("STAGE F - publication figure (v2)")
    print(f"{'='*72}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "font.family":       "DejaVu Sans",
    })

    res_dir = out_dir / "results"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Always require baseline + stats
    if not (res_dir / "baseline_summary.json").exists() \
       or not (res_dir / "stats_v2.json").exists():
        print("  [stage F] missing baseline or stats output. Run stats first.")
        return

    with open(res_dir / "baseline_summary.json") as f: base_sum = json.load(f)
    with open(res_dir / "stats_v2.json")        as f: all_stats = json.load(f)

    # Build experiment list dynamically based on what's on disk.
    exps = [("baseline", "Baseline (real only)", BASE_COLOR, base_sum)]
    for i, R in enumerate(args.ratios):
        tag = _ratio_tag(R)
        sum_path = res_dir / f"augmented_{tag}_summary.json"
        if sum_path.exists():
            with open(sum_path) as f: s = json.load(f)
            exps.append((f"augmented_{tag}",
                         f"Augmented {R:g}x (unfiltered)",
                         AUG_COLORS[i % len(AUG_COLORS)], s))
    for i, R in enumerate(args.ratios):
        tag = _ratio_tag(R)
        sum_path = res_dir / f"filtered_{tag}_summary.json"
        if sum_path.exists():
            with open(sum_path) as f: s = json.load(f)
            exps.append((f"filtered_{tag}",
                         f"Filtered {R:g}x (Mahal+FPS)",
                         FILT_COLORS[i % len(FILT_COLORS)], s))

    n_exp = len(exps)
    print(f"  Plotting {n_exp} experiments: {[e[0] for e in exps]}")

    # Per-seed vectors for bars
    base_pc = {c: [] for c in CLASSES}
    base_ov = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
    for s in SEEDS:
        with open(res_dir / f"baseline_seed{s}.json") as f: b = json.load(f)
        for c in CLASSES: base_pc[c].append(b["per_class_f1"][c])
        for k in base_ov: base_ov[k].append(b[k])

    seeds_pc = [base_pc]
    seeds_ov = [base_ov]
    for label, _, _, _ in exps[1:]:
        pc = {c: [] for c in CLASSES}
        ov = {"accuracy": [], "macro_f1": [], "weighted_f1": []}
        for s in SEEDS:
            p = res_dir / f"{label}_seed{s}.json"
            with open(p) as f: r = json.load(f)
            for c in CLASSES: pc[c].append(r["per_class_f1"][c])
            for k in ov: ov[k].append(r[k])
        seeds_pc.append(pc)
        seeds_ov.append(ov)

    # -- Figure ----------------------------------------------------------------
    n_cm  = n_exp
    fig   = plt.figure(figsize=(5 * n_cm, 11), facecolor="white")
    gs    = gridspec.GridSpec(2, n_cm, figure=fig, hspace=0.30, wspace=0.28,
                              height_ratios=[1.0, 1.05])
    fig.suptitle(
        f"Random Forest classifier - v2 (BRISC 2025, InceptionV3 pool3, {len(SEEDS)} seeds)\n"
        f"max_features={RF_MAX_FEATURES}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}, "
        f"ratios={args.ratios}",
        fontsize=12, fontweight="bold", y=0.995,
    )

    # Top-row width split: per-class F1 spans first 2 columns; overall metrics
    # in the third.  The remaining top-row cells stay empty (or extend bars
    # if N>=4 - we just stretch).
    if n_cm >= 3:
        ax_a = fig.add_subplot(gs[0, 0:max(2, n_cm - 1)])
        ax_b = fig.add_subplot(gs[0, max(2, n_cm - 1):])
    else:
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1]) if n_cm > 1 else fig.add_subplot(gs[0, 0])

    width = 0.8 / n_exp
    half  = (n_exp - 1) / 2.0
    off   = [(-half + i) * width for i in range(n_exp)]

    x = np.arange(len(CLASSES))
    for i, (label, name, col, _) in enumerate(exps):
        m = [np.mean(seeds_pc[i][c]) for c in CLASSES]
        s = [np.std(seeds_pc[i][c],  ddof=1) for c in CLASSES]
        ax_a.bar(x + off[i], m, width, yerr=s, capsize=3, color=col, label=name)

    # Stars use the strongest comparison shown for that experiment vs baseline.
    # We just take "augmented_r{max_R}" as the primary marker if present, else
    # the first non-baseline experiment.
    primary = None
    for label, _, _, _ in exps[1:]:
        if label.startswith("augmented_"):
            primary = label
            break
    primary = primary or (exps[1][0] if len(exps) > 1 else None)
    if primary and primary in all_stats:
        for i_cls, cls in enumerate(CLASSES):
            p = all_stats[primary][f"f1[{cls}]"].get(
                "p_value_holm", all_stats[primary][f"f1[{cls}]"]["p_value"]
            )
            tops = [np.mean(seeds_pc[k][cls]) + np.std(seeds_pc[k][cls], ddof=1)
                    for k in range(n_exp)]
            y_top = max(tops) + 0.02
            ax_a.text(i_cls, y_top, _stars(p), ha="center", va="bottom", fontsize=10)

    ax_a.set_xticks(x); ax_a.set_xticklabels(CLASSES)
    ax_a.set_ylabel("F1 score")
    ax_a.set_title(f"(a) Per-class F1 - mean +/- std across seeds"
                   + (f"   *: {primary} vs baseline" if primary else ""))
    ax_a.set_ylim(0, 1.05)
    ax_a.legend(loc="lower right", framealpha=0.9, fontsize=8)

    metrics_ov = ["accuracy", "macro_f1", "weighted_f1"]
    x_ov = np.arange(len(metrics_ov))
    for i, (label, name, col, _) in enumerate(exps):
        m = [np.mean(seeds_ov[i][k]) for k in metrics_ov]
        s = [np.std(seeds_ov[i][k],  ddof=1) for k in metrics_ov]
        ax_b.bar(x_ov + off[i], m, width, yerr=s, capsize=3, color=col, label=name)
    if primary and primary in all_stats:
        for i_m, met in enumerate(metrics_ov):
            p = all_stats[primary][met].get(
                "p_value_holm", all_stats[primary][met]["p_value"]
            )
            tops = [np.mean(seeds_ov[k][met]) + np.std(seeds_ov[k][met], ddof=1)
                    for k in range(n_exp)]
            y_top = max(tops) + 0.02
            ax_b.text(i_m, y_top, _stars(p), ha="center", va="bottom", fontsize=10)
    ax_b.set_xticks(x_ov); ax_b.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
    ax_b.set_ylabel("Score")
    ax_b.set_title("(b) Overall metrics")
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(loc="lower right", framealpha=0.9, fontsize=8)

    # Confusion matrices - one per experiment
    panel_letters = "(c)(d)(e)(f)(g)(h)"
    for col_i, (label, name, _, summary) in enumerate(exps):
        ax = fig.add_subplot(gs[1, col_i])
        cm        = np.array(summary["confusion_mean"])
        row_sums  = cm.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums > 0, row_sums, 1.0)
        cm_norm   = np.where(row_sums > 0, cm / safe_sums, 0.0)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=CLASSES, yticklabels=CLASSES,
                    cbar=(col_i == n_cm - 1), ax=ax, vmin=0, vmax=1, square=True,
                    annot_kws={"fontsize": 9})
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{panel_letters[col_i*3:col_i*3+3]} {name}\n(row-normalized, mean over seeds)",
                     fontsize=9)

    fig.text(0.5, 0.005,
             "Significance markers use Holm-adjusted p: *** p<=0.001  **  p<=0.01  *  p<=0.05  ns  not significant"
             + (f"   |   stars: {primary} vs baseline" if primary else ""),
             ha="center", fontsize=8, style="italic", color="#444444")

    out_svg = fig_dir / "classifier_comparison_v2.svg"
    out_pdf = fig_dir / "classifier_comparison_v2.pdf"
    saved_svg = _save_figure_with_lock_fallback(
        fig, out_svg, format="svg", bbox_inches="tight", facecolor="white")
    saved_pdf = _save_figure_with_lock_fallback(
        fig, out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote {saved_svg}")
    print(f"  Wrote {saved_pdf}")


# =============================================================================
# CLI / MAIN
# =============================================================================
STAGE_CHOICES = ("all", "manifests", "generate", "features",
                 "baseline", "augmented", "filter", "filtered",
                 "stats", "figure")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stage", default="all", choices=STAGE_CHOICES)
    p.add_argument("--project-root", type=Path,
                   default=Path(__file__).resolve().parent)
    p.add_argument("--real-data-dir",  type=Path, default=None)
    p.add_argument("--synth-data-dir", type=Path, default=None)
    p.add_argument("--synth-subdir",   default="synthetic_v2",
                   help="Subdir under stylegan3_results/<combo>/. Default 'synthetic_v2'.")
    p.add_argument("--test-data-dir",  type=Path, default=None)
    p.add_argument("--inception-weights", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                   help="Augmentation ratios to evaluate. Default: 1.0 2.0.")
    p.add_argument("--filter-pool-ratio", type=float, default=None,
                   help="Synthetic pool ratio used for generation/filtering. "
                        "Default: max(--ratios) * --filter-pool-multiplier.")
    p.add_argument("--filter-pool-multiplier", type=float, default=1.25,
                   help="Over-generate the synthetic pool so filtered ratios can "
                        "still hit their requested target after rejection.")
    p.add_argument("--mahal-threshold-pct", type=float, default=97.5,
                   help="Empirical percentile for filter_synthetic_v2 cutoff.")
    p.add_argument("--keep-train-test-duplicates", action="store_true",
                   help="Do not remove train-real rows that are pixel-identical "
                        "to test images. Default is to remove them from training.")
    p.add_argument("--resample-iters", type=int, default=5000,
                   help="Case-level bootstrap/permutation iterations for stats.")
    p.add_argument("--resample-seed", type=int, default=12345,
                   help="RNG seed for case-level resampling in stats.")
    p.add_argument("--stats-alpha", type=float, default=0.05,
                   help="Family-wise alpha used after Holm correction.")
    p.add_argument("--stats-workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1),
                   help="Worker processes for Stage E case-level resampling.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int,
                   default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="Re-run stages even if their outputs exist on disk.")
    p.add_argument("--no-filter-figures", action="store_true",
                   help="Skip filter_synthetic_v2's own figure pass.")
    return p.parse_args(argv)


def _resolve_paths(args):
    pr = args.project_root.resolve()
    args.project_root      = pr
    args.real_data_dir     = args.real_data_dir     or pr / "brisc2025_preprocessed"
    args.synth_data_dir    = args.synth_data_dir    or pr / "stylegan3_results"
    args.test_data_dir     = args.test_data_dir     or pr / "brisc2025_test_preprocessed"
    args.inception_weights = args.inception_weights or pr / "inception-2015-12-05.pt"
    args.out_dir           = args.out_dir           or pr / "outputs_v2"


def _print_header(args):
    print(f"\n{'='*72}")
    print("CLASSIFY_RF_V2.PY - Augmentation experiments at multiple ratios")
    print(f"{'='*72}")
    print(f"  Project root        : {args.project_root}")
    print(f"  Real data dir       : {args.real_data_dir}")
    print(f"  Synthetic data dir  : {args.synth_data_dir}")
    print(f"  Synth subdir        : {args.synth_subdir}")
    print(f"  Test data dir       : {args.test_data_dir}")
    print(f"  InceptionV3         : {args.inception_weights}")
    print(f"  Output dir          : {args.out_dir}")
    print(f"  Ratios              : {args.ratios}")
    print(f"  Filter pool ratio   : {_filter_pool_ratio(args):g}")
    print(f"  Stage               : {args.stage}   (force={args.force})")


def main(argv=None):
    _setup_windows_env()
    args = parse_args(argv)
    _resolve_paths(args)
    if args.resample_iters <= 0:
        raise SystemExit("--resample-iters must be > 0.")
    if any(R <= 0 for R in args.ratios):
        raise SystemExit("--ratios must all be > 0.")
    if args.filter_pool_multiplier < 1.0:
        raise SystemExit("--filter-pool-multiplier must be >= 1.0.")
    if args.filter_pool_ratio is not None and args.filter_pool_ratio < max(args.ratios):
        raise SystemExit("--filter-pool-ratio must be >= max(--ratios).")
    if not (0.0 < args.stats_alpha < 1.0):
        raise SystemExit("--stats-alpha must be between 0 and 1.")
    if args.stats_workers <= 0:
        raise SystemExit("--stats-workers must be > 0.")
    _print_header(args)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {
        "train_real": out_dir / "manifests" / "train_real.csv",
        "test":       out_dir / "manifests" / "test.csv",
    }
    for R in _manifest_ratios(args):
        tag = _ratio_tag(R)
        manifests[f"train_augmented_{tag}"] = out_dir / "manifests" / f"train_augmented_{tag}.csv"

    run_all = args.stage == "all"

    if args.stage == "generate":
        stage_generate(args, out_dir)
        return

    if run_all or args.stage == "manifests":
        all_present = all(p.exists() for p in manifests.values())
        if all_present and not args.force:
            print(f"\n  [stage A] manifests cached, skipping (use --force to rebuild)")
        else:
            stage_a_manifests(args, out_dir)

    if run_all or args.stage == "features":
        stage_b_features(args, out_dir, manifests)

    if run_all or args.stage == "baseline":
        stage_c_baseline(args, out_dir)

    if run_all or args.stage == "augmented":
        stage_d_augmented(args, out_dir)

    if run_all or args.stage == "filter":
        stage_filter(args, out_dir)
        # The filter stage produced new manifests; extract their features now.
        if run_all:
            filtered_manifests = {}
            for R in args.ratios:
                tag = _ratio_tag(R)
                p = out_dir / "manifests" / f"train_augmented_{tag}_filtered.csv"
                if p.exists():
                    filtered_manifests[f"train_augmented_{tag}_filtered"] = p
            if filtered_manifests:
                # Filtered features were ALREADY written by filter_synthetic_v2
                # (it slices the pool feature array). But the .manifest_hash
                # file uses the filtered manifest, so the cache check passes
                # without re-extraction. Just confirm presence.
                feat_dir = out_dir / "features"
                for name in filtered_manifests:
                    if not (feat_dir / f"{name}_X.npy").exists():
                        print(f"\n  [{name}] features missing post-filter - "
                              f"this should have been written by filter_synthetic_v2.")

    if run_all or args.stage == "filtered":
        stage_d_filtered(args, out_dir)

    if run_all or args.stage == "stats":
        stage_e_stats(args, out_dir)

    if run_all or args.stage == "figure":
        stage_f_figure(args, out_dir)

    print(f"\n{'='*72}")
    print("DONE (classify_rf_v2)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
