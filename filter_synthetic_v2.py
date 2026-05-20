"""
filter_synthetic_v2.py
======================
v2 of filter_synthetic.py - diversity-aware synthetic selector for the
1:1 / 1:2 augmentation experiments.

WHY V2 EXISTS
-------------
The v1 filter was a one-sided outlier remover: it fit a PCA-reduced
LedoitWolf model on real pool3 features and rejected synthetic samples
whose squared Mahalanobis distance (m^2) exceeded chi2.ppf(p, df). With a
generator whose recall is bounded by patient-level repetition in the
training set, almost every synthetic sample lands NEAR the real centroid,
so the v1 filter was effectively a no-op - it could not recognise the
problem (low coverage), only the absence of one (high typicality).

v2 reframes filtering as a TWO-STAGE selection:

    1. Empirical Mahalanobis cutoff
       Threshold = the (threshold_pct)th percentile of the *real* m^2
       distribution, not chi2.ppf. Self-calibrating, no normality
       assumption. Removes only synthetic that is more atypical than the
       least-typical real image.

    2. Farthest-Point Sampling (FPS) diversity selection
       Among the survivors, greedily pick a subset that maximally covers
       the real manifold in PCA space. Starts from the survivor closest
       to the real centroid and adds the survivor farthest from any
       already-selected point until the per-ratio target is hit.
       Deterministic.

Because FPS adds points sequentially, slicing the same FPS ordering at
different sizes produces NESTED subsets - the 1:1 filtered set is exactly
the first N points of the 1:2 filtered set. This makes 1:1 vs 1:2
comparisons clean: the only difference is how many points are added past
the same starting selection.

OTHER V2 CHANGES
----------------
  * PCA n_components                : 50 -> 200
        Captures finer texture variance. With ~400 real samples per combo
        and LedoitWolf shrinkage, n/d ~ 2 keeps the precision matrix
        well-conditioned. The 50-d space modelled only global structure,
        on which low-recall generators look most "valid."
  * --ratios                         : new argument
        Default [1.0, 2.0]. Produces one filtered manifest+feature pair
        per ratio in a single pass.

PREREQUISITE
------------
    python generate_synthetic_v2.py --combo <combo> --ratio 2.0     # for each combo
    python classify_rf_v2.py --stage features                          # builds the pool

INPUTS  (under <out-dir>, default <project-root>/outputs_v2/)
------
    manifests/train_augmented_r{pool}.csv   (the pool; ratio is formatted
    features/train_augmented_r{pool}_{X,y}.npy with ":g", so 2.0 -> "r2", 2.5 -> "r2.5")
    manifests/train_real.csv
    features/train_real_{X,y}.npy

OUTPUTS  (one set per ratio)
-------
    manifests/train_augmented_r{R}_filtered.csv
    features/train_augmented_r{R}_filtered_{X,y}.npy
    features/train_augmented_r{R}_filtered.manifest_hash
    results/filter_audit_v2.json
    figures/mahalanobis_distances_v2.{svg,pdf}
    figures/mahalanobis_summary_v2.{svg,pdf}
    figures/umap_source_v2.{svg,pdf}
    figures/umap_class_v2.{svg,pdf}

USAGE
-----
    python filter_synthetic_v2.py
    python filter_synthetic_v2.py --ratios 1.0 2.0
    python filter_synthetic_v2.py --threshold-pct 95 --no-figures
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)


# =============================================================================
# CONSTANTS
# =============================================================================
CLASSES     = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PLANES      = ["ax", "co", "sa"]
COMBOS      = [f"{c}_{p}" for c in CLASSES for p in PLANES]

DEFAULT_THRESHOLD_PCT = 97.5
DEFAULT_RATIOS        = [1.0, 2.0]
PCA_MAX_COMPONENTS    = 200    # was 50 in v1

BASE_COLOR  = "#4C72B0"
AUG_COLOR   = "#DD8452"
REJ_COLOR   = "#C44E52"
FPS_COLOR   = "#2CA02C"
THRES_COLOR = "#9467BD"


# =============================================================================
# CORE
# =============================================================================
def _split_combo(combo: str):
    cls, plane = combo.rsplit("_", 1)
    return cls, plane


def _threshold_tag(threshold_pct: float) -> str:
    return f"p{threshold_pct:g}"


def _filtered_name(ratio_tag: str, threshold_tag: str = "") -> str:
    suffix = f"_{threshold_tag}" if threshold_tag else ""
    return f"train_augmented_{ratio_tag}_filtered{suffix}"


def _file_md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _manifest_data_hash(manifest_path: Path, project_root: Path) -> str:
    """Hash the manifest plus the bytes of every referenced image."""
    import csv
    import hashlib

    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    with open(manifest_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rel = row.get("filepath", "")
            h.update(b"\0filepath\0")
            h.update(rel.encode("utf-8", errors="surrogateescape"))
            fp = _resolve_manifest_path(rel, project_root)
            if not fp.is_file():
                raise FileNotFoundError(
                    f"Manifest {manifest_path} references missing image: {fp}"
                )
            with open(fp, "rb") as img_f:
                for chunk in iter(lambda: img_f.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]


def _require_manifest_hash(feat_dir: Path, name: str, manifest_path: Path,
                           project_root: Path) -> str:
    expected = _manifest_data_hash(manifest_path, project_root)
    hash_path = feat_dir / f"{name}.manifest_hash"
    if not hash_path.exists():
        raise RuntimeError(
            f"[{name}] missing manifest hash file: {hash_path}. "
            f"Re-run classify_rf_v2.py --stage features."
        )
    observed = hash_path.read_text().strip()
    if observed != expected:
        raise RuntimeError(
            f"[{name}] feature cache does not match the current manifest. "
            f"expected={expected}, found={observed}. "
            f"Re-run classify_rf_v2.py --stage features --force."
        )
    return expected


def _fit_combo_model(X_real_combo):
    """PCA-reduced LedoitWolf model on pool3 real features.

    v2 dimensionality is 200 (was 50). With ~400 samples and shrinkage,
    this stays well-conditioned and captures texture-level information
    that the 50-d model collapses away.
    """
    import numpy as np
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA

    n_samples = X_real_combo.shape[0]
    n_components = min(n_samples - 1, PCA_MAX_COMPONENTS)

    pca = PCA(n_components=n_components, whiten=False, random_state=0)
    X_reduced = pca.fit_transform(X_real_combo)

    lw = LedoitWolf(assume_centered=False)
    lw.fit(X_reduced)

    mean_  = X_reduced.mean(axis=0)
    prec_  = lw.get_precision()
    n_feat = n_components

    return mean_, prec_, n_feat, pca


def _mahalanobis_sq(X, mean, precision):
    import numpy as np
    diff = X - mean
    m2 = np.einsum("ij,jk,ik->i", diff, precision, diff)
    return m2


def _farthest_point_sampling(X, n_select, start_point=None):
    """Greedy farthest-point sampling in Euclidean space.

    Deterministic: starts from the point closest to the centroid (no RNG),
    then repeatedly adds the point with the largest minimum distance to
    any already-selected point.

    Returns an ordered array of n_select indices into X. Slicing the
    returned array at any K <= n_select gives a valid K-point FPS subset
    (the first K added). This nesting property is what lets the v2 filter
    derive the 1:1 and 1:2 filtered sets from a single FPS pass.
    """
    import numpy as np
    N = X.shape[0]
    if n_select >= N:
        # Order remaining points by FPS even when keeping them all - gives
        # a meaningful "progression" for nested slicing later.
        n_select = N

    if start_point is None:
        start_point = X.mean(axis=0)
    start_dists = np.linalg.norm(X - start_point, axis=1)
    start       = int(np.argmin(start_dists))

    selected = [start]
    dists    = np.linalg.norm(X - X[start], axis=1)

    for _ in range(n_select - 1):
        idx = int(np.argmax(dists))
        if dists[idx] == 0.0:
            # All remaining points coincide with already-selected ones; stop.
            break
        selected.append(idx)
        new_d = np.linalg.norm(X - X[idx], axis=1)
        dists = np.minimum(dists, new_d)

    return np.array(selected, dtype=int)


def _filter_one_combo(X_real_combo, X_synth_combo, threshold_pct, ratios, real_n):
    """Apply the two-stage v2 filter to one combo.

    Returns
    -------
    per_ratio_local_keep : dict {R -> np.ndarray of indices into X_synth_combo}
        The local synth indices kept for each requested ratio.
    info : dict
        Audit info shared across ratios + per-ratio breakdown.
    """
    import numpy as np

    n_synth = X_synth_combo.shape[0]
    if n_synth == 0:
        return ({R: np.array([], dtype=int) for R in ratios},
                {"n_real": X_real_combo.shape[0], "n_synth": 0,
                 "n_after_mahal": 0, "threshold": None, "n_features": 0,
                 "skipped": True, "per_ratio": {}, "m2_real": np.array([]),
                 "m2_synth": np.array([])})

    mean_, prec_, n_feat, pca = _fit_combo_model(X_real_combo)

    X_real_reduced  = pca.transform(X_real_combo)
    X_synth_reduced = pca.transform(X_synth_combo)

    m2_real  = _mahalanobis_sq(X_real_reduced,  mean_, prec_)
    m2_synth = _mahalanobis_sq(X_synth_reduced, mean_, prec_)

    threshold = float(np.percentile(m2_real, threshold_pct))

    keep_mask    = m2_synth <= threshold
    survivor_idx = np.where(keep_mask)[0]
    n_survivors  = len(survivor_idx)

    per_ratio_local_keep = {}
    per_ratio_audit      = {}

    if n_survivors == 0:
        for R in ratios:
            per_ratio_local_keep[R] = np.array([], dtype=int)
            per_ratio_audit[R] = {
                "n_target": int(round(R * real_n)),
                "n_after_mahal": 0,
                "n_after_fps":   0,
                "n_kept":        0,
            }
    else:
        max_target = max(int(round(R * real_n)) for R in ratios)
        n_to_pick  = min(max_target, n_survivors)

        # FPS in real-calibrated Mahalanobis geometry.
        X_survivors = X_synth_reduced[survivor_idx]
        try:
            chol = np.linalg.cholesky(prec_)
            X_survivors_geom = (X_survivors - mean_) @ chol
        except np.linalg.LinAlgError:
            warnings.warn(
                "Precision matrix was not positive definite for FPS geometry; "
                "falling back to centered PCA coordinates.",
                RuntimeWarning,
                stacklevel=2,
            )
            X_survivors_geom = X_survivors - mean_
        fps_order = _farthest_point_sampling(
            X_survivors_geom,
            n_to_pick,
            start_point=np.zeros(X_survivors_geom.shape[1]),
        )
        # fps_order indexes into survivor_idx; map back to local synth indices.
        survivor_keep_global = survivor_idx[fps_order]

        for R in ratios:
            target = int(round(R * real_n))
            actual = min(target, len(survivor_keep_global))
            per_ratio_local_keep[R] = survivor_keep_global[:actual]
            per_ratio_audit[R] = {
                "n_target":     target,
                "n_after_mahal": n_survivors,
                "n_after_fps":   actual,
                "n_kept":        actual,
            }

    info = {
        "n_real":          X_real_combo.shape[0],
        "n_synth":         n_synth,
        "n_rejected_mahal": int(n_synth - n_survivors),
        "n_after_mahal":   int(n_survivors),
        "threshold":       round(threshold, 4),
        "median_m2_real":  round(float(np.median(m2_real)),  4),
        "p975_m2_real":    round(float(np.percentile(m2_real, 97.5)), 4),
        "median_m2_synth": round(float(np.median(m2_synth)), 4),
        "n_features":      n_feat,
        "skipped":         False,
        "per_ratio":       per_ratio_audit,
        "m2_real":         m2_real,
        "m2_synth":        m2_synth,
    }
    return per_ratio_local_keep, info


# =============================================================================
# DRIVER
# =============================================================================
def run_filter(args, out_dir: Path):
    import numpy as np
    import pandas as pd

    print(f"\n{'='*72}")
    print("FILTER_SYNTHETIC_V2 - empirical threshold + FPS diversity")
    print(f"{'='*72}")
    print(f"  Threshold pct    : {args.threshold_pct}%   (empirical, real-m^2 percentile)")
    print(f"  Ratios           : {args.ratios}")
    print(f"  PCA components   : up to {PCA_MAX_COMPONENTS}")
    print(f"  Out dir          : {out_dir}")

    feat_dir  = out_dir / "features"
    manif_dir = out_dir / "manifests"
    res_dir   = out_dir / "results"
    fig_dir   = out_dir / "figures"
    for d in (res_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    project_root = args.project_root.resolve()
    pool_ratio = args.pool_ratio if args.pool_ratio is not None else max(args.ratios)
    pool_tag   = f"r{pool_ratio:g}"
    output_threshold_tag = getattr(args, "output_threshold_tag", "")

    pool_X_path = feat_dir  / f"train_augmented_{pool_tag}_X.npy"
    pool_y_path = feat_dir  / f"train_augmented_{pool_tag}_y.npy"
    pool_csv    = manif_dir / f"train_augmented_{pool_tag}.csv"
    real_X_path = feat_dir  / "train_real_X.npy"
    real_y_path = feat_dir  / "train_real_y.npy"
    real_csv    = manif_dir / "train_real.csv"

    missing = [p for p in (pool_X_path, pool_y_path, pool_csv,
                           real_X_path, real_y_path, real_csv) if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing prerequisite files - run classify_rf_v2.py --stage features first:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    pool_name = f"train_augmented_{pool_tag}"
    pool_manifest_hash = _require_manifest_hash(feat_dir, pool_name, pool_csv, project_root)
    real_manifest_hash = _require_manifest_hash(feat_dir, "train_real", real_csv, project_root)
    pool_feature_hash = _file_md5(pool_X_path)
    pool_label_hash = _file_md5(pool_y_path)
    real_feature_hash = _file_md5(real_X_path)
    real_label_hash = _file_md5(real_y_path)

    print("\n  Loading features...")
    X_pool  = np.load(pool_X_path)
    y_pool  = np.load(pool_y_path)
    X_real  = np.load(real_X_path)
    df_pool = pd.read_csv(pool_csv)
    df_real = pd.read_csv(real_csv)

    if len(df_pool) != X_pool.shape[0]:
        raise RuntimeError(
            f"Pool manifest row count ({len(df_pool)}) does not match feature "
            f"array length ({X_pool.shape[0]})."
        )
    if len(df_real) != X_real.shape[0]:
        raise RuntimeError(
            f"Real manifest row count ({len(df_real)}) does not match feature "
            f"array length ({X_real.shape[0]})."
        )

    print(f"    train_real          X={X_real.shape}")
    print(f"    {pool_csv.stem:<20}X={X_pool.shape}")

    # Per-combo filter
    print(f"\n{'-'*88}")
    print(f"  {'combo':<14} {'real':>5} {'synth':>5} {'mahal':>5} "
          f"{'thresh':>8} {'p975_m2':>9} {'med_synth':>10} "
          + " ".join(f"r{R:g}" for R in args.ratios))
    print(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*9} {'-'*10} "
          + " ".join("-" * len(f"r{R:g}") for R in args.ratios))

    audit       = {"_meta": {
        "threshold_pct":  args.threshold_pct,
        "ratios":         args.ratios,
        "pool_ratio":     pool_ratio,
        "pca_components": PCA_MAX_COMPONENTS,
        "output_threshold_tag": output_threshold_tag,
        "input_hashes": {
            "pool_name":           pool_name,
            "pool_manifest_hash":  pool_manifest_hash,
            "pool_feature_hash":   pool_feature_hash,
            "pool_label_hash":     pool_label_hash,
            "real_manifest_hash":  real_manifest_hash,
            "real_feature_hash":   real_feature_hash,
            "real_label_hash":     real_label_hash,
        },
    }}
    combo_data  = {}

    # global keep masks per ratio (over the pool manifest rows)
    keep_masks = {R: np.zeros(len(df_pool), dtype=bool) for R in args.ratios}
    # always keep real rows in any ratio's filtered manifest
    real_pool_mask = (df_pool["source"] == "real").to_numpy()
    for R in args.ratios:
        keep_masks[R][real_pool_mask] = True

    for combo in COMBOS:
        cls, plane = _split_combo(combo)

        # Real features for this combo (from train_real)
        real_mask_df = (df_real["class"] == cls) & (df_real["plane"] == plane)
        real_idxs    = real_mask_df[real_mask_df].index.to_numpy()
        X_real_combo = X_real[real_idxs]
        n_real       = len(real_idxs)

        # Synthetic rows for this combo (from pool)
        synth_mask_df = (
            (df_pool["source"] == "synthetic")
            & (df_pool["class"] == cls)
            & (df_pool["plane"] == plane)
        )
        synth_pool_idxs = np.where(synth_mask_df.to_numpy())[0]
        X_synth_combo   = X_pool[synth_pool_idxs]

        if n_real < 2:
            warnings.warn(
                f"[{combo}] only {n_real} real image(s) - skipping filter, "
                f"keeping all synthetic at every ratio.",
                RuntimeWarning, stacklevel=2,
            )
            for R in args.ratios:
                keep_masks[R][synth_pool_idxs] = True
            audit[combo] = {
                "n_real": n_real, "n_synth": int(len(synth_pool_idxs)),
                "skipped": True, "threshold": None, "per_ratio": {},
            }
            combo_data[combo] = None
            continue

        if len(synth_pool_idxs) == 0:
            audit[combo] = {
                "n_real": n_real, "n_synth": 0, "skipped": True,
                "threshold": None, "per_ratio": {},
            }
            combo_data[combo] = None
            continue

        per_ratio_local_keep, info = _filter_one_combo(
            X_real_combo, X_synth_combo, args.threshold_pct,
            args.ratios, real_n=n_real,
        )

        # Map local keeps back to global pool indices
        for R, local_keep in per_ratio_local_keep.items():
            global_keep = synth_pool_idxs[local_keep]
            keep_masks[R][global_keep] = True

        # Pretty row
        row_ratio_cols = "  ".join(
            f"{info['per_ratio'][R]['n_kept']:>{max(2,len(f'r{R:g}'))}}"
            for R in args.ratios
        )
        print(f"  {combo:<14} {n_real:>5} {info['n_synth']:>5} "
              f"{info['n_after_mahal']:>5} {info['threshold']:>8.2f} "
              f"{info['p975_m2_real']:>9.2f} {info['median_m2_synth']:>10.2f} "
              f"{row_ratio_cols}")

        # Strip the heavy m^2 arrays from the audit JSON; keep them for figures only.
        audit_entry = {k: v for k, v in info.items()
                       if k not in ("m2_real", "m2_synth")}
        audit[combo] = audit_entry
        combo_data[combo] = {
            "m2_real":   info["m2_real"],
            "m2_synth":  info["m2_synth"],
            "threshold": info["threshold"],
            "n_comp":    info["n_features"],
            "per_ratio_local_keep": per_ratio_local_keep,
            "synth_pool_idxs":       synth_pool_idxs,
        }

    # -- Per-ratio totals ------------------------------------------------------
    n_total_synth = int((df_pool["source"] == "synthetic").sum())
    print(f"\n  {'-'*14}")
    for R in args.ratios:
        n_kept_synth   = int(keep_masks[R].sum() - real_pool_mask.sum())
        n_drop         = n_total_synth - n_kept_synth
        pct            = 100 * n_drop / max(n_total_synth, 1)
        print(f"  r{R:g}: kept {n_kept_synth} of {n_total_synth} synth  "
              f"(dropped {n_drop}, {pct:.1f}%)")
        audit.setdefault("_totals", {})[f"r{R:g}"] = {
            "n_synth_pool":  n_total_synth,
            "n_synth_kept":  n_kept_synth,
            "n_synth_drop":  n_drop,
            "pct_dropped":   round(pct, 2),
        }

    # -- Write filtered manifests + features per ratio -------------------------
    for R in args.ratios:
        tag         = f"r{R:g}"
        out_name    = _filtered_name(tag, output_threshold_tag)
        df_filt     = df_pool[keep_masks[R]].reset_index(drop=True)
        out_manif   = manif_dir / f"{out_name}.csv"
        df_filt.to_csv(out_manif, index=False)
        print(f"\n  Wrote manifest -> {out_manif.name}  ({len(df_filt)} rows)")

        X_filt = X_pool[keep_masks[R]]
        y_filt = y_pool[keep_masks[R]]
        np.save(feat_dir / f"{out_name}_X.npy", X_filt)
        np.save(feat_dir / f"{out_name}_y.npy", y_filt)
        print(f"  Wrote features  -> {out_name}_{{X,y}}.npy "
              f"shape={X_filt.shape}")

        h = _manifest_data_hash(out_manif, project_root)
        (feat_dir / f"{out_name}.manifest_hash").write_text(h)
        lineage = {
            "filtered_manifest_hash": h,
            "filtered_name":          out_name,
            "pool_name":              pool_name,
            "pool_manifest_hash":     pool_manifest_hash,
            "pool_feature_hash":      pool_feature_hash,
            "pool_label_hash":        pool_label_hash,
            "real_manifest_hash":     real_manifest_hash,
            "real_feature_hash":      real_feature_hash,
            "real_label_hash":        real_label_hash,
            "threshold_pct":          args.threshold_pct,
            "output_threshold_tag":    output_threshold_tag,
            "ratios":                 args.ratios,
            "ratio":                  R,
        }
        with open(feat_dir / f"{out_name}.lineage.json", "w") as f:
            json.dump(lineage, f, indent=2)

    audit_suffix = f"_{output_threshold_tag}" if output_threshold_tag else ""
    audit_path = res_dir / f"filter_audit_v2{audit_suffix}.json"
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Wrote audit     -> {audit_path}")

    return audit, combo_data


# =============================================================================
# FIGURES
# =============================================================================
def make_figures(audit: dict, combo_data: dict, out_dir: Path,
                 threshold_pct: float, ratios, output_threshold_tag: str = ""):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "font.family":       "DejaVu Sans",
    })

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -- 4x3 per-combo Mahalanobis histograms ---------------------------------
    fig, axes = plt.subplots(
        4, 3, figsize=(14, 16), facecolor="white",
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig.suptitle(
        f"Squared Mahalanobis distance - real vs synthetic (pool3, v2)\n"
        f"PCA -> LedoitWolf (n_components <= {PCA_MAX_COMPONENTS}) | "
        f"empirical {threshold_pct}% threshold from real m^2",
        fontsize=12, fontweight="bold", y=0.995,
    )

    for row, cls in enumerate(CLASSES):
        for col, plane in enumerate(PLANES):
            combo = f"{cls}_{plane}"
            ax    = axes[row][col]
            info  = audit.get(combo, {})
            data  = combo_data.get(combo)

            ax.set_title(f"{combo}", fontsize=9, fontweight="bold")
            ax.set_xlabel("m^2", fontsize=8)
            ax.set_ylabel("density", fontsize=8)
            ax.tick_params(labelsize=7)

            if data is None or info.get("skipped"):
                ax.text(0.5, 0.5, "no data / skipped",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#888888")
                continue

            m2_real  = data["m2_real"]
            m2_synth = data["m2_synth"]
            threshold = data["threshold"]

            # Use 99th percentile of m2_synth for x-range to avoid one outlier
            # squashing the kept distribution to the left edge.
            x_max = max(
                np.percentile(m2_real, 99) if len(m2_real) else 0,
                np.percentile(m2_synth, 99) if len(m2_synth) else 0,
                threshold * 1.10,
            )
            bins = np.linspace(0, x_max, 40)

            # Split synth into kept (largest ratio) vs rejected by Mahal
            largest_R       = max(ratios)
            local_keep      = data["per_ratio_local_keep"][largest_R]
            keep_mask_local = np.zeros(len(m2_synth), dtype=bool)
            keep_mask_local[local_keep] = True

            mahal_keep_mask = m2_synth <= threshold
            m2_synth_kept   = m2_synth[keep_mask_local]
            m2_synth_mahal_only = m2_synth[mahal_keep_mask & ~keep_mask_local]
            m2_synth_rej    = m2_synth[~mahal_keep_mask]

            ax.hist(m2_real, bins=bins, density=True,
                    color=BASE_COLOR, alpha=0.55, label=f"real (n={len(m2_real)})")
            if len(m2_synth_kept):
                ax.hist(m2_synth_kept, bins=bins, density=True,
                        color=AUG_COLOR, alpha=0.55,
                        label=f"FPS-kept (n={len(m2_synth_kept)})")
            if len(m2_synth_mahal_only):
                ax.hist(m2_synth_mahal_only, bins=bins, density=True,
                        color=FPS_COLOR, alpha=0.55,
                        label=f"FPS-dropped (n={len(m2_synth_mahal_only)})")
            if len(m2_synth_rej):
                ax.hist(m2_synth_rej, bins=bins, density=True,
                        color=REJ_COLOR, alpha=0.55,
                        label=f"m^2 > thresh (n={len(m2_synth_rej)})")

            ax.axvline(threshold, color=THRES_COLOR, lw=1.4, ls="--",
                       label=f"thresh={threshold:.1f}")

            ax.legend(fontsize=5, loc="upper right",
                      framealpha=0.85, handlelength=1.2)

    fig.text(0.5, 0.001,
             "Blue: real   Orange: kept by FPS   Green: passed Mahal but cut by FPS"
             "   Red: rejected by Mahal   Purple dashed: empirical threshold",
             ha="center", fontsize=7.5, style="italic", color="#555555")

    suffix = f"_{output_threshold_tag}" if output_threshold_tag else ""
    _save_fig(fig, fig_dir / f"mahalanobis_distances_v2{suffix}")
    print(f"  Wrote figure    -> mahalanobis_distances_v2{suffix}.{{svg,pdf}}")

    # -- Per-combo summary bar chart (kept fraction per ratio) -----------------
    fig2, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    fig2.suptitle("Synthetic kept per (class x plane) - v2",
                  fontsize=12, fontweight="bold")

    combo_labels = COMBOS
    x = np.arange(len(combo_labels))
    width = 0.8 / max(len(ratios), 1)
    palette = [AUG_COLOR, FPS_COLOR, "#6f42c1", "#17a2b8"]

    for i, R in enumerate(ratios):
        kept = []
        for c in combo_labels:
            info = audit.get(c, {})
            if info.get("skipped") or "per_ratio" not in info:
                kept.append(0)
            else:
                kept.append(info["per_ratio"].get(R, {}).get("n_kept", 0))
        offset = (i - (len(ratios) - 1) / 2) * width
        ax.bar(x + offset, kept, width,
               color=palette[i % len(palette)], alpha=0.85,
               label=f"ratio {R:g}  (n_kept)")

    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Synthetic images kept")
    ax.set_title("Per-combo synthetic count after Mahalanobis + FPS")
    ax.legend(fontsize=8)

    _save_fig(fig2, fig_dir / f"mahalanobis_summary_v2{suffix}")
    print(f"  Wrote figure    -> mahalanobis_summary_v2{suffix}.{{svg,pdf}}")


def _save_fig(fig, stem: Path):
    fig.savefig(str(stem) + ".svg", format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(str(stem) + ".pdf", format="pdf", bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)


# =============================================================================
# UMAP - same intent as v1, refactored to handle multi-ratio kept-masks.
# =============================================================================
def make_umap_figures(audit: dict, combo_data: dict, out_dir: Path,
                      X_real, y_real, X_pool, y_pool, df_pool, df_real,
                      ratios, output_threshold_tag: str = ""):
    try:
        import umap as umap_lib
    except ImportError:
        print(
            "\n  [UMAP] umap-learn is not installed - skipping UMAP figures.\n"
            "         Install with:  pip install umap-learn\n"
        )
        return

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
        "font.family":       "DejaVu Sans",
    })

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("  Fitting UMAP on pool feature matrix - this may take a few minutes...")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
    embedding = reducer.fit_transform(X_pool)
    print("  UMAP fit complete.")

    largest_R = max(ratios)
    pool_aug_keep = np.zeros(len(df_pool), dtype=bool)
    pool_aug_keep[df_pool["source"].to_numpy() == "real"] = True
    for combo in COMBOS:
        data = combo_data.get(combo)
        if data is None:
            continue
        local_keep = data["per_ratio_local_keep"][largest_R]
        global_keep = data["synth_pool_idxs"][local_keep]
        pool_aug_keep[global_keep] = True

    aug_is_synth = (df_pool["source"] == "synthetic").to_numpy()

    fig1, axes = plt.subplots(
        4, 3, figsize=(14, 17), facecolor="white",
        gridspec_kw={"hspace": 0.45, "wspace": 0.30},
    )
    fig1.suptitle(
        f"UMAP of pool3 features - source label per combo (filter v2, ratio {largest_R:g})\n"
        "Blue: real   Orange: synth kept (after Mahal + FPS)   Red: synth dropped",
        fontsize=12, fontweight="bold", y=0.995,
    )

    for row, cls in enumerate(CLASSES):
        for col, plane in enumerate(PLANES):
            combo = f"{cls}_{plane}"
            ax    = axes[row][col]
            info  = audit.get(combo, {})

            ax.set_title(combo, fontsize=9, fontweight="bold")
            ax.set_xlabel("UMAP-1", fontsize=7)
            ax.set_ylabel("UMAP-2", fontsize=7)
            ax.tick_params(labelsize=6)

            if info.get("skipped"):
                ax.text(0.5, 0.5, "no data / skipped",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#888888")
                continue

            real_mask_df = (
                (df_pool["source"] == "real")
                & (df_pool["class"] == cls)
                & (df_pool["plane"] == plane)
            )
            real_aug_idxs = np.where(real_mask_df.to_numpy())[0]

            synth_mask_df = (
                (df_pool["source"] == "synthetic")
                & (df_pool["class"] == cls)
                & (df_pool["plane"] == plane)
            )
            synth_aug_idxs = np.where(synth_mask_df.to_numpy())[0]
            synth_kept_idxs = synth_aug_idxs[pool_aug_keep[synth_aug_idxs]]
            synth_drop_idxs = synth_aug_idxs[~pool_aug_keep[synth_aug_idxs]]

            def _scatter(idxs, color, label, marker="o", alpha=0.55, s=6):
                if len(idxs):
                    ax.scatter(embedding[idxs, 0], embedding[idxs, 1],
                               c=color, label=label, marker=marker,
                               s=s, alpha=alpha, linewidths=0)

            _scatter(real_aug_idxs,   BASE_COLOR, f"real (n={len(real_aug_idxs)})")
            _scatter(synth_kept_idxs, AUG_COLOR,  f"kept (n={len(synth_kept_idxs)})")
            _scatter(synth_drop_idxs, REJ_COLOR,  f"drop (n={len(synth_drop_idxs)})")

            ax.legend(fontsize=5, loc="best", framealpha=0.80, handlelength=1.0,
                      markerscale=1.5)

    suffix = f"_{output_threshold_tag}" if output_threshold_tag else ""
    _save_fig(fig1, fig_dir / f"umap_source_v2{suffix}")
    print(f"  Wrote figure    -> umap_source_v2{suffix}.{{svg,pdf}}")

    fig2, ax2 = plt.subplots(figsize=(10, 8), facecolor="white")
    fig2.suptitle(
        "UMAP of pool3 features - class colour, source shape (v2)\n"
        "Circle: real   x: synthetic",
        fontsize=13, fontweight="bold",
    )
    ax2.set_xlabel("UMAP-1", fontsize=10)
    ax2.set_ylabel("UMAP-2", fontsize=10)

    CLASS_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    cls_tag_aug   = df_pool["class"].map(CLASS_TO_IDX).to_numpy()

    for ci, cls_name in enumerate(CLASSES):
        color = CLASS_PALETTE[ci]
        cls_mask = cls_tag_aug == ci

        idxs = np.where(cls_mask & ~aug_is_synth)[0]
        if len(idxs):
            ax2.scatter(embedding[idxs, 0], embedding[idxs, 1],
                        c=color, marker="o", s=5, alpha=0.45, linewidths=0,
                        label=f"{cls_name} (real)")

        idxs_s = np.where(cls_mask & aug_is_synth)[0]
        if len(idxs_s):
            ax2.scatter(embedding[idxs_s, 0], embedding[idxs_s, 1],
                        c=color, marker="x", s=8, alpha=0.50, linewidths=0.6,
                        label=f"{cls_name} (synth)")

    class_handles = [
        mlines.Line2D([], [], color=CLASS_PALETTE[i], marker="o",
                      linestyle="None", markersize=6, label=CLASSES[i])
        for i in range(len(CLASSES))
    ]
    shape_handles = [
        mlines.Line2D([], [], color="#555555", marker="o",
                      linestyle="None", markersize=6, label="real"),
        mlines.Line2D([], [], color="#555555", marker="x",
                      linestyle="None", markersize=6, label="synthetic"),
    ]
    leg1 = ax2.legend(handles=class_handles, title="Class",  fontsize=8,
                      loc="upper left",  framealpha=0.85)
    ax2.add_artist(leg1)
    ax2.legend(handles=shape_handles, title="Source", fontsize=8,
               loc="upper right", framealpha=0.85)

    _save_fig(fig2, fig_dir / f"umap_class_v2{suffix}")
    print(f"  Wrote figure    -> umap_class_v2{suffix}.{{svg,pdf}}")


# =============================================================================
# CLI
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-root", type=Path,
                   default=Path(__file__).resolve().parent,
                   help="Workspace root.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory produced by classify_rf_v2.py. "
                        "Default: <project-root>/outputs_v2")
    p.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT,
                   help=f"Empirical percentile of REAL m^2 used as the rejection "
                        f"threshold. Synthetic with m^2 > this percentile is "
                        f"discarded. Default: {DEFAULT_THRESHOLD_PCT}.")
    p.add_argument("--sensitivity-thresholds", type=float, nargs="+", default=None,
                   help="Optional prespecified threshold grid. Writes suffixed "
                        "filtered artifacts such as *_filtered_p95.csv for each "
                        "threshold instead of overwriting the primary filtered set.")
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                   help="One filtered manifest+feature pair per ratio is written. "
                        "Default: 1.0 2.0.")
    p.add_argument("--pool-ratio", type=float, default=None,
                   help="Augmented feature pool ratio to filter from. Default: "
                        "max(--ratios). Use a larger precomputed pool to replace "
                        "synthetic rows rejected by the filter.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--no-umap", action="store_true",
                   help="Skip UMAP figures only.")
    return p.parse_args(argv)


def main(argv=None):
    args    = parse_args(argv)
    out_dir = (args.out_dir or args.project_root / "outputs_v2").resolve()

    if not (50.0 <= args.threshold_pct < 100.0):
        raise SystemExit("--threshold-pct must be in [50, 100).")
    if args.sensitivity_thresholds is not None:
        for pct in args.sensitivity_thresholds:
            if not (50.0 <= pct < 100.0):
                raise SystemExit("--sensitivity-thresholds must all be in [50, 100).")
    for R in args.ratios:
        if R <= 0:
            raise SystemExit(f"--ratios must all be > 0; got {R}")
    if args.pool_ratio is not None and args.pool_ratio < max(args.ratios):
        raise SystemExit("--pool-ratio must be >= max(--ratios).")

    threshold_runs = [(args.threshold_pct, "")]
    if args.sensitivity_thresholds is not None:
        threshold_runs = [
            (pct, _threshold_tag(pct)) for pct in args.sensitivity_thresholds
        ]

    print(f"\n{'='*72}")
    print("FILTER_SYNTHETIC_V2 - empirical threshold + FPS diversity")
    print(f"{'='*72}")
    print(f"  Project root     : {args.project_root.resolve()}")
    print(f"  Output dir       : {out_dir}")
    print(f"  Threshold pct    : {args.threshold_pct}%  (empirical)")
    if args.sensitivity_thresholds is not None:
        print(f"  Sensitivity grid : {args.sensitivity_thresholds}")
    print(f"  Ratios           : {args.ratios}")
    print(f"  Pool ratio       : {(args.pool_ratio if args.pool_ratio is not None else max(args.ratios)):g}")

    for threshold_pct, output_threshold_tag in threshold_runs:
        args.threshold_pct = threshold_pct
        args.output_threshold_tag = output_threshold_tag
        if output_threshold_tag:
            print(f"\n{'='*72}")
            print(f"SENSITIVITY FILTER RUN - threshold {threshold_pct:g}% "
                  f"({output_threshold_tag})")
            print(f"{'='*72}")

        audit, combo_data = run_filter(args, out_dir)

        if not args.no_figures:
            import numpy as np
            import pandas as pd

            print(f"\n{'-'*72}")
            print("  Generating figures...")
            make_figures(audit, combo_data, out_dir, args.threshold_pct,
                         args.ratios, output_threshold_tag)

            if not args.no_umap:
                print(f"\n{'-'*72}")
                print("  Generating UMAP figures...")
                feat_dir   = out_dir / "features"
                manif_dir  = out_dir / "manifests"
                pool_ratio = (
                    args.pool_ratio if args.pool_ratio is not None
                    else max(args.ratios)
                )
                pool_tag   = f"r{pool_ratio:g}"
                X_real  = np.load(feat_dir / "train_real_X.npy")
                y_real  = np.load(feat_dir / "train_real_y.npy")
                X_pool  = np.load(feat_dir / f"train_augmented_{pool_tag}_X.npy")
                y_pool  = np.load(feat_dir / f"train_augmented_{pool_tag}_y.npy")
                df_pool = pd.read_csv(manif_dir / f"train_augmented_{pool_tag}.csv")
                df_real = pd.read_csv(manif_dir / "train_real.csv")
                make_umap_figures(
                    audit, combo_data, out_dir,
                    X_real, y_real, X_pool, y_pool, df_pool, df_real,
                    args.ratios, output_threshold_tag,
                )

    print(f"\n{'='*72}")
    print("DONE (filter_synthetic_v2)")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
