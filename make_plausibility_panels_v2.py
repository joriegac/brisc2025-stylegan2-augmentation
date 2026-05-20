"""
make_plausibility_panels_v2.py
==============================
Visual QC panels for label-fidelity / plausibility review.

For each class-plane combo, this script samples:
  * real training images
  * unfiltered synthetic images
  * filtered synthetic images
  * nearest real neighbors for the filtered synthetic images in pool3 space

The panels are supporting evidence only; they do not replace clinical review.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLANES = ["ax", "co", "sa"]
COMBOS = [f"{c}_{p}" for c in CLASSES for p in PLANES]


def _ratio_tag(r: float) -> str:
    return f"r{r:g}"


def _threshold_tag(p: float) -> str:
    return f"p{p:g}"


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _read_manifest(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _require_files(paths, hint: str) -> None:
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            hint + "\n" + "\n".join(f"  missing: {p}" for p in missing)
        )


def _combo_mask(rows, cls, plane, source=None):
    idxs = []
    for i, row in enumerate(rows):
        if row.get("class") != cls or row.get("plane") != plane:
            continue
        if source is not None and row.get("source") != source:
            continue
        idxs.append(i)
    return idxs


def _choose(idxs, n, rng):
    import numpy as np

    if len(idxs) <= n:
        return list(idxs)
    return list(rng.choice(idxs, size=n, replace=False))


def _load_gray(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("L")


def _nearest_real_indices(X_real_combo, X_query):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, metric="cosine")
    nn.fit(X_real_combo)
    _, idxs = nn.kneighbors(X_query)
    return idxs[:, 0].astype(int).tolist()


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--ratio", type=float, default=2.0)
    p.add_argument("--filter-threshold", type=float, default=None,
                   help="Optional sensitivity suffix, e.g. 95 uses *_filtered_p95.")
    p.add_argument("--n-per-combo", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.n_per_combo <= 0:
        raise SystemExit("--n-per-combo must be > 0.")
    project_root = args.project_root.resolve()
    out_dir = (args.out_dir or project_root / "outputs_v2").resolve()
    manif_dir = out_dir / "manifests"
    feat_dir = out_dir / "features"
    fig_dir = out_dir / "audits" / "plausibility_panels_v2"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ratio_tag = _ratio_tag(args.ratio)
    threshold_suffix = (
        f"_{_threshold_tag(args.filter_threshold)}"
        if args.filter_threshold is not None else ""
    )

    aug_name = f"train_augmented_{ratio_tag}"
    filt_name = f"train_augmented_{ratio_tag}_filtered{threshold_suffix}"
    _require_files(
        [
            manif_dir / "train_real.csv",
            manif_dir / f"{aug_name}.csv",
            manif_dir / f"{filt_name}.csv",
            feat_dir / "train_real_X.npy",
            feat_dir / f"{filt_name}_X.npy",
        ],
        "Missing prerequisite files for plausibility panels. Run the RF "
        "manifest/features/filter stages first, and for sensitivity panels run "
        "filter_synthetic_v2.py with the matching threshold.",
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.unicode_minus": False})
    import numpy as np

    real_rows = _read_manifest(manif_dir / "train_real.csv")
    aug_rows = _read_manifest(manif_dir / f"{aug_name}.csv")
    filt_rows = _read_manifest(manif_dir / f"{filt_name}.csv")

    X_real = np.load(feat_dir / "train_real_X.npy")
    X_filt = np.load(feat_dir / f"{filt_name}_X.npy")

    rng = np.random.default_rng(args.seed)
    for combo in COMBOS:
        cls, plane = combo.rsplit("_", 1)
        real_idxs = _combo_mask(real_rows, cls, plane, "real")
        aug_synth_idxs = _combo_mask(aug_rows, cls, plane, "synthetic")
        filt_synth_idxs = _combo_mask(filt_rows, cls, plane, "synthetic")
        if not real_idxs or not aug_synth_idxs or not filt_synth_idxs:
            continue

        real_pick = _choose(real_idxs, args.n_per_combo, rng)
        aug_pick = _choose(aug_synth_idxs, args.n_per_combo, rng)
        filt_pick = _choose(filt_synth_idxs, args.n_per_combo, rng)

        real_combo_local = real_idxs
        X_real_combo = X_real[real_combo_local]
        X_query = X_filt[filt_pick]
        nn_local = _nearest_real_indices(X_real_combo, X_query)
        nn_real_pick = [real_combo_local[i] for i in nn_local]

        rows = [
            ("Real", [real_rows[i] for i in real_pick]),
            ("Unfiltered synth", [aug_rows[i] for i in aug_pick]),
            ("Filtered synth", [filt_rows[i] for i in filt_pick]),
            ("Nearest real to filtered", [real_rows[i] for i in nn_real_pick]),
        ]

        fig, axes = plt.subplots(4, args.n_per_combo, figsize=(2.2 * args.n_per_combo, 8),
                                 facecolor="white")
        axes = np.asarray(axes).reshape(4, args.n_per_combo)
        fig.suptitle(f"{combo} plausibility panel ({filt_name})",
                     fontsize=11, fontweight="bold")
        for row_i, (label, panel_rows) in enumerate(rows):
            for col_i in range(args.n_per_combo):
                ax = axes[row_i, col_i]
                ax.axis("off")
                if col_i >= len(panel_rows):
                    continue
                fp = _resolve_manifest_path(panel_rows[col_i]["filepath"], project_root)
                ax.imshow(_load_gray(fp), cmap="gray")
                if col_i == 0:
                    # axis("off") hides set_ylabel; use a text annotation so
                    # the row label still renders.
                    ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                            rotation=90, va="center", ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{combo}_{filt_name}.png",
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)

    print(f"Wrote plausibility panels to {fig_dir}")


if __name__ == "__main__":
    main()
