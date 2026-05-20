"""
audit_dataset_independence_v2.py
================================
Reviewer-facing leakage audit for the preprocessed BRISC train/test images.

This script intentionally audits only the preprocessed images used by the
StyleGAN/classifier pipeline. It checks what is knowable from the released
files:

  * exact filepath/basename/file-hash overlap between train_real and test manifests
  * exact canonical pixel-array overlap, confirmed with np.array_equal
  * pHash nearest-neighbor proximity in preprocessed pixel space
  * Inception pool3 nearest-neighbor proximity using outputs_v2 feature caches
  * whether patient identifiers are available in the BRISC manifest

It does not claim patient-level independence when no patient identifier exists.
Instead, it writes that limitation explicitly into the Markdown report.
pHash and Inception proximity are descriptive only; they are not leakage or
exclusion criteria.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
CNN_IMAGE_SIZE = (128, 128)


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_pixel_array(path: Path):
    """Return the uint8 RGB tensor seen by classify_cnn_v2.py for saved images."""
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != CNN_IMAGE_SIZE:
            im = im.resize(CNN_IMAGE_SIZE, Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _pixel_sha256(arr) -> str:
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(b"\0")
    h.update(str(arr.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


def _read_manifest(path: Path, project_root: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["_resolved_path"] = str(_resolve_manifest_path(row["filepath"], project_root))
            row["_basename"] = Path(row["filepath"]).name
            rows.append(row)
    return rows


def _hash_rows(rows):
    out = []
    for row in rows:
        fp = Path(row["_resolved_path"])
        out.append({
            "filepath": row["filepath"],
            "basename": row["_basename"],
            "class": row.get("class", ""),
            "plane": row.get("plane", ""),
            "sha256": _file_sha256(fp),
        })
    return out


def _pixel_identity_rows(train_rows, test_rows):
    import numpy as np

    pixel_cache = {}

    def _arr(row):
        fp = Path(row["_resolved_path"])
        key = str(fp)
        if key not in pixel_cache:
            pixel_cache[key] = _canonical_pixel_array(fp)
        return pixel_cache[key]

    train_by_hash = {}
    for row in train_rows:
        arr = _arr(row)
        h = _pixel_sha256(arr)
        train_by_hash.setdefault(h, []).append((row, arr))

    candidate_pairs = 0
    exact_rows = []
    for test_row in test_rows:
        test_arr = _arr(test_row)
        h = _pixel_sha256(test_arr)
        candidates = train_by_hash.get(h, [])
        candidate_pairs += len(candidates)
        for train_row, train_arr in candidates:
            if not np.array_equal(train_arr, test_arr):
                continue
            exact_rows.append({
                "pixel_sha256": h,
                "pixel_shape": "x".join(str(v) for v in test_arr.shape),
                "pixel_dtype": str(test_arr.dtype),
                "train_filepath": train_row["filepath"],
                "train_class": train_row.get("class", ""),
                "train_plane": train_row.get("plane", ""),
                "test_filepath": test_row["filepath"],
                "test_class": test_row.get("class", ""),
                "test_plane": test_row.get("plane", ""),
            })
    return exact_rows, candidate_pairs


def _pixel_identity_clusters(exact_rows):
    clusters = []
    by_hash = {}
    for row in exact_rows:
        by_hash.setdefault(row["pixel_sha256"], []).append(row)
    for pixel_hash, rows in sorted(by_hash.items()):
        train_paths = sorted({r["train_filepath"] for r in rows})
        test_paths = sorted({r["test_filepath"] for r in rows})
        clusters.append({
            "pixel_sha256": pixel_hash,
            "pixel_shape": rows[0]["pixel_shape"],
            "pixel_dtype": rows[0]["pixel_dtype"],
            "unique_train_rows": len(train_paths),
            "unique_test_rows": len(test_paths),
            "pairwise_links": len(rows),
            "train_filepaths": " | ".join(train_paths),
            "test_filepaths": " | ".join(test_paths),
        })
    return clusters


def _phash_neighbors(train_rows, test_rows, project_root: Path, max_rows: int):
    from PIL import Image
    import numpy as np

    try:
        import imagehash
    except ImportError:
        imagehash = None

    def _hash(row):
        with Image.open(Path(row["_resolved_path"])) as im:
            im = im.convert("L")
            if imagehash is not None:
                return imagehash.phash(im)
            try:
                from scipy.fftpack import dct
                arr = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32)
                coeff = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
                low = coeff[:8, :8]
                med = np.median(low[1:, 1:])
            except ImportError:
                low = np.asarray(im.resize((8, 8), Image.BILINEAR), dtype=np.float32)
                med = np.mean(low)
            bits = (low > med).astype(np.uint8).ravel()
            value = 0
            for bit in bits:
                value = (value << 1) | int(bit)
            return value

    def _dist(a, b):
        if imagehash is not None:
            return abs(a - b)
        return int((a ^ b).bit_count())

    train_hashes = [_hash(row) for row in train_rows]
    test_hashes = [_hash(row) for row in test_rows]
    nearest = []
    for i, h_test in enumerate(test_hashes):
        best_j, best_d = -1, 10**9
        for j, h_train in enumerate(train_hashes):
            d = _dist(h_test, h_train)
            if d < best_d:
                best_j, best_d = j, d
        nearest.append({
            "test_filepath": test_rows[i]["filepath"],
            "test_class": test_rows[i].get("class", ""),
            "test_plane": test_rows[i].get("plane", ""),
            "train_filepath": train_rows[best_j]["filepath"],
            "train_class": train_rows[best_j].get("class", ""),
            "train_plane": train_rows[best_j].get("plane", ""),
            "phash_distance": int(best_d),
        })
    nearest.sort(key=lambda r: r["phash_distance"])
    return nearest[:max_rows], nearest


def _feature_neighbors(out_dir: Path, train_rows, test_rows, max_rows: int):
    import numpy as np

    feat_dir = out_dir / "features"
    train_x_path = feat_dir / "train_real_X.npy"
    test_x_path = feat_dir / "test_X.npy"
    missing = [p for p in (train_x_path, test_x_path) if not p.exists()]
    if missing:
        return [], [], {
            "skipped": True,
            "reason": (
                "feature cache files are missing; rerun "
                "classify_rf_v2.py --stage features"
            ),
            "missing": [str(p) for p in missing],
        }
    train_x = np.load(train_x_path)
    test_x = np.load(test_x_path)
    if train_x.shape[0] != len(train_rows) or test_x.shape[0] != len(test_rows):
        return [], [], {
            "skipped": True,
            "reason": (
                "feature cache row count does not match current manifests; "
                "rerun classify_rf_v2.py --stage features --force"
            ),
            "train_feature_rows": int(train_x.shape[0]),
            "train_manifest_rows": int(len(train_rows)),
            "test_feature_rows": int(test_x.shape[0]),
            "test_manifest_rows": int(len(test_rows)),
        }
    train_norm = train_x / np.maximum(np.linalg.norm(train_x, axis=1, keepdims=True), 1e-12)
    test_norm = test_x / np.maximum(np.linalg.norm(test_x, axis=1, keepdims=True), 1e-12)

    nearest = []
    for start in range(0, test_norm.shape[0], 128):
        stop = min(start + 128, test_norm.shape[0])
        sims = test_norm[start:stop] @ train_norm.T
        idxs = np.argmax(sims, axis=1)
        dists = 1.0 - sims[np.arange(stop - start), idxs]
        for offset, (j, dist) in enumerate(zip(idxs, dists)):
            i = start + offset
            nearest.append({
                "test_filepath": test_rows[i]["filepath"],
                "test_class": test_rows[i].get("class", ""),
                "test_plane": test_rows[i].get("plane", ""),
                "train_filepath": train_rows[int(j)]["filepath"],
                "train_class": train_rows[int(j)].get("class", ""),
                "train_plane": train_rows[int(j)].get("plane", ""),
                "cosine_distance": float(dist),
            })
    nearest.sort(key=lambda r: r["cosine_distance"])
    return nearest[:max_rows], nearest, {"skipped": False}


def _patient_id_columns(project_root: Path):
    manifest = project_root / "brisc2025" / "manifest.csv"
    if not manifest.exists():
        return []
    with open(manifest, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
    patientish = []
    for col in cols:
        low = col.lower()
        if any(token in low for token in ("patient", "subject", "case", "study", "series")):
            patientish.append(col)
    return patientish


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--top-k", type=int, default=50,
                   help="Nearest-neighbor rows to write for pHash/features.")
    p.add_argument("--phash-warning-threshold", type=int, default=6,
                   help="Diagnostic pHash proximity cutoff for reporting only.")
    p.add_argument("--feature-warning-threshold", type=float, default=0.01,
                   help="Diagnostic feature-space proximity cutoff for reporting only.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    out_dir = (args.out_dir or project_root / "outputs_v2").resolve()
    audit_dir = out_dir / "audits"
    audit_dir.mkdir(parents=True, exist_ok=True)

    train_csv = out_dir / "manifests" / "train_real.csv"
    test_csv = out_dir / "manifests" / "test.csv"
    missing_manifests = [p for p in (train_csv, test_csv) if not p.exists()]
    if missing_manifests:
        raise SystemExit(
            "Missing prerequisite manifest(s); run "
            "classify_rf_v2.py --stage manifests first:\n"
            + "\n".join(f"  {p}" for p in missing_manifests)
        )
    train_rows = _read_manifest(train_csv, project_root)
    test_rows = _read_manifest(test_csv, project_root)
    if not train_rows or not test_rows:
        raise SystemExit(
            f"Cannot audit empty manifests: train rows={len(train_rows)}, "
            f"test rows={len(test_rows)}."
        )

    train_hashes = _hash_rows(train_rows)
    test_hashes = _hash_rows(test_rows)

    train_sha = {r["sha256"]: r for r in train_hashes}
    test_sha = {r["sha256"]: r for r in test_hashes}
    exact_sha_overlap = sorted(set(train_sha) & set(test_sha))
    train_sha_multi = {}
    for row in train_hashes:
        train_sha_multi.setdefault(row["sha256"], []).append(row)
    exact_overlap_rows = []
    for test_row in test_hashes:
        for train_row in train_sha_multi.get(test_row["sha256"], []):
            exact_overlap_rows.append({
                "sha256": test_row["sha256"],
                "train_filepath": train_row["filepath"],
                "train_class": train_row["class"],
                "train_plane": train_row["plane"],
                "test_filepath": test_row["filepath"],
                "test_class": test_row["class"],
                "test_plane": test_row["plane"],
            })
    exact_pixel_rows, pixel_candidate_pairs = _pixel_identity_rows(
        train_rows, test_rows)
    exact_pixel_clusters = _pixel_identity_clusters(exact_pixel_rows)

    train_paths = {r["filepath"] for r in train_rows}
    test_paths = {r["filepath"] for r in test_rows}
    train_basenames = {r["_basename"] for r in train_rows}
    test_basenames = {r["_basename"] for r in test_rows}

    phash_top, phash_all = _phash_neighbors(
        train_rows, test_rows, project_root, args.top_k)
    feature_top, feature_all, feature_status = _feature_neighbors(
        out_dir, train_rows, test_rows, args.top_k)

    phash_flagged = [r for r in phash_all
                     if r["phash_distance"] <= args.phash_warning_threshold]
    feature_flagged = [
        r for r in feature_all
        if r["cosine_distance"] <= args.feature_warning_threshold
    ] if feature_all else []

    _write_csv(audit_dir / "train_test_phash_nearest_v2.csv", phash_top)
    _write_csv(audit_dir / "train_test_feature_nearest_v2.csv", feature_top)
    _write_csv(audit_dir / "train_test_exact_sha_overlap_v2.csv", exact_overlap_rows)
    _write_csv(audit_dir / "train_test_exact_pixel_overlap_v2.csv", exact_pixel_rows)
    _write_csv(audit_dir / "train_test_exact_pixel_overlap_clusters_v2.csv",
               exact_pixel_clusters)

    patientish_cols = _patient_id_columns(project_root)
    summary = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "exact_filepath_overlap": len(train_paths & test_paths),
        "basename_overlap": len(train_basenames & test_basenames),
        "file_sha256_overlap": len(exact_sha_overlap),
        "file_sha256_overlap_pairs": len(exact_overlap_rows),
        "pixel_hash_candidate_pairs": pixel_candidate_pairs,
        "pixel_exact_overlap_pairs": len(exact_pixel_rows),
        "pixel_exact_overlap_unique_train_rows": len({
            r["train_filepath"] for r in exact_pixel_rows
        }),
        "pixel_exact_overlap_unique_test_rows": len({
            r["test_filepath"] for r in exact_pixel_rows
        }),
        "pixel_exact_overlap_identities": len(exact_pixel_clusters),
        "phash_warning_threshold": args.phash_warning_threshold,
        "phash_close_neighbors": len(phash_flagged),
        "min_phash_distance": min(r["phash_distance"] for r in phash_all),
        "feature_warning_threshold": args.feature_warning_threshold,
        "feature_check": feature_status,
        "feature_close_neighbors": len(feature_flagged),
        "min_feature_cosine_distance": (
            min(r["cosine_distance"] for r in feature_all)
            if feature_all else None
        ),
        "patient_identifier_columns_found": patientish_cols,
        "patient_level_independence_claim_supported": bool(patientish_cols),
    }
    with open(audit_dir / "dataset_independence_audit_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Dataset Independence Audit v2",
        "",
        "Scope: preprocessed images used by the StyleGAN/classifier pipeline.",
        "",
        f"- Train rows: {summary['train_rows']}",
        f"- Test rows: {summary['test_rows']}",
        f"- Exact filepath overlap: {summary['exact_filepath_overlap']}",
        f"- Basename overlap: {summary['basename_overlap']}",
        f"- File SHA256 overlap: {summary['file_sha256_overlap']}",
        f"- File SHA256 overlap pairs: {summary['file_sha256_overlap_pairs']}",
        f"- Pixel-hash candidate pairs: {summary['pixel_hash_candidate_pairs']}",
        f"- Pixel-exact overlap pairs (pairwise train-test links): {summary['pixel_exact_overlap_pairs']}",
        f"- Pixel-exact unique train rows involved: {summary['pixel_exact_overlap_unique_train_rows']}",
        f"- Pixel-exact unique test rows involved: {summary['pixel_exact_overlap_unique_test_rows']}",
        f"- Pixel-exact identity clusters: {summary['pixel_exact_overlap_identities']}",
        f"- pHash close neighbors, diagnostic only (<= {args.phash_warning_threshold}): {len(phash_flagged)}",
        f"- Minimum pHash distance: {summary['min_phash_distance']}",
        f"- Feature-space close neighbors, diagnostic only (cosine <= {args.feature_warning_threshold}): {len(feature_flagged)}",
        "",
        "Identity criterion:",
        "- Image identity is tested on canonical uint8 RGB pixel arrays after loading preprocessed images the same way the CNN does.",
        "- Pixel-hash matches are confirmed with exact np.array_equal comparison.",
        "- Pair counts are many-to-many links; use the cluster CSV for unique pixel identities.",
        "- pHash and Inception cosine distances describe proximity only; they are not used as exclusion criteria.",
        "",
        "Patient-level note:",
    ]
    if feature_status.get("skipped"):
        md.insert(-2, f"- Feature nearest-neighbor check skipped: {feature_status['reason']}")
        if "missing" in feature_status:
            md.insert(-2, "  missing files: " + ", ".join(feature_status["missing"]))
        else:
            md.insert(-2, f"  train features/manifests: {feature_status['train_feature_rows']} / {feature_status['train_manifest_rows']}")
            md.insert(-2, f"  test features/manifests: {feature_status['test_feature_rows']} / {feature_status['test_manifest_rows']}")
    else:
        md.insert(-2, f"- Minimum feature cosine distance: {summary['min_feature_cosine_distance']:.6f}")
    if patientish_cols:
        md.append(f"- Candidate patient/study columns found: {patientish_cols}")
        md.append("- Patient-level split auditing should be performed using those fields.")
    else:
        md.append("- No patient/subject/study identifier column was found in the BRISC manifest.")
        md.append("- This audit can support image-level non-overlap only; it cannot prove patient-level independence.")
    md.extend([
        "",
        "CSV outputs:",
        "- `train_test_phash_nearest_v2.csv`",
        "- `train_test_feature_nearest_v2.csv`",
        "- `train_test_exact_pixel_overlap_v2.csv`",
        "- `train_test_exact_pixel_overlap_clusters_v2.csv`",
        "",
    ])
    (audit_dir / "dataset_independence_audit_v2.md").write_text(
        "\n".join(md), encoding="utf-8")
    print(f"Wrote audit to {audit_dir}")


if __name__ == "__main__":
    main()
