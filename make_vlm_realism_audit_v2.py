"""
make_vlm_realism_audit_v2.py
============================
Build a blinded image package for a VLM real-vs-synthetic realism audit.

The public package contains only plain numbered PNGs, an audit protocol, and
a response template. The private key maps each number back to the source image,
class, plane, and real/synthetic label for scoring after responses are collected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


VALID_CLASSES = {"glioma", "meningioma", "no_tumor", "pituitary"}
VALID_PLANES = {"ax", "co", "sa"}
VALID_SOURCES = {"real", "synthetic"}


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    return path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(manifest_path: Path):
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row.get("class", "").strip()
            plane = row.get("plane", "").strip()
            source = row.get("source", "").strip()
            if cls not in VALID_CLASSES:
                raise ValueError(f"Unsupported class in {manifest_path}: {cls!r}")
            if plane not in VALID_PLANES:
                raise ValueError(f"Unsupported plane in {manifest_path}: {plane!r}")
            if source not in VALID_SOURCES:
                raise ValueError(f"Unsupported source in {manifest_path}: {source!r}")
            rows.append(dict(row))
    if not rows:
        raise ValueError(f"Manifest has no auditable rows: {manifest_path}")
    return rows


def _select_rows(rows, mode: str, seed: int):
    if mode == "all":
        selected = list(rows)
    elif mode == "balanced-source-per-class-plane":
        rng = random.Random(seed)
        selected = []
        for cls in sorted(VALID_CLASSES):
            for plane in sorted(VALID_PLANES):
                real_rows = [
                    r for r in rows
                    if r["class"] == cls and r["plane"] == plane and r["source"] == "real"
                ]
                synth_rows = [
                    r for r in rows
                    if r["class"] == cls and r["plane"] == plane and r["source"] == "synthetic"
                ]
                n = min(len(real_rows), len(synth_rows))
                if n == 0:
                    continue
                rng.shuffle(real_rows)
                rng.shuffle(synth_rows)
                selected.extend(real_rows[:n])
                selected.extend(synth_rows[:n])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if not selected:
        raise ValueError(f"Selection mode produced no rows: {mode}")
    return selected


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_public_protocol(path: Path, audit_name: str, n_images: int):
    text = f"""# VLM Realism Audit Protocol

Audit package: `{audit_name}`

You are given {n_images} blinded, preprocessed 128x128 brain MRI images. The
file names are random numeric identifiers only. Do not infer anything from file
order or file name. Use image content only.

For each image, answer the prerequisite content-validity questions first:

1. Tumour class: one of `glioma`, `meningioma`, `no_tumor`, `pituitary`.
2. Anatomical plane: one of `ax`, `co`, `sa`.

Then answer the source-realism question:

3. Source prediction: one of `real`, `synthetic`.

Use the exact labels above. If uncertain, provide your best forced-choice
answer and lower confidence. Do not skip images.

Artifact categories may use zero or more of:

- `none_obvious`
- `texture_artifact`
- `anatomy_implausible`
- `edge_or_crop_artifact`
- `contrast_or_intensity_artifact`
- `noise_pattern_artifact`
- `repetition_or_symmetry_artifact`
- `other`

Write responses to `responses.csv` using exactly these columns:

`image_id,filename,tumour_class,tumour_confidence,plane,plane_confidence,source_prediction,source_confidence,artifact_categories,rationale`

Rules:

- `image_id` must match the numeric stem, for example `000001`.
- `filename` must match the PNG file name, for example `000001.png`.
- Confidence values must be numeric from 0 to 1.
- `artifact_categories` should be semicolon-separated if multiple apply.
- `rationale` should be concise, maximum 40 words.
- Do not add columns.
- Do not include markdown fences in the CSV.

Scoring note for the analyst:

Source-realism scoring should be performed only after responses are complete.
The private key must not be shown to the VLM. The prerequisite tumour/plane
answers are used as the content-validity gate before source-realism scoring.
"""
    path.write_text(text, encoding="utf-8")


def _copy_blinded_images(selected_rows, project_root: Path, images_dir: Path,
                         image_size: int):
    from PIL import Image

    key_rows = []
    public_rows = []
    response_rows = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(selected_rows, start=1):
        image_id = f"{idx:06d}"
        filename = f"{image_id}.png"
        src_path = _resolve_manifest_path(row["filepath"], project_root)
        if not src_path.is_file():
            raise FileNotFoundError(f"Manifest references missing image: {src_path}")

        dst_path = images_dir / filename
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            if im.size != (image_size, image_size):
                im = im.resize((image_size, image_size), Image.BILINEAR)
            im.save(dst_path, format="PNG")

        public_rows.append({"image_id": image_id, "filename": filename})
        response_rows.append({
            "image_id": image_id,
            "filename": filename,
            "tumour_class": "",
            "tumour_confidence": "",
            "plane": "",
            "plane_confidence": "",
            "source_prediction": "",
            "source_confidence": "",
            "artifact_categories": "",
            "rationale": "",
        })
        key_rows.append({
            "image_id": image_id,
            "filename": filename,
            "source": row["source"],
            "class": row["class"],
            "plane": row["plane"],
            "original_filepath": row["filepath"],
            "original_sha256": _sha256_file(src_path),
            "blinded_sha256": _sha256_file(dst_path),
        })

    return public_rows, response_rows, key_rows


def _summarize(key_rows):
    out = {
        "n_images": len(key_rows),
        "by_source": {},
        "by_class": {},
        "by_plane": {},
        "by_class_plane_source": {},
    }
    for row in key_rows:
        out["by_source"][row["source"]] = out["by_source"].get(row["source"], 0) + 1
        out["by_class"][row["class"]] = out["by_class"].get(row["class"], 0) + 1
        out["by_plane"][row["plane"]] = out["by_plane"].get(row["plane"], 0) + 1
        key = f"{row['class']}|{row['plane']}|{row['source']}"
        out["by_class_plane_source"][key] = (
            out["by_class_plane_source"].get(key, 0) + 1
        )
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create a blinded numbered-image package for VLM realism audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Pipeline output dir. Default: <project-root>/outputs_v2")
    parser.add_argument("--manifest", default="train_augmented_r2.csv",
                        help="Manifest name under outputs_v2/manifests or a path.")
    parser.add_argument("--audit-name", default=None,
                        help="Output package name. Default derives from manifest.")
    parser.add_argument("--mode", default="all",
                        choices=("all", "balanced-source-per-class-plane"),
                        help="Use all manifest rows, or a source-balanced subset.")
    parser.add_argument("--seed", type=int, default=20260502,
                        help="Shuffle/selection seed.")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Output image side length.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing audit package.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    out_dir = (args.out_dir or (project_root / "outputs_v2")).resolve()
    manifest_arg = Path(args.manifest)
    if manifest_arg.is_absolute() or manifest_arg.exists():
        manifest_path = manifest_arg.resolve()
    else:
        manifest_path = out_dir / "manifests" / args.manifest
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be > 0.")

    stem = manifest_path.stem
    audit_name = args.audit_name or f"vlm_realism_{stem}_{args.mode}"
    package_root = out_dir / "vlm_realism_audits" / audit_name
    public_dir = package_root / "public"
    private_dir = package_root / "private"
    images_dir = public_dir / "images"

    if package_root.exists():
        if not args.force:
            raise SystemExit(
                f"Audit package already exists: {package_root}. Pass --force to overwrite."
            )
        shutil.rmtree(package_root)

    rows = _read_manifest(manifest_path)
    selected = _select_rows(rows, args.mode, args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(selected)

    public_rows, response_rows, key_rows = _copy_blinded_images(
        selected, project_root, images_dir, args.image_size)

    _write_csv(public_dir / "image_index.csv", public_rows,
               ["image_id", "filename"])
    _write_csv(public_dir / "responses_template.csv", response_rows,
               ["image_id", "filename", "tumour_class", "tumour_confidence",
                "plane", "plane_confidence", "source_prediction",
                "source_confidence", "artifact_categories", "rationale"])
    _write_public_protocol(public_dir / "audit.md", audit_name, len(key_rows))

    _write_csv(private_dir / "answer_key.csv", key_rows,
               ["image_id", "filename", "source", "class", "plane",
                "original_filepath", "original_sha256", "blinded_sha256"])
    summary = _summarize(key_rows)
    summary.update({
        "audit_name": audit_name,
        "mode": args.mode,
        "seed": args.seed,
        "image_size": args.image_size,
        "manifest": str(manifest_path),
        "public_dir": str(public_dir),
        "private_answer_key": str(private_dir / "answer_key.csv"),
    })
    (private_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote public audit package : {public_dir}")
    print(f"Wrote private answer key   : {private_dir / 'answer_key.csv'}")
    print(f"Images                     : {summary['n_images']}")
    print(f"By source                  : {summary['by_source']}")


if __name__ == "__main__":
    main()
