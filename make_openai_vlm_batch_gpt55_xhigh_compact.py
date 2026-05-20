"""
make_openai_vlm_batch_gpt55_xhigh_compact.py
============================================
Create a token-optimized OpenAI Batch JSONL for the blinded VLM realism audit.

This version groups multiple blinded images into each Responses API request, so
the prompt, schema, and max-output budget are not repeated once per image.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import random
from pathlib import Path


DEFAULT_AUDIT_ROOT = Path(
    "outputs_v2/vlm_realism_audits/vlm_realism_train_augmented_r2_all"
)
DEFAULT_OUT_DIR = Path(
    "outputs_v2/openai_batch/vlm_realism_train_augmented_r2_gpt55_xhigh_compact"
)

SYSTEM_PROMPT = (
    "You are a blinded visual-realism judge for a research-only audit of "
    "preprocessed 128x128 brain-MRI-like images. Some images are real and some "
    "are synthetic. Do not provide medical advice. Use image content only. "
    "Think carefully, but output only valid JSON matching the schema."
)

TASK_TEXT = """\
For each labeled image below, return one result. File names and IDs are blinded
random identifiers. Forced choices:

tumour_class: glioma | meningioma | no_tumor | pituitary
plane: ax | co | sa
source_prediction: real | synthetic
artifact_categories: none_obvious | texture_artifact | anatomy_implausible |
edge_or_crop_artifact | contrast_or_intensity_artifact |
noise_pattern_artifact | repetition_or_symmetry_artifact | other

Do not call an image synthetic just because it is small, preprocessed,
normalized, grayscale/RGB encoded, or cropped. Those properties apply to both
real and synthetic images. Return every image_id exactly once.
"""

FINAL_ONLY_SYSTEM_PROMPT = (
    "You are a blinded visual-realism judge for a research-only audit of "
    "preprocessed 128x128 brain-MRI-like images. Some images are real and some "
    "are synthetic. Do not provide medical advice. Return only the requested "
    "JSON. Do not include explanations."
)

FINAL_ONLY_TASK_TEXT = """\
For each labeled image, return one compact result. Use image content only.
File names and IDs are blinded random identifiers.

Allowed values:
tumour_class: glioma | meningioma | no_tumor | pituitary
plane: ax | co | sa
source_prediction: real | synthetic
artifact_categories: none_obvious | texture_artifact | anatomy_implausible |
edge_or_crop_artifact | contrast_or_intensity_artifact |
noise_pattern_artifact | repetition_or_symmetry_artifact | other

Small, cropped, grayscale/RGB-encoded, normalized, or preprocessed appearance
applies to both real and synthetic images; do not use that alone.
Do not explain. Do not add prose. Return every image_id exactly once.
"""


def result_schema(include_rationale: bool = True) -> dict:
    artifact_enum = [
        "none_obvious",
        "texture_artifact",
        "anatomy_implausible",
        "edge_or_crop_artifact",
        "contrast_or_intensity_artifact",
        "noise_pattern_artifact",
        "repetition_or_symmetry_artifact",
        "other",
    ]
    properties = {
        "image_id": {"type": "string"},
        "filename": {"type": "string"},
        "usable_brain_mri_like_image": {"type": "boolean"},
        "tumour_class": {
            "type": "string",
            "enum": ["glioma", "meningioma", "no_tumor", "pituitary"],
        },
        "tumour_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "plane": {"type": "string", "enum": ["ax", "co", "sa"]},
        "plane_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "source_prediction": {"type": "string", "enum": ["real", "synthetic"]},
        "source_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "realism_score": {"type": "integer", "minimum": 1, "maximum": 5},
        "artifact_categories": {
            "type": "array",
            "items": {"type": "string", "enum": artifact_enum},
        },
    }
    required = [
        "image_id",
        "filename",
        "usable_brain_mri_like_image",
        "tumour_class",
        "tumour_confidence",
        "plane",
        "plane_confidence",
        "source_prediction",
        "source_confidence",
        "realism_score",
        "artifact_categories",
    ]
    if include_rationale:
        properties["rationale"] = {"type": "string"}
        required.append("rationale")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def response_schema(n_images: int, include_rationale: bool = True) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "minItems": n_images,
                "maxItems": n_images,
                "items": result_schema(include_rationale=include_rationale),
            }
        },
        "required": ["results"],
    }


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty image index: {path}")
    return rows


def read_excluded_ids(paths: list[Path]) -> set[str]:
    excluded = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_id = row.get("image_id")
                if image_id:
                    excluded.add(image_id)
    return excluded


def read_private_key(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def balanced_sample(rows: list[dict[str, str]], key_by_id: dict[str, dict[str, str]],
                    per_source_class_plane: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = key_by_id[row["image_id"]]
        bucket_key = (key["source"], key["class"], key["plane"])
        buckets.setdefault(bucket_key, []).append(row)

    selected = []
    for source in ["real", "synthetic"]:
        for cls in ["glioma", "meningioma", "no_tumor", "pituitary"]:
            for plane in ["ax", "co", "sa"]:
                bucket = buckets.get((source, cls, plane), [])
                if len(bucket) < per_source_class_plane:
                    raise ValueError(
                        f"Not enough rows for {source}/{cls}/{plane}: "
                        f"{len(bucket)} < {per_source_class_plane}"
                    )
                bucket = list(bucket)
                rng.shuffle(bucket)
                selected.extend(bucket[:per_source_class_plane])
    rng.shuffle(selected)
    return selected


def chunks(rows: list[dict[str, str]], n: int):
    for i in range(0, len(rows), n):
        yield rows[i:i + n]


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def make_request(group_no: int, group: list[dict[str, str]], images_dir: Path,
                 model: str, effort: str, max_output_tokens: int,
                 image_detail: str, text_verbosity: str,
                 include_rationale: bool, final_only: bool) -> dict:
    task_text = FINAL_ONLY_TASK_TEXT if final_only else TASK_TEXT
    system_prompt = FINAL_ONLY_SYSTEM_PROMPT if final_only else SYSTEM_PROMPT
    content = [{"type": "input_text", "text": task_text}]
    ids = []
    for row in group:
        image_id = row["image_id"]
        filename = row["filename"]
        ids.append(image_id)
        image_path = images_dir / filename
        content.append({
            "type": "input_text",
            "text": f"image_id={image_id}; filename={filename}",
        })
        content.append({
            "type": "input_image",
            "image_url": data_url(image_path),
            "detail": image_detail,
        })

    custom_id = f"vlm-realism-group-{group_no:04d}-{ids[0]}-{ids[-1]}"
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": effort},
            "max_output_tokens": max_output_tokens,
            "store": False,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "text": {
                "verbosity": text_verbosity,
                "format": {
                    "type": "json_schema",
                    "name": "vlm_realism_group_response",
                    "strict": True,
                    "schema": response_schema(
                        len(group),
                        include_rationale=include_rationale,
                    ),
                }
            },
        },
    }


def write_submitter(path: Path, shard_names: list[str], model: str,
                    effort: str, audit_label: str) -> None:
    shard_list = ",\n    ".join(repr(name) for name in shard_names)
    path.write_text(f"""\
from pathlib import Path

from openai import OpenAI

HERE = Path(__file__).resolve().parent
SHARDS = [
    {shard_list}
]

client = OpenAI()
for shard_name in SHARDS:
    shard_path = HERE / "jsonl_shards" / shard_name
    batch_input_file = client.files.create(
        file=open(shard_path, "rb"),
        purpose="batch",
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={{
            "description": f"BRISC grouped VLM audit, {model} {effort}, {{shard_name}}",
            "project": "RA1",
            "audit": "{audit_label}",
            "shard": shard_name,
        }},
    )
    print(shard_name)
    print("  input_file_id:", batch_input_file.id)
    print("  batch_id:", batch.id)
    print("  status:", batch.status)
""", encoding="utf-8")


def write_readme(path: Path, shard_names: list[str], images_per_request: int,
                 model: str, effort: str, submitter_path: Path,
                 final_only: bool) -> None:
    shard_lines = "\n".join(f"- `jsonl_shards/{name}`" for name in shard_names)
    mode = "final-only" if final_only else "compact"
    path.write_text(f"""\
# {model} {effort} {mode} blinded VLM audit batch

This is the token-optimized version. Each request contains up to
{images_per_request} blinded images and returns a JSON array of per-image
judgments. This avoids repeating the prompt and schema once per image.
The final-only mode omits free-text rationales and asks for low verbosity.

Files:

{shard_lines}

Submit:

```powershell
python "{submitter_path}"
```
""", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).parent)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--images-per-request", type=int, default=16)
    parser.add_argument("--requests-per-file", type=int, default=400)
    parser.add_argument("--max-output-tokens", type=int, default=1800)
    parser.add_argument("--image-detail", default="original")
    parser.add_argument("--text-verbosity", default="medium")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Use a terse prompt and omit explanations from the schema.",
    )
    parser.add_argument(
        "--omit-rationale",
        action="store_true",
        help="Remove rationale from each result to reduce output tokens.",
    )
    parser.add_argument(
        "--shard-prefix",
        default="vlm_realism_gpt55_xhigh_compact",
        help="Filename prefix for generated JSONL shards.",
    )
    parser.add_argument(
        "--exclude-batch-index",
        type=Path,
        action="append",
        default=[],
        help="Optional prior batch_index.csv whose image_id values should be excluded.",
    )
    parser.add_argument(
        "--per-source-class-plane",
        type=int,
        default=None,
        help=(
            "Optional balanced sample size per source x class x plane. "
            "For example 10 gives 240 images."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    audit_root = args.audit_root
    if not audit_root.is_absolute():
        audit_root = project_root / audit_root
    audit_root = audit_root.resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir = out_dir.resolve()

    image_index = audit_root / "public" / "image_index.csv"
    images_dir = audit_root / "public" / "images"
    private_key = audit_root / "private" / "answer_key.csv"
    if not image_index.is_file():
        raise SystemExit(f"Missing image index: {image_index}")
    if not images_dir.is_dir():
        raise SystemExit(f"Missing image dir: {images_dir}")
    if not private_key.is_file():
        raise SystemExit(f"Missing private key: {private_key}")
    if args.images_per_request <= 0:
        raise SystemExit("--images-per-request must be positive.")
    if args.requests_per_file <= 0:
        raise SystemExit("--requests-per-file must be positive.")

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "jsonl_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for stale in out_dir.glob("*"):
            if stale.is_file():
                stale.unlink()
        for stale in shard_dir.glob("*.jsonl"):
            stale.unlink()

    exclude_paths = []
    for path in args.exclude_batch_index:
        if not path.is_absolute():
            path = project_root / path
        path = path.resolve()
        if not path.is_file():
            raise SystemExit(f"Missing exclude batch index: {path}")
        exclude_paths.append(path)

    excluded_ids = read_excluded_ids(exclude_paths)
    rows = [
        row for row in read_index(image_index)
        if row["image_id"] not in excluded_ids
    ]
    key_by_id = read_private_key(private_key)
    original_n_rows = len(read_index(image_index))
    if args.per_source_class_plane is not None:
        if args.per_source_class_plane <= 0:
            raise SystemExit("--per-source-class-plane must be positive.")
        rows = balanced_sample(
            rows,
            key_by_id,
            per_source_class_plane=args.per_source_class_plane,
            seed=args.seed,
        )
    grouped = list(chunks(rows, args.images_per_request))
    shard_paths = []
    shard_counts = []
    total_png_bytes = 0
    all_custom_ids = set()

    batch_index_path = out_dir / "batch_index.csv"
    with batch_index_path.open("w", newline="", encoding="utf-8") as f_idx:
        writer = csv.DictWriter(
            f_idx,
            fieldnames=[
                "custom_id", "group_no", "image_id", "filename",
                "image_path", "shard",
            ],
        )
        writer.writeheader()

        shard_file = None
        shard_count = 0
        shard_no = 0
        try:
            for group_no, group in enumerate(grouped, start=1):
                if shard_file is None or shard_count >= args.requests_per_file:
                    if shard_file is not None:
                        shard_file.close()
                        shard_counts.append(shard_count)
                    shard_no += 1
                    shard_path = shard_dir / (
                        f"{args.shard_prefix}_{shard_no:03d}.jsonl"
                    )
                    shard_paths.append(shard_path)
                    shard_file = shard_path.open("w", encoding="utf-8", newline="\n")
                    shard_count = 0

                for row in group:
                    image_path = images_dir / row["filename"]
                    if not image_path.is_file():
                        raise FileNotFoundError(f"Missing image: {image_path}")
                    total_png_bytes += image_path.stat().st_size

                request = make_request(
                    group_no=group_no,
                    group=group,
                    images_dir=images_dir,
                    model=args.model,
                    effort=args.reasoning_effort,
                    max_output_tokens=args.max_output_tokens,
                    image_detail=args.image_detail,
                    text_verbosity=args.text_verbosity,
                    include_rationale=not args.omit_rationale,
                    final_only=args.final_only,
                )
                custom_id = request["custom_id"]
                if custom_id in all_custom_ids:
                    raise ValueError(f"Duplicate custom_id: {custom_id}")
                all_custom_ids.add(custom_id)
                shard_file.write(json.dumps(request, separators=(",", ":")) + "\n")
                shard_count += 1

                for row in group:
                    writer.writerow({
                        "custom_id": custom_id,
                        "group_no": group_no,
                        "image_id": row["image_id"],
                        "filename": row["filename"],
                        "image_path": str(images_dir / row["filename"]),
                        "shard": shard_paths[-1].name,
                    })
        finally:
            if shard_file is not None:
                shard_file.close()
                shard_counts.append(shard_count)

    for shard_path, count in zip(shard_paths, shard_counts):
        if count > args.requests_per_file:
            raise ValueError(f"Shard exceeds request cap: {shard_path}")
        with shard_path.open(encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj["url"] != "/v1/responses":
                    raise ValueError(f"Wrong endpoint in {shard_path}")
                if obj["body"]["model"] != args.model:
                    raise ValueError(f"Wrong model in {shard_path}")
                if obj["body"]["reasoning"]["effort"] != args.reasoning_effort:
                    raise ValueError(f"Wrong effort in {shard_path}")

    shard_sizes = [path.stat().st_size for path in shard_paths]
    summary = {
        "jsonl_shards": [str(path) for path in shard_paths],
        "batch_index": str(batch_index_path),
        "audit_root": str(audit_root),
        "image_index": str(image_index),
        "private_answer_key_not_embedded": str(private_key),
        "endpoint": "/v1/responses",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "image_detail": args.image_detail,
        "text_verbosity": args.text_verbosity,
        "final_only_prompt": args.final_only,
        "include_rationale": not args.omit_rationale,
        "n_images": len(rows),
        "original_n_images": original_n_rows,
        "excluded_image_ids": len(excluded_ids),
        "exclude_batch_indexes": [str(path) for path in exclude_paths],
        "balanced_sample_per_source_class_plane": args.per_source_class_plane,
        "sample_seed": args.seed,
        "images_per_request": args.images_per_request,
        "n_requests": len(grouped),
        "requests_per_file_cap": args.requests_per_file,
        "n_jsonl_shards": len(shard_paths),
        "shard_request_counts": shard_counts,
        "shard_bytes": shard_sizes,
        "total_indexed_png_bytes": total_png_bytes,
        "total_jsonl_bytes": sum(shard_sizes),
        "total_jsonl_mb": round(sum(shard_sizes) / 1024 / 1024, 3),
        "max_output_tokens_per_group_request": args.max_output_tokens,
        "old_one_image_request_count": len(rows),
        "request_count_reduction_factor": round(len(rows) / len(grouped), 3),
    }
    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    shard_names = [path.name for path in shard_paths]
    submitter_path = out_dir / "submit_compact_gpt55.py"
    audit_label = args.shard_prefix
    write_submitter(
        submitter_path,
        shard_names,
        model=args.model,
        effort=args.reasoning_effort,
        audit_label=audit_label,
    )
    write_readme(
        out_dir / "README.md",
        shard_names,
        args.images_per_request,
        model=args.model,
        effort=args.reasoning_effort,
        submitter_path=submitter_path,
        final_only=args.final_only,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
