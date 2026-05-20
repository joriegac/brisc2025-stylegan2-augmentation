from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENTS = ["baseline", "augmented_r1", "filtered_r1", "augmented_r2", "filtered_r2"]
SEEDS = list(range(10))

PLOTS = [
    {
        "code": "P023",
        "slug": "cnn_v2_training_validation_curves",
        "title": "CNN v2 training and validation loss/accuracy curves",
        "family": "CNN v2",
        "model": "CNN v2",
        "script": ROOT / "classify_cnn_v2.py",
        "results_dir": ROOT / "outputs_v2" / "results_cnn_v2",
        "output": OUT_DIR / "P023_cnn_v2_training_validation_curves.jsonl",
        "expected_source": "training_history",
    },
    {
        "code": "P028",
        "slug": "mobilevitv2_training_validation_curves",
        "title": "MobileViTV2 training and validation loss/accuracy curves",
        "family": "MobileViTV2",
        "model": "MobileViTV2",
        "script": ROOT / "classify_mobilevitv2.py",
        "results_dir": ROOT / "outputs_v2" / "results_mobilevitv2",
        "output": OUT_DIR / "P028_mobilevitv2_training_validation_curves.jsonl",
        "expected_source": "training_history",
    },
]

EXPECTED_FIELDS = [
    "seed",
    "experiment",
    "step",
    "real_epoch",
    "tr_loss",
    "vl_loss",
    "vl_tumour",
    "vl_plane",
    "patience_counter",
]


def _seed_jsons(results_dir: Path) -> list[tuple[str, int, Path]]:
    return [
        (experiment, seed, results_dir / f"{experiment}_seed{seed}.json")
        for experiment in EXPERIMENTS
        for seed in SEEDS
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _history_for_seed(path: Path, experiment: str, seed: int,
                      results_dir: Path) -> list[dict[str, Any]]:
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        history = result.get("training_history")
        if isinstance(history, list) and history:
            return [dict(row) for row in history]

    history_path = results_dir / "training_history" / f"{experiment}_seed{seed}_history.jsonl"
    if history_path.exists():
        return _load_jsonl(history_path)

    return []


def _numeric_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    return None


def _normalize_record(plot: dict[str, Any], experiment: str, seed: int,
                      source_json: Path, row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "record_type": "data",
        "code": plot["code"],
        "model": plot["model"],
        "experiment": row.get("experiment", experiment),
        "seed": int(row.get("seed", seed)),
        "source": row.get("source", "step" if "step" in row else "epoch"),
        "step": _numeric_or_none(row.get("step")),
        "epoch": _numeric_or_none(row.get("epoch")),
        "real_epoch": _numeric_or_none(row.get("real_epoch")),
        "tr_loss": _numeric_or_none(row.get("tr_loss")),
        "vl_loss": _numeric_or_none(row.get("vl_loss")),
        "vl_tumour": _numeric_or_none(row.get("vl_tumour")),
        "vl_plane": _numeric_or_none(row.get("vl_plane")),
        "patience_counter": _numeric_or_none(row.get("patience_counter")),
        "best_epoch": _numeric_or_none(row.get("best_epoch")),
        "best_step": _numeric_or_none(row.get("best_step")),
        "best_val_loss": _numeric_or_none(row.get("best_val_loss")),
        "learning_rate": _numeric_or_none(row.get("learning_rate")),
        "source_json": str(source_json),
    }

    for key in (
        "training_budget",
        "max_steps",
        "budget_epoch_steps",
        "val_every",
        "patience_evals",
        "batch_real",
        "batch_synthetic",
        "real_examples_seen",
        "synthetic_examples_seen",
    ):
        if key in row:
            normalized[key] = row[key]

    return normalized


def _build_plot(plot: dict[str, Any]) -> None:
    seed_jsons = _seed_jsons(plot["results_dir"])
    existing_sources = [plot["script"], *[path for _, _, path in seed_jsons if path.exists()]]

    data_records = []
    missing_history = []
    for experiment, seed, path in seed_jsons:
        history = _history_for_seed(path, experiment, seed, plot["results_dir"])
        if not history:
            missing_history.append(str(path))
            continue
        data_records.extend(
            _normalize_record(plot, experiment, seed, path, row)
            for row in history
        )

    metadata = {
        "record_type": "plot_metadata",
        "code": plot["code"],
        "slug": plot["slug"],
        "title": plot["title"],
        "family": plot["family"],
        "plot_type": "training curve grid",
        "arxiv_role": "Training dynamics for loss and validation accuracy.",
        "data_status": "available" if data_records and not missing_history else (
            "partial" if data_records else "not_persisted"
        ),
        "notes": (
            "Built from persisted per-validation training_history rows saved by the training script."
            if data_records else
            "No persisted per-evaluation tr_loss/vl_loss/vl_tumour/vl_plane history was found."
        ),
        "expected_fields": EXPECTED_FIELDS,
        "source_paths": [str(path) for path in existing_sources],
        "missing_history_sources": missing_history,
        "current_or_expected_figure_outputs": [],
    }

    records = [metadata]
    if data_records:
        records.extend(data_records)
    else:
        records.append({
            "record_type": "data_availability",
            "code": plot["code"],
            "model": plot["model"],
            "data_status": "No persisted per-evaluation tr_loss/vl_loss/vl_tumour/vl_plane history was found in seed JSON files or training_history JSONL files.",
            "expected_fields": EXPECTED_FIELDS,
            "recommended_source": "Rerun the relevant classifier stages with history logging enabled.",
        })

    plot["output"].write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {plot['output']} ({len(records)} records; status={metadata['data_status']})")


def main() -> None:
    for plot in PLOTS:
        _build_plot(plot)


if __name__ == "__main__":
    main()
