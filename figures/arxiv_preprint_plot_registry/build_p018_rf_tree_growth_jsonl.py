from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

EXPERIMENTS = ["baseline", "augmented_r1", "filtered_r1", "augmented_r2", "filtered_r2"]
HISTORY_DIR = ROOT / "outputs_v2" / "results" / "training_history"
OUTPUT = OUT_DIR / "P018_rf_v2_training_validation_curves.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _numeric_or_none(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    return None


def _normalize_row(experiment: str, source_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    out = {
        "record_type": "data",
        "code": "P018",
        "model": "RF/classic classifier v2",
        "experiment": row.get("experiment", experiment),
        "seed": int(row["seed"]),
        "source": row.get("source", "rf_tree_growth"),
        "n_estimators": int(row["n_estimators"]),
        "oob_score": _numeric_or_none(row.get("oob_score")),
        "test_accuracy": _numeric_or_none(row.get("test_accuracy")),
        "test_macro_f1": _numeric_or_none(row.get("test_macro_f1")),
        "test_weighted_f1": _numeric_or_none(row.get("test_weighted_f1")),
        "source_jsonl": str(source_path),
    }
    for key in sorted(row):
        if key.startswith("f1_"):
            out[key] = _numeric_or_none(row.get(key))
    return out


def main() -> None:
    source_paths = []
    records: list[dict[str, Any]] = []
    missing = []

    for experiment in EXPERIMENTS:
        path = HISTORY_DIR / f"{experiment}_history.jsonl"
        source_paths.append(str(path))
        if not path.exists():
            missing.append(str(path))
            continue
        for row in _load_jsonl(path):
            records.append(_normalize_row(experiment, path, row))

    status = "available" if records and not missing else ("partial" if records else "not_persisted")
    metadata = {
        "record_type": "plot_metadata",
        "code": "P018",
        "slug": "rf_v2_training_validation_curves",
        "title": "RF/classic classifier v2 tree-growth diagnostic curves",
        "family": "RF/classic classifier v2",
        "plot_type": "tree-growth/OOB diagnostic curve",
        "arxiv_role": "Random forest stabilization diagnostics across tree counts.",
        "data_status": status,
        "source_paths": [str(ROOT / "classify_rf_v2.py"), *source_paths],
        "current_or_expected_figure_outputs": [
            str(ROOT / "figures" / "pgfplots" / "pgfplots_p018" / "P018_rf_v2_tree_growth_oob.tex"),
            str(ROOT / "figures" / "pgfplots" / "pgfplots_p018" / "P018_rf_v2_tree_growth_oob.pdf"),
        ],
        "notes": (
            "RF has no neural train/validation loss. This registry file stores "
            "per-seed checkpoint diagnostics as trees are added: OOB score, test "
            "accuracy, macro F1, weighted F1, and per-class F1."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(metadata, sort_keys=True) + "\n")
        if records:
            for record in sorted(records, key=lambda r: (r["experiment"], r["seed"], r["n_estimators"])):
                fh.write(json.dumps(record, sort_keys=True) + "\n")
        else:
            fh.write(
                json.dumps(
                    {
                        "record_type": "data_availability",
                        "code": "P018",
                        "model": "RF/classic classifier v2",
                        "data_status": "No RF tree-growth history JSONL files were found.",
                        "recommended_source": str(HISTORY_DIR),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    print(f"wrote {OUTPUT} ({len(records)} data records, status={status})")
    if missing:
        print("missing:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
