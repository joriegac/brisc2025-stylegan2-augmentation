from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
HISTORY_DIR = PROJECT_ROOT / "outputs_v2" / "results_mobilevitv2" / "training_history"
DATA_DIR = ROOT / "data"

CONDITIONS = [
    ("baseline", "Baseline", "black!72"),
    ("augmented_r1", "Augmented 1:1", "condAugOne"),
    ("filtered_r1", "Filtered 1:1", "condFiltOne"),
    ("augmented_r2", "Augmented 1:2", "condAugTwo"),
    ("filtered_r2", "Filtered 1:2", "condFiltTwo"),
]

CI_Z = 1.96

FIELDS = [
    "experiment",
    "seed",
    "step",
    "real_epoch",
    "tr_loss",
    "vl_loss",
    "vl_tumour",
    "vl_plane",
    "learning_rate",
    "patience_counter",
    "best_step",
    "best_val_loss",
    "batch_real",
    "batch_synthetic",
]


def quantile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def mean_sd_ci(values: list[float]) -> tuple[float, float, float, float, float]:
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half_ci = CI_Z * sd / math.sqrt(arr.size) if arr.size > 1 else 0.0
    return mean, sd, half_ci, mean - half_ci, mean + half_ci


def load_rows() -> list[dict[str, float | int | str]]:
    paths = sorted(HISTORY_DIR.glob("*_seed*_history.jsonl"))
    if not paths:
        raise SystemExit(f"No per-seed MobileViTV2 history JSONL files found in {HISTORY_DIR}")

    rows: list[dict[str, float | int | str]] = []
    required = {"experiment", "seed", "step", "real_epoch", "tr_loss", "vl_loss", "vl_tumour", "vl_plane"}
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required - row.keys()
                if missing:
                    raise ValueError(f"{path}:{line_no} missing required fields: {sorted(missing)}")
                rows.append(row)
    return rows


def write_raw(rows: list[dict[str, float | int | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "mobilevitv2_training_history_raw.dat"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter=" ", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: list[dict[str, float | int | str]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), int(row["step"]))].append(row)

    for experiment, _, _ in CONDITIONS:
        out = DATA_DIR / f"mobilevitv2_validation_summary_{experiment}.dat"
        steps = sorted(step for exp, step in grouped if exp == experiment)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=" ")
            writer.writerow(
                [
                    "step",
                    "real_epoch_median",
                    "vl_q1",
                    "vl_median",
                    "vl_q3",
                    "vl_mean",
                    "vl_sd",
                    "vl_half_ci",
                    "vl_ci_low",
                    "vl_ci_high",
                    "vl_tumour_q1",
                    "vl_tumour_median",
                    "vl_tumour_q3",
                    "vl_tumour_mean",
                    "vl_tumour_sd",
                    "vl_tumour_half_ci",
                    "vl_tumour_ci_low",
                    "vl_tumour_ci_high",
                    "n",
                ]
            )
            for step in steps:
                records = grouped[(experiment, step)]
                real_epoch = [float(r["real_epoch"]) for r in records]
                vl = [float(r["vl_loss"]) for r in records]
                tumour_acc = [100.0 * float(r["vl_tumour"]) for r in records]
                vl_mean, vl_sd, vl_half_ci, vl_ci_low, vl_ci_high = mean_sd_ci(vl)
                (
                    tumour_mean,
                    tumour_sd,
                    tumour_half_ci,
                    tumour_ci_low,
                    tumour_ci_high,
                ) = mean_sd_ci(tumour_acc)
                writer.writerow(
                    [
                        step,
                        f"{quantile(real_epoch, 0.50):.6f}",
                        f"{quantile(vl, 0.25):.6f}",
                        f"{quantile(vl, 0.50):.6f}",
                        f"{quantile(vl, 0.75):.6f}",
                        f"{vl_mean:.6f}",
                        f"{vl_sd:.6f}",
                        f"{vl_half_ci:.6f}",
                        f"{vl_ci_low:.6f}",
                        f"{vl_ci_high:.6f}",
                        f"{quantile(tumour_acc, 0.25):.6f}",
                        f"{quantile(tumour_acc, 0.50):.6f}",
                        f"{quantile(tumour_acc, 0.75):.6f}",
                        f"{tumour_mean:.6f}",
                        f"{tumour_sd:.6f}",
                        f"{tumour_half_ci:.6f}",
                        f"{tumour_ci_low:.6f}",
                        f"{tumour_ci_high:.6f}",
                        len(records),
                    ]
                )


def write_smooth_summary() -> None:
    columns = [
        "real_epoch_median",
        "vl_q1",
        "vl_median",
        "vl_q3",
        "vl_mean",
        "vl_half_ci",
        "vl_tumour_q1",
        "vl_tumour_median",
        "vl_tumour_q3",
        "vl_tumour_mean",
        "vl_tumour_half_ci",
    ]
    smooth_steps = np.unique(np.rint(np.linspace(1, 1705, 180)).astype(int))
    log_smooth = np.log(smooth_steps)

    for experiment, _, _ in CONDITIONS:
        src = DATA_DIR / f"mobilevitv2_validation_summary_{experiment}.dat"
        with src.open("r", encoding="utf-8") as fh:
            rows = [row for row in csv.DictReader(fh, delimiter=" ") if row]

        steps = np.array([float(row["step"]) for row in rows], dtype=float)
        log_steps = np.log(steps)
        out_cols: dict[str, np.ndarray] = {"step": smooth_steps.astype(float)}

        for col in columns:
            values = np.array([float(row[col]) for row in rows], dtype=float)
            coeff = np.polyfit(log_steps, values, 3)
            fitted = np.polyval(coeff, log_smooth)
            fitted[0] = values[0]
            fitted[-1] = values[-1]
            if col.startswith("vl_tumour"):
                fitted = np.clip(fitted, 0.0, 100.0)
            out_cols[col] = fitted

        out_cols["vl_q1"], out_cols["vl_q3"] = (
            np.minimum(out_cols["vl_q1"], out_cols["vl_q3"]),
            np.maximum(out_cols["vl_q1"], out_cols["vl_q3"]),
        )
        out_cols["vl_tumour_q1"], out_cols["vl_tumour_q3"] = (
            np.minimum(out_cols["vl_tumour_q1"], out_cols["vl_tumour_q3"]),
            np.maximum(out_cols["vl_tumour_q1"], out_cols["vl_tumour_q3"]),
        )
        out_cols["vl_half_ci"] = np.maximum(out_cols["vl_half_ci"], 0.0)
        out_cols["vl_ci_low"] = out_cols["vl_mean"] - out_cols["vl_half_ci"]
        out_cols["vl_ci_high"] = out_cols["vl_mean"] + out_cols["vl_half_ci"]
        out_cols["vl_tumour_half_ci"] = np.maximum(out_cols["vl_tumour_half_ci"], 0.0)
        out_cols["vl_tumour_ci_low"] = np.clip(
            out_cols["vl_tumour_mean"] - out_cols["vl_tumour_half_ci"], 0.0, 100.0
        )
        out_cols["vl_tumour_ci_high"] = np.clip(
            out_cols["vl_tumour_mean"] + out_cols["vl_tumour_half_ci"], 0.0, 100.0
        )
        columns_out = [
            "real_epoch_median",
            "vl_q1",
            "vl_median",
            "vl_q3",
            "vl_mean",
            "vl_half_ci",
            "vl_ci_low",
            "vl_ci_high",
            "vl_tumour_q1",
            "vl_tumour_median",
            "vl_tumour_q3",
            "vl_tumour_mean",
            "vl_tumour_half_ci",
            "vl_tumour_ci_low",
            "vl_tumour_ci_high",
        ]

        out = DATA_DIR / f"smooth_mobilevitv2_validation_summary_{experiment}.dat"
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=" ")
            writer.writerow(["step", *columns_out, "n"])
            for i, step in enumerate(smooth_steps):
                writer.writerow(
                    [
                        int(step),
                        *[f"{out_cols[col][i]:.6f}" for col in columns_out],
                        rows[0].get("n", "10"),
                    ]
                )


def rel_data(name: str) -> str:
    return f"data/{name}"


def add_metric_plots(y_mid: str, y_low: str, y_high: str, prefix: str, legend: bool) -> str:
    lines: list[str] = []
    for experiment, label, colour in CONDITIONS:
        path = rel_data(f"smooth_mobilevitv2_validation_summary_{experiment}.dat")
        upper = f"{prefix}_{experiment}_upper"
        lower = f"{prefix}_{experiment}_lower"
        lines.extend(
            [
                rf"\addplot[name path={upper}, draw=none, forget plot] table[x=step, y={y_high}] {{{path}}};",
                rf"\addplot[name path={lower}, draw=none, forget plot] table[x=step, y={y_low}] {{{path}}};",
                rf"\addplot[{colour}, fill opacity=0.14, draw=none, forget plot] fill between[of={upper} and {lower}];",
                rf"\addplot[{colour}, line width=0.72pt] table[x=step, y={y_mid}] {{{path}}};",
            ]
        )
        if legend:
            lines.append(rf"\addlegendentry{{{label}}}")
    return "\n".join(lines)


def write_tex() -> Path:
    tex = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\input{{../pgfplots_font_setup.tex}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots,fillbetween}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}

% Condition colours -- same palette as P017/P018/P022/P023.
\definecolor{{condAugOne}}{{HTML}}{{7B1E3A}}
\definecolor{{condAugTwo}}{{HTML}}{{2D708E}}
\definecolor{{condFiltOne}}{{HTML}}{{20A387}}
\definecolor{{condFiltTwo}}{{HTML}}{{73D055}}

\pgfplotsset{{
  every axis/.append style={{
    width=90mm,
    height=39mm,
    scale only axis,
    xmin=1,
    xmax=1705,
    enlargelimits=false,
    xtick={{1,400,800,1200,1600}},
    scaled x ticks=false,
    axis line style={{black, line width=0.35pt}},
    tick style={{black, line width=0.35pt}},
    tick label style={{font=\scriptsize\sffamily, /pgf/number format/assume math mode=true, /pgf/number format/fixed}},
    label style={{font=\scriptsize\sffamily}},
    tick align=outside,
    tick pos=left,
    major tick length=2.2pt,
    legend style={{
      font=\pgfplotlegendfont,
      draw=none,
      fill=none,
      cells={{anchor=west}},
      /tikz/every even column/.append style={{column sep=3pt}},
      row sep=0pt
    }},
    grid=both,
    major grid style={{black!10, line width=0.25pt}},
    minor grid style={{black!5, line width=0.2pt}},
    minor tick num=1,
    unbounded coords=discard,
  }},
}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=1 by 2, vertical sep=8mm}},
]

\nextgroupplot[
  ylabel={{Validation loss}},
  ymin=-0.90,
  ymax=2.70,
  ytick={{-0.5,0,0.5,1.0,1.5,2.0,2.5}},
  xticklabels={{}},
  legend columns=5,
  legend style={{
    at={{(0.5,1.20)}}, anchor=south,
    font=\pgfplotlegendfont, draw=none, fill=none,
    cells={{anchor=west}},
    /tikz/every even column/.append style={{column sep=3pt}},
    row sep=0pt
  }},
  after end axis/.code={{
    \node[anchor=south east, font=\scriptsize\sffamily\bfseries, yshift=2pt]
      at (current axis.north east) {{(a)}};
  }},
]
{add_metric_plots("vl_mean", "vl_ci_low", "vl_ci_high", "valid_loss", True)}

\nextgroupplot[
  xlabel={{Training step}},
  ylabel={{Tumour validation accuracy (\%)}},
  ymin=20,
  ymax=101,
  ytick={{20,40,60,80,100}},
  after end axis/.code={{
    \node[anchor=south east, font=\scriptsize\sffamily\bfseries, yshift=2pt]
      at (current axis.north east) {{(b)}};
  }},
]
{add_metric_plots("vl_tumour_mean", "vl_tumour_ci_low", "vl_tumour_ci_high", "valid_acc", False)}

\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    out = ROOT / "P028_mobilevitv2_training_validation_curves.tex"
    out.write_text(tex, encoding="utf-8")
    return out


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=ROOT,
        check=True,
    )


def make_preview() -> None:
    pdf = ROOT / "P028_mobilevitv2_training_validation_curves.pdf"
    png = ROOT / "P028_mobilevitv2_training_validation_curves_preview.png"
    magick = shutil.which("magick")
    if magick:
        subprocess.run([magick, "-density", "220", str(pdf), "-quality", "95", str(png)], cwd=ROOT, check=True)
        return

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise SystemExit("Preview renderer not found: expected magick or pdftoppm")
    prefix = ROOT / "P028_mobilevitv2_training_validation_curves_preview"
    generated = ROOT / "P028_mobilevitv2_training_validation_curves_preview-1.png"
    subprocess.run([pdftoppm, "-png", "-r", "220", str(pdf), str(prefix)], cwd=ROOT, check=True)
    if generated.exists():
        generated.replace(png)


def update_registry(rows: list[dict[str, float | int | str]], paths: list[Path]) -> None:
    registry = PROJECT_ROOT / "figures" / "arxiv_preprint_plot_registry" / "P028_mobilevitv2_training_validation_curves.jsonl"
    outputs = [
        ROOT / "P028_mobilevitv2_training_validation_curves.tex",
        ROOT / "P028_mobilevitv2_training_validation_curves.pdf",
        ROOT / "P028_mobilevitv2_training_validation_curves_preview.png",
        DATA_DIR / "mobilevitv2_training_history_raw.dat",
        *[DATA_DIR / f"smooth_mobilevitv2_validation_summary_{experiment}.dat" for experiment, _, _ in CONDITIONS],
    ]
    meta = {
        "record_type": "plot_metadata",
        "code": "P028",
        "slug": "mobilevitv2_training_validation_curves",
        "title": "MobileViTV2 validation loss and tumour validation accuracy curves",
        "family": "MobileViTV2",
        "plot_type": "validation loss and validation accuracy curves",
        "data_status": "available",
        "status": "available",
        "arxiv_role": "Validation dynamics for MobileViTV2 loss and tumour accuracy across augmentation conditions.",
        "expected_fields": ["seed", "experiment", "step", "real_epoch", "tr_loss", "vl_loss", "vl_tumour", "vl_plane"],
        "source_paths": [str(PROJECT_ROOT / "classify_mobilevitv2.py")] + [str(p) for p in paths],
        "current_or_expected_figure_outputs": [str(p) for p in outputs],
        "notes": "Built from persisted per-step training_history JSONL rows. Curves keep the full compute-matched step range and show cubic-polynomial smoothed means with estimated 95% CI shading across seeds.",
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", encoding="utf-8") as out:
        out.write(json.dumps(meta, sort_keys=True) + "\n")
        for row in rows:
            record = dict(row)
            record.update({"record_type": "data", "code": "P028", "model": "MobileViTV2"})
            out.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    paths = sorted(HISTORY_DIR.glob("*_seed*_history.jsonl"))
    rows = load_rows()
    write_raw(rows)
    write_summary(rows)
    write_smooth_summary()
    tex_path = write_tex()
    compile_tex(tex_path)
    make_preview()
    update_registry(rows, paths)

    seeds_by_condition: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        seeds_by_condition[str(row["experiment"])].add(int(row["seed"]))
    print(f"Loaded records: {len(rows)}")
    for experiment, _, _ in CONDITIONS:
        print(f"{experiment}: {len(seeds_by_condition[experiment])} seeds")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
