from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
HISTORY_DIR = PROJECT_ROOT / "outputs_v2" / "results_cnn_v2" / "training_history"
DATA_DIR = ROOT / "data"

CONDITIONS = [
    ("baseline", "Baseline", "black!72"),
    ("augmented_r1", "Augmented 1:1", "condAugOne"),
    ("filtered_r1", "Filtered 1:1", "condFiltOne"),
    ("augmented_r2", "Augmented 1:2", "condAugTwo"),
    ("filtered_r2", "Filtered 1:2", "condFiltTwo"),
]

FIELDS = [
    "experiment",
    "seed",
    "epoch",
    "real_epoch",
    "tr_loss",
    "vl_loss",
    "vl_tumour",
    "vl_plane",
    "learning_rate",
    "patience_counter",
    "best_epoch",
    "best_val_loss",
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


def load_rows() -> list[dict[str, float | int | str]]:
    paths = sorted(HISTORY_DIR.glob("*_seed*_history.jsonl"))
    if not paths:
        raise SystemExit(f"No per-seed CNN history JSONL files found in {HISTORY_DIR}")

    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = {"experiment", "seed", "epoch", "tr_loss", "vl_loss"} - row.keys()
                if missing:
                    raise ValueError(f"{path}:{line_no} missing required fields: {sorted(missing)}")
                rows.append(row)
    return rows


def write_raw(rows: list[dict[str, float | int | str]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "cnn_v2_training_history_raw.dat"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, delimiter=" ", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: list[dict[str, float | int | str]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), int(row["epoch"]))].append(row)

    for experiment, _, _ in CONDITIONS:
        out = DATA_DIR / f"cnn_v2_loss_summary_{experiment}.dat"
        keys = sorted(epoch for exp, epoch in grouped if exp == experiment)
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=" ")
            writer.writerow(
                [
                    "epoch",
                    "tr_q1",
                    "tr_median",
                    "tr_q3",
                    "vl_q1",
                    "vl_median",
                    "vl_q3",
                    "vl_tumour_q1",
                    "vl_tumour_median",
                    "vl_tumour_q3",
                    "n",
                ]
            )
            for epoch in keys:
                records = grouped[(experiment, epoch)]
                tr = [float(r["tr_loss"]) for r in records]
                vl = [float(r["vl_loss"]) for r in records]
                tumour_acc = [100.0 * float(r["vl_tumour"]) for r in records]
                writer.writerow(
                    [
                        epoch,
                        f"{quantile(tr, 0.25):.6f}",
                        f"{quantile(tr, 0.50):.6f}",
                        f"{quantile(tr, 0.75):.6f}",
                        f"{quantile(vl, 0.25):.6f}",
                        f"{quantile(vl, 0.50):.6f}",
                        f"{quantile(vl, 0.75):.6f}",
                        f"{quantile(tumour_acc, 0.25):.6f}",
                        f"{quantile(tumour_acc, 0.50):.6f}",
                        f"{quantile(tumour_acc, 0.75):.6f}",
                        len(records),
                    ]
                )


def rel_data(name: str) -> str:
    return f"data/{name}"


def add_metric_plots(y_mid: str, y_low: str, y_high: str, prefix: str, legend: bool) -> str:
    lines: list[str] = []
    for experiment, label, colour in CONDITIONS:
        path = rel_data(f"cnn_v2_loss_summary_{experiment}.dat")
        upper = f"{prefix}_{experiment}_upper"
        lower = f"{prefix}_{experiment}_lower"
        lines.extend(
            [
                rf"\addplot[name path={upper}, draw=none, forget plot] table[x=epoch, y={y_high}] {{{path}}};",
                rf"\addplot[name path={lower}, draw=none, forget plot] table[x=epoch, y={y_low}] {{{path}}};",
                rf"\addplot[{colour}, fill opacity=0.14, draw=none, forget plot] fill between[of={upper} and {lower}];",
                rf"\addplot[{colour}, line width=0.72pt] table[x=epoch, y={y_mid}] {{{path}}};",
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

% Condition colours -- same palette as P017/P018/P022.
\definecolor{{condAugOne}}{{HTML}}{{440154}}
\definecolor{{condAugTwo}}{{HTML}}{{2D708E}}
\definecolor{{condFiltOne}}{{HTML}}{{20A387}}
\definecolor{{condFiltTwo}}{{HTML}}{{73D055}}

\pgfplotsset{{
  every axis/.append style={{
    width=90mm,
    height=39mm,
    scale only axis,
    xmin=1,
    xmax=200,
    enlargelimits=false,
    xtick={{1,40,80,120,160,200}},
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
  ymin=-0.35,
  ymax=4.40,
  ytick={{0,1,2,3,4}},
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
{add_metric_plots("vl_median", "vl_q1", "vl_q3", "valid_loss", True)}

\nextgroupplot[
  xlabel={{Epoch}},
  ylabel={{Tumour validation accuracy (\%)}},
  ymin=20,
  ymax=101,
  ytick={{20,40,60,80,100}},
  after end axis/.code={{
    \node[anchor=south east, font=\scriptsize\sffamily\bfseries, yshift=2pt]
      at (current axis.north east) {{(b)}};
  }},
]
{add_metric_plots("vl_tumour_median", "vl_tumour_q1", "vl_tumour_q3", "valid_acc", False)}

\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    out = ROOT / "P023_cnn_v2_training_validation_curves.tex"
    out.write_text(tex, encoding="utf-8")
    return out


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=ROOT,
        check=True,
    )


def make_preview() -> None:
    pdf = ROOT / "P023_cnn_v2_training_validation_curves.pdf"
    png = ROOT / "P023_cnn_v2_training_validation_curves_preview.png"
    magick = shutil.which("magick")
    if magick:
        subprocess.run(
            [magick, "-density", "220", str(pdf), "-quality", "95", str(png)],
            cwd=ROOT,
            check=True,
        )
        return

    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise SystemExit("Preview renderer not found: expected magick or pdftoppm")

    prefix = ROOT / "P023_cnn_v2_training_validation_curves_preview"
    generated = ROOT / "P023_cnn_v2_training_validation_curves_preview-1.png"
    subprocess.run(
        [pdftoppm, "-png", "-r", "220", str(pdf), str(prefix)],
        cwd=ROOT,
        check=True,
    )
    if generated.exists():
        generated.replace(png)


def main() -> None:
    rows = load_rows()
    write_raw(rows)
    write_summary(rows)
    tex_path = write_tex()
    compile_tex(tex_path)
    make_preview()

    seeds_by_condition: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        seeds_by_condition[str(row["experiment"])].add(int(row["seed"]))
    print(f"Loaded records: {len(rows)}")
    for experiment, _, _ in CONDITIONS:
        print(f"{experiment}: {len(seeds_by_condition[experiment])} seeds")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
