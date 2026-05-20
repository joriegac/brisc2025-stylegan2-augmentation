from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
REGISTRY = ROOT / "figures" / "arxiv_preprint_plot_registry" / "P018_rf_v2_training_validation_curves.jsonl"
TEX = HERE / "P018_rf_v2_tree_growth_oob.tex"
PDF = HERE / "P018_rf_v2_tree_growth_oob.pdf"
PREVIEW = HERE / "P018_rf_v2_tree_growth_oob_preview.png"

EXPERIMENT_LABELS = {
    "baseline": "Baseline",
    "augmented_r1": "Augmented 1:1",
    "filtered_r1": "Filtered 1:1",
    "augmented_r2": "Augmented 1:2",
    "filtered_r2": "Filtered 1:2",
}
EXPERIMENT_ORDER = ["baseline", "augmented_r1", "filtered_r1", "augmented_r2", "filtered_r2"]
METRICS = {
    "test_accuracy": "Test accuracy",
    "oob_score": "OOB score",
}


def _load_records() -> list[dict[str, Any]]:
    rows = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record_type") == "data":
            rows.append(row)
    return rows


def _mean_half_ci(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0, 0.0
    sd = statistics.stdev(values)
    half_ci = 1.96 * sd / math.sqrt(len(values))
    return mean, sd, half_ci


def _summarize(rows: list[dict[str, Any]], metric: str) -> dict[str, list[dict[str, float]]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        value = row.get(metric)
        if not isinstance(value, (int, float)):
            continue
        key = (row["experiment"], int(row["n_estimators"]))
        grouped.setdefault(key, []).append(100.0 * float(value))

    summary: dict[str, list[dict[str, float]]] = {experiment: [] for experiment in EXPERIMENT_ORDER}
    for (experiment, n_estimators), values in sorted(grouped.items()):
        mean, sd, half_ci = _mean_half_ci(values)
        summary.setdefault(experiment, []).append(
            {
                "n_estimators": float(n_estimators),
                "mean": mean,
                "sd": sd,
                "half_ci": half_ci,
                "n": float(len(values)),
            }
        )
    return summary


def _write_data_files(rows: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for metric in METRICS:
        summary = _summarize(rows, metric)
        for experiment in EXPERIMENT_ORDER:
            path = DATA_DIR / f"{metric}_{experiment}.dat"
            with path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write("n_estimators mean sd half_ci ci_low ci_high n\n")
                for item in summary.get(experiment, []):
                    ci_low = item["mean"] - item["half_ci"]
                    ci_high = item["mean"] + item["half_ci"]
                    fh.write(
                        f"{int(item['n_estimators'])} "
                        f"{item['mean']:.5f} {item['sd']:.5f} "
                        f"{item['half_ci']:.5f} {ci_low:.5f} {ci_high:.5f} "
                        f"{int(item['n'])}\n"
                    )


def _plot_band(metric: str, experiment: str, color: str) -> str:
    path = f"data/{metric}_{experiment}.dat"
    upper = f"{metric}_{experiment}_upper"
    lower = f"{metric}_{experiment}_lower"
    return rf"""
\addplot[name path={upper}, draw=none, forget plot]
  table[x=n_estimators, y=ci_high] {{{path}}};
\addplot[name path={lower}, draw=none, forget plot]
  table[x=n_estimators, y=ci_low] {{{path}}};
\addplot[
  {color},
  fill opacity=0.14,
  draw=none,
  forget plot,
] fill between[of={upper} and {lower}];
"""


def _plot_line(metric: str, experiment: str, color: str,
               include_legend: bool = True) -> str:
    path = f"data/{metric}_{experiment}.dat"
    label = EXPERIMENT_LABELS[experiment]
    legend = rf"\addlegendentry{{{label}}}" if include_legend else ""
    return rf"""
\addplot+[
  {color},
  line width=0.95pt,
  mark=none,
] table[x=n_estimators, y=mean] {{{path}}};
{legend}
"""


def _write_tex() -> None:
    colors = {
        "baseline": "black!72",
        "augmented_r1": "condAugOne",
        "augmented_r2": "condAugTwo",
        "filtered_r1": "condFiltOne",
        "filtered_r2": "condFiltTwo!72!black",
    }
    bands_accuracy = "\n".join(
        _plot_band("test_accuracy", exp, colors[exp]) for exp in EXPERIMENT_ORDER
    )
    lines_accuracy = "\n".join(
        _plot_line("test_accuracy", exp, colors[exp]) for exp in EXPERIMENT_ORDER
    )
    bands_oob = "\n".join(
        _plot_band("oob_score", exp, colors[exp]) for exp in EXPERIMENT_ORDER
    )
    lines_oob = "\n".join(
        _plot_line("oob_score", exp, colors[exp], include_legend=False)
        for exp in EXPERIMENT_ORDER
    )

    tex = rf"""\documentclass[tikz,border=2pt]{{standalone}}
\input{{../pgfplots_font_setup.tex}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots,fillbetween}}
\pgfplotsset{{compat=1.18}}

% Condition colours and confidence-band shading match the other condition plots.
\definecolor{{condAugOne}}{{HTML}}{{7B1E3A}}
\definecolor{{condAugTwo}}{{HTML}}{{2D708E}}
\definecolor{{condFiltOne}}{{HTML}}{{20A387}}
\definecolor{{condFiltTwo}}{{HTML}}{{73D055}}

\pgfplotsset{{
  every axis/.append style={{
    width=90mm,
    height=38mm,
    scale only axis,
    xmin=25,
    xmax=500,
    enlargelimits=false,
    xtick={{25,100,200,300,400,500}},
    scaled x ticks=false,
    axis line style={{black, line width=0.35pt}},
    tick style={{black, line width=0.35pt}},
    tick label style={{font=\scriptsize, /pgf/number format/fixed}},
    label style={{font=\scriptsize}},
    tick align=outside,
    tick pos=left,
    major tick length=2.2pt,
    legend style={{
      font=\scriptsize,
      draw=none,
      fill=none,
      cells={{anchor=west}},
      /tikz/every even column/.append style={{column sep=4pt}}
    }},
    grid=both,
    major grid style={{black!10, line width=0.25pt}},
    minor grid style={{black!5, line width=0.2pt}},
    minor tick num=1,
    unbounded coords=discard,
  }},
  every axis plot/.append style={{line width=0.72pt}},
}}

\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
  group style={{group size=1 by 2, vertical sep=8mm}},
]

\nextgroupplot[
  ylabel={{Held-out accuracy (\%)}},
  ymin=76.2,
  ymax=81.0,
  ytick={{77,78,79,80,81}},
  xticklabels={{}},
  legend columns=5,
  legend style={{
    at={{(0.5,1.20)}}, anchor=south,
    font=\scriptsize, draw=none, fill=none,
    cells={{anchor=west}},
    /tikz/every even column/.append style={{column sep=5pt}},
    row sep=0.4pt
  }},
  after end axis/.code={{
    \node[anchor=south east, font=\scriptsize\bfseries, yshift=2pt]
      at (current axis.north east) {{(a)}};
  }},
]
{bands_accuracy}
{lines_accuracy}

\nextgroupplot[
  xlabel={{Trees ($n_{{estimators}}$)}},
  ylabel={{OOB accuracy (\%)}},
  ymin=76.4,
  ymax=90.8,
  ytick={{78,82,86,90}},
  after end axis/.code={{
    \node[anchor=south east, font=\scriptsize\bfseries, yshift=2pt]
      at (current axis.north east) {{(b)}};
  }},
]
{bands_oob}
{lines_oob}

\end{{groupplot}}
\end{{tikzpicture}}
\end{{document}}
"""
    TEX.write_text(tex, encoding="utf-8", newline="\n")


def _compile() -> None:
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", TEX.name],
        cwd=HERE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (HERE / "P018_rf_v2_tree_growth_oob.compile.log").write_text(
        result.stdout, encoding="utf-8", newline="\n"
    )
    if result.returncode != 0:
        raise SystemExit(f"pdflatex failed with exit code {result.returncode}")

    subprocess.run(
        ["pdftoppm", "-png", "-singlefile", "-r", "220", str(PDF), str(PREVIEW.with_suffix(""))],
        cwd=HERE,
        check=True,
    )


def main() -> None:
    rows = _load_records()
    if not rows:
        raise SystemExit(f"no data records found in {REGISTRY}")
    _write_data_files(rows)
    _write_tex()
    _compile()
    print(f"wrote {TEX}")
    print(f"wrote {PDF}")
    print(f"wrote {PREVIEW}")


if __name__ == "__main__":
    main()
