#!/usr/bin/env python3
"""Regenerate SVG insert PDFs with Fira-rendered labels.

The SVG converter available in this build environment substitutes several SVG
text runs with Arial, Helvetica, or STIX.  This helper keeps the original SVG
artwork, exports a temporary textless background, and overlays the labels with
LuaLaTeX so the cached PDFs used by ``\\includesvg`` embed Fira Sans and Fira
Math.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "figures"
SVG_CACHE_DIR = ROOT / "arXiv" / "svg-inkscape"

SOURCES = (
    ("StyleGAN2-ADA_illustration.svg", "StyleGAN2-ADA_illustration_svg-raw.pdf"),
    ("CNN_illustration.svg", "CNN_illustration_svg-raw.pdf"),
    ("Mobile_Illustration.svg", "Mobile_Illustration_svg-raw.pdf"),
)

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

UNIT_TO_MM = 0.1
MM_TO_PT = 72.0 / 25.4
UNIT_TO_PT = UNIT_TO_MM * MM_TO_PT

NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
CSS_RULE_RE = re.compile(r"([^{}]+)\{([^{}]+)\}", re.S)
TRANSFORM_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")

STYLE_KEYS = (
    "fill",
    "font-family",
    "font-size",
    "font-style",
    "font-weight",
    "text-anchor",
    "dominant-baseline",
    "baseline-shift",
)


@dataclass
class CssRules:
    tag: dict[str, dict[str, str]]
    klass: dict[str, dict[str, str]]


@dataclass
class TextSegment:
    text: str
    style: dict[str, str]
    dx: float = 0.0
    math_font: bool = False


@dataclass
class TextLine:
    x: float
    y: float
    anchor: str
    baseline: str
    segments: list[TextSegment]


Matrix = tuple[float, float, float, float, float, float]


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def parse_numbers(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(match.group(0)) for match in NUMBER_RE.finditer(value)]


def parse_length(value: str | None, default: float = 0.0) -> float:
    numbers = parse_numbers(value)
    return numbers[0] if numbers else default


def parse_declarations(value: str | None) -> dict[str, str]:
    declarations: dict[str, str] = {}
    if not value:
        return declarations
    for part in value.split(";"):
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        declarations[key.strip()] = val.strip()
    return declarations


def parse_css(root: ET.Element) -> CssRules:
    tag: dict[str, dict[str, str]] = {}
    klass: dict[str, dict[str, str]] = {}
    css_text = "\n".join(
        style.text or "" for style in root.iter() if local_name(style.tag) == "style"
    )
    css_text = re.sub(r"/\*.*?\*/", "", css_text, flags=re.S)
    for selector_text, body in CSS_RULE_RE.findall(css_text):
        declarations = parse_declarations(body)
        for selector in selector_text.split(","):
            selector = selector.strip()
            if not selector:
                continue
            if selector.startswith("."):
                class_name = selector[1:].split()[0].split(".")[0].split(":")[0]
                klass.setdefault(class_name, {}).update(declarations)
            elif re.fullmatch(r"[A-Za-z][\w-]*", selector):
                tag.setdefault(selector, {}).update(declarations)
    return CssRules(tag=tag, klass=klass)


def element_style(
    elem: ET.Element, parent: dict[str, str], rules: CssRules
) -> dict[str, str]:
    style = dict(parent)
    tag_name = local_name(elem.tag)
    style.update(rules.tag.get(tag_name, {}))
    for class_name in (elem.get("class") or "").split():
        style.update(rules.klass.get(class_name, {}))
    for key in STYLE_KEYS:
        if key in elem.attrib:
            style[key] = elem.attrib[key]
    style.update(parse_declarations(elem.get("style")))
    return style


def matrix_multiply(left: Matrix, right: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def parse_transform(value: str | None) -> Matrix:
    matrix: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    if not value:
        return matrix
    for name, arg_text in TRANSFORM_RE.findall(value):
        args = parse_numbers(arg_text)
        op: Matrix | None = None
        name = name.lower()
        if name == "translate":
            tx = args[0] if args else 0.0
            ty = args[1] if len(args) > 1 else 0.0
            op = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            op = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "matrix" and len(args) >= 6:
            op = (args[0], args[1], args[2], args[3], args[4], args[5])
        elif name == "rotate" and args:
            angle = math.radians(args[0])
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            rotation = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(args) >= 3:
                cx, cy = args[1], args[2]
                op = matrix_multiply(
                    matrix_multiply((1.0, 0.0, 0.0, 1.0, cx, cy), rotation),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                op = rotation
        if op is not None:
            matrix = matrix_multiply(matrix, op)
    return matrix


def apply_matrix(matrix: Matrix, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def flush_line(
    lines: list[TextLine],
    segments: list[TextSegment],
    matrix: Matrix,
    x: float,
    y: float,
    anchor: str,
    baseline: str,
) -> list[TextSegment]:
    if any(segment.text for segment in segments):
        gx, gy = apply_matrix(matrix, x, y)
        lines.append(
            TextLine(
                x=gx,
                y=gy,
                anchor=anchor,
                baseline=baseline,
                segments=list(segments),
            )
        )
    return []


def is_math_segment(elem: ET.Element, style: dict[str, str]) -> bool:
    classes = (elem.get("class") or "").split()
    family = style.get("font-family", "")
    return any("math" in class_name.lower() for class_name in classes) or "Fira Math" in family


def add_text_segment(
    segments: list[TextSegment],
    text: str | None,
    style: dict[str, str],
    dx: float = 0.0,
    math_font: bool = False,
) -> None:
    if text is None or text == "":
        return
    segments.append(TextSegment(text=text, style=style, dx=dx, math_font=math_font))


def collect_text_element(
    elem: ET.Element,
    matrix: Matrix,
    style: dict[str, str],
    rules: CssRules,
    lines: list[TextLine],
) -> None:
    anchor = elem.get("text-anchor") or style.get("text-anchor", "start")
    baseline = elem.get("dominant-baseline") or style.get("dominant-baseline", "baseline")
    line_x = parse_length(elem.get("x"))
    current_y = parse_length(elem.get("y"))
    segments: list[TextSegment] = []

    if elem.text and elem.text.strip():
        add_text_segment(segments, elem.text, style, math_font=is_math_segment(elem, style))

    for child in list(elem):
        if local_name(child.tag) != "tspan":
            continue

        child_style = element_style(child, style, rules)
        has_x = child.get("x") is not None
        has_dy = child.get("dy") is not None

        if has_x and has_dy and segments:
            segments = flush_line(lines, segments, matrix, line_x, current_y, anchor, baseline)

        if has_dy:
            current_y += parse_length(child.get("dy"))
        if has_x:
            line_x = parse_length(child.get("x"), line_x)

        dx = parse_length(child.get("dx"))
        add_text_segment(
            segments,
            child.text,
            child_style,
            dx=dx,
            math_font=is_math_segment(child, child_style),
        )
        if child.tail and child.tail.strip():
            add_text_segment(segments, child.tail, style)

    flush_line(lines, segments, matrix, line_x, current_y, anchor, baseline)


def collect_text_lines(root: ET.Element, rules: CssRules) -> list[TextLine]:
    lines: list[TextLine] = []
    default_style = {
        "fill": "#000000",
        "font-family": "Fira Sans",
        "font-size": "16px",
        "font-style": "normal",
        "font-weight": "400",
        "text-anchor": "start",
        "dominant-baseline": "baseline",
    }

    def walk(elem: ET.Element, matrix: Matrix, inherited: dict[str, str]) -> None:
        local_style = element_style(elem, inherited, rules)
        local_matrix = matrix_multiply(matrix, parse_transform(elem.get("transform")))
        if local_name(elem.tag) == "text":
            collect_text_element(elem, local_matrix, local_style, rules, lines)
            return
        for child in list(elem):
            walk(child, local_matrix, local_style)

    walk(root, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), default_style)
    return lines


def remove_text_elements(elem: ET.Element) -> None:
    for child in list(elem):
        if local_name(child.tag) == "text":
            elem.remove(child)
        else:
            remove_text_elements(child)


def viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    values = parse_numbers(root.get("viewBox"))
    if len(values) == 4:
        return values[0], values[1], values[2], values[3]
    width = parse_length(root.get("width"))
    height = parse_length(root.get("height"))
    return 0.0, 0.0, width / UNIT_TO_MM, height / UNIT_TO_MM


def font_size_pt(style: dict[str, str]) -> float:
    return parse_length(style.get("font-size"), 16.0) * UNIT_TO_PT


def is_bold(style: dict[str, str]) -> bool:
    weight = style.get("font-weight", "400").strip().lower()
    if weight in {"bold", "bolder"}:
        return True
    try:
        return int(float(weight)) >= 600
    except ValueError:
        return False


def normalize_hex_color(fill: str | None) -> str:
    if not fill:
        return "000000"
    fill = fill.strip()
    if fill.startswith("#"):
        fill = fill[1:]
    if len(fill) == 3:
        fill = "".join(part * 2 for part in fill)
    if re.fullmatch(r"[0-9A-Fa-f]{6}", fill):
        return fill.upper()
    named = {
        "black": "000000",
        "white": "FFFFFF",
    }
    return named.get(fill.lower(), "000000")


def escape_latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        " ": r"\space{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def segment_tex(segment: TextSegment, color_names: dict[str, str]) -> str:
    size = font_size_pt(segment.style)
    leading = size * 1.2
    color = color_names[normalize_hex_color(segment.style.get("fill"))]
    family = r"\svgmathfont" if segment.math_font else r"\svgtextfont"
    weight = r"\bfseries " if is_bold(segment.style) and not segment.math_font else ""
    text = escape_latex_text(segment.text)
    dx = segment.dx * UNIT_TO_PT
    dx_tex = rf"\hspace{{{dx:.3f}pt}}" if abs(dx) > 1e-6 else ""
    body = (
        rf"{{{family}\fontsize{{{size:.3f}pt}}{{{leading:.3f}pt}}\selectfont "
        rf"{weight}\textcolor{{{color}}}{{{text}}}}}"
    )
    if segment.style.get("baseline-shift", "").strip().lower() == "sub":
        body = rf"\raisebox{{-0.45ex}}{{{body}}}"
    return dx_tex + body


def node_anchor(line: TextLine) -> str:
    anchor = line.anchor.strip().lower()
    baseline = line.baseline.strip().lower()
    if baseline == "middle":
        return {
            "middle": "center",
            "end": "east",
            "right": "east",
        }.get(anchor, "west")
    return {
        "middle": "base",
        "end": "base east",
        "right": "base east",
    }.get(anchor, "base west")


def build_overlay_tex(
    lines: list[TextLine],
    background_pdf: Path,
    width_units: float,
    height_units: float,
) -> str:
    colors = sorted(
        {normalize_hex_color(segment.style.get("fill")) for line in lines for segment in line.segments}
    )
    color_names = {hex_color: f"svgcolor{index}" for index, hex_color in enumerate(colors)}
    color_defs = "\n".join(
        rf"\definecolor{{{name}}}{{HTML}}{{{hex_color}}}"
        for hex_color, name in color_names.items()
    )
    node_lines = []
    for line in lines:
        content = "".join(segment_tex(segment, color_names) for segment in line.segments)
        node_lines.append(
            rf"\node[inner sep=0pt,outer sep=0pt,anchor={node_anchor(line)}] "
            rf"at ({line.x:.3f},{line.y:.3f}) {{{content}}};"
        )

    width_mm = width_units * UNIT_TO_MM
    height_mm = height_units * UNIT_TO_MM
    return rf"""\documentclass[tikz,border=0pt]{{standalone}}
\usepackage{{fontspec}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\newfontfamily\svgtextfont[Ligatures=TeX]{{Fira Sans}}
\newfontfamily\svgmathfont{{Fira Math}}
{color_defs}
\pagestyle{{empty}}
\begin{{document}}
\begin{{tikzpicture}}[x={UNIT_TO_MM}mm,y=-{UNIT_TO_MM}mm]
\path[use as bounding box] (0,0) rectangle ({width_units:.3f},{height_units:.3f});
\node[inner sep=0pt,outer sep=0pt,anchor=north west] at (0,0)
  {{\includegraphics[width={width_mm:.3f}mm,height={height_mm:.3f}mm]{{{background_pdf.name}}}}};
{chr(10).join(node_lines)}
\end{{tikzpicture}}
\end{{document}}
"""


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_one(svg_path: Path, out_pdf: Path, keep_temp: bool = False) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    rules = parse_css(root)
    lines = collect_text_lines(root, rules)
    _, _, width_units, height_units = viewbox(root)

    with tempfile.TemporaryDirectory(prefix=f"{svg_path.stem}-fira-") as temp_name:
        temp_dir = Path(temp_name)
        textless_tree = ET.ElementTree(root)
        remove_text_elements(textless_tree.getroot())
        textless_svg = temp_dir / "textless.svg"
        background_pdf = temp_dir / "background.pdf"
        overlay_tex = temp_dir / "overlay.tex"
        textless_tree.write(textless_svg, encoding="utf-8", xml_declaration=True)

        env = os.environ.copy()
        cache_paths = {
            "XDG_CACHE_HOME": str(temp_dir / "xdg-cache"),
            # LuaTeX checks write permissions conservatively; a local cache
            # under the compilation directory is the most reliable option.
            "TEXMFCACHE": ".",
            "TEXMFVAR": ".",
        }
        for key, value in cache_paths.items():
            if value != ".":
                Path(value).mkdir(parents=True, exist_ok=True)
            env[key] = value

        run(["rsvg-convert", "-f", "pdf", "-o", str(background_pdf), str(textless_svg)], ROOT, env)
        overlay_tex.write_text(
            build_overlay_tex(lines, background_pdf, width_units, height_units),
            encoding="utf-8",
        )
        run(
            [
                "lualatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                overlay_tex.name,
            ],
            temp_dir,
            env,
        )
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_dir / "overlay.pdf", out_pdf)
        if keep_temp:
            kept = out_pdf.with_suffix(".fira-build")
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(temp_dir, kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate files beside each output PDF for inspection.",
    )
    args = parser.parse_args()

    for svg_name, pdf_name in SOURCES:
        svg_path = FIGURE_DIR / svg_name
        out_pdf = SVG_CACHE_DIR / pdf_name
        print(f"Building {out_pdf.relative_to(ROOT)}")
        build_one(svg_path, out_pdf, keep_temp=args.keep_temp)


if __name__ == "__main__":
    main()
