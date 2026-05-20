"""
png_to_svg.py — Trace a black-on-white PNG to SVG using vtracer.

Usage:
    python png_to_svg.py input.png output.svg
    python png_to_svg.py input.png output.svg --speckle 4 --corner 60
"""

import argparse
import sys
from pathlib import Path

import vtracer


def png_to_svg(
    input_path: Path,
    output_path: Path,
    filter_speckle: int = 4,
    corner_threshold: int = 60,
    path_precision: int = 3,
):
    vtracer.convert_image_to_svg_py(
        str(input_path),
        str(output_path),
        colormode="binary",          # black-and-white line art
        hierarchical="stacked",
        mode="spline",               # smooth curves
        filter_speckle=filter_speckle,
        corner_threshold=corner_threshold,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=path_precision,
    )
    size = output_path.stat().st_size
    print(f"Wrote {output_path} ({size:,} bytes)")


def main():
    p = argparse.ArgumentParser(description="Trace a black-on-white PNG to SVG.")
    p.add_argument("input", type=Path, help="Input PNG path")
    p.add_argument("output", type=Path, help="Output SVG path")
    p.add_argument("--speckle", type=int, default=4,
                   help="Suppress speckles smaller than N pixels (default 4). "
                        "Lower if tick marks vanish; raise if noise appears.")
    p.add_argument("--corner", type=int, default=60,
                   help="Corner threshold 0-180 (default 60). "
                        "Lower preserves more sharp corners.")
    p.add_argument("--precision", type=int, default=3,
                   help="Coordinate decimal precision (default 3).")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: {args.input} not found")

    png_to_svg(
        args.input,
        args.output,
        filter_speckle=args.speckle,
        corner_threshold=args.corner,
        path_precision=args.precision,
    )


if __name__ == "__main__":
    main()