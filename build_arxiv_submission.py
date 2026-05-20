"""
Build an arXiv-ready submission tarball.

Output location: C:\\Users\\norie\\OneDrive\\arXiv\\submission\\

What this script does
---------------------
1. Creates a clean submission folder.
2. Copies the manuscript and rewrites it for arXiv:
     - ``\\includesvg{../figures/svg_embedded/X}``  ->  ``\\includegraphics{figures/svg_embedded/X}``
     - Drops the ``svg`` package (no longer needed) and ``inkscapelatex`` option.
     - Rewrites ``../figures/`` references to ``figures/`` (paths flattened).
3. For the three vector schematics (StyleGAN2-ADA, CNN, MobileViTV2): copies the
   pre-converted PDF from ``arXiv/svg-inkscape/`` under the un-suffixed name.
4. For the two raster mosaics: reads the source PNG from ``figures/``, downsamples
   it to ~1500-pixel width with Lanczos resampling, and saves as a compact PDF.
   This cuts each from ~85-100 MB down to ~3-5 MB without visible quality loss
   at print resolution.
5. Copies all ``.dat`` files referenced by the manuscript into
   ``figures/pgfplots/.../data/`` preserving subdirectory structure.
6. Copies the UbuntuMono font folder.
7. Bundles everything into ``noriega2026_arxiv.tar.gz``.

Run from the project root:
    python build_arxiv_submission.py
"""

from __future__ import annotations

import re
import shutil
import tarfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT  = Path(__file__).resolve().parent
SOURCE_TEX    = PROJECT_ROOT / "arXiv" / "noriega2026.tex"
SVG_CACHE     = PROJECT_ROOT / "arXiv" / "svg-inkscape"
FONTS_DIR     = PROJECT_ROOT / "arXiv" / "fonts"
SOURCE_PNG_DIR = PROJECT_ROOT / "figures"

BUILD_DIR = Path(r"C:\Users\norie\OneDrive\arXiv\submission")
TARBALL   = Path(r"C:\Users\norie\OneDrive\arXiv\noriega2026_arxiv.tar.gz")

# Three vector schematics — copy the pre-converted PDF from svg-inkscape/ as-is.
VECTOR_SVG_MAP = {
    "StyleGAN2-ADA": "StyleGAN2-ADA_svg-raw.pdf",
    "CNN":           "CNN_svg-raw.pdf",
    "MobileViTV2":   "MobileViTV2_svg-raw.pdf",
}

# Two raster mosaics — downsample the source PNG and embed as compact PDF.
RASTER_MOSAIC_MAP = {
    "brisc_raw_vs_preprocessed_mosaic": "brisc_raw_vs_preprocessed_mosaic.png",
    "brisc_synthetic_mosaic":           "brisc_synthetic_mosaic.png",
}

# Target longer-edge resolution for downsampled mosaics (pixels).
# 1800 px gives ~300 DPI for a 6-inch print width, ample for print and PDF viewers.
MOSAIC_MAX_DIM = 1800

# JPEG quality for the embedded image inside the PDF (PIL saves PDF as JPEG).
MOSAIC_JPEG_QUALITY = 92


def reset_build_dir() -> None:
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)


def copy_vector_pdfs() -> None:
    dest = BUILD_DIR / "figures" / "svg_embedded"
    dest.mkdir(parents=True, exist_ok=True)
    for stem, raw_pdf in VECTOR_SVG_MAP.items():
        src = SVG_CACHE / raw_pdf
        if not src.exists():
            raise FileNotFoundError(
                f"Pre-converted SVG missing: {src}. "
                f"Run a local LuaLaTeX build with -shell-escape first."
            )
        shutil.copy2(src, dest / f"{stem}.pdf")
        size_mb = (dest / f"{stem}.pdf").stat().st_size / (1024 * 1024)
        print(f"  {stem}.pdf  ({size_mb:.2f} MB, vector)")


def downsample_mosaics() -> None:
    dest = BUILD_DIR / "figures" / "svg_embedded"
    dest.mkdir(parents=True, exist_ok=True)
    for stem, png_name in RASTER_MOSAIC_MAP.items():
        src = SOURCE_PNG_DIR / png_name
        if not src.exists():
            raise FileNotFoundError(f"Source mosaic PNG missing: {src}")

        with Image.open(src) as im:
            w0, h0 = im.size
            scale = MOSAIC_MAX_DIM / max(w0, h0)
            if scale < 1.0:
                new_size = (round(w0 * scale), round(h0 * scale))
                im = im.resize(new_size, Image.Resampling.LANCZOS)
            else:
                new_size = (w0, h0)

            # PIL's PDF writer keeps the image as a single embedded raster.
            # Convert to RGB for JPEG-compressed embedding inside the PDF.
            if im.mode != "RGB":
                im = im.convert("RGB")

            out = dest / f"{stem}.pdf"
            im.save(
                out,
                "PDF",
                resolution=300.0,
                quality=MOSAIC_JPEG_QUALITY,
            )

        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  {stem}.pdf  ({size_mb:.2f} MB, raster {new_size[0]}x{new_size[1]})")


def copy_dat_files(tex_text: str) -> None:
    pattern = re.compile(r"\.\./figures/([A-Za-z0-9_./]+\.dat)")
    seen: set[str] = set()
    for match in pattern.finditer(tex_text):
        rel = match.group(1)
        if rel in seen:
            continue
        seen.add(rel)
        src = PROJECT_ROOT / "figures" / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing data file: {src}")
        dst = BUILD_DIR / "figures" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"  copied {len(seen)} .dat files")


def copy_fonts() -> None:
    dest = BUILD_DIR / "fonts"
    shutil.copytree(FONTS_DIR, dest)


def rewrite_tex(tex_text: str) -> str:
    tex_text = re.sub(
        r"\\usepackage\[inkscapelatex=false\]\{svg\}\s*\n",
        "% svg package removed for arXiv submission (figures are pre-built PDFs)\n",
        tex_text,
    )

    def rewrite_includesvg(match: re.Match[str]) -> str:
        opts = match.group(1) or ""
        path = match.group(2)
        new_path = path.replace("../figures/", "figures/", 1)
        return rf"\includegraphics{opts}{{{new_path}}}"

    tex_text = re.sub(
        r"\\includesvg(\[[^\]]*\])?\{([^}]+)\}",
        rewrite_includesvg,
        tex_text,
    )

    # PGFPlots data table paths
    tex_text = tex_text.replace("../figures/pgfplots/", "figures/pgfplots/")

    return tex_text


def write_tex() -> None:
    tex_text = SOURCE_TEX.read_text(encoding="utf-8")
    copy_dat_files(tex_text)
    new_text = rewrite_tex(tex_text)
    (BUILD_DIR / "noriega2026.tex").write_text(new_text, encoding="utf-8")


def make_tarball() -> None:
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tar:
        for path in sorted(BUILD_DIR.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(BUILD_DIR)
                tar.add(path, arcname=str(arcname))
    size_mb = TARBALL.stat().st_size / (1024 * 1024)
    print(f"  tarball: {TARBALL.name}  ({size_mb:.2f} MB)")


def main() -> None:
    print(f"Submission tree:  {BUILD_DIR}")
    print(f"Tarball output:   {TARBALL}")
    print()

    print("Resetting build directory...")
    reset_build_dir()

    print("Copying vector schematic PDFs...")
    copy_vector_pdfs()

    print("Downsampling and rebuilding raster mosaic PDFs...")
    downsample_mosaics()

    print("Copying fonts...")
    copy_fonts()

    print("Rewriting .tex and copying data files...")
    write_tex()

    print("Building tarball...")
    make_tarball()

    print()
    print("Local compile test (no shell-escape required):")
    print(f"  cd {BUILD_DIR}")
    print( "  lualatex -interaction=nonstopmode noriega2026.tex")
    print( "  lualatex -interaction=nonstopmode noriega2026.tex   # second pass for refs")


if __name__ == "__main__":
    main()
