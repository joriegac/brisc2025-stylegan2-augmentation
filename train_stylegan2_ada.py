"""
train_stylegan2_ada.py
======================
Script version of the original notebook implementation through model training.

This covers the notebook path from environment/import setup through:
  - StyleGAN3 repository/dependency checks
  - notebook-faithful BRISC preprocessing for GAN training
  - StyleGAN-compatible ZIP creation
  - CUDA plugin verification
  - StyleGAN2-ADA training calls

Synthetic image generation, real-vs-synthetic comparison, and checkpoint
selection are intentionally left to the dedicated project scripts.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
import zipfile
from datetime import datetime, timedelta
from pathlib import Path


warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", r"C:\tmp\torch_ext")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLANES = ["ax", "co", "sa"]
ALL_COMBOS = [f"{cls}_{plane}" for cls in CLASSES for plane in PLANES]


def _run(cmd, cwd: Path, check: bool = True):
    print(f"  > {' '.join(str(c) for c in cmd)}", flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd))
    if check and rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(str(c) for c in cmd)}")
    return rc


def _print_header(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _verify_system():
    _print_header("SYSTEM VERIFICATION")
    try:
        import torch
        import torchvision
        import PIL
        import scipy
        import skimage
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")

    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  torch       : {torch.__version__}  CUDA={torch.cuda.is_available()}")
    print(f"  torchvision : {torchvision.__version__}")
    print(f"  Pillow      : {PIL.__version__}")
    print(f"  scipy       : {scipy.__version__}")
    print(f"  scikit-image: {skimage.__version__}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {props.total_memory / (1024 ** 3):.1f} GB")
    else:
        print("  WARNING     : CUDA is not available; StyleGAN training will not be practical.")


def _ensure_stylegan3(args):
    _print_header("STYLEGAN3 CODEBASE CHECK")
    stylegan_dir = args.stylegan3_dir
    repo = "https://github.com/NVlabs/stylegan3.git"

    if stylegan_dir.exists() and (stylegan_dir / "train.py").exists():
        print(f"  Found StyleGAN3 at {stylegan_dir}")
    elif args.clone_stylegan3:
        if stylegan_dir.exists():
            shutil.rmtree(stylegan_dir)
        _run(["git", "clone", repo, stylegan_dir], cwd=args.project_root)
    else:
        raise SystemExit(
            f"StyleGAN3 train.py not found at {stylegan_dir}. "
            f"Pass --clone-stylegan3 to clone it."
        )

    if args.install_deps:
        deps = [
            "click", "requests", "tqdm", "pyspng", "ninja",
            "imageio-ffmpeg==0.4.3", "pillow", "numpy", "matplotlib",
            "scipy", "scikit-image", "psutil", "munch",
        ]
        _print_header("INSTALLING STYLEGAN DEPENDENCIES")
        for dep in deps:
            _run([sys.executable, "-m", "pip", "install", dep], cwd=args.project_root)

    stylegan_path = str(stylegan_dir.resolve())
    if stylegan_path not in sys.path:
        sys.path.insert(0, stylegan_path)

    required = [
        "train.py",
        "dataset_tool.py",
        "training/training_loop.py",
        "training/networks_stylegan2.py",
        "torch_utils/ops/conv2d_gradfix.py",
    ]
    missing = [rel for rel in required if not (stylegan_dir / rel).exists()]
    if missing:
        raise SystemExit(f"StyleGAN3 checkout is incomplete; missing: {missing}")
    print("  StyleGAN3 repository is ready.")


def _verify_cuda_plugins(args):
    _print_header("CUDA PLUGIN VERIFICATION")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch not importable: {exc}")

    if not torch.cuda.is_available():
        print("  CUDA unavailable; skipping plugin verification.")
        return

    stylegan_path = str(args.stylegan3_dir.resolve())
    if stylegan_path not in sys.path:
        sys.path.insert(0, stylegan_path)

    checks = {}
    try:
        from torch_utils.ops import bias_act
        x = torch.randn(1, 64, 8, 8, device="cuda")
        b = torch.randn(64, device="cuda")
        _ = bias_act.bias_act(x, b, act="lrelu", impl="cuda")
        checks["bias_act"] = True
    except Exception as exc:
        checks["bias_act"] = str(exc)

    try:
        from torch_utils.ops import upfirdn2d
        x = torch.randn(1, 64, 8, 8, device="cuda")
        f = torch.ones(4, device="cuda") / 4
        _ = upfirdn2d.upfirdn2d(x, f, up=2, impl="cuda")
        checks["upfirdn2d"] = True
    except Exception as exc:
        checks["upfirdn2d"] = str(exc)

    try:
        from torch_utils.ops import filtered_lrelu
        x = torch.randn(1, 64, 8, 8, device="cuda")
        _ = filtered_lrelu.filtered_lrelu(x, impl="cuda")
        checks["filtered_lrelu"] = True
    except Exception as exc:
        checks["filtered_lrelu"] = str(exc)

    for name, status in checks.items():
        print(f"  {name:<18}: {'OK' if status is True else 'FALLBACK'}")
        if status is not True:
            print(f"    {status}")

    torch.cuda.empty_cache()


def _stage_preprocess(args):
    _print_header("PREPROCESS TRAINING IMAGES")
    script = args.project_root / "preprocess_v2.py"
    cmd = [
        sys.executable, script,
        "--dataset-type", "train",
        "--project-root", args.project_root,
        "--input-dir", args.train_dir,
        "--output-dir", args.preprocessed_dir,
        "--layout", "class-plane",
        "--workers", str(args.workers),
        "--image-size", str(args.resolution),
        "--comparison-dir", args.project_root / "figures" / "preprocessing_train",
    ]
    if args.force_preprocess:
        cmd.append("--force")
    _run(cmd, cwd=args.project_root)


def _create_stylegan_zip(src_folder: Path, dst_zip: Path, resolution: int):
    from PIL import Image

    png_files = sorted(src_folder.glob("*.png"))
    if not png_files:
        return 0
    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    if dst_zip.exists():
        dst_zip.unlink()

    count = 0
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_STORED) as zf:
        for png_path in png_files:
            try:
                with Image.open(png_path) as img:
                    if img.size != (resolution, resolution):
                        img = img.resize((resolution, resolution), Image.LANCZOS)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG", compress_level=0)
                zf.writestr(f"{count:05d}.png", buf.getvalue())
                count += 1
            except Exception as exc:
                print(f"    [WARN] {png_path.name}: {exc}")
    return count


def _stage_zip(args):
    _print_header("PREPARE STYLEGAN2-ADA ZIP DATASETS")
    args.stylegan_data.mkdir(parents=True, exist_ok=True)
    stats = {}
    total = 0
    for combo in ALL_COMBOS:
        src = args.preprocessed_dir / combo
        dst = args.stylegan_data / f"{combo}.zip"
        src_count = len(list(src.glob("*.png"))) if src.exists() else 0
        if not src.exists() or src_count == 0:
            print(f"  {combo:<16}: SKIP, no preprocessed images")
            stats[combo] = {"status": "missing", "count": 0}
            continue
        if dst.exists() and not args.force_zip:
            with zipfile.ZipFile(dst, "r") as zf:
                zip_count = sum(1 for name in zf.namelist() if name.endswith(".png"))
            print(f"  {combo:<16}: cached {zip_count} images")
            stats[combo] = {"status": "cached", "count": zip_count}
            total += zip_count
            continue
        processed = _create_stylegan_zip(src, dst, args.resolution)
        size_mb = dst.stat().st_size / (1024 * 1024) if dst.exists() else 0.0
        status = "OK" if processed == src_count else "WARN"
        print(f"  {combo:<16}: {processed}/{src_count} images, {size_mb:.1f} MB [{status}]")
        stats[combo] = {"status": status.lower(), "count": processed, "size_mb": size_mb}
        total += processed

    out = args.stylegan_data / "dataset_zip_audit.json"
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  Total images in ZIPs: {total}")
    print(f"  Wrote {out}")


def _flush_gpu():
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


def _has_existing_snapshot(out_root: Path, combo: str):
    combo_root = out_root / combo
    return combo_root.exists() and any(combo_root.glob("**/network-snapshot-*.pkl"))


def _train_combo(args, combo: str):
    data_zip = args.stylegan_data / f"{combo}.zip"
    if not data_zip.exists():
        raise FileNotFoundError(f"Missing dataset ZIP for {combo}: {data_zip}")
    if args.skip_trained and _has_existing_snapshot(args.stylegan_out, combo):
        print(f"  [{combo}] existing snapshot found; skipping.")
        return {"combo": combo, "status": "skipped_existing"}

    _flush_gpu()
    run_root = args.stylegan_out / combo
    run_root.mkdir(parents=True, exist_ok=True)
    train_script = args.stylegan3_dir / "train.py"
    cmd = [
        sys.executable, train_script,
        f"--outdir={run_root}",
        f"--data={data_zip}",
        "--cfg=stylegan2",
        "--gpus=1",
        f"--batch={args.batch}",
        f"--batch-gpu={args.batch_gpu}",
        f"--gamma={args.gamma}",
        f"--kimg={args.kimg}",
        f"--snap={args.snap}",
        "--aug=ada",
        f"--augpipe={args.augpipe}",
        f"--target={args.ada_target}",
        "--mirror=0",
        "--metrics=none",
        f"--workers={args.training_workers}",
        "--nobench=1",
    ]
    if args.resume:
        cmd.append(f"--resume={args.resume}")

    _print_header(f"STYLEGAN2-ADA TRAINING: {combo}")
    print(f"  Data     : {data_zip}")
    print(f"  Output   : {run_root}")
    print(f"  kimg     : {args.kimg}")
    print(f"  batch    : {args.batch} total / {args.batch_gpu} per GPU")
    print(f"  augpipe  : {args.augpipe}")

    if args.dry_run:
        print(f"  DRY RUN  : {' '.join(str(c) for c in cmd)}")
        return {"combo": combo, "status": "dry_run", "cmd": [str(c) for c in cmd]}

    start = time.time()
    rc = _run(cmd, cwd=args.project_root, check=False)
    elapsed = str(timedelta(seconds=int(time.time() - start)))
    status = "success" if rc == 0 else "failed"
    ckpts = sorted(run_root.glob("**/network-snapshot-*.pkl"))
    result = {
        "combo": combo,
        "status": status,
        "return_code": rc,
        "elapsed": elapsed,
        "final_checkpoint": str(ckpts[-1]) if ckpts else None,
    }
    history_path = run_root / "training_history_from_script.json"
    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"Training failed for {combo} with exit code {rc}")
    return result


def _stage_train(args):
    _print_header("TRAIN STYLEGAN2-ADA MODELS")
    combos = ALL_COMBOS if args.combos == ["all"] else args.combos
    unknown = [combo for combo in combos if combo not in ALL_COMBOS]
    if unknown:
        raise SystemExit(f"Unknown combo(s): {unknown}")

    results = []
    for combo in combos:
        results.append(_train_combo(args, combo))
    out = args.stylegan_out / "training_manifest_from_script.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Wrote {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", choices=("setup", "preprocess", "zip", "train", "all"),
                   default="all")
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--stylegan3-dir", type=Path, default=None)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--preprocessed-dir", type=Path, default=None)
    p.add_argument("--stylegan-data", type=Path, default=None)
    p.add_argument("--stylegan-out", type=Path, default=None)
    p.add_argument("--combos", nargs="+", default=["all"],
                   help="One or more class-plane combos, or 'all'.")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--training-workers", type=int, default=3)
    p.add_argument("--kimg", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--batch-gpu", type=int, default=4)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--snap", type=int, default=10)
    p.add_argument("--augpipe", default="bgcfnc")
    p.add_argument("--ada-target", type=float, default=0.6)
    p.add_argument("--resume", default=None)
    p.add_argument("--clone-stylegan3", action="store_true")
    p.add_argument("--install-deps", action="store_true")
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--force-zip", action="store_true")
    p.add_argument("--skip-trained", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _resolve_args(args):
    args.project_root = args.project_root.resolve()
    args.stylegan3_dir = (args.stylegan3_dir or args.project_root / "stylegan3").resolve()
    args.train_dir = (args.train_dir or args.project_root / "brisc2025" / "classification_task" / "train").resolve()
    args.preprocessed_dir = (args.preprocessed_dir or args.project_root / "brisc2025_preprocessed").resolve()
    args.stylegan_data = (args.stylegan_data or args.project_root / "stylegan3_data").resolve()
    args.stylegan_out = (args.stylegan_out or args.project_root / "stylegan3_results").resolve()
    if args.workers <= 0:
        raise SystemExit("--workers must be > 0.")
    if args.training_workers < 0:
        raise SystemExit("--training-workers must be >= 0.")
    if args.kimg <= 0 or args.batch <= 0 or args.batch_gpu <= 0:
        raise SystemExit("--kimg, --batch, and --batch-gpu must be > 0.")
    return args


def main(argv=None):
    args = _resolve_args(parse_args(argv))
    _print_header("STYLEGAN2-ADA TRAINING SCRIPT")
    print(f"  Project root : {args.project_root}")
    print(f"  Stage        : {args.stage}")
    print(f"  Combos       : {args.combos}")
    print(f"  Date         : {datetime.now().isoformat(timespec='seconds')}")

    if args.stage in ("setup", "all"):
        _verify_system()
        _ensure_stylegan3(args)
        _verify_cuda_plugins(args)
    else:
        _ensure_stylegan3(args)

    if args.stage in ("preprocess", "all"):
        _stage_preprocess(args)
    if args.stage in ("zip", "all"):
        _stage_zip(args)
    if args.stage in ("train", "all"):
        _stage_train(args)


if __name__ == "__main__":
    main()
