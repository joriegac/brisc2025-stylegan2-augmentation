"""
audit_synthetic_artifacts_v2.py
===============================
Synthetic shortcut/artifact audit.

This is not a radiologist-equivalence test. It asks a narrower, defensible
reviewer-facing question: can a CNN trained only to distinguish source
(real vs synthetic) detect synthetic images above chance when class and plane
are stratified? High AUROC/accuracy indicates visible or learnable synthetic
artifacts; near-chance performance is supportive evidence that shortcuts are
not obvious to this model, but does not prove clinical realism.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path


CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLANES = ["ax", "co", "sa"]
IMG_SIZE = 128


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _read_rows(manifest: Path):
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source") not in ("real", "synthetic"):
                continue
            if row.get("class") not in CLASSES or row.get("plane") not in PLANES:
                continue
            rows.append(dict(row))
    return rows


def _split_rows(rows, seed: int):
    from sklearn.model_selection import train_test_split

    y_source = [r["source"] for r in rows]
    strat_full = [f"{r['source']}|{r['class']}|{r['plane']}" for r in rows]
    try:
        train_rows, temp_rows = train_test_split(
            rows, test_size=0.30, random_state=seed, stratify=strat_full)
        temp_strat = [f"{r['source']}|{r['class']}|{r['plane']}" for r in temp_rows]
        val_rows, test_rows = train_test_split(
            temp_rows, test_size=0.50, random_state=seed, stratify=temp_strat)
    except ValueError:
        train_rows, temp_rows = train_test_split(
            rows, test_size=0.30, random_state=seed, stratify=y_source)
        temp_source = [r["source"] for r in temp_rows]
        val_rows, test_rows = train_test_split(
            temp_rows, test_size=0.50, random_state=seed, stratify=temp_source)
    return train_rows, val_rows, test_rows


class SourceDataset:
    def __init__(self, rows, project_root: Path):
        self.rows = rows
        self.project_root = project_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import numpy as np
        import torch
        from PIL import Image

        row = self.rows[idx]
        fp = _resolve_manifest_path(row["filepath"], self.project_root)
        with Image.open(fp) as im:
            im = im.convert("RGB")
            if im.size != (IMG_SIZE, IMG_SIZE):
                im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr.transpose(2, 0, 1))
        y = torch.tensor(1 if row["source"] == "synthetic" else 0, dtype=torch.long)
        return x, y


def _make_loader(rows, project_root: Path, batch_size: int, shuffle: bool,
                 workers: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        SourceDataset(rows, project_root),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def make_source_cnn():
    import torch.nn as nn

    def conv_block(in_f, out_c):
        return nn.Sequential(
            nn.Conv2d(in_f, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        conv_block(3, 32),
        conv_block(32, 64),
        conv_block(64, 128),
        conv_block(128, 256),
        conv_block(256, 512),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, 2),
    )


def _eval(model, loader, device):
    import numpy as np
    import torch

    model.eval()
    y_true, y_prob, y_pred = [], [], []
    loss_total, n_total = 0.0, 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            loss_total += loss.item() * xb.size(0)
            n_total += xb.size(0)
            y_true.append(yb.cpu().numpy())
            y_prob.append(probs.cpu().numpy())
            y_pred.append(logits.argmax(1).cpu().numpy())
    return {
        "loss": loss_total / max(n_total, 1),
        "y_true": np.concatenate(y_true),
        "y_prob": np.concatenate(y_prob),
        "y_pred": np.concatenate(y_pred),
    }


def _metrics(eval_out):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, roc_auc_score)

    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]
    y_prob = eval_out["y_prob"]
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = None
    return {
        "loss": float(eval_out["loss"]),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": auc,
        "confusion_real_synth": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _source_counts(rows):
    out = {}
    for row in rows:
        key = f"{row['source']}|{row['class']}|{row['plane']}"
        out[key] = out.get(key, 0) + 1
    return out


def _save_panel(rows, probs, out_path: Path, project_root: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.unicode_minus": False})
    from PIL import Image

    if not rows:
        return
    n = min(len(rows), 12)
    fig, axes = plt.subplots(2, 6, figsize=(12, 4), facecolor="white")
    axes = axes.ravel()
    fig.suptitle(title, fontsize=11, fontweight="bold")
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        row = rows[i]
        fp = _resolve_manifest_path(row["filepath"], project_root)
        with Image.open(fp) as im:
            axes[i].imshow(im.convert("L"), cmap="gray")
        axes[i].set_title(
            f"{row['source']} {row['class']} {row['plane']}\nP(synth)={probs[i]:.2f}",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_one_seed(args, rows, manifest_name: str, seed: int):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_rows, val_rows, test_rows = _split_rows(rows, seed)
    train_loader = _make_loader(train_rows, args.project_root, args.batch_size,
                                True, args.workers, seed)
    val_loader = _make_loader(val_rows, args.project_root, args.batch_size * 2,
                              False, args.workers, seed)
    test_loader = _make_loader(test_rows, args.project_root, args.batch_size * 2,
                               False, args.workers, seed)

    device = torch.device(args.device if torch.cuda.is_available()
                          and "cuda" in args.device else "cpu")
    model = make_source_cnn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    steps_per_epoch = max(1, math.ceil(len(train_rows) / args.batch_size))
    max_steps = args.max_steps or max(args.min_steps,
                                      steps_per_epoch * args.epochs)
    val_every = args.val_every_steps or max(1, steps_per_epoch * args.val_every_epochs)

    best_state, best_val, bad_evals = None, float("inf"), 0
    step, t0 = 0, time.time()
    while step < max_steps:
        model.train()
        for xb, yb in train_loader:
            if step >= max_steps:
                break
            step += 1
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            if step == 1 or step == max_steps or step % val_every == 0:
                val_eval = _eval(model, val_loader, device)
                if val_eval["loss"] < best_val - args.min_delta:
                    best_val, bad_evals = val_eval["loss"], 0
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad_evals += 1
                print(f"  seed={seed} step={step}/{max_steps} "
                      f"val_loss={val_eval['loss']:.4f}")
                if step >= args.min_steps and bad_evals >= args.patience_evals:
                    step = max_steps
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    test_eval = _eval(model, test_loader, device)
    metrics = _metrics(test_eval)
    metrics.update({
        "seed": seed,
        "manifest": manifest_name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "steps_run": max_steps,
        "elapsed_sec": time.time() - t0,
        "split_counts": {
            "train": _source_counts(train_rows),
            "val": _source_counts(val_rows),
            "test": _source_counts(test_rows),
        },
    })
    return metrics, test_rows, test_eval["y_prob"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--manifest", default="train_augmented_r2.csv",
                   help="Manifest under outputs_v2/manifests to audit.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="Training seeds for the source-discriminator audit.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--min-steps", type=int, default=400)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--val-every-epochs", type=int, default=2)
    p.add_argument("--val-every-steps", type=int, default=None)
    p.add_argument("--patience-evals", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-panels", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    import numpy as np

    args = parse_args(argv)
    args.project_root = args.project_root.resolve()
    out_dir = (args.out_dir or args.project_root / "outputs_v2").resolve()
    manifest_path = out_dir / "manifests" / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not args.seeds:
        raise SystemExit("--seeds must include at least one seed.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0.")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0.")
    if args.epochs <= 0:
        raise SystemExit("--epochs must be > 0.")
    if args.min_steps < 0:
        raise SystemExit("--min-steps must be >= 0.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be > 0.")
    if args.val_every_epochs <= 0:
        raise SystemExit("--val-every-epochs must be > 0.")
    if args.val_every_steps is not None and args.val_every_steps <= 0:
        raise SystemExit("--val-every-steps must be > 0.")
    if args.patience_evals <= 0:
        raise SystemExit("--patience-evals must be > 0.")

    rows = _read_rows(manifest_path)
    if not rows:
        raise SystemExit(f"Manifest has no auditable rows: {manifest_path}")
    if not any(r["source"] == "synthetic" for r in rows):
        raise SystemExit("Manifest has no synthetic rows; cannot audit source shortcuts.")

    audit_dir = out_dir / "audits" / "synthetic_artifacts_v2"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest_stem = Path(args.manifest).stem

    per_seed = []
    last_test_rows, last_probs = None, None
    for seed in args.seeds:
        metrics, test_rows, probs = run_one_seed(args, rows, args.manifest, seed)
        per_seed.append(metrics)
        last_test_rows, last_probs = test_rows, probs

    scalar_keys = ["accuracy", "balanced_accuracy", "macro_f1", "auroc"]
    summary = {
        "manifest": args.manifest,
        "interpretation": (
            "This is a source-discriminator artifact audit, not a radiologist "
            "equivalence test. Higher-than-chance performance means synthetic "
            "source cues are learnable by this CNN."
        ),
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
        "summary": {},
    }
    for key in scalar_keys:
        vals = [m[key] for m in per_seed if m[key] is not None]
        if vals:
            summary["summary"][key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

    json_path = audit_dir / f"{manifest_stem}_source_discriminator.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Synthetic Artifact / Shortcut Audit v2",
        "",
        f"Manifest: `{args.manifest}`",
        "",
        "Interpretation: this tests whether a CNN can distinguish real from synthetic images.",
        "It is not a radiologist Turing test and should not be described as one.",
        "",
    ]
    for key, val in summary["summary"].items():
        md.append(f"- {key}: {val['mean']:.4f} +/- {val['std']:.4f}")
    md.append("")
    md.append("Chance expectation is approximately 0.5 for accuracy, balanced accuracy, and AUROC.")
    (audit_dir / f"{manifest_stem}_source_discriminator.md").write_text(
        "\n".join(md), encoding="utf-8")

    if not args.no_panels and last_test_rows is not None:
        order_synth = np.argsort(-last_probs)
        order_real = np.argsort(last_probs)
        synth_rows = [last_test_rows[i] for i in order_synth[:12]]
        synth_probs = [float(last_probs[i]) for i in order_synth[:12]]
        real_rows = [last_test_rows[i] for i in order_real[:12]]
        real_probs = [float(last_probs[i]) for i in order_real[:12]]
        _save_panel(synth_rows, synth_probs,
                    audit_dir / f"{manifest_stem}_most_synthetic_like.png",
                    args.project_root, "Most synthetic-like held-out audit images")
        _save_panel(real_rows, real_probs,
                    audit_dir / f"{manifest_stem}_most_real_like.png",
                    args.project_root, "Most real-like held-out audit images")

    print(f"Wrote synthetic artifact audit to {audit_dir}")


if __name__ == "__main__":
    main()
