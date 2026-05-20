"""
classify_cnn_v2.py
==================
v2 of classify_cnn.py — adds native 1:1 and 1:2 augmentation experiments
and corrects a class-weight asymmetry between baseline and augmented runs.

EXPERIMENTAL MATRIX
-------------------
For each ratio R in --ratios (default 1.0 2.0):

    baseline             real-only, run once
    augmented_r{R}       real + R*real_n synthetic (no filter)
    filtered_r{R}        real + R*real_n synthetic from filter_synthetic_v2
                          (Mahalanobis cutoff + farthest-point sampling)

Same 10 seeds across all five experiments per classifier.

CHANGES vs classify_cnn.py
--------------------------
  * --ratios <R> [<R> ...]
        List, default [1.0 2.0]. Drives manifest, result and figure
        naming. Sources manifests from classify_v2.py's outputs_v2/.

  * Class weights derived from real-only training rows
        v1 computed weights from `yt_train[idx_tr]`, which in the
        augmented case includes ALL synthetic rows. Synthetic balance is
        roughly uniform across classes, so the weighted CE loss in
        augmented runs becomes near-uniform — different from baseline,
        which sees the natural imbalance via real-only weights. v2
        always derives weights from `yt_train[idx_tr][is_real_in_idx_tr]`,
        so both experiments train against the same loss landscape and
        the comparison is apples-to-apples.

  * Sequential dataset loading
        For each ratio R, the augmented and filtered image tensors are
        loaded right before the experiment that needs them and released
        immediately after — avoids holding 6 image tensors in RAM
        simultaneously when both ratios are in play.

  * Output dir
        Default <project-root>/outputs_v2 (was outputs/), with
        results_cnn_v2/ and figures_cnn_v2/ inside.

UNCHANGED
---------
  Architecture (5-block conv backbone + tumour/plane heads),
  optimizer, scheduler, AMP wiring, real-only validation split,
  reproducibility seeding. Preprocessing is byte-identical to v1.

USAGE
-----
    # Pre-requisite: run classify_v2.py at least through --stage filter so
    # the augmented + filtered manifests exist under outputs_v2/manifests/.

    python classify_cnn_v2.py
    python classify_cnn_v2.py --stage baseline
    python classify_cnn_v2.py --stage augmented   # all ratios
    python classify_cnn_v2.py --stage filtered    # all ratios
    python classify_cnn_v2.py --stage stats
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*Parameter `min_size` is deprecated.*",
    category=FutureWarning,
)


# =============================================================================
# CONSTANTS
# =============================================================================
CLASSES        = ["glioma", "meningioma", "no_tumor", "pituitary"]
PLANES         = ["ax", "co", "sa"]
COMBOS         = [f"{c}_{p}" for c in CLASSES for p in PLANES]
CLASS_TO_IDX   = {c: i for i, c in enumerate(CLASSES)}
TUMOUR_TO_IDX  = {c: i for i, c in enumerate(CLASSES)}
PLANE_TO_IDX   = {p: i for i, p in enumerate(PLANES)}
SEEDS          = list(range(10))
IMG_SIZE       = 128
DEFAULT_RATIOS = [1.0, 2.0]

DEFAULT_EPOCHS          = 200
DEFAULT_BATCH_SIZE      = 256
DEFAULT_LR              = 0.001
DEFAULT_WEIGHT_DECAY    = 1e-3
DEFAULT_OPTIMIZER       = "adam"        # "adam" | "adamw"
DEFAULT_SCHEDULER       = "plateau"     # "plateau" | "cosine"
DEFAULT_WARMUP_EPOCHS   = 0.0           # 0 = disabled
DEFAULT_LABEL_SMOOTHING = 0.1
DEFAULT_MIXUP_ALPHA     = 0.0           # 0 = disabled

# Output path hooks — override these before calling main() to redirect outputs.
SCRIPT_NAME      = "classify_cnn_v2.py"
MODEL_LABEL      = "CNN v2"
RESULTS_SUBDIR   = "results_cnn_v2"
FIGURES_SUBDIR   = "figures_cnn_v2"
FIGURE_BASENAME  = "classifier_comparison_cnn_v2"

# Normalization hook — set to ImageNet constants for pretrained backbones.
# None means compute from real training images (original behaviour).
NORM_MEAN_OVERRIDE = None   # e.g. (0.485, 0.456, 0.406)
NORM_STD_OVERRIDE  = None   # e.g. (0.229, 0.224, 0.225)


def _ratio_tag(R: float) -> str:
    return f"r{R:g}"


# =============================================================================
# PREPROCESSING — verbatim from classify_cnn.py (DO NOT MODIFY)
# =============================================================================
def _import_preprocessing_libs():
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from scipy.ndimage import label, binary_closing
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import flood_fill as sk_flood_fill
    return (np, Image, ndimage, label, binary_closing,
            threshold_otsu, remove_small_objects, sk_flood_fill)


def _flood_fill_background(closed_binary, np, sk_flood_fill):
    padded = np.pad(closed_binary, 2, mode="constant", constant_values=0)
    bg_map = (padded == 0).astype(np.uint8)
    filled = sk_flood_fill(bg_map, (0, 0), 2)
    return (filled == 2)[2:-2, 2:-2]


def _try_mask(img_array, thresh_fraction, close_iters, libs):
    np, _, ndimage, label, binary_closing, threshold_otsu, \
        remove_small_objects, sk_flood_fill = libs
    h, w = img_array.shape
    try:
        thresh = threshold_otsu(img_array) * thresh_fraction
    except Exception:
        thresh = 20.0
    binary = (img_array > thresh).astype(np.uint8)
    struct = ndimage.generate_binary_structure(2, 1)
    closed = binary_closing(
        binary, structure=ndimage.iterate_structure(struct, close_iters)
    ).astype(np.uint8)
    external_bg = _flood_fill_background(closed, np, sk_flood_fill)
    brain_mask  = (~external_bg).astype(np.uint8)
    lbl, n = label(brain_mask)
    if n == 0:
        return None, 0.0
    sizes      = ndimage.sum(brain_mask, lbl, range(1, n + 1))
    brain_mask = (lbl == int(np.argmax(sizes) + 1)).astype(np.uint8)
    brain_mask = remove_small_objects(
        brain_mask.astype(bool), min_size=500
    ).astype(np.uint8)
    brain_mask = binary_closing(
        brain_mask, structure=ndimage.iterate_structure(struct, 3)
    ).astype(np.uint8)
    return brain_mask, float(brain_mask.mean())


def _compute_skull_strip_mask(img_array, libs):
    np = libs[0]
    h, w = img_array.shape
    attempts = [(0.50, 5), (0.75, 5), (1.00, 7), (1.25, 7), (1.50, 5)]
    best_mask, best_dist = None, 1.0
    for thresh_frac, close_iters in attempts:
        mask, coverage = _try_mask(img_array, thresh_frac, close_iters, libs)
        if mask is None:
            continue
        if 0.15 <= coverage <= 0.85:
            return mask
        dist = abs(coverage - 0.50)
        if dist < best_dist:
            best_mask, best_dist = mask, dist
    if best_mask is None:
        warnings.warn("Skull stripping failed. Using full mask.", RuntimeWarning)
        return np.ones((h, w), dtype=np.uint8)
    return best_mask


def _normalize_intensity_with_mask(img_array, mask, np):
    brain_pixels = img_array[mask > 0]
    if len(brain_pixels) == 0:
        rng = img_array.max() - img_array.min()
        return ((img_array - img_array.min()) / (rng + 1e-8) * 255).astype(np.uint8)
    p1, p99    = np.percentile(brain_pixels, 1), np.percentile(brain_pixels, 99)
    normalized = np.clip(img_array, p1, p99)
    normalized = ((normalized - p1) / (p99 - p1 + 1e-8) * 255).astype(np.uint8)
    return normalized


def _preprocess_pil(pil_img):
    libs = _import_preprocessing_libs()
    np, Image, *_ = libs
    gray       = np.array(pil_img.convert("L"), dtype=np.float32)
    mask       = _compute_skull_strip_mask(gray, libs)
    normalized = _normalize_intensity_with_mask(gray, mask, np)
    normalized = normalized * mask
    normalized[normalized < 11] = 0
    rgb = np.stack([normalized, normalized, normalized], axis=-1)
    return rgb.astype(np.uint8)


def _resolve_manifest_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = project_root / p
    return p


def _load_image(path_str: str, project_root: Path, needs_preprocessing: bool):
    from PIL import Image
    import numpy as np
    fp  = _resolve_manifest_path(path_str, project_root)
    pil = Image.open(fp).convert("RGB")
    if needs_preprocessing:
        arr = _preprocess_pil(pil)
        pil = Image.fromarray(arr)
    if pil.size != (IMG_SIZE, IMG_SIZE):
        pil = pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(pil, dtype=np.uint8)


# =============================================================================
# MODEL — verbatim from classify_cnn.py
# =============================================================================
def make_model(n_tumour: int = 4, n_plane: int = 3):
    import torch
    import torch.nn as nn

    class TwoHeadCNN(nn.Module):
        def __init__(self, n_tumour, n_plane):
            super().__init__()
            def conv_block(in_f, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_f, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )

            self.features = nn.Sequential(
                conv_block(3, 32),
                conv_block(32, 64),
                conv_block(64, 128),
                conv_block(128, 256),
                conv_block(256, 512)
            )
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.shared = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.ReLU(inplace=True),
            )
            self.head_tumour = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, n_tumour)
            )
            self.head_plane = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(1024, n_plane)
            )
            self.log_vars = torch.nn.Parameter(torch.zeros(2))

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = self.shared(x)
            return self.head_tumour(x), self.head_plane(x)

    return TwoHeadCNN(n_tumour, n_plane)


# =============================================================================
# DATASET LOADING
# =============================================================================
def _bulk_load(paths, tumour_labels, plane_labels, is_real, project_root: Path,
               n_workers: int, log_label: str, needs_preprocessing: bool):
    import numpy as np
    from multiprocessing import get_context

    print(f"  [{log_label}] loading {len(paths)} images...", flush=True)
    t0 = time.time()
    if n_workers <= 1:
        arrs = [_load_image(p, project_root, needs_preprocessing) for p in paths]
    else:
        ctx           = get_context("spawn")
        proj_root_str = str(project_root)
        with ctx.Pool(processes=n_workers) as pool:
            arrs = pool.map(_bulk_worker,
                            [(p, proj_root_str, needs_preprocessing) for p in paths])

    X  = np.stack(arrs, axis=0)
    yt = np.asarray(tumour_labels, dtype=np.int64)
    yp = np.asarray(plane_labels,  dtype=np.int64)
    ir = np.asarray(is_real,       dtype=bool)
    print(f"    [{log_label}] done — X={X.shape} yt={yt.shape} yp={yp.shape} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return X, yt, yp, ir


def _bulk_worker(args_tuple):
    path, root_str, needs_preprocessing = args_tuple
    return _load_image(path, Path(root_str), needs_preprocessing)


def _read_manifest(csv_path: Path):
    paths, tumour_labels, plane_labels, is_real = [], [], [], []
    skipped = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            cls   = row.get("class", "").strip()
            plane = row.get("plane", "").strip()
            if cls not in TUMOUR_TO_IDX or plane not in PLANE_TO_IDX:
                skipped += 1
                continue
            paths.append(row["filepath"])
            tumour_labels.append(TUMOUR_TO_IDX[cls])
            plane_labels.append(PLANE_TO_IDX[plane])
            is_real.append(row.get("source", "real") == "real")
    if skipped:
        warnings.warn(
            f"_read_manifest: skipped {skipped} unrecognised row(s) in {csv_path.name}",
            RuntimeWarning,
        )
    return paths, tumour_labels, plane_labels, is_real


# =============================================================================
# TRAINING / EVAL HELPERS
# =============================================================================
def _setup_windows_env():
    if os.name != 'nt':
        return
    ext_dir = os.path.join(tempfile.gettempdir(), 'torch_ext')
    os.environ['TORCH_EXTENSIONS_DIR'] = ext_dir
    try:
        os.makedirs(ext_dir, exist_ok=True)
    except OSError:
        pass
    os.environ['PYTHONWARNINGS'] = 'ignore'


def _make_autocast(device_type: str, use_amp: bool):
    import contextlib
    import torch
    if not use_amp:
        return contextlib.nullcontext()
    return torch.autocast(device_type, dtype=torch.float16, enabled=True)


def _make_grad_scaler(use_amp: bool):
    import torch
    if not use_amp:
        class _NullScaler:
            def scale(self, loss): return loss
            def step(self, opt): opt.step()
            def update(self): pass
        return _NullScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _evaluate(model, loader, device, norm_mean, norm_std, use_amp):
    import numpy as np
    import torch
    model.eval()
    yt_true, yt_pred, yp_true, yp_pred = [], [], [], []
    with torch.no_grad():
        for xb, ybt, ybp in loader:
            xb = xb.to(device)
            xb = (xb - norm_mean) / norm_std
            with _make_autocast(device.type, use_amp):
                logits_t, logits_p = model(xb)
            yt_true.append(ybt.numpy())
            yt_pred.append(logits_t.argmax(1).cpu().numpy())
            yp_true.append(ybp.numpy())
            yp_pred.append(logits_p.argmax(1).cpu().numpy())
    return (np.concatenate(yt_true), np.concatenate(yt_pred),
            np.concatenate(yp_true), np.concatenate(yp_pred))


def _np_to_torch_chw(X_np):
    import numpy as np
    import torch
    return torch.from_numpy(np.ascontiguousarray(np.transpose(X_np, (0, 3, 1, 2))))


def _compute_norm_stats(Xt):
    X = Xt.float() / 255.0
    return X.mean(dim=(0, 2, 3)), X.std(dim=(0, 2, 3)).clamp(min=1e-8)


class _CollateFn:
    def __call__(self, batch):
        import torch
        xs  = torch.stack([b[0] for b in batch]).to(torch.float32) / 255.0
        yts = torch.stack([b[1] for b in batch])
        yps = torch.stack([b[2] for b in batch])
        return xs, yts, yps


def _make_loader(X, yt, yp, batch_size, num_workers, seed, shuffle, drop_last=False):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    collate = _CollateFn()
    ds = TensorDataset(X, yt, yp)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=g if shuffle else None, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0,
                      drop_last=drop_last)


# =============================================================================
# TRAIN / EVAL ONE SEED
# =============================================================================
def _train_eval_one_seed(Xt_train, yt_train, yp_train, is_real_train,
                         Xt_test, yt_test, yp_test, seed, args, device):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.model_selection import train_test_split

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    combined_y    = (yt_train * len(PLANES) + yp_train).numpy()
    real_indices  = np.where(is_real_train.numpy())[0]
    synth_indices = np.where(~is_real_train.numpy())[0]

    try:
        idx_tr_real, idx_val_real = train_test_split(
            real_indices, test_size=0.2, random_state=seed,
            stratify=combined_y[real_indices],
        )
    except ValueError:
        warnings.warn(
            f"[seed={seed}] Stratified split failed — falling back to "
            f"unstratified 80/20 split.",
            RuntimeWarning,
        )
        idx_tr_real, idx_val_real = train_test_split(
            real_indices, test_size=0.2, random_state=seed,
        )

    idx_tr  = np.concatenate([idx_tr_real, synth_indices])
    idx_val = idx_val_real

    if NORM_MEAN_OVERRIDE is not None and NORM_STD_OVERRIDE is not None:
        norm_mean = torch.tensor(NORM_MEAN_OVERRIDE, dtype=torch.float32).to(device).view(1, -1, 1, 1)
        norm_std  = torch.tensor(NORM_STD_OVERRIDE,  dtype=torch.float32).to(device).view(1, -1, 1, 1)
    else:
        norm_mean, norm_std = _compute_norm_stats(Xt_train[real_indices])
        norm_mean = norm_mean.to(device).view(1, -1, 1, 1)
        norm_std  = norm_std.to(device).view(1, -1, 1, 1)

    train_loader = _make_loader(Xt_train[idx_tr],  yt_train[idx_tr],  yp_train[idx_tr],
                                args.batch_size, args.train_workers, seed,
                                shuffle=True, drop_last=False)
    val_loader   = _make_loader(Xt_train[idx_val], yt_train[idx_val], yp_train[idx_val],
                                args.batch_size * 2, args.train_workers, seed, shuffle=False)
    test_loader  = _make_loader(Xt_test, yt_test, yp_test,
                                args.batch_size * 2, args.train_workers, seed, shuffle=False)

    model     = make_model(n_tumour=len(CLASSES), n_plane=len(PLANES)).to(device)
    _opt = DEFAULT_OPTIMIZER.lower()
    if _opt == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)

    # ── v2 fix: class weights from REAL-only training rows ────────────────────
    # In v1, weights were derived from `yt_train[idx_tr]`, which in the
    # augmented case includes synthetic rows that balance the per-class
    # counts. The result: baseline trained on real-distribution-weighted
    # CE while augmented effectively trained on uniform CE — different
    # loss landscapes, no longer a fair comparison. v2 always derives
    # weights from the real subset of idx_tr, so both runs see the same
    # loss prior.
    is_real_in_idx_tr = is_real_train.numpy()[idx_tr]
    real_in_idx_tr    = idx_tr[is_real_in_idx_tr]

    counts_t = torch.bincount(yt_train[real_in_idx_tr], minlength=len(CLASSES)).clamp(min=1)
    weight_t = torch.clamp(counts_t.float().mean() / counts_t.float(), max=5.0)

    counts_p = torch.bincount(yp_train[real_in_idx_tr], minlength=len(PLANES)).clamp(min=1)
    weight_p = torch.clamp(counts_p.float().mean() / counts_p.float(), max=5.0)

    try:
        criterion_t = nn.CrossEntropyLoss(weight=weight_t.to(device), label_smoothing=DEFAULT_LABEL_SMOOTHING)
        criterion_p = nn.CrossEntropyLoss(weight=weight_p.to(device), label_smoothing=DEFAULT_LABEL_SMOOTHING)
    except TypeError:
        criterion_t = nn.CrossEntropyLoss(weight=weight_t.to(device))
        criterion_p = nn.CrossEntropyLoss(weight=weight_p.to(device))

    _warmup_steps = int(DEFAULT_WARMUP_EPOCHS * (len(idx_tr) / max(args.batch_size, 1)))
    _total_steps  = args.epochs * (len(idx_tr) / max(args.batch_size, 1))

    _sched = DEFAULT_SCHEDULER.lower()
    if _sched == "cosine":
        # Linear warmup then cosine decay to 0
        def _lr_lambda(step):
            if _warmup_steps > 0 and step < _warmup_steps:
                return float(step) / float(max(_warmup_steps, 1))
            progress = float(step - _warmup_steps) / float(max(_total_steps - _warmup_steps, 1))
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        _step_counter = [0]
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10)
        _step_counter = None

    use_amp = args.amp and device.type == "cuda"
    scaler  = _make_grad_scaler(use_amp)

    best_val_loss, best_epoch, patience_counter, best_state = float('inf'), -1, 0, None
    training_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss, tr_total = 0.0, 0
        for xb, ybt, ybp in train_loader:
            xb, ybt, ybp = xb.to(device), ybt.to(device), ybp.to(device)
            xb = (xb - norm_mean) / norm_std

            # Mixup augmentation (disabled when DEFAULT_MIXUP_ALPHA == 0)
            if DEFAULT_MIXUP_ALPHA > 0:
                import numpy as _np
                lam = float(_np.random.beta(DEFAULT_MIXUP_ALPHA, DEFAULT_MIXUP_ALPHA))
                idx = torch.randperm(xb.size(0), device=device)
                xb  = lam * xb + (1 - lam) * xb[idx]
                ybt_b, ybp_b = ybt[idx], ybp[idx]
            else:
                lam, ybt_b, ybp_b = 1.0, None, None

            optimizer.zero_grad(set_to_none=True)
            with _make_autocast(device.type, use_amp):
                logits_t, logits_p = model(xb)
                if lam < 1.0:
                    loss_t = lam * criterion_t(logits_t, ybt) + (1 - lam) * criterion_t(logits_t, ybt_b)
                    loss_p = lam * criterion_p(logits_p, ybp) + (1 - lam) * criterion_p(logits_p, ybp_b)
                else:
                    loss_t = criterion_t(logits_t, ybt)
                    loss_p = criterion_p(logits_p, ybp)
                loss = (torch.exp(-model.log_vars[0]) * loss_t + model.log_vars[0] +
                        torch.exp(-model.log_vars[1]) * loss_p + model.log_vars[1])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if _step_counter is not None:
                _step_counter[0] += 1
                scheduler.step()

            tr_loss  += loss.item() * xb.size(0)
            tr_total += xb.size(0)

        model.eval()
        vl_loss, vl_t_correct, vl_p_correct, vl_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for xb, ybt, ybp in val_loader:
                xb, ybt, ybp = xb.to(device), ybt.to(device), ybp.to(device)
                xb = (xb - norm_mean) / norm_std

                with _make_autocast(device.type, use_amp):
                    logits_t, logits_p = model(xb)
                    loss_t = criterion_t(logits_t, ybt)
                    loss_p = criterion_p(logits_p, ybp)
                    loss = (torch.exp(-model.log_vars[0]) * loss_t + model.log_vars[0] +
                            torch.exp(-model.log_vars[1]) * loss_p + model.log_vars[1])

                vl_loss      += loss.item() * xb.size(0)
                vl_t_correct += (logits_t.argmax(1) == ybt).sum().item()
                vl_p_correct += (logits_p.argmax(1) == ybp).sum().item()
                vl_total     += xb.size(0)

        avg_tr_loss = tr_loss / tr_total
        avg_vl_loss = vl_loss / vl_total
        vl_t_acc = vl_t_correct / vl_total
        vl_p_acc = vl_p_correct / vl_total
        if _step_counter is None:   # ReduceLROnPlateau
            scheduler.step(avg_vl_loss)

        if avg_vl_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = avg_vl_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        training_history.append({
            "source": "epoch",
            "seed": int(seed),
            "epoch": int(epoch),
            "step": int(epoch),
            "real_epoch": float(epoch),
            "tr_loss": float(avg_tr_loss),
            "vl_loss": float(avg_vl_loss),
            "vl_tumour": float(vl_t_acc),
            "vl_plane": float(vl_p_acc),
            "patience_counter": int(patience_counter),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(f"      epoch {epoch:>3}/{args.epochs}  tr_loss={avg_tr_loss:.4f}"
                  f"  vl_loss={avg_vl_loss:.4f}"
                  f"  vl_tumour={vl_t_acc:.4f}"
                  f"  vl_plane={vl_p_acc:.4f}", flush=True)
        if patience_counter >= 30:
            break

    if best_state is None:
        warnings.warn(
            "best_state is None — no training epoch was completed. "
            "Using initial weights for evaluation.",
            RuntimeWarning,
        )
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    yt_true, yt_pred, yp_true, yp_pred = _evaluate(
        model, test_loader, device, norm_mean, norm_std, use_amp)

    tumour_f1 = f1_score(yt_true, yt_pred, labels=list(range(len(CLASSES))),
                         average=None, zero_division=0).tolist()
    plane_f1  = f1_score(yp_true, yp_pred, labels=list(range(len(PLANES))),
                         average=None, zero_division=0).tolist()
    return {
        "seed":               seed,
        "tumour_accuracy":    float(accuracy_score(yt_true, yt_pred)),
        "plane_accuracy":     float(accuracy_score(yp_true, yp_pred)),
        "tumour_macro_f1":    float(f1_score(yt_true, yt_pred, average="macro",    zero_division=0)),
        "tumour_weighted_f1": float(f1_score(yt_true, yt_pred, average="weighted", zero_division=0)),
        "plane_macro_f1":     float(f1_score(yp_true, yp_pred, average="macro",    zero_division=0)),
        "plane_weighted_f1":  float(f1_score(yp_true, yp_pred, average="weighted", zero_division=0)),
        "per_tumour_f1":      {CLASSES[i]: tumour_f1[i] for i in range(len(CLASSES))},
        "per_plane_f1":       {PLANES[i]:  plane_f1[i]  for i in range(len(PLANES))},
        "tumour_confusion":   confusion_matrix(yt_true, yt_pred,
                                               labels=list(range(len(CLASSES)))).tolist(),
        "plane_confusion":    confusion_matrix(yp_true, yp_pred,
                                               labels=list(range(len(PLANES)))).tolist(),
        "best_epoch":         best_epoch,
        "training_history":   training_history,
    }


def _write_history_table(path_base: Path, rows: list[dict]):
    if not rows:
        return

    jsonl_path = path_base.with_suffix(".jsonl")
    csv_path = path_base.with_suffix(".csv")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_history(res_dir: Path, label: str, seed: int, history: list[dict]) -> list[dict]:
    if not history:
        return []

    hist_dir = res_dir / "training_history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in history:
        saved = dict(row)
        saved.setdefault("experiment", label)
        saved.setdefault("seed", int(seed))
        rows.append(saved)

    _write_history_table(hist_dir / f"{label}_seed{seed}_history", rows)
    return rows


def _summarize_seeds(per_seed):
    import numpy as np
    summary = {"n_seeds": len(per_seed)}
    for k in ("tumour_accuracy", "plane_accuracy",
              "tumour_macro_f1", "tumour_weighted_f1",
              "plane_macro_f1",  "plane_weighted_f1"):
        vals = np.array([s[k] for s in per_seed])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    pt, pp = {}, {}
    for cls in CLASSES:
        vals = np.array([s["per_tumour_f1"][cls] for s in per_seed])
        pt[cls] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    for plane in PLANES:
        vals = np.array([s["per_plane_f1"][plane] for s in per_seed])
        pp[plane] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1))}
    summary["per_tumour_f1"]         = pt
    summary["per_plane_f1"]          = pp
    summary["tumour_confusion_mean"] = (
        np.array([s["tumour_confusion"] for s in per_seed]).mean(axis=0).tolist()
    )
    summary["plane_confusion_mean"]  = (
        np.array([s["plane_confusion"]  for s in per_seed]).mean(axis=0).tolist()
    )
    return summary


def _run_experiment(label, Xt_train, yt_train, yp_train, is_real_train,
                    Xt_test, yt_test, yp_test, args, device, res_dir):
    print(f"\nEXPERIMENT — {label.upper()} (CNN, two-headed, v2)", flush=True)
    results = []
    all_history = []
    for seed in SEEDS:
        print(f"  seed={seed}", flush=True)
        r = _train_eval_one_seed(Xt_train, yt_train, yp_train, is_real_train,
                                 Xt_test, yt_test, yp_test, seed, args, device)
        for row in r.get("training_history", []):
            row.setdefault("experiment", label)
        all_history.extend(_write_training_history(
            res_dir, label, seed, r.get("training_history", [])
        ))
        results.append(r)
        print(f"    → tumour_acc={r['tumour_accuracy']:.4f}  "
              f"plane_acc={r['plane_accuracy']:.4f}  "
              f"(best epoch {r['best_epoch']})", flush=True)
        with open(res_dir / f"{label}_seed{seed}.json", "w") as f:
            json.dump(r, f, indent=2)

    if all_history:
        _write_history_table(res_dir / "training_history" / f"{label}_history", all_history)

    summary = _summarize_seeds(results)
    with open(res_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return results, summary


# =============================================================================
# STATS / FIGURE
# =============================================================================
def stage_e_stats(args, res_dir: Path):
    import numpy as np
    from scipy import stats

    base_files = [res_dir / f"baseline_seed{s}.json" for s in SEEDS]
    if not all(p.exists() for p in base_files):
        raise FileNotFoundError(
            f"stage_e_stats: baseline result files missing under {res_dir}. "
            f"Run --stage baseline first."
        )
    base_seeds = [json.load(open(p)) for p in base_files]

    out = {}
    scalar_metrics = ["tumour_accuracy", "plane_accuracy",
                      "tumour_macro_f1", "tumour_weighted_f1",
                      "plane_macro_f1",  "plane_weighted_f1"]
    per_tumour = [f"tumour_f1[{c}]" for c in CLASSES]
    per_plane  = [f"plane_f1[{p}]"  for p in PLANES]

    labels_to_compare = []
    for R in args.ratios:
        tag = _ratio_tag(R)
        labels_to_compare.append(f"augmented_{tag}")
        labels_to_compare.append(f"filtered_{tag}")

    for label in labels_to_compare:
        seed_files = [res_dir / f"{label}_seed{s}.json" for s in SEEDS]
        if not all(p.exists() for p in seed_files):
            print(f"  [{label}] missing seed files — skipping comparison.")
            continue
        comp_seeds = [json.load(open(p)) for p in seed_files]

        comp_out = {}
        for m in scalar_metrics + per_tumour + per_plane:
            if m.startswith("tumour_f1["):
                cls = m[len("tumour_f1["):-1]
                b_v = np.array([s["per_tumour_f1"][cls] for s in base_seeds])
                a_v = np.array([s["per_tumour_f1"][cls] for s in comp_seeds])
            elif m.startswith("plane_f1["):
                plane = m[len("plane_f1["):-1]
                b_v = np.array([s["per_plane_f1"][plane] for s in base_seeds])
                a_v = np.array([s["per_plane_f1"][plane] for s in comp_seeds])
            else:
                b_v = np.array([s[m] for s in base_seeds])
                a_v = np.array([s[m] for s in comp_seeds])
            try:
                t_stat, p_val = stats.ttest_rel(a_v, b_v, alternative="greater")
            except TypeError:
                t = stats.ttest_rel(a_v, b_v)
                t_stat, p_val = t.statistic, (t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)
            comp_out[m] = {
                "mean_a":    float(a_v.mean()),
                "mean_b":    float(b_v.mean()),
                "delta":     float(a_v.mean() - b_v.mean()),
                "p_value":   float(p_val),
                "t":         float(t_stat),
            }
        out[label] = comp_out

    with open(res_dir / f"stats_{FIGURE_BASENAME}.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def _safe_normalise_cm(cm):
    import numpy as np
    row_sums  = cm.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return np.where(row_sums > 0, cm / safe_sums, 0.0)


def stage_f_figure(args, res_dir: Path, fig_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if not (res_dir / "baseline_summary.json").exists():
        raise FileNotFoundError(
            f"stage_f_figure: baseline_summary.json missing in {res_dir}. "
            f"Run --stage baseline first."
        )

    base_sum = json.load(open(res_dir / "baseline_summary.json"))

    exps = [("baseline", "Baseline (real only)", "#4C72B0", base_sum)]
    aug_colors  = ["#DD8452", "#E89A65"]
    filt_colors = ["#2CA02C", "#5DBE5D"]

    for i, R in enumerate(args.ratios):
        tag = _ratio_tag(R)
        p = res_dir / f"augmented_{tag}_summary.json"
        if p.exists():
            exps.append((f"augmented_{tag}", f"Augmented {R:g}×",
                         aug_colors[i % len(aug_colors)], json.load(open(p))))
    for i, R in enumerate(args.ratios):
        tag = _ratio_tag(R)
        p = res_dir / f"filtered_{tag}_summary.json"
        if p.exists():
            exps.append((f"filtered_{tag}", f"Filtered {R:g}×",
                         filt_colors[i % len(filt_colors)], json.load(open(p))))

    n_exp = len(exps)
    width = 0.8 / n_exp
    half  = (n_exp - 1) / 2.0

    fig, axes = plt.subplots(2, max(4, n_exp + 1), figsize=(7 * max(4, n_exp + 1), 13))

    # Row 0: per-tumour F1, per-plane F1, overall metrics
    ax = axes[0, 0]
    x = np.arange(len(CLASSES))
    for i, (lbl, name, col, summ) in enumerate(exps):
        m = [summ["per_tumour_f1"][c]["mean"] for c in CLASSES]
        ax.bar(x + (-half + i) * width, m, width, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=20, ha="right")
    ax.set_title("Per-Tumour F1"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 1]
    x = np.arange(len(PLANES))
    for i, (lbl, name, col, summ) in enumerate(exps):
        m = [summ["per_plane_f1"][p]["mean"] for p in PLANES]
        ax.bar(x + (-half + i) * width, m, width, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(PLANES)
    ax.set_title("Per-Plane F1"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[0, 2]
    metrics = ["tumour_accuracy", "plane_accuracy", "tumour_macro_f1", "plane_macro_f1"]
    labels  = ["Tumour Acc", "Plane Acc", "Tumour F1", "Plane F1"]
    x = np.arange(len(metrics))
    for i, (lbl, name, col, summ) in enumerate(exps):
        m = [summ[k]["mean"] for k in metrics]
        ax.bar(x + (-half + i) * width, m, width, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Overall Metrics"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, loc="lower right")

    for col in range(3, axes.shape[1]):
        axes[0, col].axis("off")

    # Row 1: tumour confusion matrices, one per experiment
    for col, (lbl, name, _, summ) in enumerate(exps):
        if col >= axes.shape[1]:
            break
        ax = axes[1, col]
        cm = np.array(summ["tumour_confusion_mean"])
        sns.heatmap(_safe_normalise_cm(cm), annot=True, fmt=".2f",
                    cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
        ax.set_title(f"{name} — Tumour Confusion", fontsize=9)

    for col in range(len(exps), axes.shape[1]):
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(fig_dir / f"{FIGURE_BASENAME}.pdf")
    plt.savefig(fig_dir / f"{FIGURE_BASENAME}.svg")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None):
    _setup_windows_env()
    import torch

    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    args = parse_args(argv)
    out_dir = args.out_dir or (args.project_root.resolve() / "outputs_v2")
    out_dir = Path(out_dir)
    res_dir = out_dir / RESULTS_SUBDIR
    fig_dir = out_dir / FIGURES_SUBDIR
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    manif_dir = out_dir / "manifests"
    manifests = {
        "real": manif_dir / "train_real.csv",
        "test": manif_dir / "test.csv",
    }
    for R in args.ratios:
        tag = _ratio_tag(R)
        manifests[f"augmented_{tag}"] = manif_dir / f"train_augmented_{tag}.csv"
        manifests[f"filtered_{tag}"]  = manif_dir / f"train_augmented_{tag}_filtered.csv"

    requested = args.device
    if requested != "cpu" and not torch.cuda.is_available():
        warnings.warn(
            f"Requested device '{requested}' but CUDA is not available. "
            f"Falling back to CPU.",
            RuntimeWarning,
        )
        requested = "cpu"
    device = torch.device(requested)

    needs_images = args.stage in ("all", "baseline", "augmented", "filtered")

    Xt_test = yt_test = yp_test = None
    Xt_real = yt_real = yp_real = is_real_real = None

    if needs_images:
        # Real + test are loaded once and reused.
        if not manifests["real"].exists():
            raise SystemExit(
                f"Missing manifest: {manifests['real']}. "
                f"Run classify_v2.py --stage manifests first."
            )
        if not manifests["test"].exists():
            raise SystemExit(
                f"Missing manifest: {manifests['test']}. "
                f"Run classify_v2.py --stage manifests first."
            )
        for key in ("real", "test"):
            paths, tl, pl, ir = _read_manifest(manifests[key])
            X_np, yt_np, yp_np, ir_np = _bulk_load(
                paths, tl, pl, ir,
                args.project_root, args.load_workers, key,
                needs_preprocessing=False,
            )
            if key == "real":
                Xt_real      = _np_to_torch_chw(X_np)
                yt_real      = torch.from_numpy(yt_np)
                yp_real      = torch.from_numpy(yp_np)
                is_real_real = torch.from_numpy(ir_np)
            else:
                Xt_test = _np_to_torch_chw(X_np)
                yt_test = torch.from_numpy(yt_np)
                yp_test = torch.from_numpy(yp_np)

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.stage in ("all", "baseline"):
        _run_experiment("baseline",
                        Xt_real, yt_real, yp_real, is_real_real,
                        Xt_test, yt_test, yp_test,
                        args, device, res_dir)

    # ── Per-ratio augmented ───────────────────────────────────────────────────
    if args.stage in ("all", "augmented"):
        for R in args.ratios:
            tag = _ratio_tag(R)
            if not manifests[f"augmented_{tag}"].exists():
                print(f"\n  [augmented_{tag}] manifest missing — "
                      f"run classify_v2.py --stage manifests first.")
                continue
            paths, tl, pl, ir = _read_manifest(manifests[f"augmented_{tag}"])
            X_np, yt_np, yp_np, ir_np = _bulk_load(
                paths, tl, pl, ir,
                args.project_root, args.load_workers, f"augmented_{tag}",
                needs_preprocessing=False,
            )
            Xt_aug      = _np_to_torch_chw(X_np)
            yt_aug      = torch.from_numpy(yt_np)
            yp_aug      = torch.from_numpy(yp_np)
            is_real_aug = torch.from_numpy(ir_np)

            _run_experiment(f"augmented_{tag}",
                            Xt_aug, yt_aug, yp_aug, is_real_aug,
                            Xt_test, yt_test, yp_test,
                            args, device, res_dir)

            del Xt_aug, yt_aug, yp_aug, is_real_aug, X_np, yt_np, yp_np, ir_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Per-ratio filtered ────────────────────────────────────────────────────
    if args.stage in ("all", "filtered"):
        for R in args.ratios:
            tag = _ratio_tag(R)
            if not manifests[f"filtered_{tag}"].exists():
                print(f"\n  [filtered_{tag}] manifest missing — "
                      f"run classify_v2.py --stage filter first.")
                continue
            paths, tl, pl, ir = _read_manifest(manifests[f"filtered_{tag}"])
            X_np, yt_np, yp_np, ir_np = _bulk_load(
                paths, tl, pl, ir,
                args.project_root, args.load_workers, f"filtered_{tag}",
                needs_preprocessing=False,
            )
            Xt_filt      = _np_to_torch_chw(X_np)
            yt_filt      = torch.from_numpy(yt_np)
            yp_filt      = torch.from_numpy(yp_np)
            is_real_filt = torch.from_numpy(ir_np)

            _run_experiment(f"filtered_{tag}",
                            Xt_filt, yt_filt, yp_filt, is_real_filt,
                            Xt_test, yt_test, yp_test,
                            args, device, res_dir)

            del Xt_filt, yt_filt, yp_filt, is_real_filt, X_np, yt_np, yp_np, ir_np
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.stage in ("all", "stats"):
        stage_e_stats(args, res_dir)

    if args.stage in ("all", "figure"):
        stage_f_figure(args, res_dir, fig_dir)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all",
                   choices=["all", "baseline", "augmented", "filtered", "stats", "figure"])
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS,
                   help="Augmentation ratios. Default: 1.0 2.0.")
    p.add_argument("--epochs",       type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",   type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",           type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--load-workers",  type=int, default=8)
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true",
                   help="Enable AMP. Off by default to avoid float16 dynamic-range "
                        "loss on low-contrast MRI.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()
