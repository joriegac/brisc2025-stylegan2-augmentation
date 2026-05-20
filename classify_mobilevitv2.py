"""
classify_mobilevitv2.py
=======================
MobileViTV2 architecture-ceiling control for the v2 classifier pipeline.

This file intentionally reuses the image loading, manifest handling, figures,
and basic result layout from classify_cnn_v2.py, but overrides the parts that
must differ for a fair SOTA-control run:

* MobileViTV2/timm backbone with two heads for tumour class and plane.
* Compute-matched step-budget training across baseline/augmented/filtered arms.
* Real-only validation split for checkpoint selection.
* Per-case test predictions saved in each seed JSON.
* Case-level paired resampling for primary accuracy statistics.

Requires: torch, timm, numpy, sklearn, scipy stack used by classify_cnn_v2.py.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import warnings
from pathlib import Path

import numpy as np

import classify_cnn_v2 as pipeline


MOBILEVITV2_MODEL_CANDIDATES = (
    os.environ.get("MOBILEVITV2_MODEL", "").strip(),
    "mobilevitv2_100.cvnets_in1k",
    "mobilevitv2_100",
    "mobilevitv2_075.cvnets_in1k",
    "mobilevitv2_075",
    "mobilevitv2_050.cvnets_in1k",
    "mobilevitv2_050",
)


def _pretrained_enabled() -> bool:
    value = os.environ.get("MOBILEVITV2_PRETRAINED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _create_backbone():
    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "classify_mobilevitv2.py requires timm. Install timm in the same "
            "environment used for PyTorch before running this stage."
        ) from exc

    tried = []
    last_exc = None
    pretrained = _pretrained_enabled()
    for name in MOBILEVITV2_MODEL_CANDIDATES:
        if not name:
            continue
        tried.append(name)
        try:
            try:
                model = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                    img_size=pipeline.IMG_SIZE,
                )
            except TypeError:
                model = timm.create_model(
                    name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
            return name, model
        except Exception as exc:
            last_exc = exc

    try:
        available = ", ".join(timm.list_models("*mobilevitv2*"))
    except Exception:
        available = "unavailable"
    raise SystemExit(
        "Could not create a timm MobileViTV2 backbone. Tried: "
        f"{', '.join(tried)}. Available timm matches: {available}. "
        f"Last error: {last_exc}"
    )


def make_model(n_tumour: int = 4, n_plane: int = 3):
    import torch
    import torch.nn as nn

    class TwoHeadMobileViTV2(nn.Module):
        def __init__(self, n_tumour: int, n_plane: int):
            super().__init__()
            self.backbone_name, self.backbone = _create_backbone()
            self.pool = nn.AdaptiveAvgPool2d(1)
            feature_dim = getattr(self.backbone, "num_features", None)
            if feature_dim is None:
                feature_dim = self._infer_feature_dim()
            self.dropout = nn.Dropout(0.2)
            self.head_tumour = nn.Linear(int(feature_dim), n_tumour)
            self.head_plane = nn.Linear(int(feature_dim), n_plane)
            self.log_vars = torch.nn.Parameter(torch.zeros(2))

        def _features(self, x):
            x = self.backbone(x)
            if isinstance(x, (tuple, list)):
                x = x[-1]
            if x.ndim > 2:
                x = self.pool(x).flatten(1)
            return x

        def _infer_feature_dim(self) -> int:
            was_training = self.backbone.training
            self.backbone.eval()
            try:
                with torch.no_grad():
                    x = torch.zeros(1, 3, pipeline.IMG_SIZE, pipeline.IMG_SIZE)
                    y = self._features(x)
                return int(y.shape[1])
            finally:
                if was_training:
                    self.backbone.train()

        def forward(self, x):
            x = self.dropout(self._features(x))
            return self.head_tumour(x), self.head_plane(x)

    return TwoHeadMobileViTV2(n_tumour, n_plane)


def _ceil_div(a: int, b: int) -> int:
    return int(math.ceil(float(a) / float(max(b, 1))))


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
            def unscale_(self, opt): pass
            def step(self, opt): opt.step()
            def update(self): pass
        return _NullScaler()
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _make_optimizer_param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if (
            name.endswith(".bias")
            or param.ndim <= 1
            or "log_vars" in lname
            or "norm" in lname
            or ".bn" in lname
            or "batchnorm" in lname
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _compute_batch_mix(batch_size: int, n_real_total: int, n_synth_total: int):
    if n_synth_total <= 0:
        return batch_size, 0, 0.0
    ratio = n_synth_total / max(n_real_total, 1)
    real_bs = max(1, int(round(batch_size / (1.0 + ratio))))
    real_bs = min(real_bs, batch_size - 1)
    synth_bs = batch_size - real_bs
    return real_bs, synth_bs, ratio


def _evaluate_with_predictions(model, loader, device, norm_mean, norm_std, use_amp):
    import numpy as np
    import torch

    model.eval()
    yt_true, yt_pred, yp_true, yp_pred = [], [], [], []
    with torch.no_grad():
        for xb, ybt, ybp in loader:
            xb = xb.to(device, non_blocking=True)
            xb = (xb - norm_mean) / norm_std
            with _make_autocast(device.type, use_amp):
                logits_t, logits_p = model(xb)
            yt_true.append(ybt.numpy())
            yp_true.append(ybp.numpy())
            yt_pred.append(logits_t.argmax(1).cpu().numpy())
            yp_pred.append(logits_p.argmax(1).cpu().numpy())
    return (
        np.concatenate(yt_true),
        np.concatenate(yt_pred),
        np.concatenate(yp_true),
        np.concatenate(yp_pred),
    )


def _validation_pass(model, val_loader, device, norm_mean, norm_std,
                     criterion_t, criterion_p, use_amp):
    import torch

    model.eval()
    vl_loss, vl_t_correct, vl_p_correct, vl_total = 0.0, 0, 0, 0
    with torch.no_grad():
        for xb, ybt, ybp in val_loader:
            xb = xb.to(device, non_blocking=True)
            ybt = ybt.to(device, non_blocking=True)
            ybp = ybp.to(device, non_blocking=True)
            xb = (xb - norm_mean) / norm_std
            with _make_autocast(device.type, use_amp):
                logits_t, logits_p = model(xb)
                loss_t = criterion_t(logits_t, ybt)
                loss_p = criterion_p(logits_p, ybp)
                loss = (
                    torch.exp(-model.log_vars[0]) * loss_t + model.log_vars[0]
                    + torch.exp(-model.log_vars[1]) * loss_p + model.log_vars[1]
                )
            vl_loss += loss.item() * xb.size(0)
            vl_t_correct += (logits_t.argmax(1) == ybt).sum().item()
            vl_p_correct += (logits_p.argmax(1) == ybp).sum().item()
            vl_total += xb.size(0)
    return vl_loss / vl_total, vl_t_correct / vl_total, vl_p_correct / vl_total


def _train_eval_one_seed(Xt_train, yt_train, yp_train, is_real_train,
                         Xt_test, yt_test, yp_test, seed, args, device):
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    combined_y = (yt_train * len(pipeline.PLANES) + yp_train).numpy()
    real_indices = np.where(is_real_train.numpy())[0]
    synth_indices = np.where(~is_real_train.numpy())[0]

    try:
        idx_tr_real, idx_val_real = train_test_split(
            real_indices,
            test_size=0.2,
            random_state=seed,
            stratify=combined_y[real_indices],
        )
    except ValueError:
        warnings.warn(
            f"[seed={seed}] Stratified split failed; using unstratified real-only split.",
            RuntimeWarning,
        )
        idx_tr_real, idx_val_real = train_test_split(
            real_indices,
            test_size=0.2,
            random_state=seed,
        )

    if pipeline.NORM_MEAN_OVERRIDE is not None and pipeline.NORM_STD_OVERRIDE is not None:
        norm_mean = torch.tensor(
            pipeline.NORM_MEAN_OVERRIDE, dtype=torch.float32, device=device
        ).view(1, -1, 1, 1)
        norm_std = torch.tensor(
            pipeline.NORM_STD_OVERRIDE, dtype=torch.float32, device=device
        ).view(1, -1, 1, 1)
    else:
        norm_mean, norm_std = pipeline._compute_norm_stats(Xt_train[real_indices])
        norm_mean = norm_mean.to(device).view(1, -1, 1, 1)
        norm_std = norm_std.to(device).view(1, -1, 1, 1)

    val_loader = pipeline._make_loader(
        Xt_train[idx_val_real],
        yt_train[idx_val_real],
        yp_train[idx_val_real],
        args.batch_size * 2,
        args.train_workers,
        seed,
        shuffle=False,
    )
    test_loader = pipeline._make_loader(
        Xt_test,
        yt_test,
        yp_test,
        args.batch_size * 2,
        args.train_workers,
        seed,
        shuffle=False,
    )

    model = make_model(n_tumour=len(pipeline.CLASSES), n_plane=len(pipeline.PLANES)).to(device)
    if getattr(args, "torch_compile", False):
        try:
            model = torch.compile(model)
        except Exception as exc:
            warnings.warn(f"torch.compile failed; continuing eager mode. Error: {exc}")

    if pipeline.DEFAULT_OPTIMIZER.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            _make_optimizer_param_groups(model, args.weight_decay), lr=args.lr
        )
    else:
        optimizer = torch.optim.Adam(
            _make_optimizer_param_groups(model, args.weight_decay), lr=args.lr
        )

    counts_t = torch.bincount(yt_train[idx_tr_real], minlength=len(pipeline.CLASSES)).clamp(min=1)
    weight_t = torch.clamp(counts_t.float().mean() / counts_t.float(), max=5.0)
    counts_p = torch.bincount(yp_train[idx_tr_real], minlength=len(pipeline.PLANES)).clamp(min=1)
    weight_p = torch.clamp(counts_p.float().mean() / counts_p.float(), max=5.0)

    try:
        criterion_t = nn.CrossEntropyLoss(
            weight=weight_t.to(device),
            label_smoothing=pipeline.DEFAULT_LABEL_SMOOTHING,
        )
        criterion_p = nn.CrossEntropyLoss(
            weight=weight_p.to(device),
            label_smoothing=pipeline.DEFAULT_LABEL_SMOOTHING,
        )
    except TypeError:
        criterion_t = nn.CrossEntropyLoss(weight=weight_t.to(device))
        criterion_p = nn.CrossEntropyLoss(weight=weight_p.to(device))

    n_real_total = len(real_indices)
    n_synth_total = len(synth_indices)
    real_bs, synth_bs, synth_ratio = _compute_batch_mix(
        args.batch_size, n_real_total, n_synth_total
    )
    training_budget = getattr(args, "training_budget", "compute")
    if training_budget == "real_exposure":
        budget_epoch_steps = _ceil_div(len(idx_tr_real), real_bs)
    else:
        budget_epoch_steps = _ceil_div(len(idx_tr_real), args.batch_size)
    max_steps = args.max_steps or int(args.epochs * budget_epoch_steps)
    max_steps = max(max_steps, 1)
    val_every = max(1, args.val_every)
    patience_evals = max(1, args.early_stop_patience_evals)
    min_delta = max(0.0, args.min_delta)

    warmup_steps = int(pipeline.DEFAULT_WARMUP_EPOCHS * budget_epoch_steps)
    total_steps = max_steps
    if pipeline.DEFAULT_SCHEDULER.lower() == "cosine":
        def _lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(warmup_steps, 1))
            progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        plateau_scheduler = False
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        plateau_scheduler = True

    use_amp = args.amp and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + 1009)

    best_val_loss = float("inf")
    patience_best_loss = float("inf")
    best_step = -1
    best_state = None
    patience_counter = 0
    loss_window = []
    n_window = 0
    real_seen = 0
    synthetic_seen = 0
    training_history = []

    print(
        f"      step budget: max_steps={max_steps}  "
        f"budget_epoch_steps={budget_epoch_steps}  val_every={val_every}  "
        f"patience_evals={patience_evals}  training_budget={training_budget}",
        flush=True,
    )
    if n_synth_total:
        print(
            f"      train mix: compute-matched batch real={real_bs} "
            f"synthetic={synth_bs} (synth/real rows={synth_ratio:.2f})",
            flush=True,
        )

    for step in range(1, max_steps + 1):
        model.train()
        real_pick = torch.randint(len(idx_tr_real), (real_bs,), generator=gen)
        batch_idx_np = idx_tr_real[real_pick.numpy()]
        if synth_bs > 0:
            synth_pick = torch.randint(len(synth_indices), (synth_bs,), generator=gen)
            batch_idx_np = np.concatenate([batch_idx_np, synth_indices[synth_pick.numpy()]])
        np.random.shuffle(batch_idx_np)
        batch_idx = torch.as_tensor(batch_idx_np, dtype=torch.long)

        xb = Xt_train[batch_idx].to(device, non_blocking=True).float() / 255.0
        ybt = yt_train[batch_idx].to(device, non_blocking=True)
        ybp = yp_train[batch_idx].to(device, non_blocking=True)
        xb = (xb - norm_mean) / norm_std

        if pipeline.DEFAULT_MIXUP_ALPHA > 0:
            lam = float(np.random.beta(pipeline.DEFAULT_MIXUP_ALPHA, pipeline.DEFAULT_MIXUP_ALPHA))
            perm = torch.randperm(xb.size(0), device=device)
            xb = lam * xb + (1.0 - lam) * xb[perm]
            ybt_b, ybp_b = ybt[perm], ybp[perm]
        else:
            lam, ybt_b, ybp_b = 1.0, None, None

        optimizer.zero_grad(set_to_none=True)
        with _make_autocast(device.type, use_amp):
            logits_t, logits_p = model(xb)
            if lam < 1.0:
                loss_t = lam * criterion_t(logits_t, ybt) + (1.0 - lam) * criterion_t(logits_t, ybt_b)
                loss_p = lam * criterion_p(logits_p, ybp) + (1.0 - lam) * criterion_p(logits_p, ybp_b)
            else:
                loss_t = criterion_t(logits_t, ybt)
                loss_p = criterion_p(logits_p, ybp)
            loss = (
                torch.exp(-model.log_vars[0]) * loss_t + model.log_vars[0]
                + torch.exp(-model.log_vars[1]) * loss_p + model.log_vars[1]
            )

        scaler.scale(loss).backward()
        if args.grad_clip and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if not plateau_scheduler:
            scheduler.step()

        loss_window.append(float(loss.item()) * xb.size(0))
        n_window += xb.size(0)
        real_seen += real_bs
        synthetic_seen += synth_bs

        do_val = step == 1 or step % val_every == 0 or step == max_steps
        if not do_val:
            continue

        avg_vl_loss, vl_t_acc, vl_p_acc = _validation_pass(
            model, val_loader, device, norm_mean, norm_std,
            criterion_t, criterion_p, use_amp,
        )
        if plateau_scheduler:
            scheduler.step(avg_vl_loss)

        if avg_vl_loss < best_val_loss:
            best_val_loss = avg_vl_loss
            best_step = step
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if avg_vl_loss <= patience_best_loss - min_delta:
            patience_best_loss = avg_vl_loss
            patience_counter = 0
        else:
            patience_counter += 1

        tr_loss = sum(loss_window) / max(n_window, 1)
        real_epoch = real_seen / max(len(idx_tr_real), 1)
        training_history.append({
            "source": "step",
            "seed": int(seed),
            "step": int(step),
            "real_epoch": float(real_epoch),
            "tr_loss": float(tr_loss),
            "vl_loss": float(avg_vl_loss),
            "vl_tumour": float(vl_t_acc),
            "vl_plane": float(vl_p_acc),
            "patience_counter": int(patience_counter),
            "best_step": int(best_step),
            "best_val_loss": float(best_val_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "training_budget": training_budget,
            "max_steps": int(max_steps),
            "budget_epoch_steps": int(budget_epoch_steps),
            "val_every": int(val_every),
            "patience_evals": int(patience_evals),
            "batch_real": int(real_bs),
            "batch_synthetic": int(synth_bs),
            "real_examples_seen": int(real_seen),
            "synthetic_examples_seen": int(synthetic_seen),
        })
        print(
            f"      step {step:>5}/{max_steps}  real_epoch={real_epoch:>6.1f}  "
            f"tr_loss={tr_loss:.4f}  vl_loss={avg_vl_loss:.4f}  "
            f"vl_tumour={vl_t_acc:.4f}  vl_plane={vl_p_acc:.4f}  "
            f"patience={patience_counter}/{patience_evals}",
            flush=True,
        )
        loss_window.clear()
        n_window = 0

        if patience_counter >= patience_evals:
            break

    if best_state is None:
        warnings.warn("No validation checkpoint was saved; using final weights.")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_step = step

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    yt_true, yt_pred, yp_true, yp_pred = _evaluate_with_predictions(
        model, test_loader, device, norm_mean, norm_std, use_amp
    )

    tumour_f1 = f1_score(
        yt_true, yt_pred, labels=list(range(len(pipeline.CLASSES))),
        average=None, zero_division=0,
    ).tolist()
    plane_f1 = f1_score(
        yp_true, yp_pred, labels=list(range(len(pipeline.PLANES))),
        average=None, zero_division=0,
    ).tolist()
    return {
        "seed": int(seed),
        "tumour_accuracy": float(accuracy_score(yt_true, yt_pred)),
        "plane_accuracy": float(accuracy_score(yp_true, yp_pred)),
        "tumour_macro_f1": float(f1_score(yt_true, yt_pred, average="macro", zero_division=0)),
        "tumour_weighted_f1": float(f1_score(yt_true, yt_pred, average="weighted", zero_division=0)),
        "plane_macro_f1": float(f1_score(yp_true, yp_pred, average="macro", zero_division=0)),
        "plane_weighted_f1": float(f1_score(yp_true, yp_pred, average="weighted", zero_division=0)),
        "per_tumour_f1": {pipeline.CLASSES[i]: tumour_f1[i] for i in range(len(pipeline.CLASSES))},
        "per_plane_f1": {pipeline.PLANES[i]: plane_f1[i] for i in range(len(pipeline.PLANES))},
        "tumour_confusion": confusion_matrix(
            yt_true, yt_pred, labels=list(range(len(pipeline.CLASSES)))
        ).tolist(),
        "plane_confusion": confusion_matrix(
            yp_true, yp_pred, labels=list(range(len(pipeline.PLANES)))
        ).tolist(),
        "best_step": int(best_step),
        "best_epoch": int(best_step),
        "max_steps": int(max_steps),
        "budget_epoch_steps": int(budget_epoch_steps),
        "training_budget": training_budget,
        "batch_real": int(real_bs),
        "batch_synthetic": int(synth_bs),
        "real_examples_seen": int(real_seen),
        "synthetic_examples_seen": int(synthetic_seen),
        "effective_train_examples_seen": int(real_seen + synthetic_seen),
        "training_history": training_history,
        "y_tumour_true": yt_true.astype(int).tolist(),
        "y_tumour_pred": yt_pred.astype(int).tolist(),
        "y_plane_true": yp_true.astype(int).tolist(),
        "y_plane_pred": yp_pred.astype(int).tolist(),
    }


def _f1_scores(y_true: np.ndarray, y_pred: np.ndarray, n_labels: int) -> tuple[np.ndarray, np.ndarray]:
    f1 = np.zeros(n_labels, dtype=np.float64)
    support = np.zeros(n_labels, dtype=np.float64)
    for label in range(n_labels):
        true_pos = np.sum((y_true == label) & (y_pred == label))
        false_pos = np.sum((y_true != label) & (y_pred == label))
        false_neg = np.sum((y_true == label) & (y_pred != label))
        support[label] = np.sum(y_true == label)
        denom = 2 * true_pos + false_pos + false_neg
        f1[label] = (2 * true_pos / denom) if denom else 0.0
    return f1, support


def _metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric in {"tumour_accuracy", "plane_accuracy"}:
        return float(np.mean(y_true == y_pred))

    if metric.startswith("per_tumour_f1:"):
        label = pipeline.CLASSES.index(metric.split(":", 1)[1])
        return float(_f1_scores(y_true, y_pred, len(pipeline.CLASSES))[0][label])

    if metric.startswith("per_plane_f1:"):
        label = pipeline.PLANES.index(metric.split(":", 1)[1])
        return float(_f1_scores(y_true, y_pred, len(pipeline.PLANES))[0][label])

    if metric in {"tumour_macro_f1", "tumour_weighted_f1"}:
        f1, support = _f1_scores(y_true, y_pred, len(pipeline.CLASSES))
    elif metric in {"plane_macro_f1", "plane_weighted_f1"}:
        f1, support = _f1_scores(y_true, y_pred, len(pipeline.PLANES))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if metric.endswith("_macro_f1"):
        return float(np.mean(f1))
    return float(np.average(f1, weights=support))


def _metric_family(metric: str) -> str:
    return "plane" if metric.startswith("plane_") or metric.startswith("per_plane_") else "tumour"


def _mean_seed_delta(base_preds: list[np.ndarray], comp_preds: list[np.ndarray],
                     y_true: np.ndarray, metric: str, idx: np.ndarray | None = None) -> float:
    if idx is None:
        return float(np.mean([
            _metric_value(y_true, comp_pred, metric) - _metric_value(y_true, base_pred, metric)
            for base_pred, comp_pred in zip(base_preds, comp_preds)
        ]))

    y_subset = y_true[idx]
    return float(np.mean([
        _metric_value(y_subset, comp_pred[idx], metric) - _metric_value(y_subset, base_pred[idx], metric)
        for base_pred, comp_pred in zip(base_preds, comp_preds)
    ]))


def _case_resampling_metric_worker(job: dict) -> dict:
    metric = job["metric"]
    comparison = job["comparison"]
    iters = int(job["iters"])
    seed = int(job["seed"])
    y_true = np.asarray(job["y_true"], dtype=np.int16)
    base_preds = [np.asarray(pred, dtype=np.int16) for pred in job["base_preds"]]
    comp_preds = [np.asarray(pred, dtype=np.int16) for pred in job["comp_preds"]]
    n_cases = y_true.shape[0]
    rng = np.random.default_rng(seed)

    observed = _mean_seed_delta(base_preds, comp_preds, y_true, metric)

    boot = np.empty(iters, dtype=np.float64)
    perm = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n_cases, size=n_cases)
        boot[i] = _mean_seed_delta(base_preds, comp_preds, y_true, metric, idx)

        swap = rng.random(n_cases) < 0.5
        deltas = []
        for base_pred, comp_pred in zip(base_preds, comp_preds):
            base_swapped = base_pred.copy()
            comp_swapped = comp_pred.copy()
            base_swapped[swap], comp_swapped[swap] = comp_swapped[swap], base_swapped[swap]
            deltas.append(
                _metric_value(y_true, comp_swapped, metric)
                - _metric_value(y_true, base_swapped, metric)
            )
        perm[i] = float(np.mean(deltas))

    p_two_sided = float((np.sum(np.abs(perm) >= abs(observed)) + 1) / (iters + 1))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return {
        "comparison": comparison,
        "metric": metric,
        "delta": observed,
        "ci95": [float(ci_low), float(ci_high)],
        "p_raw": p_two_sided,
        "resample_iters": iters,
    }


def _load_seed_results(res_dir: Path, label: str) -> list[dict]:
    files = [res_dir / f"{label}_seed{s}.json" for s in pipeline.SEEDS]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing seed files for {label}: {missing[:3]}")
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _prediction_arrays(seed_runs: list[dict], family: str) -> tuple[list[int], list[list[int]]]:
    true_key = f"y_{family}_true"
    pred_key = f"y_{family}_pred"
    y_true = seed_runs[0][true_key]
    for run in seed_runs[1:]:
        if run[true_key] != y_true:
            raise ValueError(f"{family}: y_true differs across seeds; cannot pair cases.")
    return y_true, [run[pred_key] for run in seed_runs]


def _holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda idx: p_values[idx])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def _metric_names() -> list[str]:
    return [
        "tumour_accuracy",
        "plane_accuracy",
        "tumour_macro_f1",
        "tumour_weighted_f1",
        "plane_macro_f1",
        "plane_weighted_f1",
        *[f"per_tumour_f1:{name}" for name in pipeline.CLASSES],
        *[f"per_plane_f1:{name}" for name in pipeline.PLANES],
    ]


def stage_e_stats(args, res_dir: Path):
    base_runs = _load_seed_results(res_dir, "baseline")

    comparisons = []
    for ratio in args.ratios:
        tag = pipeline._ratio_tag(ratio)
        for prefix in ("augmented", "filtered"):
            label = f"{prefix}_{tag}"
            files = [res_dir / f"{label}_seed{s}.json" for s in pipeline.SEEDS]
            if all(path.exists() for path in files):
                comparisons.append((label, _load_seed_results(res_dir, label)))
            else:
                print(f"  [{label}] missing seed files; skipping comparison.")

    metrics = _metric_names()
    jobs = []
    for comparison_idx, (label, comp_runs) in enumerate(comparisons):
        for metric_idx, metric in enumerate(metrics):
            family = _metric_family(metric)
            base_true, base_preds = _prediction_arrays(base_runs, family)
            comp_true, comp_preds = _prediction_arrays(comp_runs, family)
            if comp_true != base_true:
                raise ValueError(f"{label}: {family} y_true differs from baseline.")
            jobs.append({
                "comparison": f"{label} vs baseline",
                "metric": metric,
                "iters": args.resample_iters,
                "seed": args.resample_seed + 1009 * comparison_idx + 37 * metric_idx,
                "y_true": base_true,
                "base_preds": base_preds,
                "comp_preds": comp_preds,
            })

    if not jobs:
        raise FileNotFoundError(f"No complete augmented/filtered comparisons found under {res_dir}.")

    workers = max(1, min(args.stats_workers, len(jobs)))
    print(f"  Running {len(jobs)} resampling jobs on {workers} worker(s).", flush=True)
    if workers == 1:
        results = [_case_resampling_metric_worker(job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            results = list(pool.imap_unordered(_case_resampling_metric_worker, jobs))

    adjusted = _holm_adjust([result["p_raw"] for result in results])
    for result, p_holm in zip(results, adjusted):
        result["p_holm"] = p_holm

    out = {
        "method": "two-sided paired case-level permutation over test cases; paired bootstrap CI over test cases",
        "resample_iters": int(args.resample_iters),
        "workers": int(workers),
        "multiplicity": "Holm correction across all comparisons and metrics",
        "comparisons": {},
    }
    for result in sorted(results, key=lambda item: (item["comparison"], item["metric"])):
        out["comparisons"].setdefault(result["comparison"], {})[result["metric"]] = {
            "delta": float(result["delta"]),
            "ci95": [float(result["ci95"][0]), float(result["ci95"][1])],
            "p_raw": float(result["p_raw"]),
            "p_holm": float(result["p_holm"]),
        }

    for name in (f"stats_{pipeline.FIGURE_BASENAME}.json", "stats_v2.json"):
        path = res_dir / name
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  Wrote {path}", flush=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="MobileViTV2 compute-matched v2 classifier pipeline.",
    )
    p.add_argument("--stage", default="all",
                   choices=["all", "baseline", "augmented", "filtered", "stats", "figure"])
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: <project-root>/outputs_v2")
    p.add_argument("--ratios", type=float, nargs="+", default=pipeline.DEFAULT_RATIOS,
                   help="Augmentation ratios.")
    p.add_argument("--epochs", type=int, default=pipeline.DEFAULT_EPOCHS,
                   help="Used to derive default max_steps as epochs * real-only budget_epoch_steps.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override compute-matched optimizer-step budget per seed.")
    p.add_argument("--training-budget", choices=["compute", "real_exposure"], default="compute",
                   help=(
                       "compute keeps optimizer steps matched to the real-only baseline; "
                       "real_exposure increases augmented-arm steps so real examples are "
                       "seen as often as in the baseline."
                   ))
    p.add_argument("--val-every", type=int, default=80,
                   help="Validate every N optimizer steps.")
    p.add_argument("--early-stop-patience-evals", type=int, default=8,
                   help="Early-stop patience measured in validation evaluations.")
    p.add_argument("--min-delta", type=float, default=0.005,
                   help="Validation-loss improvement needed to reset patience.")
    p.add_argument("--batch-size", type=int, default=pipeline.DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=pipeline.DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=pipeline.DEFAULT_WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping max norm. Use 0 to disable.")
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--train-workers", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", dest="amp", action="store_true",
                   help="Enable AMP. Default for MobileViTV2 is enabled.")
    p.add_argument("--no-amp", dest="amp", action="store_false",
                   help="Disable AMP for maximum numerical conservatism.")
    p.set_defaults(amp=True)
    p.add_argument("--torch-compile", action="store_true",
                   help="Optionally use torch.compile. Disabled by default on Windows.")
    p.add_argument("--resample-iters", type=int, default=5000,
                   help="Case-level bootstrap/permutation iterations for stats.")
    p.add_argument("--resample-seed", type=int, default=12345)
    p.add_argument("--stats-workers", type=int, default=max(1, min(19, os.cpu_count() or 1)),
                   help="Parallel workers for case-level resampling.")
    args = p.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be > 0.")
    if args.val_every <= 0:
        raise SystemExit("--val-every must be > 0.")
    if args.early_stop_patience_evals <= 0:
        raise SystemExit("--early-stop-patience-evals must be > 0.")
    if args.batch_size <= 1:
        raise SystemExit("--batch-size must be > 1.")
    if args.grad_clip < 0:
        raise SystemExit("--grad-clip must be >= 0.")
    if args.resample_iters <= 0:
        raise SystemExit("--resample-iters must be > 0.")
    if args.stats_workers <= 0:
        raise SystemExit("--stats-workers must be > 0.")
    return args


pipeline.SCRIPT_NAME = "classify_mobilevitv2.py"
pipeline.MODEL_LABEL = "MobileViTV2"
pipeline.RESULTS_SUBDIR = "results_mobilevitv2"
pipeline.FIGURES_SUBDIR = "figures_mobilevitv2"
pipeline.FIGURE_BASENAME = "classifier_comparison_mobilevitv2"

pipeline.DEFAULT_EPOCHS = 55
pipeline.DEFAULT_BATCH_SIZE = 128
pipeline.DEFAULT_LR = 5e-4
pipeline.DEFAULT_WEIGHT_DECAY = 1e-4
pipeline.DEFAULT_OPTIMIZER = "adamw"
pipeline.DEFAULT_SCHEDULER = "cosine"
pipeline.DEFAULT_WARMUP_EPOCHS = 5.0
pipeline.DEFAULT_LABEL_SMOOTHING = 0.001
pipeline.DEFAULT_MIXUP_ALPHA = 0.2

# Pretrained MobileViTV2 expects ImageNet normalization.
pipeline.NORM_MEAN_OVERRIDE = (0.485, 0.456, 0.406)
pipeline.NORM_STD_OVERRIDE = (0.229, 0.224, 0.225)

pipeline.make_model = make_model
pipeline._train_eval_one_seed = _train_eval_one_seed
pipeline.stage_e_stats = stage_e_stats
pipeline.parse_args = parse_args


def main(argv=None):
    return pipeline.main(argv)


if __name__ == "__main__":
    main()
