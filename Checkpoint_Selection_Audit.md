# Checkpoint Selection Audit & Global Status

**Project:** StyleGAN2-ADA synthetic MRI augmentation, BRISC 2025
**Audit date:** 2026-04-26
**Auditor:** José Rafael (with Claude)
**Scope:** Verify the per-(class × plane) optimal checkpoint records produced from `checkpoint_metrics_*.json`, and audit the algorithm that selected them.

---

## TL;DR

1. **All 12 models trained to completion** (1,000 kimgs, 22 evaluated checkpoints each, all with FID + Precision + Recall). The `no_tumor_sa` row that was marked **TBD** in the original briefing is now complete: optimal at **240 kimgs**, FID = 48.90, EXCELLENT tier.
2. **The 3-stage selection algorithm reproduces the briefing's results table exactly for all 11 previously-tabulated models.** The algorithm is sound and is the one actually used at synthetic-image generation time.
3. **There is a documentation/figure inconsistency that matters for the paper.** The checkpoint-evaluation cell (notebook Cell 23) labels the green "Optimal" line in `checkpoint_selection_*.png` using the *pure-FID-minimum*, not the 3-stage selection. In **7 of 12 models** these disagree, so the published per-model figures point to the wrong checkpoint.
4. **No synthetic images have been generated yet.** The `synthetic/` directory tree is empty across all 12 combos.
5. **One file is misplaced on disk:** `meningioma_sa` is nested under `no_tumor_ax/meningioma_sa/` instead of living at the top level of `stylegan3_results/`. The 3-stage selector still finds it (glob walks recursively), but the directory layout is fragile and should be fixed before the public artifact upload.

---

## 1. Global status — where we stand

### 1.1 Per-model selected checkpoints (3-stage algorithm)

Re-derived from each model's `checkpoint_metrics_*.json` using the algorithm in notebook Cell 21:

| Model | Real N | Optimal kimg | % budget | Tier | FID | Precision | Recall | Composite |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| glioma_ax | 394 | 960 | 96.0% | EXTRAORDINARY | 25.36 | 0.5128 | 0.0863 | 0.2996 |
| glioma_co | 430 | 336 | 33.6% | EXCELLENT | 35.27 | 0.6178 | 0.0814 | 0.3496 |
| glioma_sa | 323 | 240 | 24.0% | GOOD | 53.05 | 0.5693 | 0.0279 | 0.2986 |
| meningioma_ax | 423 | 1000 | 100.0% | EXCELLENT | 36.11 | 0.6066 | 0.0567 | 0.3317 |
| meningioma_co | 426 | 240 | 24.0% | GOOD | 54.18 | 0.4243 | 0.0352 | 0.2298 |
| meningioma_sa | 480 | 240 | 24.0% | EXCELLENT | 43.93 | 0.4931 | 0.0229 | 0.2580 |
| no_tumor_ax | 352 | 672 | 67.2% | EXCELLENT | 48.28 | 0.3446 | 0.0767 | 0.2107 |
| no_tumor_co | 310 | 288 | 28.8% | GOOD | 51.84 | 0.4839 | 0.0097 | 0.2468 |
| **no_tumor_sa** | **405** | **240** | **24.0%** | **EXCELLENT** | **48.90** | **0.3308** | **0.0349** | **0.1828** |
| pituitary_ax | 426 | 288 | 28.8% | EXTRAORDINARY | 28.40 | 0.4811 | 0.0211 | 0.2511 |
| pituitary_co | 510 | 192 | 19.2% | EXTRAORDINARY | 29.38 | 0.3590 | 0.0314 | 0.1952 |
| pituitary_sa | 521 | 192 | 19.2% | EXCELLENT | 46.39 | 0.3007 | 0.0134 | 0.1571 |

The `no_tumor_sa` row replaces the TBD placeholder in the briefing; the other 11 rows match the briefing values to all reported decimal places.

### 1.2 Tier distribution (per model, across the 21 evaluated checkpoints with kimg > 0)

How many checkpoints in each model fell into each FID tier — useful as a per-model "training quality" signal.

| Model | EXTRAORDINARY | EXCELLENT | GOOD | FAIR | Best tier achieved |
|---|---:|---:|---:|---:|---|
| glioma_ax | 13 | 5 | 1 | 2 | EXTRAORDINARY |
| glioma_co | 0 | 18 | 1 | 2 | EXCELLENT |
| glioma_sa | 0 | 0 | 13 | 8 | GOOD |
| meningioma_ax | 0 | 17 | 2 | 2 | EXCELLENT |
| meningioma_co | 0 | 0 | 19 | 2 | GOOD |
| meningioma_sa | 0 | 6 | 13 | 2 | EXCELLENT |
| no_tumor_ax | 0 | 11 | 7 | 3 | EXCELLENT |
| no_tumor_co | 0 | 0 | 19 | 2 | GOOD |
| no_tumor_sa | 0 | 1 | 13 | 7 | EXCELLENT |
| pituitary_ax | 4 | 14 | 1 | 2 | EXTRAORDINARY |
| pituitary_co | 1 | 18 | 1 | 1 | EXTRAORDINARY |
| pituitary_sa | 0 | 2 | 16 | 3 | EXCELLENT |

Summary across all 12 models: 3 EXTRAORDINARY, 6 EXCELLENT, 3 GOOD. No model finished in the FAIR tier. The `no_tumor_sa` model is the weakest of the EXCELLENT-tier set (only 1 checkpoint qualified for that tier, and it sits right at the EXCELLENT/GOOD boundary at FID = 48.90).

### 1.3 Synthetic generation status

| Combo | Synthetic images on disk | Target | To generate |
|---|---:|---:|---:|
| glioma_ax | 0 (empty subdir) | 1,606 | 1,606 |
| all other 11 combos | 0 (no subdir) | varies | varies |
| **TOTAL** | **0** | **19,000** | **19,000** |

Nothing has been generated yet. Cell 21 of the notebook is wired up and will use the 3-stage algorithm above the moment it's run with `CHECKPOINT = "optimal"`.

---

## 2. Audit of the selection algorithm

### 2.1 Where the selection logic actually lives

There are **two** "checkpoint selection" code paths in the notebook, and they do not implement the same algorithm:

| Path | Purpose | Algorithm |
|---|---|---|
| **Cell 23** — *Checkpoint Selection — Full Evaluation + Publication Plot* | Runs `calc_metrics.py` over every snapshot, persists JSON, draws figure | **Pure-FID minimum** (`best_fid_kimg = argmin FID`) |
| **Cell 21** — *Generate Synthetic Images (Self-Contained)* | Picks the checkpoint and produces synthetic images | **Three-stage consensus** (FID tier → composite P/R → recall tiebreak) |

The briefing's results table was generated by **Cell 21's** algorithm (re-derived value-for-value above). **Cell 23's** algorithm is what drives the green "Optimal (XXX kimgs)" annotation in the published per-model `checkpoint_selection_*.png` figures.

In 7 of 12 models, the two algorithms select **different** checkpoints, so the per-model figures currently shipped in `stylegan3_results/<combo>/<run>/checkpoint_selection_<combo>.png` annotate the wrong kimg as the optimum.

| Model | Pure-FID kimg (Cell 23 figure) | 3-stage kimg (Cell 21 generation) | Matches? |
|---|---:|---:|---|
| glioma_ax | 1000 | 960 | ✗ |
| glioma_co | 432 | 336 | ✗ |
| glioma_sa | 240 | 240 | ✓ |
| meningioma_ax | 912 | 1000 | ✗ |
| meningioma_co | 192 | 240 | ✗ |
| meningioma_sa | 240 | 240 | ✓ |
| no_tumor_ax | 1000 | 672 | ✗ |
| no_tumor_co | 288 | 288 | ✓ |
| no_tumor_sa | 240 | 240 | ✓ |
| pituitary_ax | 336 | 288 | ✗ |
| pituitary_co | 192 | 192 | ✓ |
| pituitary_sa | 192 | 192 | ✓ |

This is a **publication-grade bug** — the methods section will describe the 3-stage algorithm, the table will show 3-stage results, but the per-model evidence figures will point to a different checkpoint with no caption explaining why. Fix before submission.

### 2.2 Three-stage algorithm (Cell 21) — line-by-line audit

```
def resolve_checkpoint(model_dir, combo, checkpoint_spec):
    candidates = [(int(k), v) for k, v in metrics.items() if int(k) > 0]
    candidates.sort(key=lambda x: x[0])
    for k, v in candidates:
        v["tier"], v["tier_rank"] = get_fid_tier(v["fid"])

    best_tier_rank = min(v["tier_rank"] for _, v in candidates)
    plateau        = [(k, v) for k, v in candidates
                      if v["tier_rank"] == best_tier_rank]

    if len(plateau) == 1:
        best_kimg, best_entry = plateau[0]
    else:
        scored = sorted(
            [(k, v, PRECISION_WEIGHT * v["precision"] +
                     RECALL_WEIGHT   * v["recall"])
             for k, v in plateau],
            key=lambda x: x[2], reverse=True
        )
        top_k, top_v, top_score = scored[0]
        sec_k, sec_v, sec_score = scored[1]
        gap = top_score - sec_score
        if gap <= SCORE_TIE_THRESHOLD:
            if sec_v["recall"] > top_v["recall"]:
                best_kimg, best_entry = sec_k, sec_v
            else:
                best_kimg, best_entry = top_k, top_v
        else:
            best_kimg, best_entry = top_k, top_v
```

**What is correct:**

The algorithm exactly matches the methods description in `Instructions_Classifier_Pipeline.txt`. Tiers are medically-calibrated (Skandarani 2023): EXTRAORDINARY < 30, EXCELLENT 30–50, GOOD 50–75, FAIR ≥ 75, with `lo <= fid < hi` (so FID = 30 → EXCELLENT, FID = 50 → GOOD; both are reasonable conventions and consistent with the briefing). The `kimg > 0` filter correctly drops the initialization snapshot, which has FID ≈ 290 and zero P/R. Composite weights are 0.5/0.5 with a 0.005 score-gap tiebreak — both empirically validated per the briefing. End-to-end re-derivation reproduces the briefing's 11 tabulated rows to within rounding.

**What is fragile or worth flagging:**

1. **Pairwise tiebreak only.** The composite-tie check compares `scored[0]` vs `scored[1]` only. If three or more checkpoints land within 0.005 of each other, the third+ are silently ignored even if one of them has the highest recall. Not currently triggered by any of the 12 datasets (verified — no model has a 3-way tie within 0.005), but worth tightening to "all checkpoints within ε of the top" for robustness.

2. **No defensive None handling on P/R.** If a JSON entry has `precision: null` (which would happen if Cell 23 was run in `LIGHTWEIGHT=True` mode), the line `PRECISION_WEIGHT * v["precision"]` raises `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'`. The current JSONs all have valid P/R, so this is not biting now, but a `LIGHTWEIGHT` re-evaluation would silently break generation.

3. **`json_candidates[0]` after recursive glob.** `model_dir.glob("**/checkpoint_metrics_{combo}.json")` returns matches in arbitrary order; `[0]` picks the first. If a stale JSON ever exists in an older run dir (e.g., `00000-…/` left over from a restarted training), the wrong file could be loaded silently. In the current state every model has exactly one JSON, but the misplaced `meningioma_sa` directory under `no_tumor_ax/` shows that ad-hoc directory layouts do happen. Recommend explicitly picking the JSON from the highest-numbered run dir.

4. **No reproducibility seed printed in the audit log.** The audit print block reports the chosen checkpoint, weights, and tie threshold, but does not stamp the JSON path or timestamp it was read from. For the methods section it would be valuable to log `json_candidates[0]` and the file's mtime so the selection can be re-traced.

5. **Single-tier-member shortcut.** When only one checkpoint qualifies for the best tier, it is selected without any P/R sanity check. By design this is consistent with the "all checkpoints in the best tier are FID-equivalent" framing, but it means a single EXCELLENT checkpoint with miserable P/R (e.g., `no_tumor_sa` at P=0.33, R=0.035) will beat a slightly-worse-FID GOOD checkpoint with much higher P/R. This is the **correct** behavior under the published algorithm — flagging it only because it should be acknowledged as a *deliberate* asymmetry in the paper, not glossed over.

### 2.3 Pure-FID algorithm (Cell 23) — line-by-line audit

```
if r.get('fid') is not None and r['fid'] < best_fid_val:
    best_fid_val  = r['fid']
    best_fid_kimg = r['kimg']
```

This is just `argmin FID`. The cell prints "✅ Use network-snapshot-XXXXXX.pkl for generation" based on this value, which is **misleading guidance** because Cell 21's actual generation pipeline ignores this recommendation and uses the 3-stage choice instead. Two specific concerns:

- The printed "Use … for generation" message tells the user to use, e.g., k=1000 for `no_tumor_ax`, when the generation cell will actually use k=672. Anyone running cells in order would believe k=1000 was the intended pick.
- The plot's green "Optimal" annotation is similarly bound to `best_fid_kimg`, not the 3-stage choice.

### 2.4 Other miscellaneous notes

- **All models share identical training config**: `--cfg=stylegan2 --gpus=1 --batch=16 --batch-gpu=4 --gamma=2.0 --kimg=3000 --snap=10 --aug=ada --mirror=0 --augpipe=bgcfnc --target=0.6`. The `--kimg=3000` was a ceiling — actual training stopped at 1,000 kimgs in every case (consistent with the JSON having checkpoints up to 1000).
- **Hardcoded `REAL_DATASET_SIZE` in Cell 21** must be edited per combo. Easy to forget; consider replacing with the dictionary literal already present in the cell's docstring header (`REAL_N = {'glioma_ax': 394, ...}; REAL_DATASET_SIZE = REAL_N[COMBO]`).
- **Hardcoded Windows iCloud path** in `PROJECT_ROOT`. Fine for execution on your machine, but it propagates into stored artifacts (the `cmd` field of `batch_training_results.json` is full of these paths) and would block reproduction by a peer reviewer running from a clone.

---

## 3. Recommended actions before generation

The minimum set to clean up before kicking off the 19,000-image generation:

1. **Fix the Cell 23 figure annotation** so the "Optimal" line uses the 3-stage selection, not pure-FID. Otherwise the per-model evidence figures for the paper will contradict the methods text.
2. **Move `stylegan3_results/no_tumor_ax/meningioma_sa/`** up to `stylegan3_results/meningioma_sa/`. Cell 21's recursive glob saves us today, but the layout is wrong.
3. **(Optional) Patch the pairwise tiebreak** in `resolve_checkpoint` to consider all checkpoints within `SCORE_TIE_THRESHOLD` of the top, not just rank-2. Cosmetic for the current dataset; defensible for the methods section.
4. **(Optional) Replace `json_candidates[0]`** with explicit "pick highest-numbered run dir" logic.

After those fixes, `CHECKPOINT = "optimal"` in Cell 21 is safe to use for all 12 combos in turn.

---

## 4. Artifacts produced by this audit

- `Checkpoint_Selection_Audit.md` — this document
- `outputs/checkpoint_selection_audit.json` — machine-readable per-model record (pure-FID vs 3-stage vs briefing)
- `outputs/all_checkpoint_metrics.csv` — flat 264-row table of every (model, kimg) with FID, P, R, tier, composite score
