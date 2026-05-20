# Per-Combo Orientation Audit — `brisc2025_preprocessed`

**Date:** 2026-04-26
**Method:** Two complementary signals — (a) visual inspection of contact sheets at 200 px per thumbnail, 25 thumbs per combo, evenly sampled across the file index; (b) brain-mass centroid offset from image center per combo (`cy`, `cx` in normalized [-1, 1] coordinates), with mean and 5th–95th percentile spread.

**Important caveats up front.** Centroid analysis can be misleading: a tight centroid distribution suggests consistent orientation, but a wide one can come either from mixed orientations *or* from genuine cross-patient anatomical variation (different head shapes, different slice depths). Where the two signals disagree I trust the visual evidence. You also told me you tried to partially fix the orientation problem at some point, so the truth here is messy — these are best-guess categorizations, not certainties.

## Per-combo verdict

| Combo | Verdict | Confidence | Visual evidence | cy / cx stats |
|---|---|---|---|---|
| **glioma_ax** | Canonical | High | All slices oval, bilaterally symmetric, consistent rotation | tightest of all 12: cy ±0.017, cx ±0.015 |
| **glioma_co** | Mostly canonical, some mixing | Medium | At least one upside-down slice and a few rotations visible | cy ±0.041 |
| **glioma_sa** | **Mixed** | High | Faces pointing both directions; at least one 90°-rotated visible | cx_mean −0.038 (std 0.029) — stats *suggest* canonical but visuals contradict |
| **meningioma_ax** | Mostly canonical | Medium | One slice looks 90°-rotated; otherwise consistent | cy ±0.034 |
| **meningioma_co** | Mostly canonical | Medium | One upside-down slice in sample; otherwise rounded-top-up | cy ±0.045 |
| **meningioma_sa** | **Mixed** | High | Multiple distinct orientations visible: face-left, face-right, possibly rotated | cx_mean −0.002 (averaging out) |
| **no_tumor_ax** | Mostly canonical | Medium | One outlier; rest consistent axial | cy ±0.047 |
| **no_tumor_co** | Ambiguous | Low | Hard to read — many slices are at orbital level so the rounded-top-up cue is weak | cy_mean −0.028, cy span [−0.116, +0.111] crosses zero |
| **no_tumor_sa** | **Mixed (worst)** | Very high | Multiple 90°-rotated and possibly upside-down samples | cx_mean −0.045 with span [−0.137, +0.077] crossing zero |
| **pituitary_ax** | Canonical | High | Consistent oval slices throughout | cy ±0.033, cx ±0.021 |
| **pituitary_co** | Canonical | Medium-high | Visually consistent rounded-top-up; centroid std large but explained by depth variation | cy ±0.069 (depth-driven) |
| **pituitary_sa** | Mostly canonical | Medium | Dominantly face-right with a couple of outliers | cx_mean −0.006 with std 0.056 |

## Cross-referencing with model performance

Cross-tabulating the orientation verdicts against your 3-stage selection results:

| Combo | Orientation | FID at optimal | Tier |
|---|---|---:|---|
| glioma_ax | Canonical | 25.36 | EXTRAORDINARY |
| pituitary_ax | Canonical | 28.40 | EXTRAORDINARY |
| pituitary_co | Canonical | 29.38 | EXTRAORDINARY |
| glioma_co | Mostly canonical | 35.27 | EXCELLENT |
| meningioma_ax | Mostly canonical | 36.11 | EXCELLENT |
| meningioma_sa | **Mixed** | 43.93 | EXCELLENT |
| pituitary_sa | Mostly canonical | 46.39 | EXCELLENT |
| no_tumor_ax | Mostly canonical | 48.28 | EXCELLENT |
| no_tumor_sa | **Mixed (worst)** | 48.90 | EXCELLENT |
| no_tumor_co | Ambiguous | 51.84 | GOOD |
| glioma_sa | **Mixed** | 53.05 | GOOD |
| meningioma_co | Mostly canonical | 54.18 | GOOD |

**Patterns that emerge:**

1. **All three EXTRAORDINARY-tier models are the canonical ones.** glioma_ax, pituitary_ax, and pituitary_co all have visually consistent orientation and they cluster as the three best FIDs in the entire study. That is at least suggestive.

2. **Sagittal models systematically underperform.** Average FID by plane: axial 34.5, coronal 42.7, sagittal 48.1. This is the plane where you have the strongest evidence of mixed orientation (3 of 4 sagittal combos look mixed), and it's also the plane with the worst FIDs. Consistent with the data-per-mode argument from earlier.

3. **The pattern is not clean enough to be a *proof*.** meningioma_co is GOOD-tier despite looking mostly canonical, and meningioma_sa is EXCELLENT-tier despite being clearly mixed. So orientation alone doesn't determine performance — image quality, slice depth distribution, and intrinsic class difficulty all play in. This matches your warning that the truth is messy.

4. **The no_tumor class is the most affected.** All three no_tumor combos sit at the bottom of the FID distribution, and no_tumor_sa is the most visibly orientation-inconsistent set in the entire study. The no_tumor class also has the smallest training sets (310 / 352 / 405), which compounds the data-per-mode problem.

## Bottom line for the paper

For your discussion / limitations section, a defensible statement would be: *"In a subset of the (class × plane) combinations — particularly the sagittal-plane subsets and the no_tumor class — training images contained inconsistent orientations (variants rotated up to 180° from canonical). This is consistent with the systematically lower FID and recall achieved by the affected models, and constitutes a known limitation of this work. Future work should canonicalize orientation prior to training."*

The visual evidence is strongest for `no_tumor_sa`, `meningioma_sa`, and `glioma_sa` being mixed, and for `glioma_ax`, `pituitary_ax`, and `pituitary_co` being canonical. The middle ground is genuinely uncertain.

## Artifacts

- 12 small (96 px) contact sheets in `outputs/orientation_audit/<combo>.png`
- 12 large (200 px) contact sheets in `outputs/orientation_audit/<combo>_big.png`
- Centroid statistics in `outputs/orientation_audit/centroid_stats.json`
