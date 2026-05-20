# Do Synthetic Brain MRIs Reliably Improve Tumour Classification?
### A StyleGAN2-ADA Class-Plane Augmentation Study on BRISC 2025

**Author:** José Rafael Noriega Cedeño  
**Institution:** Faculty of Applied Sciences and Technology, Humber Polytechnic, Toronto, ON M9W 5L7  
**Paper:** [`arXiv/noriega2026.pdf`](arXiv/noriega2026.pdf) — source at [`arXiv/noriega2026.tex`](arXiv/noriega2026.tex)

---

## Abstract

We trained twelve class-plane StyleGAN2-ADA generators on constrained BRISC 2025 partitions and tested whether their output, with or without InceptionV3 feature-space filtering, improves held-out tumour classification across three classifier families: a random forest (RF) on InceptionV3 features, a compact two-headed CNN, and MobileViTV2. Each was evaluated at 1:1 and 1:2 real-to-synthetic ratios. An independent GPT-5.5 blind test placed gated real-versus-synthetic discrimination at 57.73% (95% CI: 54.48–60.92%) on the model-legible subset. The RF did not benefit from the synthetic MRIs. The CNN showed consistent mean gains that did not survive Holm correction. MobileViTV2 showed the clearest benefit: filtered 1:1 augmentation improved tumour classification accuracy by 1.02% absolute (95% CI: 0.54–1.54%; Holm-corrected *p* = 0.0104). Augmentation utility was found to be architecture- and ratio-dependent, not guaranteed by visual fidelity alone.

---

## Repository Structure

```
RA1/
├── arXiv/                          # Paper source and compiled PDF
│   ├── noriega2026.tex             # LuaLaTeX manuscript source
│   ├── noriega2026.pdf             # Compiled preprint
│   ├── fonts/ubuntu-mono/          # Ubuntu Mono font for monospace listings
│   └── svg-inkscape/               # SVG-converted PDF figures used in paper
│
├── figures/                        # Figure generation pipeline
│   ├── pgfplots/                   # PGFPlots figure sources (one folder per figure)
│   │   ├── pgfplots_font_setup.tex # Shared typography setup for all plots
│   │   ├── pgfplots_p001_p012_consolidated/  # StyleGAN2-ADA training curves
│   │   ├── pgfplots_p013/          # FID / Precision / Recall
│   │   ├── pgfplots_p014/          # RF aggregate metrics
│   │   ├── pgfplots_p015/          # RF confusion matrices
│   │   ├── pgfplots_p017/          # RF paired deltas
│   │   ├── pgfplots_p018/          # RF tree-growth OOB curves
│   │   ├── pgfplots_p019/          # CNN aggregate metrics
│   │   ├── pgfplots_p020_p021/     # CNN confusion matrices
│   │   ├── pgfplots_p022_p027/     # CNN/MobileViTV2 paired deltas
│   │   ├── pgfplots_p023/          # CNN training/validation curves
│   │   ├── pgfplots_p025_p026/     # MobileViTV2 confusion matrices
│   │   ├── pgfplots_p028/          # MobileViTV2 training/validation curves
│   │   └── pgfplots_p034_umap_class_v2/  # UMAP class embeddings
│   ├── arxiv_preprint_plot_registry/  # JSONL data registries for reproducibility
│   ├── mosaics/                    # Mosaic selection JSON for figure panels
│   ├── svg_embedded/               # SVG embedding utilities
│   ├── brisc_raw_vs_preprocessed_mosaic.tex
│   └── brisc_synthetic_mosaic.tex
│
├── brisc2025/                      # Raw BRISC 2025 dataset (train + test splits)
├── brisc2025_preprocessed/         # Preprocessed training images (class_plane folders)
├── brisc2025_test_preprocessed/    # Preprocessed test images (class folders)
├── data/                           # StyleGAN2-ADA training ZIPs (128×128, one per combo)
├── stylegan3/                      # NVIDIA StyleGAN3 codebase (StyleGAN2-ADA backend)
├── stylegan3_results/              # Per-combo GAN training artefacts and synthetic images
├── outputs_v2/                     # All downstream classifier outputs and audit files
│
├── preprocess_v2.py                # BRISC preprocessing pipeline (skull-strip → normalise)
├── train_stylegan2_ada.py          # GAN training orchestration
├── generate_synthetic_v2.py        # Synthetic image generation with quality/dedup gates
├── filter_synthetic_v2.py          # InceptionV3 feature-space diversity filtering (FPS)
├── audit_dataset_independence_v2.py  # Train/test leakage audit
├── audit_synthetic_artifacts_v2.py   # Real-vs-synthetic discriminator audit
├── classify_rf_v2.py               # Random Forest classifier (all stages)
├── classify_cnn_v2.py              # Two-headed CNN classifier
├── classify_mobilevitv2.py         # MobileViTV2 classifier
├── classify_v2.py                  # Compatibility entrypoint → classify_rf_v2.py
├── make_vlm_realism_audit_v2.py    # VLM realism audit setup
├── make_openai_vlm_batch_gpt55_xhigh.py        # GPT-5.5 batch submission (original)
├── make_openai_vlm_batch_gpt55_xhigh_compact.py # GPT-5.5 batch submission (compact)
├── make_plausibility_panels_v2.py  # Plausibility panel figure generation
│
├── StyleGAN2-ADA_Implementation_Optimized.ipynb  # Notebook implementation (original)
├── Checkpoint_Selection_Audit.md   # Audit trail: 3-stage checkpoint selection
├── Orientation_Audit.md            # Audit trail: per-combo MRI orientation verification
├── Instructions_Classifier_Pipeline.txt  # Collaborator briefing for downstream pipeline
├── all_checkpoint_metrics.csv      # Consolidated GAN checkpoint evaluation metrics
├── checkpoint_selection_audit.json # Machine-readable checkpoint selection record
└── inception-2015-12-05.pt         # InceptionV3 TorchScript weights (FID/Mahalanobis)
```

---

## Reproduction Pipeline

Run the following scripts in order. All scripts assume the working directory is the project root.

### Step 1 — Preprocess the BRISC 2025 images

```bash
# Preprocess training split (creates brisc2025_preprocessed/)
python preprocess_v2.py --dataset-type train

# Preprocess test split (creates brisc2025_test_preprocessed/)
python preprocess_v2.py --dataset-type test
```

### Step 2 — Train the StyleGAN2-ADA generators

```bash
# Trains all 12 class-plane models to 1,000 kimgs
python train_stylegan2_ada.py
```

A CUDA-capable GPU is required. Results are written to `stylegan3_results/`.

### Step 3 — Generate synthetic images

```bash
# Generate for all 12 combos (ratio=2.0 feeds both 1:1 and 1:2 experiments)
for combo in glioma_ax glioma_co glioma_sa \
             meningioma_ax meningioma_co meningioma_sa \
             no_tumor_ax no_tumor_co no_tumor_sa \
             pituitary_ax pituitary_co pituitary_sa; do
    python generate_synthetic_v2.py --combo $combo --ratio 2.0
done
```

### Step 4 — Extract InceptionV3 features and run the RF baseline

```bash
python classify_rf_v2.py --stage features
python classify_rf_v2.py --stage classify
```

### Step 5 — Diversity filter the synthetic pool

```bash
python filter_synthetic_v2.py --ratios 1.0 2.0
```

### Step 6 — Run augmented RF experiments

```bash
python classify_rf_v2.py --stage classify --condition augmented_r1
python classify_rf_v2.py --stage classify --condition filtered_r1
python classify_rf_v2.py --stage classify --condition augmented_r2
python classify_rf_v2.py --stage classify --condition filtered_r2
```

### Step 7 — Train the CNN and MobileViTV2 classifiers

```bash
python classify_cnn_v2.py
python classify_mobilevitv2.py
```

### Step 8 — Audit pipeline integrity

```bash
python audit_dataset_independence_v2.py
python audit_synthetic_artifacts_v2.py
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| Python ≥ 3.10 | Runtime |
| PyTorch ≥ 2.0 (CUDA) | GAN training, CNN/MobileViTV2, InceptionV3 inference |
| torchvision | Model utilities |
| scikit-learn | Random forest, PCA, LedoitWolf |
| scikit-image | Skull-stripping (Otsu, flood-fill) |
| scipy | Morphological closing |
| Pillow | Image I/O |
| imagehash | Perceptual hash deduplication |
| pandas, numpy | Data manipulation |
| matplotlib, umap-learn | Visualisation and UMAP embedding |
| openai | GPT-5.5 VLM realism batch API |

The `stylegan3/` directory must be on `sys.path` when loading GAN checkpoints (handled automatically by `generate_synthetic_v2.py` and `train_stylegan2_ada.py`).

`inception-2015-12-05.pt` (95 MB) is the FID-standard InceptionV3 TorchScript model. It is included in this repository for reproducibility. The original source is the [pytorch-fid](https://github.com/mseitzer/pytorch-fid) project.

---

## Outputs

All classifier results, manifests, and audit files are written to `outputs_v2/`. Key sub-directories:

| Path | Contents |
|------|----------|
| `outputs_v2/results/` | RF per-seed JSON results |
| `outputs_v2/results_cnn_v2/` | CNN per-seed JSON results |
| `outputs_v2/results_mobilevitv2/` | MobileViTV2 per-seed JSON results |
| `outputs_v2/manifests/` | Train/test image manifest CSVs |
| `outputs_v2/features/` | InceptionV3 pool3 feature arrays (.npy) |
| `outputs_v2/audits/` | Dataset independence audit outputs |
| `outputs_v2/vlm_realism_audits/` | GPT-5.5 realism audit responses |
| `outputs_v2/openai_batch/` | GPT-5.5 Batch API job files |

---

## Notes

- **Compile the paper** with LuaLaTeX: `lualatex -shell-escape noriega2026.tex` (requires Inkscape for SVG conversion; Fira Sans and Fira Math fonts must be installed).
- **Figures** can be regenerated from the build scripts in `figures/pgfplots/*/build_*.py`; they read from `outputs_v2/` and write standalone PGFPlots PDFs.
- `figures/pgfplots/pgfplots_p018/.tmp/` contains an empty stale Python temp directory (`tmpopmrpqtn`) that cannot be removed without elevated privileges on Windows; it has no effect on any pipeline step.
- The `data/` directory contains the 12 StyleGAN2-ADA training ZIPs (128×128, one per class-plane combination) consumed by `train_stylegan2_ada.py`.
