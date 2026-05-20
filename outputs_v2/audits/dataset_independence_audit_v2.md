# Dataset Independence Audit v2

Scope: preprocessed images used by the StyleGAN/classifier pipeline.

- Train rows: 4896
- Test rows: 1000
- Exact filepath overlap: 0
- Basename overlap: 0
- File SHA256 overlap: 0
- File SHA256 overlap pairs: 0
- Pixel-hash candidate pairs: 0
- Pixel-exact overlap pairs (pairwise train-test links): 0
- Pixel-exact unique train rows involved: 0
- Pixel-exact unique test rows involved: 0
- Pixel-exact identity clusters: 0
- pHash close neighbors, diagnostic only (<= 6): 356
- Minimum pHash distance: 0
- Feature-space close neighbors, diagnostic only (cosine <= 0.01): 5

Identity criterion:
- Image identity is tested on canonical uint8 RGB pixel arrays after loading preprocessed images the same way the CNN does.
- Pixel-hash matches are confirmed with exact np.array_equal comparison.
- Pair counts are many-to-many links; use the cluster CSV for unique pixel identities.
- pHash and Inception cosine distances describe proximity only; they are not used as exclusion criteria.
- Minimum feature cosine distance: 0.000658

Patient-level note:
- No patient/subject/study identifier column was found in the BRISC manifest.
- This audit can support image-level non-overlap only; it cannot prove patient-level independence.

CSV outputs:
- `train_test_phash_nearest_v2.csv`
- `train_test_feature_nearest_v2.csv`
- `train_test_exact_pixel_overlap_v2.csv`
- `train_test_exact_pixel_overlap_clusters_v2.csv`
