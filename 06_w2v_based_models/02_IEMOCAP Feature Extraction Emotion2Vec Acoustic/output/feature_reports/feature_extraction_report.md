# IEMOCAP Feature Extraction Report

## Scope

- Samples: **5,531**.
- Feature cache: `/kaggle/working/06_w2v_based_models/02_IEMOCAP Feature Extraction Emotion2Vec Acoustic/output/features/iemocap_full_06d_multibranch_cache.npz`.
- Sample rate: **16000 Hz**.
- Max seconds: **12.0**.
- Target frames: **601**.
- Truncation rate after fixed-length preprocessing: **3.78%**.

## Shapes

- `X_temporal`: `(5531, 135, 601)`.
- `X_spectral`: `(5531, 3, 96, 601)`.
- `X_stats`: `(5531, 1224)`.
- `X_e2v`: `(5531, 768)`.
- `X_text_stats`: `(5531, 6)`.

## Class Counts

- angry: **1,103**
- happy: **1,636**
- neutral: **1,708**
- sad: **1,084**

## Duration

- Mean: **4.549s**.
- Median: **3.576s**.
- 95th percentile: **11.059s**.
- Max: **34.139s**.

## Downstream Usage

- 03A raw pretrained backbone: use metadata/splits and raw audio.
- 03B acoustic co-attention: use `iemocap_full_06d_multibranch_cache.npz`.
- Transcript branch: use `text/text_ready_metadata.csv` and `text/text_folds_long.csv`.
- 04 fusion: align by `train_sample_id` and `utterance_id`.

## Important Note

No augmentation is applied in notebook 02. Any augmentation should be train-only inside model notebooks.