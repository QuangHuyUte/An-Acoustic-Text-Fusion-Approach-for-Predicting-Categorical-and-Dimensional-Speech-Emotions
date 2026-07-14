# IEMOCAP Full Release Dataset Analysis Summary

## Source

- Root: `/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release`
- Annotation source: `Session*/dialog/EmoEvaluation/*.txt`.
- Audio source: `Session*/sentences/wav/**/*.wav`.
- Transcript source: `Session*/dialog/transcriptions/*.txt`.

## Scope

- Full utterances parsed: **10,039**.
- 4-class train-ready utterances: **5,531**.
- Sessions: **5**.
- Speakers: **10**.
- Conversations: **151**.
- WAV found rate: **1.0000**.
- Transcript found rate: **1.0000**.

## 4-Class Label Rule

`neu -> neutral`, `ang -> angry`, `sad -> sad`, `hap + exc -> happy`.

Class counts:
- neutral: **1,708**
- angry: **1,103**
- sad: **1,084**
- happy: **1,636**

## Duration

- Mean: **4.549s**.
- Median: **3.576s**.
- 75th percentile: **5.773s**.
- 95th percentile: **11.059s**.
- Max: **34.139s**.

Fixed-length truncation risk:
- Longer than 3s: **60.01%**
- Longer than 4s: **43.21%**
- Longer than 5s: **31.75%**
- Longer than 6s: **23.45%**
- Longer than 8s: **12.60%**
- Longer than 10s: **6.67%**
- Longer than 15s: **1.41%**

## Valence / Arousal / Dominance

Labels are on the original IEMOCAP 1-5 scale. Notebook also stores normalized 0-1 columns.

- Mean valence: **2.907**.
- Mean arousal: **3.078**.
- Mean dominance: **3.171**.

| emotion_4class   |   valence |   arousal |   dominance |
|:-----------------|----------:|----------:|------------:|
| neutral          |     2.971 |     2.726 |       2.831 |
| angry            |     1.906 |     3.636 |       3.95  |
| sad              |     2.253 |     2.563 |       2.828 |
| happy            |     3.947 |     3.411 |       3.228 |

## Split Checks

- 5-fold rows: **27,655**.
- 10-fold rows: **55,310**.
- 5-fold leakage detected: **False**.
- 10-fold leakage detected: **False**.

## Output Files

- Metadata: `/kaggle/working/06_w2v_based_models/01_IEMOCAP Dataset Analysis and Speaker-Independent Splits/output/metadata`
- Splits: `/kaggle/working/06_w2v_based_models/01_IEMOCAP Dataset Analysis and Speaker-Independent Splits/output/splits`
- Reports: `/kaggle/working/06_w2v_based_models/01_IEMOCAP Dataset Analysis and Speaker-Independent Splits/output/reports`
- Figures: `/kaggle/working/06_w2v_based_models/01_IEMOCAP Dataset Analysis and Speaker-Independent Splits/output/figures`
- Shared data folder: `/kaggle/working/06_w2v_based_models/data`