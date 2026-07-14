# IEMOCAP full multi-task data folder

Upload this `data` folder to Kaggle for the full pipeline.

## Folder structure

- `metadata/iemocap_metadata_full.csv`
- `metadata/iemocap_4class_avd_metadata.csv`
- `splits/iemocap_5fold_session_long.csv`
- `splits/iemocap_10fold_speaker_long.csv`
- `splits/iemocap_5fold_session.json`
- `splits/iemocap_10fold_speaker.json`
- `audio_wav/*.wav`
- `features/`

## How the full pipeline works

1. Run notebook `02_IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb`.
2. Notebook 02 reads `audio_wav/` and creates:

   `features/iemocap_full_emotion2vec_acoustic_cache.npz`

3. Run notebook `03_MultiTask_Emotion2Vec_CoAttention_5Fold.ipynb`.
4. Run notebook `04_MultiTask_Emotion2Vec_CoAttention_10Fold.ipynb`.

Notebook 03/04 now require the full cache. They no longer train the metadata-only sanity baseline by default.

## Full model input

The final model uses two real branches:

- Branch A: frozen Emotion2Vec utterance embedding.
- Branch B: handcrafted acoustic vector extracted from waveform.

The two branches are fused by bidirectional cross-attention and gated fusion, then split into:

- emotion classification head: neutral / angry / sad / happy
- AVD regression head: valence / arousal / dominance

## Kaggle note

If `funasr` and `modelscope` are not installed on Kaggle, turn Internet on and set:

`INSTALL_EMOTION2VEC_DEPS=1`

Notebook 02 will then install the required packages before loading `iic/emotion2vec_base`.
