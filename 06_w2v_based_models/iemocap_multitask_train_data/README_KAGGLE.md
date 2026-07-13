# Kaggle dataset: iemocap_multitask_train_data

Upload nguyên folder `iemocap_multitask_train_data` này lên Kaggle Dataset để chạy hai notebook train:

- `03_MultiTask_Emotion2Vec_CoAttention_5Fold.ipynb`
- `04_MultiTask_Emotion2Vec_CoAttention_10Fold.ipynb`

## Cấu trúc bắt buộc

- `metadata/iemocap_metadata_full.csv`
- `metadata/iemocap_4class_avd_metadata.csv`
- `splits/iemocap_5fold_session_long.csv`
- `splits/iemocap_10fold_speaker_long.csv`
- `splits/iemocap_5fold_session.json`
- `splits/iemocap_10fold_speaker.json`

Hai notebook train đã được chỉnh để tự tìm folder này trong `/kaggle/input/...`.

## Chưa bao gồm audio WAV

Folder này hiện chỉ chứa dữ liệu cần để chạy baseline multi-task trong notebook 03/04. Nó chưa chứa toàn bộ WAV vì WAV rất lớn và sẽ làm dataset upload nặng.

Khi chuyển sang train Emotion2Vec-CoAttention thật, cần thêm một trong hai hướng:

1. Upload thêm `audio_wav/` vào Kaggle Dataset này rồi chạy notebook 02 để trích feature.
2. Hoặc upload sẵn feature cache vào `features/`, ví dụ:
   - `features/iemocap_emotion2vec_tokens.npz`
   - `features/iemocap_acoustic_features.npz`

Nếu chỉ chạy bản sanity/baseline hiện tại của notebook 03/04 thì chưa cần upload audio WAV.
