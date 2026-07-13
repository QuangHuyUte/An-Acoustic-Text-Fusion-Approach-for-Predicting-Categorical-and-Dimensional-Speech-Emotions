# IEMOCAP training data inputs

Folder này gom các file đầu vào cần thiết để chạy notebook model `03` và `04`.

## Dùng ngay cho notebook 03/04

- `metadata/iemocap_metadata_full.csv`: metadata đầy đủ sau notebook 01, gồm emotion label, AVD score, speaker/session, audio path và các acoustic metadata cơ bản.
- `metadata/iemocap_4class_avd_metadata.csv`: subset 4 emotion chính kèm AVD.
- `splits/iemocap_5fold_session_long.csv`: split 5-fold theo session, dùng cho notebook `03_MultiTask_Emotion2Vec_CoAttention_5Fold.ipynb`.
- `splits/iemocap_10fold_speaker_long.csv`: split 10-fold theo speaker, dùng cho notebook `04_MultiTask_Emotion2Vec_CoAttention_10Fold.ipynb`.
- `splits/iemocap_5fold_session.json` và `splits/iemocap_10fold_speaker.json`: bản JSON của fold definition để kiểm tra/chia lại dữ liệu nếu cần.

## Dữ liệu vẫn cần cho bản Emotion2Vec đầy đủ

Notebook 03/04 hiện chạy được pipeline multi-task bằng metadata/acoustic baseline để kiểm tra split, metric, loss, prediction và report. Khi chuyển sang model Emotion2Vec-guided acoustic cross-attention thật, cần thêm:

- `../datasets/AbstractTTS_IEMOCAP/audio_wav/`: audio WAV gốc để trích embedding.
- `features/iemocap_emotion2vec_tokens.npz`: cache token/utterance embedding từ Emotion2Vec, sinh bởi notebook 02.
- `features/iemocap_acoustic_features.npz`: cache handcrafted acoustic features nếu dùng nhánh acoustic ngoài metadata.

Không copy toàn bộ WAV vào folder này để tránh nhân đôi dung lượng. WAV vẫn nằm trong `06_w2v_based_models/datasets/AbstractTTS_IEMOCAP/audio_wav/`.
