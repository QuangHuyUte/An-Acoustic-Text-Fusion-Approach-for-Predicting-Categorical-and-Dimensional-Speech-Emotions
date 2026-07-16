# Speech Emotion Studio - Presentation Feedback Demo

Web demo cho đồ án Speech Processing của Group 11. Demo tập trung vào speech emotion recognition và VAD regression, dùng mô hình 03B acoustic + pretrained speech + transcript text branch để giải thích cảm xúc, năng lượng và độ biểu cảm khi nói.

## Chạy Demo

```bash
cd "D:\UTE\Speech Programming\Speech Project\07_Web_Demo"
npm start
```

Hoặc double-click:

```text
start-demo.cmd
```

Sau đó mở:

```text
http://localhost:5174
```

Microphone và Speech-to-Text nên chạy qua `http://localhost:5174`. Không mở trực tiếp file HTML bằng `file://`.

## Backend Tăng Tốc

Demo hiện dùng service Python thường trú:

```text
backend/03b_service.py
```

Thay vì mỗi lần analyze lại mở Python và load model từ đầu, service này giữ sẵn Whisper ASR, FunASR emotion2vec, RoBERTa và checkpoint 03B trong RAM. Lần đầu có thể vẫn chậm do warm-up, nhưng các lần analyze sau sẽ nhanh hơn vì không phải load lại toàn bộ model.

Nếu service chưa sẵn sàng, `server.js` vẫn fallback sang `backend/03b_infer.py` dạng one-shot để demo không bị dừng.

## Workflow

1. Record hoặc upload audio.
2. Web chuyển audio thành WAV PCM16 base64.
3. Nếu frontend chưa có transcript, backend dùng local Whisper ASR để sinh transcript.
4. Audio được convert mono, resample 16 kHz, chia segment nếu dài.
5. Backend trích đặc trưng giống notebook 03B:
   - `X_temporal`: MFCC, delta, delta-delta, RMS, ZCR, spectral descriptors, F0, voiced flag.
   - `X_spectral`: log-Mel, delta log-Mel, delta-delta log-Mel.
   - `X_stats`: statistical functionals.
   - `X_e2v`: FunASR emotion2vec nếu có, nếu thiếu thì dùng wav2vec2 fallback.
6. Backend dùng scaler đã lưu từ train fold, không fit lại trên audio live.
7. Transcript đi qua tokenizer RoBERTa giống lúc train.
8. Model chạy `eval()` và `torch.no_grad()`.
9. Nếu audio có nhiều segment, emotion probabilities và VAD được aggregate bằng duration-weighted mean.

## Artifact Cần Có

Backend 03B dùng các file sau:

```text
models/checkpoints/03b/5fold_session/fold_1_test_Ses01_val_Ses02_acoustic_text_bridge_rmm_best.pt
models/scalers/03b_frozen_text_5fold_scalers.npz
models/docs/03B_acoustic_text_bridge_rmm_config.json
../pretrained_models/roberta-base-kaggle
../pretrained_models/whisper-tiny-en
../pretrained_models/wav2vec2-base-kaggle
../pretrained_models/funasr/models/iic--emotion2vec_base/snapshots/master
```

File registry tổng hợp:

```text
models/live_model_registry.json
```

## Task Prompt Theo IEMOCAP

Demo dùng 4 nhãn giống hướng IEMOCAP 4-class: `neutral`, `happy`, `sad`, `angry`.

Thay vì mỗi emotion có nội dung khác nhau, demo dùng cùng một bộ câu ngắn lặp lại cho mọi emotion. Người dùng thay đổi cách nói, không thay đổi nội dung. Cách này đúng hơn với mục tiêu Speech Emotion Recognition vì mô hình phải nghe prosody/acoustic emotion thay vì đoán emotion từ chữ.

## Kết Quả Trả Về

Endpoint chính:

```text
POST /api/analyze-03b
```

Backend trả:

- Emotion probabilities cho `neutral`, `happy`, `sad`, `angry`.
- Valence, Arousal, Dominance.
- Transcript do frontend gửi hoặc local Whisper ASR sinh ra.
- Segment diagnostics nếu audio dài.
- Debug feature shapes để kiểm tra pipeline có đi đúng nhánh không.
