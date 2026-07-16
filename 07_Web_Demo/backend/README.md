# 03B Web Inference Backend

Folder này chứa phần backend để web demo gọi mô hình 03B.

## Luồng Chạy Mới

1. Frontend encode audio thành WAV PCM16 base64.
2. `server.js` nhận payload tại `POST /api/analyze-03b`.
3. Node ưu tiên gọi service Python thường trú `03b_service.py`.
4. Service giữ sẵn RoBERTa, Whisper ASR, FunASR emotion2vec và checkpoint 03B trong RAM sau lần load đầu.
5. Python xử lý audio như test set: mono, 16 kHz, segmentation nếu dài, trích đặc trưng, áp dụng scaler đã lưu.
6. Nếu frontend không gửi transcript, backend dùng local Whisper ASR để sinh transcript.
7. Transcript được tokenize bằng RoBERTa giống lúc train.
8. Model 03B chạy `eval()` và `torch.no_grad()`.
9. Backend trả transcript, emotion, VAD, segment diagnostics và thông tin debug.

Nếu service Python không sẵn sàng, `server.js` vẫn fallback sang `03b_infer.py` dạng one-shot để demo không bị dừng.

## Service Thường Trú

File service:

```text
backend/03b_service.py
```

Endpoint nội bộ:

```text
GET  /health
POST /prewarm
POST /analyze
```

Biến môi trường hữu ích:

```text
USE_MODEL_03B_SERVICE=1
MODEL_03B_SERVICE_HOST=127.0.0.1
MODEL_03B_SERVICE_PORT=8765
MODEL_03B_TIMEOUT_MS=300000
MODEL_03B_PREWARM_ASR=1
MODEL_03B_PREWARM_E2V=1
```

`server.js` tự khởi động service khi chạy `npm start`, sau đó gọi prewarm nền. Vì vậy lần analyze đầu vẫn có thể chậm nếu model chưa warm-up xong, nhưng các lần sau không phải load lại Whisper/FunASR/RoBERTa/checkpoint nữa.

## Các Nhánh Đang Dùng

- Acoustic temporal branch: MFCC, delta MFCC, delta-delta MFCC, RMS, ZCR, spectral descriptors, F0, voiced flag.
- Acoustic spectral branch: log-Mel, delta log-Mel, delta-delta log-Mel.
- Acoustic statistics branch: statistical functionals của temporal/chroma/tonnetz.
- Pretrained speech branch: ưu tiên FunASR emotion2vec; nếu thiếu thì dùng local wav2vec2 fallback 768 chiều.
- ASR branch: local `whisper-tiny-en` sinh transcript khi upload audio hoặc browser STT không có.
- Text branch: local `roberta-base-kaggle`.
- Fusion: acoustic/text bridge + balanced fusion theo notebook 03B.
- Output heads: emotion classifier và VAD regressor.

## Segment Inference

Audio dài được chia thành các cửa sổ:

```text
LIVE_SEGMENT_SECONDS=12
LIVE_SEGMENT_OVERLAP_SECONDS=2
```

Mỗi segment chạy qua cùng pipeline inference. Kết quả cuối được aggregate bằng trung bình có trọng số theo thời lượng:

```text
final_probability = weighted_mean(segment_probabilities)
final_vad = weighted_mean(segment_vad)
```

Backend cũng trả `segments` để frontend hiển thị đoạn nào có tín hiệu yếu, ví dụ `low arousal`, `low valence`, `low dominance`, hoặc `uncertain emotion`.

## Scaler Và Chuẩn Hóa

Backend ưu tiên dùng scaler đã lưu:

```text
models/scalers/03b_frozen_text_5fold_scalers.npz
```

Khi demo chính thức, cần giữ scaler đã lưu từ train fold. Không fit lại scaler trên audio live, vì làm vậy sẽ lệch so với test set và không đúng protocol đánh giá.
