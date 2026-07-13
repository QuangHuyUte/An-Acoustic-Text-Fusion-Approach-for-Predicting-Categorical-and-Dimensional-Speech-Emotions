# Emotion2Vec-Guided Acoustic Cross-Attention Multi-Task SER

## 1. Mục tiêu mô hình

Mục tiêu của mô hình là nâng cấp hướng 06D hiện tại từ bài toán nhận diện cảm xúc đơn lẻ sang mô hình multi-task trên IEMOCAP:

- Task 1: phân loại cảm xúc, ví dụ neutral, angry, sad, happy/excited.
- Task 2: hồi quy các chiều cảm xúc Valence, Arousal, Dominance.
- Điều kiện đánh giá chính: speaker-independent, tránh phụ thuộc người nói.
- Hai protocol cần triển khai từ đầu:
  - 5-fold leave-one-session-out.
  - 10-fold speaker-independent, theo hướng 8 speakers train, 1 speaker validation, 1 speaker test.

Ý tưởng trung tâm: emotion2vec là backbone chính. Các đặc trưng acoustic như MFCC, log-Mel, prosody/statistical features không đứng ngang hàng để concat thô như 06D cũ, mà được dùng như tín hiệu bổ sung để điều hướng hoặc hiệu chỉnh representation từ emotion2vec.

## 2. Lý do cần đổi thiết kế từ 06D cũ

Trong 06D cũ, mô hình đã có nhiều đặc trưng:

- MFCC, delta MFCC, delta-delta MFCC.
- log-Mel spectrogram.
- RMS energy, ZCR, spectral centroid, bandwidth, rolloff, contrast.
- Chroma, tonnetz.
- Pitch/F0 và các statistical functionals như mean, std, min, max, median, IQR.
- Frozen emotion2vec embedding.

Tuy nhiên kết quả strict split cho thấy nhánh deep co-attention chưa thật sự tận dụng tốt các đặc trưng này. Mô hình emotion2vec MLP đơn giản còn có thể tốt hơn deep co-attention full. Điều đó cho thấy vấn đề chính không phải là thiếu thêm feature thủ công, mà là:

- Fusion hiện tại còn thô.
- Feature yếu có thể kéo feature mạnh xuống.
- Co-attention cũ chủ yếu tương tác temporal branch với spectral branch, trong khi emotion2vec lại chỉ được đưa vào như một nhánh MLP rồi concat cuối.
- Các auxiliary branch losses có thể ép các nhánh yếu tự classify, làm representation chung kém ổn định hơn trên strict speaker split.

Vì vậy mô hình mới cần đặt emotion2vec làm trung tâm, còn acoustic branches chỉ đóng vai trò hỗ trợ có kiểm soát.

## 3. Kiến trúc tổng quát đề xuất

Tên đề xuất:

```text
Emotion2Vec-Guided Acoustic Cross-Attention Multi-Task SER
```

Pipeline tổng quát:

```text
Audio waveform
  |
  |-- emotion2vec backbone
  |      -> emotion2vec tokens E
  |
  |-- acoustic branches
  |      |-- 1D CNN / BiLSTM branch từ MFCC, delta, delta-delta
  |      |-- 2D CNN branch từ log-Mel / log-Mel delta
  |      |-- prosody-temporal branch từ pitch, RMS, ZCR, speaking activity
  |      |-- statistical branch từ mean, std, min, max, median, IQR
  |      -> acoustic tokens A
  |
  |-- emotion2vec-guided acoustic cross-attention
  |      Q = emotion2vec tokens
  |      K = acoustic tokens
  |      V = acoustic tokens
  |
  |-- gated fusion
  |      z_shared = emotion2vec representation + gate * acoustic context
  |
  |-- task heads
         |-- emotion classification head
         |-- Valence/Arousal/Dominance regression head
```

## 4. Vai trò từng thành phần

### 4.1 Emotion2vec backbone

Emotion2vec nhận raw waveform làm đầu vào và tạo representation cảm xúc ở mức cao.

Đây là backbone chính vì:

- Đã được pretrained trên dữ liệu lớn hơn nhiều so với dataset hiện tại.
- Có khả năng học biểu diễn cảm xúc tốt hơn handcrafted features đơn lẻ.
- Trong kết quả 06D hiện tại, emotion2vec MLP là baseline mạnh và ổn định hơn deep fusion cũ.

Ở giai đoạn đầu, nên dùng emotion2vec ở chế độ frozen để giảm chi phí train và tránh overfit. Sau khi baseline ổn định, có thể thử partial fine-tune một vài layer cuối hoặc adapter fine-tuning.

### 4.2 Acoustic branches

Các branch cũ của 06D vẫn có thể dùng lại, nhưng đổi vai trò:

```text
Không dùng như các model phụ tự dự đoán ngang hàng.
Dùng như tín hiệu bổ sung để hỗ trợ emotion2vec.
```

Các branch đề xuất:

| Branch | Input | Encoder | Output |
|---|---|---|---|
| MFCC temporal branch | MFCC, delta, delta-delta | 1D CNN + BiLSTM/Transformer nhỏ | acoustic temporal tokens |
| Spectral branch | log-Mel, delta log-Mel, delta-delta log-Mel | 2D CNN / ResNet-SE nhỏ | spectral tokens |
| Prosody branch | pitch/F0, RMS, ZCR, voiced ratio, silence/pause proxy | 1D CNN hoặc MLP theo frame | prosody tokens |
| Statistical branch | mean, std, min, max, median, IQR | MLP | stats token hoặc gate vector |

Statistical branch nên dùng cẩn thận. Vì statistical features thường yếu trên strict speaker split, nên ưu tiên dùng nó để tạo gate thay vì ép nó tự classify.

## 5. Emotion2vec-guided cross-attention

Thành phần quan trọng nhất của mô hình mới là cross-attention giữa emotion2vec tokens và acoustic tokens.

Thiết kế đề xuất:

```text
E = emotion2vec tokens             # [B, T_e, d_e]
A = acoustic tokens                # [B, T_a, d_a]

E_proj = Linear(E)                 # [B, T_e, d]
A_proj = Linear(A)                 # [B, T_a, d]

C = MultiHeadAttention(
      query = E_proj,
      key   = A_proj,
      value = A_proj
    )

Z = LayerNorm(E_proj + C)
z_shared = AttentionPooling(Z)
```

Ý nghĩa:

- Emotion2vec giữ vai trò câu hỏi chính: đoạn audio này có biểu hiện cảm xúc gì?
- Acoustic features trả lời: vùng thời gian nào có pitch/energy/MFCC/log-Mel đáng chú ý?
- Cross-attention giúp emotion2vec nhìn lại các tín hiệu acoustic cụ thể thay vì concat thô toàn bộ feature.

Có thể thử thêm chiều attention ngược:

```text
Acoustic-to-emotion2vec:
  Q = acoustic tokens
  K = emotion2vec tokens
  V = emotion2vec tokens
```

Nhưng bản đầu nên làm một chiều emotion2vec-to-acoustic để giảm độ phức tạp.

## 6. Gated fusion

Sau cross-attention, không nên cộng hoặc concat acoustic context một cách cố định. Nên có gate để model tự quyết định mức độ tin vào acoustic branch.

Công thức đề xuất:

```text
z_e = AttentionPooling(E_proj)
z_c = AttentionPooling(C)
z_s = MLP(statistical_features)

gate = sigmoid(MLP([z_e, z_c, z_s]))
z_shared = LayerNorm(z_e + gate * z_c)
```

Nếu statistical feature không tốt ở một fold nào đó, gate có thể giảm ảnh hưởng của acoustic context. Đây là điểm an toàn hơn so với 06D cũ.

## 7. Hai task heads

### 7.1 Emotion classification head

Input:

```text
z_shared
```

Output:

```text
emotion logits: [B, num_classes]
```

Loss:

```text
L_emotion = CrossEntropyLoss
```

Có thể dùng class weight hoặc focal loss nếu class imbalance rõ ràng.

Metrics:

- WA / WAR.
- UA / UAR.
- Macro-F1.
- Weighted-F1.
- Confusion matrix.

### 7.2 AVD regression head

Input:

```text
z_shared
```

Output:

```text
[valence, arousal, dominance]
```

Nếu label IEMOCAP dùng scale 1-5, nên normalize về 0-1 trong training:

```text
y_norm = (y - 1) / 4
```

Loss chính:

```text
L_avd = CCCLoss(valence) + CCCLoss(arousal) + CCCLoss(dominance)
```

Có thể thêm SmoothL1 để ổn định:

```text
L_avd = L_ccc + alpha * SmoothL1
```

Gợi ý ban đầu:

```text
alpha = 0.2
```

Metrics:

- CCC cho Valence.
- CCC cho Arousal.
- CCC cho Dominance.
- MAE.
- RMSE.

## 8. Multi-task loss

Loss tổng:

```text
L_total = lambda_emotion * L_emotion + lambda_avd * L_avd
```

Thiết lập ban đầu:

```text
lambda_emotion = 1.0
lambda_avd = 1.0
```

Nếu emotion classification giảm quá mạnh vì regression, thử:

```text
lambda_emotion = 1.0
lambda_avd = 0.5
```

Nếu AVD regression quá yếu, thử:

```text
lambda_emotion = 0.7
lambda_avd = 1.0
```

Không nên bật auxiliary loss cho từng branch ở bản đầu. Chỉ tối ưu final emotion head và final AVD head. Sau khi baseline ổn định mới thử auxiliary losses.

## 9. Liên hệ với bài báo multi-task

Bài multi-task tham khảo dùng hướng:

```text
Audio -> HuBERT-large -> acoustic representation
Transcript -> DeBERTaV3-large -> linguistic representation
Self-attention theo từng modality
Cross-attention / bridge tokens
Shared emotional representation
Emotion classification head
Valence/Arousal regression head
```

Bài đó là multimodal, còn hướng hiện tại của đề tài là audio-first. Vì vậy ta không lấy nguyên kiến trúc HuBERT + DeBERTa, mà học ý tưởng:

- Có shared emotional representation.
- Có multi-task learning.
- Có classification head và regression head.
- Có cross-attention để kết hợp hai nguồn biểu diễn.
- Dùng CCC loss cho dimensional emotion regression.

Áp dụng vào đề tài:

```text
Thay HuBERT acoustic representation bằng emotion2vec.
Thay DeBERTa text representation bằng acoustic branches từ MFCC/log-Mel/prosody/statistics.
Giữ ý tưởng shared representation + 2 heads.
```

Nếu sau này có thời gian, transcript có thể là bonus, nhưng không phải trọng tâm v1.

## 10. Ablation cần chạy

Các ablation nên đi từ đơn giản đến phức tạp:

| Version | Mô hình | Mục đích |
|---|---|---|
| M0 | emotion2vec-only + 2 heads | Baseline sạch nhất |
| M1 | emotion2vec + statistical gate | Kiểm tra stats có giúp không |
| M2 | emotion2vec + MFCC/log-Mel concat | So sánh fusion thô |
| M3 | emotion2vec + gated acoustic fusion | Kiểm tra gate có giảm nhiễu không |
| M4 | emotion2vec-guided cross-attention | Kiểm tra cross-attention chính |
| M5 | cross-attention + gated fusion | Mô hình đề xuất đầy đủ |
| M6 | partial fine-tune emotion2vec | Chỉ thử nếu M5 ổn định |

Kết quả cần báo cáo riêng cho 5-fold và 10-fold. Không so sánh một kết quả random split với speaker-independent split.

## 11. Các notebook cần triển khai

### 11.1 IEMOCAP_Dataset_Analysis_and_Splits.ipynb

Mục tiêu:

- Đọc metadata IEMOCAP.
- Kiểm tra số lượng utterances.
- Kiểm tra emotion labels.
- Kiểm tra dimensional labels: Valence, Arousal, Dominance.
- Gộp happy và excited nếu dùng 4-class setting phổ biến.
- Phân tích class imbalance.
- Phân tích duration distribution.
- Phân tích số mẫu theo session, speaker, emotion.
- Tạo split speaker-independent.

Các phần cần có:

1. Load metadata.
2. Chuẩn hóa label emotion.
3. Chuẩn hóa AVD label.
4. EDA label distribution.
5. EDA duration.
6. Tạo 5-fold leave-one-session-out:

```text
Fold 1: test Session 1
Fold 2: test Session 2
...
Fold 5: test Session 5
```

7. Tạo 10-fold speaker-independent:

```text
Mỗi fold:
  8 speakers train
  1 speaker validation
  1 speaker test
```

8. Check speaker leakage.
9. Lưu split files:

```text
splits/iemocap_5fold_session.json
splits/iemocap_10fold_speaker.json
```

Output cần có:

- Bảng số lượng mẫu theo fold.
- Bảng số lượng mẫu theo emotion trong train/val/test.
- Biểu đồ duration.
- Biểu đồ phân bố Valence/Arousal/Dominance.
- File split JSON/CSV.

### 11.2 IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb

Mục tiêu:

- Trích xuất emotion2vec features.
- Trích xuất acoustic features dùng cho branch phụ.
- Lưu cache để các notebook train không phải extract lại.

Input:

```text
IEMOCAP wav files
metadata chuẩn hóa
split files
```

Preprocessing:

- Resample audio về 16 kHz.
- Convert mono.
- Trim silence nếu cần.
- Loudness normalize nếu phù hợp.
- Không crop mất thông tin một cách tùy tiện.
- Với utterance dài, ưu tiên segment + pooling thay vì chỉ lấy 3 giây đầu.

Features cần trích:

1. Emotion2vec:

```text
raw audio -> emotion2vec -> frame/timestep embeddings hoặc utterance embedding
```

2. MFCC temporal:

```text
MFCC
delta MFCC
delta-delta MFCC
```

3. Spectral:

```text
log-Mel
delta log-Mel
delta-delta log-Mel
```

4. Prosody:

```text
pitch/F0
RMS energy
ZCR
voiced ratio
silence ratio
speaking activity proxy
```

5. Statistical:

```text
mean
std
min
max
median
IQR
percentiles
```

Output cache đề xuất:

```text
features/iemocap_emotion2vec_tokens.npz
features/iemocap_acoustic_features.npz
features/iemocap_feature_manifest.csv
```

Yêu cầu kiểm tra chất lượng:

- Không có emotion2vec vector toàn 0.
- Không có NaN/Inf.
- Kiểm tra shape thống nhất.
- Kiểm tra audio path bị lỗi.
- Ghi log các file extract fail.

### 11.3 MultiTask_Emotion2Vec_CoAttention_5Fold.ipynb

Mục tiêu:

- Train và evaluate mô hình đề xuất trên 5-fold leave-one-session-out.
- Mỗi fold test một session chưa thấy.

Nội dung notebook:

1. Load feature cache.
2. Load `iemocap_5fold_session.json`.
3. Define Dataset/DataLoader.
4. Define model:

```text
Emotion2Vec encoder/loader
Acoustic branches
Emotion2vec-guided cross-attention
Gated fusion
Emotion classification head
AVD regression head
```

5. Define losses:

```text
CrossEntropyLoss cho emotion
CCCLoss cho V/A/D
SmoothL1 optional
Multi-task weighted loss
```

6. Train từng fold.
7. Early stopping theo validation:

```text
primary score = UAR + mean CCC
```

hoặc lưu riêng best classification và best regression nếu cần phân tích.

8. Evaluate test fold.
9. Save:

```text
predictions/fold_x_predictions.csv
models/fold_x_best.pt
reports/5fold_metrics.csv
reports/5fold_confusion_matrix.png
```

Metrics cần báo cáo:

- Emotion: WA, UAR, Macro-F1, Weighted-F1.
- AVD: CCC-V, CCC-A, CCC-D, mean CCC, MAE, RMSE.
- Mean và std qua 5 folds.

Các ablation nên chạy trong notebook hoặc notebook phụ:

- M0 emotion2vec-only.
- M3 gated acoustic fusion.
- M5 cross-attention + gated fusion.

### 11.4 MultiTask_Emotion2Vec_CoAttention_10Fold.ipynb

Mục tiêu:

- Train và evaluate mô hình đề xuất trên 10-fold speaker-independent.
- Đây là split khó và quan trọng nhất để chứng minh mô hình không phụ thuộc speaker.

Nội dung notebook:

1. Load feature cache.
2. Load `iemocap_10fold_speaker.json`.
3. Kiểm tra không có speaker leakage:

```text
train speakers ∩ validation speakers = empty
train speakers ∩ test speakers = empty
validation speakers ∩ test speakers = empty
```

4. Train từng fold với cùng kiến trúc như 5-fold.
5. Dùng validation speaker để chọn checkpoint.
6. Evaluate trên test speaker.
7. Save metrics và predictions.

Output:

```text
reports/10fold_metrics.csv
reports/10fold_summary.md
reports/10fold_confusion_matrix.png
predictions/10fold_all_predictions.csv
```

Metrics cần báo cáo:

- Emotion:
  - WA/WAR.
  - UA/UAR.
  - Macro-F1.
  - Confusion matrix.

- AVD:
  - CCC-Valence.
  - CCC-Arousal.
  - CCC-Dominance.
  - Mean CCC.
  - MAE.
  - RMSE.

Điểm cần chú ý:

- 10-fold speaker-independent có thể thấp hơn 5-fold, điều này bình thường.
- Không nên so trực tiếp với bài dùng random split.
- Nếu model cross-attention thấp hơn emotion2vec-only, cần xem lại gate/fusion chứ không kết luận acoustic feature vô dụng ngay.

## 12. Thứ tự triển khai khuyến nghị

Thứ tự làm thực tế:

1. Hoàn thành `IEMOCAP_Dataset_Analysis_and_Splits.ipynb`.
2. Hoàn thành `IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb`.
3. Train baseline M0 trên 5-fold.
4. Train baseline M0 trên 10-fold.
5. Thêm gated fusion M3 trên 5-fold.
6. Nếu M3 tốt hơn hoặc ngang M0, chạy M3 trên 10-fold.
7. Thêm cross-attention M4/M5.
8. Chạy ablation đầy đủ.
9. Nếu còn thời gian/GPU, thử partial fine-tune emotion2vec.
10. Làm demo inference một file audio mới.

## 13. Demo inference mong muốn

Demo nên nhận một audio mới và trả về:

```text
Predicted emotion:
  neutral / angry / sad / happy

Emotion probability:
  neutral: ...
  angry: ...
  sad: ...
  happy: ...

Dimensional emotion:
  Valence: ...
  Arousal: ...
  Dominance: ...
```

Nếu muốn giải thích thêm, có thể hiển thị:

- Vùng thời gian model chú ý nhiều nhất từ attention weights.
- Energy/pitch curve.
- Dự đoán theo segment nếu audio dài.

Nhưng phần explainability nên là bonus, không phải điều kiện bắt buộc cho model v1.

## 14. Kết luận định hướng

Mô hình đề xuất không bỏ công việc của 06D cũ. Ngược lại, nó giữ lại các branch 1D CNN, 2D CNN và statistical features, nhưng đổi vai trò của chúng:

```text
Từ: các nhánh ngang hàng rồi concat thô.
Sang: acoustic support signals giúp emotion2vec tạo representation tốt hơn.
```

Điểm cốt lõi cần bảo vệ trong báo cáo:

- Emotion2vec là pretrained emotional speech representation mạnh.
- Acoustic features vẫn có giá trị vì cung cấp tín hiệu cụ thể về pitch, energy, spectral shape, temporal dynamics.
- Cross-attention giúp kết hợp hai loại thông tin có kiểm soát.
- Multi-task learning giúp mô hình học cả categorical emotion và dimensional emotion.
- Speaker-independent 5-fold/10-fold giúp đánh giá nghiêm túc hơn random split.
