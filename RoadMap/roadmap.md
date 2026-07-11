---
title: "Roadmap V1 - Nâng cấp 06D cho Speech Emotion Recognition độc lập người nói"
subtitle: "Tối ưu mô hình 06D, strict speaker-independent split, IEMOCAP, và mở rộng Valence/Arousal/Dominance"
date: "2026-07-11"
lang: "vi"
---

# Roadmap V1 - Nâng cấp 06D cho Speech Emotion Recognition độc lập người nói

## 0. Định hướng đã chốt

Định hướng hiện tại của đề tài không còn đi theo hướng xây dựng một hệ thống đánh giá bài thuyết trình đầy đủ như PuSQ/TED Talk rating nữa. Hướng đó có nhiều ý tưởng hay, nhưng quá rộng cho giai đoạn cuối kỳ vì cần dataset thuyết trình thật, nhãn chất lượng thuyết trình, transcript, audio dài, mô hình đánh giá presentation-level và cơ chế giải thích theo từng đoạn.

Hướng hiện tại được chốt lại như sau:

```text
Tiếp tục phát triển từ notebook 06D_Emotion2Vec_CoAttention_Full_SER.

Mục tiêu chính:
Nâng cấp mô hình Speech Emotion Recognition 06D để hoạt động tốt hơn
trong điều kiện strict speaker-independent.

Mục tiêu mở rộng:
Sau khi emotion classification ổn định, mở rộng sang IEMOCAP để dự đoán
Activation/Arousal, Valence và Dominance bằng regression.
```

Nói ngắn gọn:

- **Bài toán chính**: phân loại cảm xúc từ giọng nói.
- **Mô hình lõi**: 06D Emotion2Vec + Co-Attention.
- **Điểm cần cải thiện**: mô hình hiện tại có kiến trúc khá mạnh nhưng phần feature extraction và chiến lược strict split chưa đủ tốt.
- **Dataset trọng tâm mới**: IEMOCAP.
- **Tiêu chí đánh giá quan trọng nhất**: mô hình phải generalize tốt với người nói mới, không chỉ đạt điểm cao trên random split.
- **Demo cuối kỳ**: cho người dùng nhập audio, hệ thống chia đoạn, dự đoán cảm xúc từng đoạn và hiển thị timeline; nếu kịp thì thêm đường Arousal/Valence/Dominance.

Các hướng đã loại khỏi roadmap hiện tại:

| Hướng cũ | Trạng thái | Lý do loại khỏi scope chính |
|---|---|---|
| PuSQ reproduction | Không còn là hướng chính | Có giá trị tham khảo nhưng không còn phù hợp với bài toán emotion hiện tại |
| TED Talk rating prediction | Không còn là hướng chính | Dataset khó lấy, audio dài, nhãn rating không chỉ dựa trên speech |
| Full presentation feedback system | Chỉ giữ như ý tưởng tương lai | Cần nhiều signal khác ngoài emotion, vượt quá thời gian cuối kỳ |
| Long TED audio assessment | Không triển khai ở final | Chưa có nhãn đoạn nào hay/dở nên khó đánh giá khoa học |

## 1. Vấn đề hiện tại của mô hình 06D

Notebook 06D hiện tại đã có ý tưởng tốt: kết hợp representation từ emotion2vec với các nhánh đặc trưng âm thanh và cơ chế attention/fusion. Tuy nhiên kết quả đang cho thấy một vấn đề rất quan trọng:

| Protocol | Kết quả hiện tại | Ý nghĩa |
|---|---:|---|
| Random split | khoảng 80% | Điểm khá tốt, nhưng có khả năng bị ảnh hưởng bởi speaker overlap hoặc data distribution quá dễ |
| Strict speaker split | khoảng 69% | Giảm mạnh, cho thấy mô hình chưa generalize tốt sang người nói mới |

Điều này có nghĩa là mô hình có thể đang học một phần đặc trưng của người nói, thiết bị, cách phát âm, hoặc phân bố dataset, thay vì học đủ bền các tín hiệu cảm xúc trong giọng nói.

Vì vậy, điểm cần làm ở final không phải là cố đẩy random split lên cao hơn, mà là:

1. Chuẩn hóa lại cách chia dữ liệu để tránh speaker leakage.
2. Đọc kỹ các repo mạnh trên IEMOCAP xem họ xử lý feature, split và evaluation như thế nào.
3. Cải thiện phần feature extraction của 06D.
4. Chạy ablation để chứng minh thành phần nào thực sự giúp mô hình tốt hơn.
5. So sánh với các mô hình tham chiếu có paper/code rõ ràng.

## 2. Mục tiêu cuối kỳ

### 2.1 Mục tiêu chính

Xây dựng phiên bản cải tiến của mô hình 06D cho bài toán Speech Emotion Recognition trên điều kiện strict speaker-independent.

Output chính của mô hình:

```text
Input: audio utterance hoặc audio segment
Output: xác suất các emotion class
```

Với IEMOCAP, setup chuẩn nên dùng 4 lớp:

```text
angry
sad
neutral
happy + excited
```

Lý do gộp `happy` và `excited`: đây là cách rất nhiều paper trên IEMOCAP sử dụng vì số mẫu happy riêng thường ít, còn excited có sắc thái gần happy hơn so với các nhóm còn lại.

### 2.2 Mục tiêu mở rộng

Nếu emotion classification đã ổn định, mở rộng mô hình sang regression cho ba chiều cảm xúc:

| Chiều cảm xúc | Tên trong IEMOCAP | Ý nghĩa |
|---|---|---|
| Activation / Arousal | `EmoAct` | Mức năng lượng/kích hoạt của cảm xúc |
| Valence | `EmoVal` | Sắc thái tích cực hay tiêu cực |
| Dominance | `EmoDom` | Mức kiểm soát, tự tin, chủ động trong giọng nói |

Output mở rộng:

```text
Input: audio utterance hoặc audio segment
Output:
  - emotion class probabilities
  - arousal score
  - valence score
  - dominance score
```

Phần này không nên thay thế bài toán emotion classification chính. Nó là phần mở rộng để đề tài có chiều sâu hơn và bám sát IEMOCAP hơn.

## 3. Dataset chính: IEMOCAP

### 3.1 Link dataset

Các link cần ghi trong report:

- Official IEMOCAP: <https://sail.usc.edu/iemocap/>
- HuggingFace mirror user đã tìm được: <https://huggingface.co/datasets/AbstractTTS/IEMOCAP>

Lưu ý khi viết report: IEMOCAP bản official thường cần xin quyền truy cập; bản HuggingFace có thể dùng để kiểm tra nhanh cấu trúc dữ liệu, nhưng khi báo cáo cần nói rõ phiên bản nào được dùng trong thực nghiệm.

### 3.2 Cấu trúc dữ liệu cần quan tâm

IEMOCAP gồm các đoạn hội thoại/diễn xuất cảm xúc, có audio, transcript, label emotion và các nhãn dimensional emotion.

Thông tin quan trọng:

| Nội dung | Mô tả |
|---|---|
| Tổng số utterance | khoảng 10,039 utterance |
| Subset SER phổ biến | 5,531 utterance |
| Emotion labels thường dùng | angry, sad, neutral, happy + excited |
| Dimensional labels | activation/arousal, valence, dominance |
| Speaker/session | 5 session, 10 speaker |
| Protocol phổ biến | Leave-One-Session-Out 5-fold |
| Protocol strict hơn | Leave-One-Speaker-Out 10-fold |

### 3.3 Vì sao IEMOCAP phù hợp với hướng hiện tại

IEMOCAP phù hợp hơn RAVDESS/CREMA-D/SAVEE/TESS cho giai đoạn final vì:

- Có audio thật để train mô hình speech emotion.
- Có speaker/session rõ ràng để kiểm tra speaker-independent.
- Có emotion categorical label.
- Có `EmoAct`, `EmoVal`, `EmoDom` cho regression.
- Được dùng rất nhiều trong paper SER, nên dễ so sánh với các mô hình mạnh.
- Có nhiều benchmark về WA, UA, Macro-F1, CCC.

RAVDESS, CREMA-D, SAVEE, TESS vẫn có thể dùng làm dữ liệu phụ hoặc sanity check, nhưng không nên là trọng tâm cuối cùng nếu mục tiêu là so sánh với các mô hình SER hiện đại.

## 4. Cách chia train/test cần dùng

### 4.1 Không nên chỉ dùng random split

Random split dễ cho kết quả cao vì cùng một speaker có thể xuất hiện ở cả train và test. Khi đó mô hình có thể học các đặc điểm riêng của người nói thay vì cảm xúc.

Vì vậy, random split chỉ nên dùng để:

- kiểm tra pipeline có chạy không;
- so sánh với kết quả midterm;
- chứng minh rằng random split dễ hơn strict split.

Không nên lấy random split làm kết quả chính.

### 4.2 Protocol chính: Leave-One-Session-Out 5-fold

IEMOCAP có 5 session. Mỗi fold lấy 1 session làm test, 4 session còn lại làm train/validation.

Ví dụ:

| Fold | Test session | Train/validation |
|---|---|---|
| Fold 1 | Session 1 | Session 2, 3, 4, 5 |
| Fold 2 | Session 2 | Session 1, 3, 4, 5 |
| Fold 3 | Session 3 | Session 1, 2, 4, 5 |
| Fold 4 | Session 4 | Session 1, 2, 3, 5 |
| Fold 5 | Session 5 | Session 1, 2, 3, 4 |

Ưu điểm:

- Đây là protocol phổ biến trong nhiều paper IEMOCAP.
- Dễ giải thích trong report.
- Giảm nguy cơ speaker/session leakage.
- Phù hợp để so với emotion2vec, CA-MSER, FT-w2v2 và DST.

### 4.3 Protocol mở rộng: Leave-One-Speaker-Out 10-fold

Nếu còn thời gian, chạy thêm Leave-One-Speaker-Out. Mỗi fold giữ một speaker làm test.

Ưu điểm:

- Strict hơn LOSO.
- Kiểm tra khả năng generalize sang speaker mới rõ hơn.

Nhược điểm:

- Tốn thời gian train hơn.
- Có thể mất cân bằng class nhiều hơn.
- Cần kiểm soát kỹ validation split.

Trong final, bắt buộc nên có LOSO 5-fold. Leave-One-Speaker-Out là phần cộng thêm nếu đủ thời gian.

## 5. Các metric cần báo cáo

### 5.1 Emotion classification

Với emotion classification, dùng:

| Metric | Ý nghĩa |
|---|---|
| WA / Weighted Accuracy | Accuracy có xét phân bố mẫu, thường tương đương accuracy tổng |
| UA / Unweighted Accuracy | Trung bình recall theo từng class, quan trọng khi class imbalance |
| Macro-F1 | Trung bình F1 của từng class, không để class lớn lấn át |
| Weighted-F1 | F1 có trọng số theo số mẫu từng class |
| Confusion matrix | Cho biết class nào hay bị nhầm với class nào |

Metric chính nên ưu tiên:

```text
UA + Macro-F1
```

Lý do: IEMOCAP bị lệch số mẫu giữa các emotion class, nên chỉ nhìn accuracy có thể gây hiểu nhầm.

### 5.2 Arousal/Valence/Dominance regression

Với Arousal, Valence, Dominance, không dùng accuracy nếu giữ nhãn dạng continuous score.

Nên dùng:

| Metric | Ý nghĩa |
|---|---|
| CCC | Concordance Correlation Coefficient, đo mức đồng thuận giữa dự đoán và nhãn người đánh giá |
| MAE | Sai số tuyệt đối trung bình |
| RMSE/MSE | Phạt mạnh các lỗi dự đoán lớn |

CCC phù hợp hơn correlation thông thường vì nó xét cả:

- xu hướng tương quan;
- độ lệch trung bình;
- độ lệch scale giữa prediction và ground truth.

Giá trị CCC càng gần 1 càng tốt.

## 6. Arousal, Valence, Dominance được đo như thế nào?

Trong IEMOCAP, các chiều `activation`, `valence`, `dominance` là nhãn do người annotator đánh giá. Thông thường, mỗi utterance được chấm trên thang liên tục hoặc bán liên tục, thường quy về khoảng 1-5.

| Dimension | Ý nghĩa | Điểm thấp | Điểm cao |
|---|---|---|---|
| Arousal / Activation | Mức năng lượng cảm xúc | bình tĩnh, mệt, ít năng lượng | kích động, phấn khích, giận dữ, nhiều năng lượng |
| Valence | Mức tích cực/tiêu cực | buồn, khó chịu, tiêu cực | vui, dễ chịu, tích cực |
| Dominance | Mức kiểm soát/chủ động | yếu, bị động, thiếu tự tin | mạnh, kiểm soát tốt, tự tin |

Ví dụ trực giác:

| Trạng thái | Arousal | Valence | Dominance |
|---|---:|---:|---:|
| Buồn, mệt | thấp | thấp | thấp/trung bình |
| Vui, hào hứng | cao | cao | trung bình/cao |
| Giận dữ | cao | thấp | cao |
| Bình thường | thấp/trung bình | trung bình | trung bình |

Điểm quan trọng: Arousal/Valence/Dominance không phải do mô hình tự nghĩ ra. Chúng là nhãn do con người đánh giá trong dataset. Mô hình chỉ học cách dự đoán các điểm đó từ audio.

## 7. Mô hình 06D hiện tại và vấn đề cần cải thiện

### 7.1 Ý tưởng tốt của 06D

06D có nhiều điểm mạnh:

- Dùng emotion2vec, phù hợp với speech emotion.
- Có ý tưởng multi-branch, không chỉ dùng một loại feature.
- Có co-attention/fusion, tốt hơn việc nối feature thô.
- Có thể mở rộng thành multi-task: emotion classification + AVD regression.
- Có thể dùng cho demo segment-level vì input audio có thể chia thành các đoạn nhỏ.

### 7.2 Điểm yếu hiện tại

Các vấn đề cần kiểm tra và cải thiện:

1. **Feature extraction chưa đủ mạnh hoặc chưa chuẩn hóa tốt**
   - Cần kiểm tra sampling rate, duration, padding/truncation.
   - Cần kiểm tra MFCC/log-mel có cùng shape, normalization và cache ổn định không.
   - Cần kiểm tra emotion2vec embedding lấy ở frame-level hay utterance-level.

2. **Fusion có thể chưa tận dụng tốt từng nhánh**
   - Nếu chỉ concat nhiều feature, mô hình dễ overfit.
   - Co-attention cần chứng minh bằng ablation.
   - Cần kiểm tra nhánh nào thực sự đóng góp.

3. **Strict split làm performance giảm**
   - Đây là vấn đề chính.
   - Cần kiểm tra có speaker leakage không.
   - Cần báo cáo kết quả theo từng fold.

4. **Chưa có baseline mạnh cùng protocol**
   - Cần so với emotion2vec linear, CA-MSER, FT-w2v2 P-TAPT, DST.
   - Nếu không reproduce hết được, vẫn cần bảng tham chiếu từ paper và mô tả protocol.

5. **Chưa có kết quả AVD**
   - Nếu mở rộng sang IEMOCAP, cần thêm head regression và metric CCC/MAE.

## 8. Kiến trúc đề xuất cho 06D cải tiến

### 8.1 Pipeline tổng thể

```text
Audio input
-> resample 16 kHz
-> trim/pad hoặc segment
-> feature extraction
   -> emotion2vec embedding
   -> MFCC sequence
   -> log-Mel / spectrogram
   -> optional handcrafted statistics
-> branch encoders
-> co-attention / gated fusion
-> shared representation
-> emotion classification head
-> optional AVD regression head
```

### 8.2 Nhánh emotion2vec

Vai trò:

- Là representation chính vì emotion2vec được pretrain cho speech emotion.
- Có thể dùng như frozen feature extractor trước, sau đó thử fine-tune nhẹ nếu đủ tài nguyên.

Thiết kế đề xuất:

```text
audio
-> emotion2vec
-> frame-level hoặc utterance-level embedding
-> adapter MLP / projection layer
-> shared fusion space
```

Ablation cần có:

- emotion2vec frozen + linear classifier;
- emotion2vec frozen + MLP;
- emotion2vec + co-attention với acoustic feature.

### 8.3 Nhánh MFCC

Vai trò:

- Bắt các đặc trưng âm học truyền thống.
- Có ích với các dataset nhỏ.
- Dễ so sánh với TIM-Net/CA-MSER.

Thiết kế đề xuất:

```text
audio
-> MFCC + delta + delta-delta
-> normalization
-> 1D CNN hoặc BiLSTM/GRU
-> attention pooling
```

Điểm cần kiểm tra:

- số coefficient MFCC;
- frame length/hop length;
- có dùng delta/delta-delta không;
- normalization theo utterance hay theo train set.

### 8.4 Nhánh log-Mel/spectrogram

Vai trò:

- Biểu diễn năng lượng theo thời gian-tần số.
- Phù hợp với CNN/ResNet nhỏ.
- Dùng để bổ sung thông tin spectral mà emotion2vec có thể chưa khai thác hết.

Thiết kế đề xuất:

```text
audio
-> log-Mel spectrogram
-> 2D CNN / small ResNet
-> pooling
-> projection
```

Ablation:

- log-Mel only;
- emotion2vec + log-Mel;
- emotion2vec + MFCC + log-Mel.

### 8.5 Nhánh handcrafted statistics

Vai trò:

- Bổ sung các đặc trưng ổn định ở mức utterance.
- Giúp report dễ giải thích hơn.

Có thể dùng:

| Feature | Ý nghĩa |
|---|---|
| RMS energy | mức năng lượng giọng nói |
| Zero-crossing rate | đặc trưng thô về tín hiệu |
| spectral centroid | độ sáng của âm thanh |
| spectral bandwidth | độ rộng phổ |
| pitch/F0 statistics | cao độ trung bình, biến thiên cao độ |
| silence ratio | tỷ lệ im lặng |
| duration | độ dài utterance/segment |

Lưu ý: các handcrafted features không nên là phần chính để đạt SOTA, nhưng rất hữu ích cho giải thích và demo.

### 8.6 Fusion

Không nên chỉ nối tất cả feature rồi đưa vào MLP. Cần có ít nhất một cơ chế fusion có chủ đích.

Các hướng có thể thử:

| Fusion | Mô tả | Ưu tiên |
|---|---|---|
| Concatenation | baseline đơn giản | bắt buộc để so sánh |
| Gated fusion | học trọng số cho từng nhánh | nên làm |
| Co-attention | nhánh SSL và acoustic feature tương tác với nhau | hướng chính của 06D |
| Validation-weighted ensemble | kết hợp prediction theo hiệu quả validation | optional |

Trong report, cần chứng minh co-attention có ích bằng ablation:

```text
emotion2vec only
MFCC only
log-Mel only
concat fusion
gated fusion
co-attention fusion
```

## 9. Mở rộng sang Arousal/Valence/Dominance

### 9.1 Khi nào làm phần này?

Chỉ làm AVD sau khi:

- pipeline IEMOCAP chạy ổn;
- LOSO split không bị lỗi;
- emotion classification có kết quả đủ tin cậy;
- đã có baseline emotion2vec/06D rõ ràng.

Nếu emotion classification chưa ổn, không nên vội thêm AVD vì sẽ làm project loãng.

### 9.2 Kiến trúc multi-task đề xuất

```text
shared 06D encoder
-> emotion head:
      softmax over 4 emotion classes
      loss: CrossEntropyLoss
-> AVD head:
      3 regression outputs: EmoAct, EmoVal, EmoDom
      loss: MSELoss hoặc CCCLoss
```

Loss tổng:

```text
total_loss = CE_emotion + lambda * AVD_loss
```

Trong đó:

- `CE_emotion`: loss cho emotion classification.
- `AVD_loss`: MSE hoặc CCC loss cho 3 điểm AVD.
- `lambda`: trọng số cân bằng hai task.

### 9.3 Cách báo cáo AVD

Không trộn metric emotion và AVD vào cùng một cột.

Bảng emotion:

| Model | WA | UA | Macro-F1 |
|---|---:|---:|---:|
| 06D baseline | ... | ... | ... |
| 06D improved | ... | ... | ... |

Bảng AVD:

| Model | CCC-Arousal | CCC-Valence | CCC-Dominance | MAE |
|---|---:|---:|---:|---:|
| 06D + AVD head | ... | ... | ... | ... |

Lý do: emotion classification và AVD regression là hai bài toán khác nhau.

## 10. Các mô hình tham chiếu chính

Các paper và repo đã được gom vào thư mục:

```text
Papers/Papers for main models emotion classification and valence arousal, dominance regression
```

### 10.1 Bảng model emotion classification

| Model | Paper / repo | Input | Kiến trúc | Split | Kết quả báo cáo | Vai trò với đề tài |
|---|---|---|---|---|---|---|
| emotion2vec | Paper: <https://arxiv.org/abs/2312.15185>; Repo: <https://github.com/ddlBoJack/emotion2vec> | raw speech -> emotion2vec embedding | self-supervised emotion representation + downstream classifier | IEMOCAP 4-class, LOSO/LOSpeaker | WA khoảng 71.79-72.94; một số protocol cao hơn | Backbone/feature chính để nâng 06D |
| FT-w2v2-ser P-TAPT | Paper: <https://arxiv.org/abs/2110.06309>; Repo: <https://github.com/b04901014/FT-w2v2-ser> | raw waveform | Wav2Vec2 fine-tuning + task adaptive pretraining | LOSO 5-fold | UA 74.3 | Baseline audio-only mạnh |
| CA-MSER | Paper: <https://arxiv.org/abs/2203.15326>; Repo: <https://github.com/Vincent-ZHQ/CA-MSER> | MFCC, spectrogram, wav2vec2 | multi-level acoustic features + co-attention | LOSO/LOSpeaker | LOSO WA 69.80, UA 71.05; LOSpeaker WA 71.64, UA 72.70 | Gần nhất với ý tưởng co-attention của 06D |
| DST | Paper: <https://arxiv.org/abs/2302.13729>; Repo: <https://github.com/HappyColor/DST> | WavLM features | Deformable Speech Transformer | LOSO | WA 71.8, UA 73.6 | Baseline transformer mạnh |
| TIM-Net | Paper: <https://arxiv.org/abs/2211.08233>; Repo: <https://github.com/Jiaxin-Ye/TIM-Net_SER> | MFCC | temporal-aware multi-scale dilated CNN | 10-fold CV | UAR/WAR khoảng 69-72.5 tùy setup | Baseline nhẹ, hữu ích khi compute hạn chế |
| SpeechFormer | Paper: <https://arxiv.org/abs/2203.03812>; Repo: <https://github.com/HappyColor/SpeechFormer> | spectrogram/log-Mel/wav2vec | hierarchical transformer | LOSO | không phải mạnh nhất trên IEMOCAP trong bảng hiện tại | Tham khảo kiến trúc phân cấp |

### 10.2 Bảng model AVD regression/classification

| Model/paper | Paper link | Code/repo | Input | Task | Split | Kết quả | Vai trò |
|---|---|---|---|---|---|---|---|
| Contrastive unsupervised learning / preCPC | <https://arxiv.org/abs/2102.06357> | Chưa tìm thấy official repo | raw audio/CPC | continuous AVD regression | IEMOCAP 5-fold | CCC act 0.752, val 0.752, dom 0.691 | Mốc tham chiếu mạnh cho AVD |
| Cross-modal conditional teacher-student | <https://arxiv.org/abs/2112.00158> | Chưa tìm thấy official repo | HuBERT audio, BERT text teacher | audio-only AVD regression student | 5-fold speaker-independent | CCC act 0.667, val 0.582, dom 0.545 | Tham khảo nếu muốn dùng transcript nhẹ |
| Attention-augmented multi-task learning | <https://arxiv.org/abs/1903.12424> | Chưa tìm thấy official repo | raw audio/eGeMAPS | AVD low/mid/high classification | Session 1-3 train, Session 4 dev, Session 5 test | UAR arousal 48.5, valence 63.8, dominance 51.6; không phải CCC/MAE vì bài này discretize nhãn | Tham khảo multi-task AVD |
| AV from categorical emotion labels | <https://arxiv.org/abs/2311.14816> | Chưa tìm thấy official repo | WavLM | arousal/valence từ categorical emotion | speaker-independent | valence CCC 0.529-0.566, arousal CCC 0.632-0.672; không có dominance | Rất liên quan nếu muốn suy ra AV từ emotion classifier |

Hiện tại các paper AVD/CCC đã được tải PDF về thư mục `papers/`, nhưng chưa có repo code chính thức để clone. Vì vậy, phần AVD nên được đặt là hướng mở rộng dựa trên ý tưởng và metric của paper, không nên hứa reproduce đầy đủ các paper CCC/MAE này.

## 11. Ta nên tham khảo mô hình nào nhất?

Ưu tiên thực nghiệm nên là:

1. **emotion2vec**
   - Gần nhất với 06D.
   - Có repo.
   - Có thể dùng làm feature extractor/backbone.
   - Dễ tích hợp vào pipeline hiện tại.

2. **CA-MSER**
   - Gần nhất với ý tưởng co-attention/multi-feature.
   - Có MFCC, spectrogram, wav2vec2.
   - Có code repo.
   - Hữu ích để học cách họ xử lý feature và fusion.

3. **FT-w2v2-ser P-TAPT**
   - Baseline mạnh audio-only.
   - Có repo.
   - Dùng để đối chiếu xem 06D improved có cạnh tranh được không.

4. **DST**
   - Kết quả tốt, có code.
   - Nhưng kiến trúc transformer phức tạp hơn, nên dùng làm reference hơn là tái hiện đầy đủ nếu thời gian ít.

Các model còn lại như TIM-Net, SpeechFormer dùng làm bổ sung trong related work hoặc baseline nhẹ.

## 12. Kế hoạch thực nghiệm chi tiết

### 12.1 Giai đoạn 1 - Audit lại 06D hiện tại

Mục tiêu:

```text
Biết chính xác 06D hiện tại đang đạt bao nhiêu,
trên split nào,
với dataset nào,
và có speaker leakage hay không.
```

Việc cần làm:

| Việc | Output |
|---|---|
| Kiểm tra lại notebook 06D | ghi rõ input feature, model architecture, loss, optimizer |
| Chạy lại random split | xác nhận khoảng 80% |
| Chạy lại strict split | xác nhận khoảng 69% |
| Lưu metrics per fold | CSV kết quả |
| Lưu confusion matrix | hình cho report |
| Kiểm tra speaker distribution | bảng speaker/session/class |

Kết luận cần viết trong report:

```text
Random split cho kết quả cao hơn nhưng chưa đủ thuyết phục.
Strict speaker-independent split phản ánh khả năng generalization thực tế hơn.
```

### 12.2 Giai đoạn 2 - Chuẩn hóa IEMOCAP pipeline

Mục tiêu:

```text
Tạo pipeline IEMOCAP sạch, có manifest, có fold, có label rõ ràng.
```

Việc cần làm:

| Việc | Output |
|---|---|
| Tạo manifest | audio_path, session, speaker, emotion, EmoAct, EmoVal, EmoDom |
| Chọn 4-class subset | angry, sad, neutral, happy+excited |
| Gộp happy/excited | mapping label rõ ràng |
| Tạo LOSO 5-fold | fold_id cho từng sample |
| Kiểm tra class distribution | bảng số mẫu từng class/fold |
| Chuẩn hóa audio | resample 16 kHz, mono |

Nếu dùng HuggingFace mirror, cần ghi rõ:

- tên dataset;
- cột nào dùng làm audio;
- cột nào dùng làm label;
- có khác gì so với official IEMOCAP hay không.

### 12.3 Giai đoạn 3 - Học cách các repo mạnh trích xuất feature

Mục tiêu:

```text
Không tự đoán feature nữa, mà đọc các repo/paper tốt rồi bắt chước phần hợp lý.
```

Cần đọc:

| Repo/paper | Cần học |
|---|---|
| emotion2vec | cách lấy embedding, pooling, downstream classifier |
| CA-MSER | cách kết hợp MFCC, spectrogram, wav2vec2 và co-attention |
| FT-w2v2-ser | cách chuẩn bị IEMOCAP split và fine-tune SSL model |
| DST | cách dùng WavLM feature và transformer attention |
| TIM-Net | cách dùng MFCC sequence cho model nhẹ |

Output của giai đoạn này:

- bảng feature của từng repo;
- mô tả shape feature;
- quyết định feature nào đưa vào 06D improved;
- quyết định feature nào chỉ dùng làm baseline/reference.

### 12.4 Giai đoạn 4 - Nâng cấp feature extraction của 06D

Feature đề xuất:

| Nhóm feature | Bắt buộc? | Lý do |
|---|---:|---|
| emotion2vec embedding | Có | backbone chính |
| MFCC + delta + delta-delta | Có | acoustic low-level, dễ so sánh CA-MSER/TIM-Net |
| log-Mel spectrogram | Có | spectral representation cho CNN |
| handcrafted statistics | Nên có | giúp giải thích và demo |
| WavLM/wav2vec2 | Optional | mạnh nhưng tăng compute |

Các ablation cần chạy:

| Thí nghiệm | Mục đích |
|---|---|
| 06D old feature | baseline nội bộ |
| emotion2vec only | xem backbone mạnh đến đâu |
| MFCC only | xem acoustic feature truyền thống |
| log-Mel only | xem spectral branch |
| emotion2vec + MFCC | xem SSL + acoustic |
| emotion2vec + log-Mel | xem SSL + spectral |
| all feature concat | baseline fusion đơn giản |
| all feature co-attention | kiểm tra đóng góp của 06D proposed |

### 12.5 Giai đoạn 5 - Tối ưu model

Các yếu tố cần thử:

| Thành phần | Hướng thử |
|---|---|
| Optimizer | AdamW |
| Learning rate | thử nhiều mức nhỏ, ví dụ 1e-5, 3e-5, 1e-4 tùy phần fine-tune |
| Dropout | 0.2-0.5 |
| Loss | CrossEntropy, class-weighted CE, focal loss |
| Early stopping | theo validation UA hoặc Macro-F1 |
| Batch size | tùy GPU/RAM |
| Seed | chạy nhiều seed nếu kịp |
| Freeze/unfreeze | frozen emotion2vec trước, fine-tune nhẹ sau |

Không nên thay đổi quá nhiều thứ cùng lúc. Mỗi thay đổi cần có kết quả ablation để biết nó giúp hay làm hại.

### 12.6 Giai đoạn 6 - So sánh với baseline/reference

Bảng chính trong report nên có dạng:

| Model | Speaker-independent? | Split | Input | WA | UA | Macro-F1 |
|---|---:|---|---|---:|---:|---:|
| emotion2vec linear | Có | LOSO/LOSpeaker | audio | khoảng 71.79-72.94 | khoảng 72.69 | paper báo WF1 |
| CA-MSER | Có | LOSO/LOSpeaker | audio | 69.80 / 71.64 | 71.05 / 72.70 | không báo |
| FT-w2v2 P-TAPT | Có | LOSO 5-fold | audio | không báo | 74.3 | không báo |
| DST | Có | LOSO 5-fold | audio | 71.8 | 73.6 | không báo |
| 06D current | Có | strict split | audio | khoảng 69 | cần đo | cần đo |
| 06D improved | Có | LOSO 5-fold | audio | cần đo | cần đo | cần đo |

Khi viết nhận xét:

- Nếu 06D improved chưa vượt SOTA, vẫn ổn nếu strict protocol rõ ràng và ablation tốt.
- Cần nhấn mạnh đóng góp là cải thiện từ 06D hiện tại, không phải tuyên bố SOTA.
- Nếu kết quả tăng từ strict 69% lên gần nhóm 71-74% thì rất tốt cho final.

### 12.7 Giai đoạn 7 - Mở rộng AVD nếu còn thời gian

Việc cần làm:

| Việc | Output |
|---|---|
| Đọc `EmoAct`, `EmoVal`, `EmoDom` | kiểm tra scale và missing value |
| Chuẩn hóa label | có thể normalize về 0-1 hoặc giữ 1-5 |
| Thêm regression head | 3 output |
| Train multi-task hoặc train riêng | so sánh nếu kịp |
| Báo CCC/MAE | bảng kết quả |

Nếu thời gian ít, cách an toàn:

```text
Giữ encoder 06D/emotion2vec frozen
Train một regression head nhỏ cho EmoAct, EmoVal, EmoDom
Report CCC và MAE
```

## 13. Demo cuối kỳ

Demo nên nhỏ, rõ và đúng với năng lực mô hình. Không nên demo như một hệ thống chấm điểm bài thuyết trình hoàn chỉnh.

### 13.1 Input demo

Demo có thể nhận:

- một file WAV/MP3 ngắn;
- hoặc một sample từ IEMOCAP;
- hoặc một audio người dùng tự thu.

Nếu audio dài, chia thành segment 3-5 giây.

### 13.2 Xử lý demo

```text
audio input
-> resample 16 kHz
-> split thành segment
-> extract feature
-> 06D improved prediction
-> aggregate theo segment
```

### 13.3 Output demo

Output tối thiểu:

- emotion dự đoán cho toàn audio;
- confidence score;
- emotion timeline theo từng segment;
- biểu đồ xác suất emotion;
- nếu là test sample thì hiển thị ground truth và prediction.

Output mở rộng:

- arousal curve;
- valence curve;
- dominance curve;
- biểu đồ segment nào có cảm xúc mạnh/yếu.

### 13.4 Điều cần tránh khi demo

Không nên nói:

```text
Hệ thống đánh giá bài thuyết trình hay/dở.
```

Nên nói:

```text
Hệ thống phân tích tín hiệu cảm xúc trong giọng nói.
Kết quả emotion/arousal/valence/dominance có thể là một module nền
cho hệ thống feedback thuyết trình trong tương lai.
```

## 14. Bám theo thang điểm cuối kỳ

File đánh giá cuối kỳ yêu cầu 3 nhóm lớn:

1. Complete System Implementation.
2. Experimental Evaluation.
3. Project Report.

Roadmap này tổ chức deliverable theo đúng ba nhóm đó.

### 14.1 Complete System Implementation

| Mục | Điểm | Cần làm trong đề tài hiện tại |
|---|---:|---|
| 1.1 Complete speech processing pipeline | 10 | audio input, preprocessing, segmentation, feature extraction, prediction, visualization |
| 1.2 Implement selected model | 20 | implement/clean 06D Emotion2Vec Co-Attention trên strict split |
| 1.3 Improvements | 10 | nâng cấp feature extraction và fusion dựa trên emotion2vec/CA-MSER/FT-w2v2 |
| 1.4 Impact analysis | 10 | ablation old 06D vs improved features vs co-attention |
| 1.5 UI/demo | 10 | demo upload audio và emotion timeline |

Deliverable cụ thể:

- notebook/script training rõ ràng;
- notebook/script inference rõ ràng;
- model checkpoint;
- cached feature hoặc hướng dẫn tạo feature;
- demo notebook/app;
- hình kiến trúc 06D improved.

### 14.2 Experimental Evaluation

| Mục | Điểm | Cần làm |
|---|---:|---|
| 2.1 Dataset/test scenario | 10 | IEMOCAP 4-class LOSO, optional LOSpeaker |
| 2.2 Training/testing/results | 20 | train đủ fold, lưu metrics CSV, confusion matrix |
| 2.3 Metric analysis | 10 | phân tích WA, UA, Macro-F1, confusion matrix, inference time |
| 2.4 Compare baseline/reference | 10 | so với emotion2vec, CA-MSER, FT-w2v2, DST |
| 2.5 Extended evaluation | 10 | strict speaker split, optional AVD regression, audio demo |

Deliverable cụ thể:

- bảng kết quả per fold;
- bảng average/std;
- confusion matrix;
- bảng ablation;
- bảng so sánh paper;
- phân tích lỗi theo class.

### 14.3 Project Report

| Mục | Điểm | Cần viết |
|---|---:|---|
| 3.1 Required sections | 20 | Introduction, Background, Dataset, Method, Evaluation, Conclusion, References |
| 3.2 Development/evaluation process | 10 | mô tả quy trình đọc paper, build pipeline, train/test |
| 3.3 Figures/tables/diagrams | 10 | architecture, feature pipeline, results, confusion matrix |
| 3.4 Citations/references | 10 | trích dẫn paper/repo chính |
| 3.5 Format/academic writing | 10 | trình bày rõ ràng, thuật ngữ thống nhất |

## 15. Cấu trúc report đề xuất

Report cuối kỳ nên viết theo cấu trúc:

1. **Introduction**
   - Nêu bài toán Speech Emotion Recognition.
   - Nêu vấn đề random split không đủ tin cậy.
   - Nêu mục tiêu strict speaker-independent.

2. **Theoretical Background**
   - Speech emotion recognition.
   - Acoustic features.
   - Self-supervised speech representation.
   - Co-attention/fusion.
   - Arousal/Valence/Dominance.

3. **Related Work**
   - emotion2vec.
   - Wav2Vec2/FT-w2v2 SER.
   - CA-MSER.
   - DST.
   - AVD regression papers.

4. **Dataset**
   - IEMOCAP.
   - 4-class subset.
   - speaker/session split.
   - AVD labels.

5. **Proposed Method**
   - 06D current.
   - Vấn đề của feature extraction cũ.
   - 06D improved.
   - Feature branches.
   - Co-attention fusion.
   - Optional AVD head.

6. **Experimental Setup**
   - preprocessing.
   - split protocol.
   - training settings.
   - metrics.

7. **Results and Analysis**
   - random vs strict.
   - ablation.
   - baseline comparison.
   - confusion matrix.
   - error analysis.

8. **Demo**
   - input audio.
   - segment-level prediction.
   - visualization.

9. **Discussion**
   - vì sao strict split khó hơn.
   - vì sao feature extraction quan trọng.
   - giới hạn của dataset acted emotion.
   - giới hạn của AVD nếu chưa đủ thời gian.

10. **Conclusion and Future Work**
    - tổng kết đóng góp.
    - hướng mở rộng presentation feedback sau final.

## 16. Timeline thực hiện

### Tuần 1 - Chốt dataset và audit 06D

Việc cần hoàn thành:

- đọc lại notebook 06D;
- xác định chính xác feature hiện tại;
- chạy lại random split và strict split;
- tạo IEMOCAP manifest;
- tạo LOSO fold;
- kiểm tra số mẫu từng class.

Output:

- bảng kết quả 06D hiện tại;
- bảng class distribution;
- ghi chú vấn đề feature/split.

### Tuần 2 - Nâng feature extraction

Việc cần hoàn thành:

- đọc kỹ emotion2vec và CA-MSER;
- trích emotion2vec embedding;
- chuẩn hóa MFCC/log-Mel;
- cache feature;
- chạy emotion2vec-only baseline;
- chạy MFCC/log-Mel baseline.

Output:

- feature cache;
- bảng ablation feature đầu tiên;
- chọn feature cho 06D improved.

### Tuần 3 - Nâng 06D và chạy ablation

Việc cần hoàn thành:

- implement fusion/attention improved;
- thử concat/gated/co-attention;
- tune learning rate/dropout/loss;
- chạy LOSO 5-fold;
- lưu confusion matrix và result CSV.

Output:

- bảng 06D current vs 06D improved;
- bảng ablation;
- hình confusion matrix.

### Tuần 4 - So sánh, demo, optional AVD

Việc cần hoàn thành:

- so sánh với emotion2vec, CA-MSER, FT-w2v2, DST;
- nếu kịp, thêm AVD regression head;
- xây demo audio input;
- hoàn thiện report.

Output:

- bảng baseline comparison;
- demo;
- optional bảng CCC/MAE;
- report hoàn chỉnh.

## 17. Mức độ khả thi

### 17.1 Khả thi cao

Các phần rất nên làm và có khả năng hoàn thành:

- clean lại 06D;
- chuẩn hóa strict split;
- chuyển trọng tâm sang IEMOCAP;
- emotion2vec-only baseline;
- cải thiện feature extraction;
- ablation concat vs co-attention;
- demo audio segment-level emotion timeline.

### 17.2 Khả thi trung bình

Các phần làm được nếu thời gian còn ổn:

- fine-tune nhẹ emotion2vec/Wav2Vec2;
- chạy đủ Leave-One-Speaker-Out;
- thêm AVD regression head;
- so sánh thêm với DST bằng code.

### 17.3 Rủi ro cao

Các phần không nên đặt làm deliverable bắt buộc:

- reproduce đầy đủ tất cả baseline mạnh;
- train transformer lớn từ đầu;
- làm full presentation feedback system;
- xử lý TED Talk audio dài và rating;
- giải thích đoạn nào hay/dở như hệ thống coaching thật.

## 18. Kết luận roadmap

Đề tài final nên được định hình lại như sau:

```text
Tên hướng:
Improving 06D Emotion2Vec Co-Attention for Strict Speaker-Independent
Speech Emotion Recognition and Affective Attribute Regression.

Core contribution:
Nâng cấp feature extraction và fusion của 06D để cải thiện strict
speaker-independent SER.

Optional contribution:
Mở rộng shared encoder sang regression cho Arousal, Valence, Dominance
trên IEMOCAP.

Demo:
Audio input -> segment -> emotion prediction -> timeline visualization.
```

Đây là hướng thực tế hơn, học thuật hơn, dễ bảo vệ hơn và bám sát phần nhóm đã làm ở 06D hơn so với việc mở rộng quá xa sang presentation feedback/PuSQ/TED rating.
