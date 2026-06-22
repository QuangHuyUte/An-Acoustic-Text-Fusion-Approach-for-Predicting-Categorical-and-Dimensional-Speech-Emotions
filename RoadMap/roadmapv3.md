---
title: "Roadmap v3 - Hệ thống phản hồi luyện thuyết trình dựa trên giọng nói"
subtitle: "SER 4 datasets, Strong ML Baseline, 1D-CNN Main Model, VAD, Smoothing, Emotion Timeline, Acoustic Indicators and Feedback Engine"
author: "Nhóm: Nguyễn Minh Cường, Nguyễn Tài Huy, Bùi Quang Huy"
date: "22/06/2026"
lang: "vi"
---

# Roadmap v3

# Hệ thống phản hồi luyện thuyết trình dựa trên giọng nói

**Tên đề tài đề xuất:** Hệ thống phản hồi luyện thuyết trình dựa trên phân tích giọng nói  
**Tên tiếng Anh ngắn:** Voice-Based Presentation Feedback System  
**Tên tiếng Anh đầy đủ:** Voice-Based Presentation Feedback System using Speech Emotion Recognition, VAD-guided Emotion Timeline and Acoustic Speaking Indicators  
**Tên repo Git gợi ý:** `voice-presentation-feedback`  
**Lĩnh vực:** Speech Processing, Speech Emotion Recognition, Audio Feature Analysis, Near Real-Time Inference, Human-Computer Interaction  
**Nền tảng triển khai:** Kaggle/Google Colab cho training; Gradio hoặc Streamlit cho demo  
**Ràng buộc tài nguyên:** không có GPU mạnh, ưu tiên mô hình vừa phải, inference nhanh, dễ demo, dễ giải thích trong báo cáo  

---

## 0. Mục tiêu của bản roadmap v3

Bản roadmap này được viết lại từ roadmap v2/v3, nhưng cập nhật theo tình trạng mới nhất của project:

1. Dataset hiện tại đã mở rộng thành **4 dataset chính**: CREMA-D, RAVDESS speech-only, TESS và SAVEE.
2. Nhóm đã hoàn thành notebook **04_Baseline_Models** với classical ML baseline trên 4 dataset.
3. Kết quả baseline cho thấy mô hình ML đơn giản có thể làm mốc so sánh tốt, nhưng chưa nên dùng làm final main model.
4. Main model cần được điều chỉnh từ hướng "CNN chung chung" sang một lựa chọn rõ hơn: **1D-CNN trên chuỗi đặc trưng acoustic/MFCC** hoặc **2D-CNN log-Mel**, trong đó 1D-CNN feature-fusion nên là main model ưu tiên vì sát với paper 4-dataset mà nhóm đã tham khảo.
5. Sau khi có SER model tốt, hệ thống không dừng ở classification mà đi tiếp tới **VAD, sliding window, smoothing, emotion timeline, acoustic indicators, speaking pressure score và feedback engine**.

Bản này không chỉ trả lời "làm gì", mà còn giải thích:

- tại sao làm bước đó,
- input/output của từng bước là gì,
- lỗi thường gặp là gì,
- artifact nào cần lưu,
- cách viết vào báo cáo,
- mô hình nào là baseline, main, advanced,
- đánh giá như thế nào để không bị phản biện.

---

## 1. Chốt hướng đề tài cuối cùng

### 1.1. Hướng đề tài nên dùng

Hướng đề tài nên chốt là:

> **Hệ thống phản hồi luyện thuyết trình dựa trên phân tích giọng nói**

Đây là hướng cân bằng nhất vì nó giữ được lõi Speech Emotion Recognition, đồng thời mở rộng thành một ứng dụng thực tế có demo rõ ràng. Hệ thống không chỉ nói "audio này là happy/sad/angry", mà phân tích bài nói theo thời gian và đưa ra phản hồi giúp người dùng cải thiện phong thái trình bày.

### 1.2. Vì sao không gọi là stress detection?

Không nên gọi hệ thống là "stress detection" vì bốn dataset chính của nhóm là emotion datasets, không phải stress datasets. CREMA-D, RAVDESS, TESS và SAVEE có nhãn cảm xúc, không có ground truth stress/non-stress. Nếu gọi là stress detector, thầy cô có thể hỏi:

- Stress được gán nhãn như thế nào?
- Ai xác nhận người nói bị stress?
- Có self-report, physiological signal hoặc clinical annotation không?
- Mô hình có rủi ro y khoa/đạo đức không?

Do đó, phần này phải gọi là:

- **Speaking Pressure Indicator**
- **Vocal Stability Indicator**
- **Stress-related Acoustic Indicator**
- **Confidence/Pressure Indicator for Presentation Feedback**

Nói ngắn gọn: hệ thống chỉ tính chỉ báo âm học liên quan đến sự ổn định khi nói, không kết luận trạng thái tâm lý.

### 1.3. Vì sao không chuyển sang IELTS Speaking?

IELTS Speaking cần đánh giá grammar, vocabulary, fluency, coherence và pronunciation. Emotion không phải tiêu chí chính thức. Nếu chuyển sang IELTS, đề tài sẽ phải phụ thuộc nhiều vào ASR, transcript, grammar scoring và LLM. Khi đó phần SER chỉ còn là phụ.

Presentation feedback phù hợp hơn vì emotion, energy, pitch, pause, speech rate và filler words đều liên quan trực tiếp tới cách trình bày.

### 1.4. Một câu giới thiệu đề tài có thể dùng

> Đề tài xây dựng một hệ thống phản hồi luyện thuyết trình dựa trên giọng nói. Hệ thống huấn luyện mô hình Speech Emotion Recognition trên bốn dataset công khai, sau đó áp dụng mô hình theo từng đoạn audio ngắn để tạo emotion timeline. Kết quả emotion được kết hợp với VAD, smoothing, acoustic indicators, speech rate và filler words để sinh feedback giúp người học cải thiện độ tự tin, sự liền mạch, năng lượng và ngữ điệu khi trình bày.

---

## 2. Bức tranh tổng thể của hệ thống

### 2.1. Hai pha chính

Project cần tách thành hai pha lớn:

```text
PHASE A - TRAINING / MODEL DEVELOPMENT
4 emotion datasets
→ preprocessing
→ feature extraction
→ baseline ML
→ main SER model
→ advanced SER model optional
→ model evaluation
→ save best model

PHASE B - APPLICATION / FEEDBACK SYSTEM
user presentation audio
→ preprocessing
→ VAD
→ sliding window chunks
→ SER prediction per chunk
→ smoothing
→ emotion timeline
→ acoustic indicators
→ transcript metrics optional
→ speaking pressure score
→ feedback engine
→ dashboard
```

### 2.2. Tư duy quan trọng

Không nên xây một model duy nhất đi từ audio thẳng ra feedback. Cách đúng hơn là chia hệ thống thành các module nhỏ:

| Module | Vai trò |
|---|---|
| SER model | Dự đoán cảm xúc của từng audio/chunk |
| VAD | Xác định speech/silence, tránh predict emotion trên im lặng |
| Sliding window | Chia audio dài thành nhiều đoạn ngắn để phân tích theo thời gian |
| Smoothing | Làm timeline ổn định, giảm nhảy nhãn giữa các chunk |
| Acoustic indicators | Tính RMS, pitch, silence, pause, continuity |
| Transcript analysis | Tính speech rate và filler words nếu dùng Whisper |
| Feedback engine | Biến các chỉ số thành nhận xét dễ hiểu |
| Dashboard | Hiển thị waveform, spectrogram, timeline, metrics và feedback |

Cách này phù hợp với đồ án sinh viên vì mỗi module có thể triển khai độc lập, dễ debug, dễ phân công nhóm, và dễ giải thích trước hội đồng.

---

## 3. Trạng thái hiện tại của project

### 3.1. Những gì đã làm xong

Nhóm đã hoàn thành notebook baseline classical ML:

```text
04_Baseline_Models.ipynb
```

Notebook này đã làm được các việc quan trọng:

- dùng 4 dataset: RAVDESS, CREMA-D, TESS, SAVEE,
- dùng 6 nhãn chung: neutral, happy, sad, angry, fear, disgust,
- tạo feature vector 248 chiều,
- train nhiều mô hình classical ML,
- đánh giá bằng hai protocol:
  - `paper_comparable_random`,
  - `strict_speaker_aware`,
- báo cáo accuracy, macro-F1, weighted-F1,
- phân tích per-class, per-dataset và error analysis.

### 3.2. Kết quả baseline hiện tại

| Protocol | Best model | Test accuracy | Test macro-F1 | Ý nghĩa |
|---|---:|---:|---:|---|
| Paper-comparable random split | SVM RBF tuned | 66.61% | 66.64% | Dùng để so với paper/Kaggle, nhưng dễ hơn |
| Strict speaker-aware split | Probability-average ensemble | 47.06% | 46.53% | Khó hơn, thực tế hơn, phản ánh speaker/domain shift |

Kết quả này rất hữu ích. Nó cho thấy baseline ML không quá yếu, nhưng cũng chứng minh bài toán multi-dataset SER không hề dễ. Sự chênh lệch lớn giữa random split và strict split là điểm nên đưa vào báo cáo, vì nó thể hiện nhóm hiểu vấn đề đánh giá chứ không chỉ chạy model lấy accuracy cao.

### 3.3. Cách diễn giải kết quả baseline trong báo cáo

Không nên viết:

> Baseline đạt 66.61%, mô hình nhận diện cảm xúc khá tốt.

Nên viết:

> Với random split dùng để so sánh với các paper/Kaggle, SVM RBF đạt 66.61% accuracy và 66.64% macro-F1. Tuy nhiên, khi dùng strict speaker-aware split, kết quả giảm xuống 47.06% accuracy và 46.53% macro-F1. Điều này cho thấy bài toán SER trên nhiều corpus chịu ảnh hưởng mạnh bởi speaker identity, dataset domain, accent và recording condition.

### 3.4. Baseline này có dùng làm main model không?

Không nên dùng classical ML baseline làm main model cuối cùng. Nó nên giữ vai trò:

- mốc so sánh truyền thống,
- bằng chứng rằng handcrafted features có hiệu quả ở mức cơ bản,
- mô hình dự phòng nếu deep learning gặp lỗi,
- phần giải thích học thuật trong report.

Main model tiếp theo nên là deep learning nhẹ, có softmax probability để phục vụ smoothing và emotion timeline.

---

## 4. Chiến lược dataset v3

### 4.1. Dataset chính hiện tại

Phiên bản hiện tại dùng 4 dataset emotion chính:

```text
CREMA-D
RAVDESS speech-only
TESS
SAVEE
```

Hệ thống giữ 6 nhãn chung:

```text
neutral, happy, sad, angry, fear, disgust
```

Loại bỏ hoặc không dùng trong bản chính:

```text
calm, surprise
```

Lý do loại `calm` và `surprise` là chúng không có mặt đồng đều trong tất cả dataset. Nếu cố giữ, label space sẽ lệch và model dễ học nhiễu.

### 4.2. Vai trò của từng dataset

| Dataset | Vai trò trong project | Điểm mạnh | Hạn chế |
|---|---|---|---|
| CREMA-D | Dataset lõi quan trọng nhất | Nhiều speaker, 6 emotion khớp tốt | Acted speech, domain vẫn khác presentation thật |
| RAVDESS speech-only | Dataset học thuật chuẩn | Phổ biến, dễ cite, chất lượng audio tốt | Ít hơn CREMA-D nếu chỉ lấy speech |
| TESS | Bổ sung dữ liệu sạch | Nhiều mẫu sạch, dễ train | Chỉ 2 female speakers, dễ làm kết quả ảo cao |
| SAVEE | Benchmark phụ trong 4-dataset setup | Có male British speakers, nhiều paper dùng | Chỉ 4 male speakers, 480 utterances, nhỏ |
| EmoDB | Future benchmark optional | Phổ biến trong SER | Tiếng Đức, nhỏ, không cần cho bản chính |

### 4.3. Label mapping

| Unified label | CREMA-D | RAVDESS | TESS | SAVEE |
|---|---|---|---|---|
| neutral | neutral | neutral | neutral | neutral / n |
| happy | happy | happy | happy | happiness / h |
| sad | sad | sad | sad | sadness / sa |
| angry | anger | angry | angry | anger / a |
| fear | fear | fearful | fear | fear / f |
| disgust | disgust | disgust | disgust | disgust / d |

Không map `calm` vào `neutral` trong bản chính nếu muốn giữ label sạch. Nếu làm thử, phải ghi rõ đó là ablation hoặc variant.

### 4.4. Ba protocol đánh giá bắt buộc

Do nhóm đang tham khảo các paper có kết quả khác nhau trên 4 dataset, cần dùng ít nhất ba protocol:

#### Protocol 1: Paper-comparable random split

Dùng stratified random split sau khi gộp dataset. Protocol này giúp so với paper/Kaggle vì nhiều nguồn dùng random split. Tuy nhiên, nó có thể dễ hơn thực tế vì speaker/corpus style có thể xuất hiện ở cả train và test.

#### Protocol 2: Strict speaker-aware split

Tách speaker để tránh cùng người nói xuất hiện ở cả train và test. Đây nên là protocol nghiêm túc hơn trong báo cáo. Nếu kết quả thấp hơn nhiều, đó là phân tích quan trọng chứ không phải lỗi.

#### Protocol 3: Cross-dataset evaluation

Train trên một số dataset và test trên dataset còn lại, ví dụ:

```text
Train: CREMA-D + RAVDESS + TESS
Test: SAVEE

Train: CREMA-D + RAVDESS + SAVEE
Test: TESS

Train: RAVDESS + TESS + SAVEE
Test: CREMA-D
```

Protocol này kiểm tra khả năng generalization giữa các corpus. Kết quả thường thấp, nhưng rất có giá trị học thuật.

### 4.5. Audio tự thu

Audio tự thu không dùng để train SER model. Nó dùng cho demo và kiểm thử feedback.

Nên thu khoảng 10-30 file, mỗi file 30-90 giây:

| Case | Mục đích kiểm thử |
|---|---|
| nói đều, ít pause | hệ thống báo ổn định |
| nói quá nhanh | speech rate cao |
| nói quá chậm | speech rate thấp |
| nói nhiều pause | silence ratio và pause count cao |
| nói nhỏ dần | energy decreasing |
| monotone | pitch variation thấp |
| nhiều filler | filler count cao |
| đoạn mở đầu tự tin, đoạn giữa ngập ngừng | timeline và feedback theo đoạn |

---

## 5. Literature Review và vai trò từng paper

### 5.1. Nguyên tắc đưa paper vào report

Không nên đưa paper vào chỉ để đủ số lượng. Mỗi paper cần có vai trò rõ:

- giúp chọn dataset,
- giúp chọn model,
- giúp giải thích VAD,
- giúp giải thích emotion timeline,
- giúp giải thích smoothing,
- giúp giải thích acoustic indicators,
- giúp định hướng future work.

### 5.2. Bảng paper theo module

| Paper/nguồn | Vai trò | Dataset | Model/phương pháp | Có implement không? |
|---|---|---|---|---|
| IEEE 1D-CNN dùng RAVDESS + CREMA-D + TESS + SAVEE | Gợi ý main model mới | 4 dataset | 1D-CNN, ZCR, energy, entropy of energy, RMS, MFCC | Có thể implement phiên bản chính |
| Novais/Sapientia ensemble framework | Gợi ý multi-dataset ensemble | 4 dataset hoặc tổ hợp dataset | Ensemble/aggregation primary classifiers | Không implement full, dùng để compare |
| Real-time SER using DL and Data Augmentation | Backbone cho SER + augmentation | TESS, EmoDB, RAVDESS | MLP, CNN, CNN+BiLSTM; MFCC/ZCR/Mel/RMS/Chroma | Học pipeline và augmentation |
| FLEA | Cơ sở cho segment/frame-level emotion | IEMOCAP | HuBERT, pseudo frame labels, attention pooling | Không implement full |
| Speech Emotion Diarization | Cơ sở cho emotion timeline đúng nghĩa | ZED | Emotion boundaries, diarization | Future work |
| Multimodal Emotion Diarization | Cơ sở cho smoothing/timeline | multimodal | WavLM + EmoBERTa + sliding smoothing | Không implement full |
| VAD + SER paper | Cơ sở cho VAD trước SER | IEMOCAP hoặc corpus SER | SSL features + VAD + SER | Dùng ý tưởng, implement bằng WebRTC/Silero |
| Acoustic/prosodic stress paper | Cơ sở cho speaking pressure | Cyberball/MIST | F0, jitter, speech rate, voiced segments | Dùng để giải thích acoustic indicators |
| TBDM-Net | Cơ sở temporal SER nâng cao | CASIA, EMOVO, EMO-DB, IEMOCAP, RAVDESS, SAVEE | MFCC + temporal dense blocks + gender info | Future work/reference |

### 5.3. Research gap nên viết

Một đoạn có thể đưa vào report:

> Existing SER studies often focus on classifying short utterances into a single emotion label. However, in presentation practice, users need feedback over time rather than a single global label. They need to know where their voice becomes unstable, where they pause too much, where their energy decreases, and whether the delivery sounds monotonous. Therefore, this project extends standard SER into a VAD-guided, sliding-window and smoothed emotion timeline, then combines it with acoustic speaking indicators and transcript-based metrics to generate practical presentation feedback.

---

## 6. Model strategy v3

### 6.1. Tầng model tổng thể

| Vai trò | Model | Input | Mục tiêu |
|---|---|---|---|
| Completed baseline | SVM RBF / ensemble classical ML | 248 handcrafted mean/std features | Mốc so sánh ML truyền thống |
| Main model v3 | 1D-CNN acoustic feature sequence | MFCC + ZCR + energy + entropy + RMS theo thời gian | Model chính cho SER và realtime chunks |
| Comparison model | 2D-CNN log-Mel spectrogram | log-Mel spectrogram | So sánh với 1D-CNN |
| Advanced model | CNN-GRU hoặc CNN+BiLSTM | feature sequence hoặc CNN time features | Học temporal pattern tốt hơn |
| Future model | Wav2Vec2/HuBERT/FLEA/TBDM-Net | SSL embeddings hoặc MFCC temporal blocks | Hướng nghiên cứu nâng cao |

### 6.2. Baseline model hiện tại

Notebook baseline hiện tại đã dùng feature vector 248 chiều. Vector này gồm các nhóm feature đã được pooling bằng mean/std theo thời gian.

Vai trò của baseline:

- chứng minh handcrafted features có hiệu quả cơ bản,
- so sánh random split và strict split,
- tạo kết quả để báo cáo trước khi qua deep learning,
- giúp xác định emotion nào dễ nhầm.

Không nên sửa quá nhiều notebook này nữa. Nên đóng gói nó thành baseline chính và chuyển sang main model.

### 6.3. Main model nên là 1D-CNN feature sequence

Main model v3 nên là:

```text
Frame-level feature sequence
→ 1D-CNN along time axis
→ Global pooling
→ Dense
→ Softmax 6 emotions
```

Feature đầu vào nên gồm:

```text
MFCC
Delta MFCC
Delta-delta MFCC
ZCR
Short-time energy
Entropy of energy
RMS
```

Điểm quan trọng: **không nên lấy vector 248 mean/std rồi đưa thẳng vào 1D-CNN như một chuỗi giả** nếu mục tiêu là học temporal pattern. 1D-CNN nên nhận chuỗi theo thời gian:

```text
X shape = (time_frames, feature_dim)
```

Ví dụ:

```text
3-second audio at 16 kHz
frame length = 25 ms
hop length = 10 ms
→ khoảng 300 frames
→ mỗi frame có MFCC + delta + zcr + rms + energy...
→ input shape khoảng (300, F)
```

1D-CNN sẽ học pattern theo trục thời gian, ví dụ năng lượng tăng dần, pitch/feature thay đổi, sự khác biệt giữa angry/sad/fear.

### 6.4. Vì sao 1D-CNN hợp làm main model?

1D-CNN hợp với project vì:

- sát với paper 4-dataset mà nhóm đã tham khảo,
- nhẹ hơn CNN+BiLSTM,
- inference nhanh trên chunk 3 giây,
- trả được softmax probability vector,
- feature đầu vào liên quan trực tiếp tới acoustic indicators,
- dễ giải thích hơn SSL model.

### 6.5. Kiến trúc 1D-CNN gợi ý

```text
Input: (T, F)
→ Conv1D(filters=64, kernel_size=5) + BatchNorm + ReLU
→ MaxPool1D
→ Conv1D(filters=128, kernel_size=5) + BatchNorm + ReLU
→ MaxPool1D
→ Conv1D(filters=256, kernel_size=3) + BatchNorm + ReLU
→ GlobalAveragePooling1D
→ Dropout(0.3)
→ Dense(128) + ReLU
→ Dropout(0.3)
→ Dense(6) + Softmax
```

Loss và optimizer:

```text
Loss: CrossEntropyLoss
Optimizer: AdamW
Learning rate: 1e-3 hoặc 3e-4
Metric chính: macro-F1
Early stopping: validation macro-F1
```

### 6.5.1. Phạm vi notebook 05

Notebook 05 không nên nhồi tất cả deep model vào một chỗ. Mục tiêu của 05 là tạo **main SER model đầu tiên** sau baseline, có thể chạy được trên Kaggle/Colab và có câu chuyện học thuật rõ ràng.

Các mô hình nên có trong 05:

| Model trong 05 | Vai trò | Ghi chú |
|---|---|---|
| 1D-CNN basic | Main model chính | Nhẹ, dễ debug, dùng làm mốc deep learning đầu tiên |
| 1D-CNN + BatchNorm + Dropout | Bản ổn định hơn | Giảm overfitting, phù hợp dữ liệu multi-corpus |
| 1D-CNN + GlobalAveragePooling | Bản inference nhanh | Phù hợp sliding-window và near real-time demo |
| 1D-CNN + attention pooling | Điểm cộng học thuật | Giúp model học frame/segment quan trọng hơn nếu còn thời gian |
| MLP trên pooled sequence features | Sanity check | Optional, để biết deep model đơn giản có hơn classical ML không |

Các mô hình **không nên đưa vào 05**:

| Model | Để ở đâu | Lý do |
|---|---|---|
| 2D-CNN log-Mel | Notebook 06 | Đây là hướng representation khác, nên tách riêng để so sánh công bằng |
| CNN-GRU / CNN-BiLSTM | Notebook 07 | Nặng hơn, thuộc advanced temporal model |
| Wav2Vec2 / HuBERT / emotion2vec | Future/advanced | Có giá trị nghiên cứu nhưng phụ thuộc GPU và tải model lớn |

### 6.5.2. Paper làm cơ sở cho notebook 05

| Nguồn | Họ dùng gì | Cách nhóm mình kế thừa |
|---|---|---|
| IEEE 1D-CNN dùng RAVDESS + CREMA-D + TESS + SAVEE | 1D-CNN với ZCR, energy, entropy of energy, RMS và MFCC | Chọn 1D-CNN feature fusion làm main model của 05 |
| MDPI 2022 về CNN trên nhiều dataset SER | CNN/ResNet, nhấn mạnh proper split khác random split rất nhiều | Giữ cả random và strict evaluation, không chỉ báo cáo điểm đẹp |
| Real-time SER using deep learning and augmentation | MLP, CNN, CNN+BiLSTM; MFCC, ZCR, Mel, RMS, Chroma; augmentation | Dùng augmentation nhẹ chỉ trên train: noise, gain, time shift, random crop |
| emotion2vec / SSL emotion representation | Universal speech emotion representation | Đưa vào hướng phát triển sau 05/06, không dùng làm main đầu tiên |

### 6.5.3. Ablation nên làm trong 05

Để 05 có giá trị học thuật hơn một notebook train CNN thông thường, nên thêm ablation nhỏ:

| Thí nghiệm | Input feature | Mục đích |
|---|---|---|
| A1 | MFCC only | So với baseline MFCC truyền thống |
| A2 | MFCC + delta + delta-delta | Kiểm tra lợi ích của dynamics |
| A3 | MFCC + delta + ZCR/RMS/energy | Kiểm tra feature fusion |
| A4 | Full feature fusion + attention pooling | Bản mạnh nhất nếu tài nguyên cho phép |

Kết quả 05 nên được so với 04 theo cùng hai protocol:

```text
paper_comparable_random
strict_speaker_aware
```

Nếu 1D-CNN không vượt baseline rõ ràng ở strict split, vẫn có thể giải thích rằng multi-corpus SER chịu domain shift mạnh; khi đó notebook 06 log-Mel và notebook 07 temporal model là bước cải thiện tiếp theo.

### 6.6. Comparison model: 2D-CNN log-Mel

2D-CNN vẫn nên giữ để so sánh vì nó là hướng phổ biến trong SER:

```text
Audio 3s
→ log-Mel spectrogram
→ Conv2D blocks
→ Global Average Pooling
→ Softmax 6 classes
```

Vai trò:

- nếu 2D-CNN tốt hơn 1D-CNN, có thể dùng 2D-CNN cho demo,
- nếu 1D-CNN tốt hơn, dùng 2D-CNN làm comparison,
- giúp báo cáo có phần so sánh representation: handcrafted temporal sequence vs spectrogram image.

### 6.7. Advanced model

Advanced model nên là:

```text
CNN-GRU
```

hoặc:

```text
CNN+BiLSTM
```

Thứ tự ưu tiên:

1. CNN-GRU: nhẹ hơn, hợp realtime hơn.
2. CNN-LSTM: dễ hiểu, phổ biến.
3. CNN+BiLSTM: học hai chiều tốt hơn nhưng nặng hơn.
4. HuBERT/Wav2Vec2 embeddings: future work nếu còn GPU.

Advanced model chỉ nên dùng trong demo nếu:

- macro-F1 cao hơn main model rõ ràng,
- inference time vẫn ổn,
- load model trong Gradio không lỗi.

---

## 7. Module 1 - Data Preparation

### 7.1. Mục tiêu

Chuẩn hóa 4 dataset thành một metadata thống nhất để train 6-class SER.

### 7.2. Input

```text
data/raw/CREMA-D/
data/raw/RAVDESS/
data/raw/TESS/
data/raw/SAVEE/
```

### 7.3. Output

```text
metadata_raw.csv
metadata_clean.csv
label_mapping.json
split_config.json
```

### 7.4. Các bước chi tiết

#### Bước 1: Duyệt file

Duyệt toàn bộ file `.wav`, `.mp3` nếu có. Mỗi file tạo một dòng metadata.

#### Bước 2: Xác định dataset

Dựa vào folder hoặc filename để gán `dataset`.

#### Bước 3: Parse speaker id

Speaker id cần cho strict speaker-aware split. Nếu dataset không có speaker rõ, cần tạo speaker id theo quy tắc.

Ví dụ:

- CREMA-D: speaker nằm ở đầu filename.
- RAVDESS: actor id nằm ở cuối filename.
- TESS: speaker có thể là OAF/YAF.
- SAVEE: speaker là DC/JE/JK/KL hoặc code tương tự.

#### Bước 4: Parse emotion raw

Mỗi dataset có format khác nhau. Nên viết hàm riêng:

```text
parse_cremad()
parse_ravdess()
parse_tess()
parse_savee()
```

#### Bước 5: Map label

Map tất cả về 6 nhãn chung. Bỏ calm/surprise.

#### Bước 6: Kiểm tra duration và sample rate

Lưu `duration_sec`, `sample_rate`, `num_channels`. Nếu file quá ngắn hoặc lỗi load, đánh dấu `usable=False`.

#### Bước 7: Split

Tạo hai kiểu split:

- `split_strict`: speaker-aware nếu có thể.
- `split_random`: stratified random.

Nên lưu cả hai trong metadata hoặc lưu hai file metadata riêng.

### 7.5. Metadata schema

| Cột | Ý nghĩa |
|---|---|
| sample_id | ID duy nhất |
| filepath | Đường dẫn file |
| dataset | Dataset nguồn |
| speaker_id | ID speaker |
| gender | nếu có |
| emotion_raw | nhãn gốc |
| emotion | nhãn 6 lớp |
| duration_sec | độ dài |
| sample_rate | sample rate |
| num_channels | số kênh |
| split_strict | train/validation/test |
| split_random | train/validation/test |
| usable | true/false |
| note | lỗi hoặc ghi chú |

### 7.6. Lỗi cần tránh

- augmentation trước split,
- segment trước split,
- cùng speaker nằm cả train và test trong strict protocol,
- map calm vào neutral mà không ghi rõ,
- dùng TESS random split rồi kết luận model generalize tốt.

---

## 8. Module 2 - Audio Preprocessing

### 8.1. Mục tiêu

Đưa audio về chuẩn thống nhất để feature extraction và model training ổn định.

### 8.2. Chuẩn đề xuất

| Thành phần | Giá trị |
|---|---|
| sample rate | 16 kHz |
| channel | mono |
| training duration | 3 giây |
| normalization | peak normalize hoặc RMS normalize nhẹ |
| silence trim | chỉ trim đầu/cuối nhẹ trong training |
| inference audio dài | không trim pause giữa bài |

### 8.3. Vì sao chọn 3 giây?

3 giây đủ dài để có ngữ cảnh cảm xúc, nhưng vẫn đủ ngắn cho near real-time. Khi inference, mỗi chunk 3 giây có thể được xem như một utterance ngắn.

### 8.4. Pad/crop

Nếu audio ngắn hơn 3 giây:

```text
pad silence at end
```

Nếu audio dài hơn 3 giây trong training:

```text
center crop hoặc random crop
```

Nếu audio dài trong demo:

```text
sliding window
```

### 8.5. Artifact

```text
processed_audio/
preprocess_config.json
duration_distribution.png
```

---

## 9. Module 3 - EDA và Visualization

### 9.1. Mục tiêu

EDA giúp chứng minh nhóm hiểu dữ liệu. Đây là phần rất nên đầu tư vì nó làm report có chiều sâu.

### 9.2. Biểu đồ bắt buộc

| Biểu đồ | Ý nghĩa |
|---|---|
| samples_by_emotion | kiểm tra class imbalance |
| samples_by_dataset | xem dataset nào chiếm nhiều |
| samples_by_speaker | kiểm tra speaker imbalance |
| duration_distribution | quyết định pad/crop |
| split_distribution | kiểm tra train/val/test |
| waveform_examples | minh họa raw signal |
| mel_spectrogram_examples | minh họa time-frequency |
| mfcc_heatmap_examples | minh họa feature |
| rms_by_emotion | xem năng lượng theo emotion |
| zcr_by_emotion | xem mức dao động |
| pitch_by_emotion | xem cao độ nếu extract được |

### 9.3. Nhận xét nên viết

Mỗi hình cần có 2-3 câu phân tích. Ví dụ:

- TESS thường dễ hơn vì audio sạch và ít speaker.
- CREMA-D khó hơn vì nhiều speaker hơn.
- Fear/sad có thể dễ nhầm vì đều có thể có energy thấp.
- Angry/disgust có thể dễ nhầm vì đều có sắc thái mạnh.
- Random split dễ cho kết quả cao hơn strict split.

### 9.4. Artifact

```text
figures/eda/*.png
eda_summary.md
```

---

## 10. Module 4 - Feature Extraction

### 10.1. Feature cho baseline đã hoàn thành

Baseline hiện tại dùng vector 248 chiều. Đây là feature đã pooling mean/std theo thời gian.

Nhóm feature:

- MFCC,
- delta MFCC,
- delta-delta MFCC,
- RMS,
- zero-crossing rate,
- spectral centroid,
- spectral bandwidth.

Feature này phù hợp với SVM/RF/Logistic Regression.

### 10.2. Feature cho main 1D-CNN

Main 1D-CNN không nên dùng mean/std vector. Nó nên dùng sequence:

```text
X_sequence = [frame_1, frame_2, ..., frame_T]
```

Mỗi frame gồm:

```text
MFCC
Delta MFCC
Delta-delta MFCC
ZCR
Energy
Entropy of energy
RMS
```

Output shape:

```text
(T, F)
```

Trong đó T là số frame, F là số feature mỗi frame.

### 10.3. Energy và entropy of energy

Short-time energy đo năng lượng trong từng frame. Entropy of energy đo mức phân bố năng lượng trong một đoạn. Đây là feature được nhiều pipeline SER dùng để phân biệt các cảm xúc có độ mạnh khác nhau.

### 10.4. Feature cho acoustic indicators

Khác với feature cho model, acoustic indicators dùng để feedback:

- RMS mean/std/trend,
- pitch mean/std/range,
- silence ratio,
- pause count,
- pause duration,
- speaking continuity,
- speech rate,
- filler count.

### 10.5. Artifact cần lưu

```text
baseline_features.npz
sequence_features_train.npz
sequence_features_val.npz
sequence_features_test.npz
mel_features/
feature_config.json
```

---

## 11. Module 5 - Baseline Models đã hoàn thành

### 11.1. Vai trò

Baseline là mốc so sánh truyền thống. Nó giúp trả lời câu hỏi:

> Với handcrafted features và ML đơn giản, hệ thống đạt đến đâu trước khi dùng deep learning?

### 11.2. Model zoo hiện tại

Notebook baseline đã thử:

- dummy majority,
- logistic regression,
- SVM RBF,
- random forest,
- extra trees,
- KNN,
- probability-average ensemble.

### 11.3. Kết quả nên đưa vào report

| Scenario | Best model | Validation macro-F1 | Test macro-F1 |
|---|---:|---:|---:|
| paper_comparable_random | SVM RBF tuned | 68.98% | 66.64% |
| strict_speaker_aware | probability-average ensemble | 64.18% | 46.53% |

### 11.4. Bài học từ baseline

- Random split cho kết quả cao hơn strict split.
- TESS có thể quá dễ nếu random split vì audio sạch và ít speaker.
- CREMA-D khó hơn nhưng quan trọng hơn cho generalization.
- Classical ML chưa đủ mạnh để làm model chính, nhưng rất tốt làm baseline.
- Cần deep learning/sequence model để học representation tốt hơn.

### 11.5. Việc cần làm tiếp với baseline

Không cần train thêm quá nhiều ML model. Chỉ cần:

- lưu model tốt nhất,
- lưu metrics,
- lưu confusion matrix,
- viết analysis vào report,
- dùng baseline làm mốc so sánh với 1D-CNN/2D-CNN.

---

## 12. Module 6 - Main Model 1D-CNN

### 12.1. Mục tiêu

Train model SER chính trên 4 dataset để dùng cho chunk-based emotion prediction trong demo.

### 12.2. Input

```text
sequence_features: (num_samples, time_frames, feature_dim)
labels: 6 classes
```

### 12.3. Training setup

| Thành phần | Gợi ý |
|---|---|
| loss | CrossEntropyLoss |
| optimizer | AdamW |
| learning rate | 1e-3 hoặc 3e-4 |
| batch size | 32 hoặc 64 |
| epochs | 30-80 |
| early stopping | patience 8-10 |
| metric chọn model | validation macro-F1 |
| class imbalance | class weights nếu cần |

### 12.4. Data augmentation cho 1D-CNN

Nên augmentation ở waveform trước khi extract feature:

- noise addition,
- time shift,
- volume gain,
- time stretch nhẹ,
- pitch shift nhẹ nếu không làm méo emotion quá mức.

Nếu làm augmentation trên feature:

- spec/time masking nhẹ,
- random time crop,
- Gaussian noise nhẹ vào feature.

### 12.5. Evaluation

Đánh giá theo cả hai protocol:

- paper-comparable random,
- strict speaker-aware.

Nếu có thời gian, thêm cross-dataset test.

### 12.6. Điều kiện dùng 1D-CNN làm demo model

Dùng 1D-CNN làm demo nếu:

- macro-F1 cao hơn baseline rõ ràng,
- inference trên chunk 3 giây nhanh,
- softmax probability ổn,
- không overfit quá mạnh.

### 12.7. Artifact

```text
models/ser_1dcnn.pt
models/label_encoder.pkl
models/feature_config.json
figures/metrics/1dcnn_confusion_matrix.png
figures/metrics/1dcnn_training_curves.png
results/1dcnn_metrics.json
```

---

## 13. Module 7 - Comparison Model 2D-CNN log-Mel

### 13.1. Mục tiêu

So sánh 1D-CNN feature sequence với 2D-CNN trên log-Mel spectrogram.

### 13.2. Input

```text
log_mel: (1, n_mels, time_frames)
```

### 13.3. Kiến trúc gợi ý

```text
Conv2D + BatchNorm + ReLU + MaxPool
Conv2D + BatchNorm + ReLU + MaxPool
Conv2D + BatchNorm + ReLU + MaxPool
GlobalAveragePooling
Dropout
Dense 128
Dense 6
```

### 13.4. Vai trò trong report

Nếu 2D-CNN tốt hơn:

> Chọn 2D-CNN làm model demo vì log-Mel biểu diễn time-frequency tốt hơn handcrafted sequence.

Nếu 1D-CNN tốt hơn:

> Chọn 1D-CNN làm model demo vì feature fusion phù hợp với 4-dataset setting và inference nhanh hơn.

Dù kết quả thế nào, phần so sánh này giúp report mạnh hơn.

---

## 14. Module 8 - Advanced Temporal Model

### 14.1. Mục tiêu

Advanced model dùng để thử nghiệm temporal modeling. Không bắt buộc phải dùng cho demo.

### 14.2. Model ưu tiên

```text
CNN-GRU
```

Pipeline:

```text
sequence input hoặc log-Mel
→ CNN feature extractor
→ GRU
→ Dense
→ Softmax
```

### 14.3. CNN+BiLSTM

CNN+BiLSTM có tính học thuật tốt hơn, gần với nhiều paper, nhưng nặng hơn. Dùng nếu còn thời gian.

### 14.4. Khi nào dừng advanced model?

Nếu sau 1-2 ngày train mà kết quả không vượt main model hoặc inference quá chậm, dừng lại và chuyển sang VAD/timeline/feedback. Điểm mạnh của project không chỉ nằm ở model accuracy, mà là hệ thống hoàn chỉnh.

---

## 15. Module 9 - VAD

### 15.1. VAD là gì?

VAD là Voice Activity Detection, dùng để xác định đoạn nào có tiếng nói và đoạn nào là im lặng/noise.

### 15.2. Vì sao cần VAD?

Không có VAD, hệ thống có thể predict emotion trên chunk toàn im lặng. Điều này làm timeline sai và feedback không đáng tin.

VAD giúp:

- phát hiện speech/silence,
- bỏ hoặc giảm trọng số chunk silence,
- tính silence ratio,
- tính pause count,
- tạo speaking continuity,
- hỗ trợ pressure score.

### 15.3. Công cụ đề xuất

| VAD | Ưu điểm | Nhược điểm |
|---|---|---|
| WebRTC VAD | rất nhẹ, nhanh, dễ realtime | nhạy với cấu hình frame, có thể kém trong noise |
| Silero VAD | neural pretrained, robust hơn | nặng hơn WebRTC, cần thêm dependency |

Bản demo có thể bắt đầu với WebRTC VAD, nếu có thời gian thì thử Silero VAD.

### 15.4. Output VAD

| Output | Ý nghĩa |
|---|---|
| speech_ratio | tỷ lệ speech trong chunk |
| silence_ratio | tỷ lệ silence |
| voiced_segments | danh sách đoạn có speech |
| pause_segments | danh sách đoạn pause |
| pause_count | số pause |
| mean_pause_duration | độ dài pause trung bình |

### 15.5. Rule dùng VAD trong timeline

```text
Nếu speech_ratio < 0.4:
    không cập nhật emotion mạnh
    label = silence hoặc uncertain
    vẫn dùng chunk để tính pause
Nếu speech_ratio >= 0.4:
    chạy SER bình thường
```

---

## 16. Module 10 - Sliding Window và Near Real-Time

### 16.1. Vì sao dataset chỉ có 1 label/audio vẫn làm timeline được?

Dataset train SER có một label cho một utterance ngắn. Khi demo với audio dài, ta chia audio thành nhiều chunk 2-3 giây. Mỗi chunk được xem như một utterance ngắn và model dự đoán emotion chủ đạo của chunk đó.

Đây không phải frame-level emotion ground truth, mà là:

```text
chunk-based emotion approximation
```

Cần ghi rõ trong report để tránh overclaim.

### 16.2. Cấu hình đề xuất

| Tham số | Giá trị |
|---|---|
| window size | 3 giây |
| hop size | 1 hoặc 1.5 giây |
| overlap | 50-67% |
| sample rate | 16 kHz |
| output per chunk | probability vector 6 lớp |

### 16.3. Simulated near real-time

Do Colab/Kaggle không ổn định cho microphone streaming thật, demo nên dùng:

```text
upload/record audio
→ xử lý chunk theo thứ tự
→ update timeline
```

Gọi là:

```text
simulated near real-time chunk-based analysis
```

Không nên gọi là full real-time nếu chưa streaming microphone thật.

---

## 17. Module 11 - Smoothing

### 17.1. Vì sao smoothing cần thiết?

Nếu mỗi chunk predict độc lập, timeline dễ nhảy:

```text
neutral → fear → neutral → sad → neutral
```

Smoothing giúp giảm biến động do:

- chunk ngắn,
- noise,
- confidence thấp,
- silence,
- model chưa chắc chắn.

### 17.2. Các loại smoothing nên dùng

#### Moving average probability

```text
smooth_prob[t] = mean(prob[t], prob[t-1], prob[t-2])
```

#### Majority vote

Lấy nhãn xuất hiện nhiều nhất trong 3 chunk gần nhất.

#### Confidence threshold

Nếu confidence thấp hơn 0.55 hoặc 0.60:

```text
label = uncertain
hoặc giữ label trước đó
```

#### Silence-aware weighting

Nếu speech_ratio thấp:

```text
prob_weight = speech_ratio
```

Chunk nhiều silence sẽ ít ảnh hưởng đến timeline.

### 17.3. Output

- raw timeline,
- smoothed timeline,
- confidence curve,
- uncertainty segments,
- emotion transition count trước/sau smoothing.

### 17.4. Đánh giá smoothing

Không cần ground truth timeline. Có thể đánh giá bằng:

- số lần chuyển emotion giảm,
- timeline bớt nhảy,
- feedback ít false alarm hơn,
- người nghe thấy hợp lý hơn.

---

## 18. Module 12 - Emotion Timeline

### 18.1. Mục tiêu

Emotion timeline cho biết cảm xúc/giọng điệu thay đổi ở đoạn nào trong bài nói.

### 18.2. Output table

| Start | End | Raw emotion | Smoothed emotion | Confidence | Speech ratio | Note |
|---:|---:|---|---|---:|---:|---|
| 0.0 | 3.0 | neutral | neutral | 0.72 | 0.91 | mở đầu ổn |
| 1.5 | 4.5 | neutral | neutral | 0.69 | 0.88 | ổn định |
| 3.0 | 6.0 | fear | neutral | 0.52 | 0.77 | confidence thấp |
| 4.5 | 7.5 | sad | sad | 0.64 | 0.86 | energy thấp |

### 18.3. Cách dùng timeline trong feedback

Không dùng emotion một mình. Emotion phải kết hợp với acoustic indicators:

| Pattern | Ý nghĩa feedback |
|---|---|
| neutral nhiều + pitch std thấp | giọng đều, thiếu nhấn nhá |
| fear/sad + pause cao + energy thấp | đoạn có speaking pressure cao |
| angry/disgust + energy cao | tone có thể hơi gắt |
| happy vừa phải + energy ổn | giọng có sức sống |
| confidence thấp + speech_ratio thấp | không nên kết luận emotion ở đoạn đó |

### 18.4. Hạn chế cần ghi

> Emotion timeline của nhóm là chunk-based approximation, không phải emotion diarization có ground-truth boundary.

---

## 19. Module 13 - Acoustic Indicators

### 19.1. Mục tiêu

Acoustic indicators giúp feedback có cơ sở dễ hiểu. Người dùng sẽ hiểu vì sao hệ thống nói "bạn nói hơi ngập ngừng" hoặc "giọng hơi đều".

### 19.2. RMS Energy

RMS dùng để đo năng lượng/âm lượng.

Output:

- RMS mean,
- RMS std,
- RMS trend,
- low-energy segments,
- energy decreasing flag.

Feedback:

- RMS thấp: giọng hơi nhỏ.
- RMS giảm về cuối: thiếu năng lượng khi kết bài.
- RMS dao động mạnh: âm lượng không ổn định.

### 19.3. Pitch/F0

Pitch dùng để đo cao độ và ngữ điệu.

Output:

- pitch mean,
- pitch std,
- pitch range,
- voiced frame ratio.

Feedback:

- pitch std thấp: giọng đều/monotone.
- pitch std quá cao: giọng dao động mạnh.
- voiced ratio thấp: nhiều đoạn không có speech rõ.

### 19.4. Silence ratio và pause count

Silence ratio:

```text
silence_ratio = silence_duration / total_duration
```

Pause count:

```text
số đoạn silence dài hơn 0.5s hoặc 0.8s
```

Feedback:

- silence ratio cao: nhiều khoảng lặng.
- pause count cao: bài nói đứt đoạn.
- pause dài ở giữa bài: cần luyện chuyển ý.

### 19.5. Speaking continuity

```text
speaking_continuity = 1 - silence_ratio
```

Có thể dùng để hiển thị mức độ liền mạch.

### 19.6. Acoustic report table

| Indicator | Value | Level | Feedback |
|---|---:|---|---|
| RMS mean | 0.034 | medium | âm lượng tương đối ổn |
| RMS trend | decreasing | warning | energy giảm về cuối |
| Pitch std | low | warning | giọng hơi đều |
| Silence ratio | 0.35 | high | nhiều khoảng lặng |
| Pause count | 12/min | high | ngắt quãng nhiều |

---

## 20. Module 14 - Speaking Pressure Indicator

### 20.1. Định nghĩa

Speaking Pressure Indicator là điểm tổng hợp phản ánh sự thiếu ổn định khi nói. Nó không phải stress diagnosis.

### 20.2. Công thức đề xuất

```text
pressure_score =
0.25 * pause_score
+ 0.20 * energy_instability_score
+ 0.20 * pitch_instability_score
+ 0.15 * speech_rate_score
+ 0.10 * filler_score
+ 0.10 * emotion_pressure_score
```

### 20.3. Chuẩn hóa từng thành phần

| Score | Cách hiểu |
|---|---|
| pause_score | cao khi silence ratio và pause count cao |
| energy_instability_score | cao khi RMS std cao hoặc energy giảm rõ |
| pitch_instability_score | cao khi pitch quá phẳng hoặc dao động mạnh |
| speech_rate_score | cao khi WPM quá nhanh hoặc quá chậm |
| filler_score | cao khi filler per minute cao |
| emotion_pressure_score | chỉ là tín hiệu phụ, tăng nhẹ khi fear/sad đi kèm acoustic bất ổn |

### 20.4. Mức đánh giá

| Score | Level | Ý nghĩa |
|---:|---|---|
| 0-35 | Low | giọng tương đối ổn định |
| 36-65 | Medium | có một số dấu hiệu thiếu ổn định |
| 66-100 | High | nhiều dấu hiệu ngập ngừng hoặc thiếu kiểm soát |

### 20.5. Câu tránh phản biện

Không nói:

> Người dùng đang bị stress.

Nói:

> Một số đoạn có speaking pressure cao hơn, thể hiện qua pause dài, energy thấp và pitch không ổn định.

---

## 21. Module 15 - Transcript, Speech Rate và Filler Words

### 21.1. Vai trò

Transcript không bắt buộc nhưng giúp feedback cụ thể hơn.

Nếu chưa có transcript, hệ thống vẫn chạy được bằng emotion + acoustic indicators. Nếu có transcript, thêm được:

- speech rate,
- filler words,
- repeated phrases,
- word count.

### 21.2. Công cụ

Dùng Whisper tiny hoặc base. Không train ASR.

### 21.3. Speech rate

```text
speech_rate = number_of_words / duration_minutes
```

Ngưỡng demo:

| WPM | Feedback |
|---:|---|
| < 90 | hơi chậm, có thể thiếu tự nhiên |
| 90-160 | tương đối ổn |
| > 160 | hơi nhanh, người nghe khó theo dõi |

### 21.4. Filler words

Tiếng Anh:

```text
um, uh, er, ah, like, you know, actually, basically, I mean, so
```

Tiếng Việt:

```text
ờ, ừm, à, thì, là, kiểu như, nói chung là, thật ra là
```

### 21.5. Hạn chế

Whisper có thể bỏ qua filler hoặc nhận sai, nhất là tiếng Việt. Vì vậy, phần filler nên ghi là experimental feature.

---

## 22. Module 16 - Feedback Engine

### 22.1. Nguyên tắc

Feedback không sinh trực tiếp từ CNN. CNN chỉ đưa ra emotion probabilities. Feedback nên do rule-based engine tạo ra từ nhiều chỉ số.

### 22.2. Input

```json
{
  "emotion_distribution": {},
  "emotion_timeline": [],
  "smoothed_timeline": [],
  "rms_stats": {},
  "pitch_stats": {},
  "pause_stats": {},
  "speech_rate": 0,
  "filler_count": 0,
  "pressure_score": 0,
  "high_pressure_segments": []
}
```

### 22.3. Output

1. Overall feedback.
2. Segment-level feedback.
3. Improvement suggestions.
4. Practice recommendation.

### 22.4. Rule examples

| Điều kiện | Feedback |
|---|---|
| neutral > 70% và pitch std thấp | Giọng khá đều, nên thêm ngữ điệu và nhấn keyword |
| silence ratio > 0.30 | Có nhiều khoảng lặng, nên chuẩn bị outline rõ hơn |
| pause count cao ở giữa bài | Đoạn giữa bài bị ngắt quãng, nên luyện chuyển ý |
| energy giảm về cuối | Năng lượng giọng giảm, nên giữ hơi và kết bài chắc hơn |
| speech rate > 160 WPM | Nói hơi nhanh, nên giảm tốc ở ý quan trọng |
| filler count cao | Dùng nhiều từ đệm, nên thay bằng pause ngắn có kiểm soát |
| fear/sad + pause cao + energy thấp | Đoạn này có speaking pressure cao, nên luyện lại |

### 22.5. Feedback mẫu

> Bài nói của bạn có phần mở đầu tương đối ổn định. Tuy nhiên, đoạn 12-18s có nhiều khoảng lặng, energy giảm và emotion timeline nghiêng về fear/sad với confidence trung bình. Hệ thống đánh dấu đây là đoạn có speaking pressure cao hơn. Bạn nên luyện lại phần chuyển ý ở đoạn này, giảm pause dài và giữ âm lượng ổn định hơn.

---

## 23. Module 17 - Dashboard Demo

### 23.1. Công nghệ

Ưu tiên Gradio vì upload/record audio nhanh và dễ demo.

### 23.2. Layout đề xuất

Dashboard nên có:

1. Audio input.
2. Waveform.
3. Mel-spectrogram.
4. Overall emotion prediction.
5. Emotion distribution.
6. Raw emotion timeline.
7. Smoothed emotion timeline.
8. VAD speech/silence plot.
9. Acoustic indicators table.
10. Speaking pressure score.
11. Transcript optional.
12. Speech rate/filler optional.
13. Feedback text.

### 23.3. Luồng demo

Demo nên có 3 phần:

#### Demo 1: Dataset audio ngắn

Upload một audio ngắn từ dataset để chứng minh model predict emotion.

#### Demo 2: Presentation audio tự thu

Upload audio dài 30-90 giây để hiển thị emotion timeline và acoustic indicators.

#### Demo 3: Raw vs smoothed timeline

So sánh timeline trước và sau smoothing để chứng minh smoothing có tác dụng.

### 23.4. Backup

Cần chuẩn bị:

- ảnh screenshot dashboard,
- video demo ngắn,
- audio mẫu có sẵn,
- file model đã save,
- notebook chạy lại được.

---

## 24. Evaluation Plan

### 24.1. SER model evaluation

Báo cáo:

- accuracy,
- macro-F1,
- weighted-F1,
- precision/recall per class,
- confusion matrix,
- per-dataset metrics,
- training/validation loss.

### 24.2. Split evaluation

So sánh:

| Protocol | Mục đích |
|---|---|
| random split | so với paper/Kaggle |
| strict speaker-aware | đánh giá nghiêm túc hơn |
| cross-dataset | kiểm tra generalization |

### 24.3. Model comparison table

| Model | Input | Random macro-F1 | Strict macro-F1 | Inference time | Chọn cho demo? |
|---|---|---:|---:|---:|---|
| SVM RBF baseline | 248 features | 66.64 |  | fast | no/main baseline |
| 1D-CNN | feature sequence | TBD | TBD | TBD | candidate main |
| 2D-CNN | log-Mel | TBD | TBD | TBD | candidate |
| CNN-GRU | sequence/log-Mel | TBD | TBD | TBD | advanced |

### 24.4. VAD evaluation

Nếu không có VAD ground truth, đánh giá định tính:

- có giảm predict emotion trên silence không,
- pause count có hợp lý không,
- timeline có sạch hơn không.

### 24.5. Smoothing evaluation

Dùng:

- emotion transition count trước/sau smoothing,
- số uncertain chunks,
- ví dụ timeline raw vs smoothed,
- qualitative judgement.

### 24.6. Feedback evaluation

Dùng audio tự thu theo kịch bản:

| Audio test | Kỳ vọng |
|---|---|
| nhiều pause | báo silence/pause cao |
| nói nhanh | báo WPM cao |
| nói chậm | báo WPM thấp |
| nói nhỏ dần | báo energy decreasing |
| monotone | báo pitch variation thấp |
| nhiều filler | báo filler count cao |

### 24.7. Latency evaluation

Đo:

- preprocessing time,
- feature extraction time per chunk,
- model inference time per chunk,
- total processing time for 60s audio,
- dashboard response time.

---

## 25. Notebook roadmap cập nhật

### 25.1. Notebook 01 - Data Preparation

Tasks:

- load 4 datasets,
- parse label,
- map 6 labels,
- parse speaker id,
- create metadata,
- create strict/random split.

Outputs:

```text
metadata_clean.csv
label_mapping.json
split_config.json
```

### 25.2. Notebook 02 - EDA

Tasks:

- dataset distribution,
- label distribution,
- duration distribution,
- waveform examples,
- MFCC/Mel examples,
- initial acoustic analysis.

Outputs:

```text
figures/eda/
eda_summary.md
```

### 25.3. Notebook 03 - Feature Extraction

Tasks:

- extract 248 baseline features,
- extract sequence features for 1D-CNN,
- extract log-Mel for 2D-CNN,
- extract acoustic indicators prototype.

Outputs:

```text
baseline_features.npz
sequence_features.npz
mel_features/
feature_config.json
```

### 25.4. Notebook 04 - Baseline Models

Status:

```text
DONE
```

Tasks already completed:

- train classical ML,
- evaluate strict/random,
- save metrics,
- error analysis.

Next action:

- clean notebook markdown,
- export figures,
- write report subsection.

### 25.5. Notebook 05 - Main 1D-CNN SER

Tasks:

- build dataset loader for sequence features,
- extract/load frame-level features: MFCC, delta, delta-delta, ZCR, RMS, energy, entropy of energy,
- implement 1D-CNN basic and 1D-CNN regularized,
- optional: attention pooling head,
- run small ablation: MFCC only vs MFCC+delta vs full feature fusion,
- train with early stopping,
- evaluate random and strict split,
- compare against 04_Baseline_Models,
- save model, metrics, confusion matrices, per-dataset reports and downloadable zip.

### 25.6. Notebook 06 - 2D-CNN log-Mel Comparison

Tasks:

- build log-Mel dataset,
- train 2D-CNN,
- compare with 1D-CNN and baseline.

### 25.7. Notebook 07 - Advanced Temporal Model

Tasks:

- train CNN-GRU or CNN+BiLSTM,
- compare macro-F1 and latency,
- decide whether to use for demo.

### 25.8. Notebook 08 - VAD + Sliding Window + Smoothing

Tasks:

- load demo audio,
- run VAD,
- split chunks,
- predict per chunk,
- smooth probabilities,
- plot raw/smoothed timeline.

### 25.9. Notebook 09 - Acoustic Indicators + Feedback

Tasks:

- compute RMS, pitch, silence, pause,
- compute pressure score,
- add transcript if available,
- generate feedback.

### 25.10. Notebook 10 - Gradio Dashboard

Tasks:

- load best model,
- accept upload/record audio,
- show dashboard outputs,
- export demo report.

---

## 26. Source code structure

```text
voice-presentation-feedback/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── demo_audio/
│   └── metadata_clean.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_train_1dcnn_ser.ipynb
│   ├── 06_train_2dcnn_logmel.ipynb
│   ├── 07_train_advanced_temporal.ipynb
│   ├── 08_vad_timeline_smoothing.ipynb
│   ├── 09_feedback_engine.ipynb
│   └── 10_gradio_demo.ipynb
│
├── src/
│   ├── config.py
│   ├── label_parsers.py
│   ├── audio_io.py
│   ├── preprocessing.py
│   ├── features_baseline.py
│   ├── features_sequence.py
│   ├── features_mel.py
│   ├── models_1dcnn.py
│   ├── models_2dcnn.py
│   ├── models_temporal.py
│   ├── train_utils.py
│   ├── evaluation.py
│   ├── vad.py
│   ├── smoothing.py
│   ├── timeline.py
│   ├── acoustic_indicators.py
│   ├── pressure_score.py
│   ├── transcript_analysis.py
│   └── feedback_engine.py
│
├── models/
│   ├── baseline_svm.pkl
│   ├── ser_1dcnn.pt
│   ├── ser_2dcnn.pt
│   ├── ser_cnn_gru.pt
│   ├── label_encoder.pkl
│   └── feature_config.json
│
├── app/
│   └── gradio_app.py
│
├── figures/
│   ├── eda/
│   ├── metrics/
│   └── demo/
│
├── results/
│   ├── baseline_metrics.json
│   ├── model_comparison.csv
│   └── ablation_results.csv
│
└── reports/
    ├── final_report.docx
    ├── slides.pptx
    └── demo_script.md
```

---

## 27. Timeline triển khai 4 tuần từ hiện tại

### Tuần 1 - Main model 1D-CNN

Mục tiêu: có model deep learning chính.

Việc cần làm:

- tạo sequence feature extraction,
- kiểm tra shape và label,
- train 1D-CNN trên random split,
- train/evaluate trên strict split,
- lưu model và metrics,
- so sánh với baseline.

Deliverables:

- `ser_1dcnn.pt`,
- `1dcnn_metrics.json`,
- confusion matrix,
- model comparison table v1.

### Tuần 2 - 2D-CNN và advanced optional

Mục tiêu: có comparison model và quyết định model demo.

Việc cần làm:

- extract log-Mel,
- train 2D-CNN,
- compare 1D-CNN vs 2D-CNN,
- thử CNN-GRU nếu còn thời gian,
- đo inference time.

Deliverables:

- `ser_2dcnn.pt`,
- comparison table,
- decision note: model nào dùng cho demo.

### Tuần 3 - VAD, sliding window, smoothing, timeline

Mục tiêu: biến SER model thành timeline.

Việc cần làm:

- thu audio demo,
- implement VAD,
- implement chunking 3s/hop 1.5s,
- predict per chunk,
- smooth probabilities,
- plot timeline,
- compare raw vs smoothed.

Deliverables:

- `emotion_timeline_demo.png`,
- `vad_speech_silence_plot.png`,
- `timeline_results.csv`.

### Tuần 4 - Acoustic indicators, feedback, dashboard, report

Mục tiêu: hoàn thiện hệ thống ứng dụng.

Việc cần làm:

- compute RMS/pitch/silence/pause,
- pressure score,
- transcript/speech rate nếu kịp,
- feedback engine,
- Gradio dashboard,
- viết report và slide.

Deliverables:

- Gradio app,
- feedback demo screenshots,
- final report draft,
- slides,
- demo script.

---

## 28. Timeline mở rộng 6 tuần

### Tuần 5

- Cross-dataset evaluation.
- Ablation augmentation.
- VAD/no VAD comparison.
- Smoothing/no smoothing comparison.
- Improve dashboard UI.

### Tuần 6

- Clean notebooks.
- Write final discussion and limitation.
- Prepare Q&A.
- Record backup demo video.
- Package repo.

---

## 29. Task assignment

### Nguyễn Minh Cường

Vai trò chính: EDA, acoustic indicators, timeline, dashboard.

Công việc:

- hoàn thiện EDA figures,
- implement RMS/pitch/silence/pause,
- implement pressure score,
- plot emotion timeline,
- build Gradio dashboard,
- chuẩn bị demo audio.

### Nguyễn Tài Huy

Vai trò chính: dataset, preprocessing, feature extraction, baseline.

Công việc:

- duy trì 4-dataset metadata,
- kiểm tra label mapping,
- kiểm tra split strict/random,
- extract sequence features,
- hỗ trợ 1D-CNN input pipeline,
- đóng gói baseline results.

### Bùi Quang Huy

Vai trò chính: literature, main model, methodology/report.

Công việc:

- viết literature map,
- train 1D-CNN/2D-CNN,
- compare models,
- viết methodology,
- viết experiment results,
- chuẩn bị phần phản biện.

---

## 30. Must-have, Should-have, Nice-to-have

### Must-have

- 4-dataset metadata clean.
- Label mapping 6 classes.
- EDA figures.
- Baseline results đã có.
- Main 1D-CNN hoặc 2D-CNN model.
- Model comparison table.
- Emotion prediction per chunk.
- VAD hoặc silence detection cơ bản.
- Smoothing đơn giản.
- Emotion timeline.
- Acoustic indicators: RMS, pitch, silence ratio, pause count.
- Rule-based feedback.
- Gradio upload demo.

### Should-have

- Strict speaker-aware evaluation.
- Cross-dataset test.
- Augmentation noise + shift.
- Raw vs smoothed timeline comparison.
- Speaking pressure score.
- Audio tự thu nhiều case.
- Transcript speech rate.

### Nice-to-have

- Filler words.
- CNN-GRU/CNN+BiLSTM.
- Silero VAD.
- Vietnamese sample.
- Wav2Vec2/HuBERT embeddings.
- Video demo backup.

### Future work

- Full real-time microphone streaming.
- Emotion diarization dataset có boundary.
- Vietnamese public speaking dataset.
- Multimodal video + audio.
- IELTS rubric.
- Clinical stress/depression datasets, nếu có ethical handling.

---

## 31. Rủi ro và cách xử lý

| Rủi ro | Tác động | Cách xử lý |
|---|---|---|
| Baseline strict split thấp | nhìn kết quả không đẹp | giải thích speaker/domain shift, dùng làm động lực cho CNN |
| TESS quá dễ | random accuracy ảo cao | báo per-dataset metrics, không overclaim |
| SAVEE nhỏ | model dễ bias | dùng per-dataset/cross-dataset analysis |
| 1D-CNN không vượt SVM nhiều | khó chọn main | thử 2D-CNN log-Mel, kiểm tra feature sequence |
| Data leakage | kết quả sai lệch | split trước augmentation/segmentation |
| Timeline nhảy | feedback rối | smoothing + threshold |
| Predict trên silence | timeline sai | thêm VAD |
| Feedback bị chủ quan | khó đánh giá | dùng rule rõ ràng và audio test có kịch bản |
| Stress bị hiểu nhầm | phản biện y khoa | gọi là speaking pressure, không stress diagnosis |
| Whisper sai filler | feedback sai | để optional/experimental |

---

## 32. Câu hỏi phản biện và câu trả lời

### Câu 1: Vì sao random split cao hơn strict split?

Vì random split có thể để cùng speaker hoặc cùng corpus style xuất hiện trong cả train và test. Strict split khó hơn vì kiểm tra khả năng generalize sang speaker/corpus khác. Sự chênh lệch này là hiện tượng thường gặp trong SER.

### Câu 2: Vì sao baseline chưa cao?

Vì baseline dùng handcrafted mean/std features và classical ML. Nó không học temporal representation sâu như CNN/GRU. Mục tiêu của baseline là mốc so sánh, không phải model cuối cùng.

### Câu 3: Main model là gì?

Main model là 1D-CNN trên chuỗi đặc trưng MFCC/acoustic theo thời gian. Nếu 2D-CNN log-Mel cho kết quả tốt hơn trong thí nghiệm, nhóm sẽ dùng model tốt hơn cho demo và giữ 1D-CNN làm comparison.

### Câu 4: Dataset chỉ có label cho toàn audio, sao làm timeline?

Nhóm dùng chunk-based approximation. Audio dài được chia thành các chunk 3 giây. Mỗi chunk được xem như một utterance ngắn và model dự đoán emotion chủ đạo. Nhóm không khẳng định đây là frame-level emotion ground truth.

### Câu 5: Vì sao cần VAD?

VAD giúp tránh dự đoán emotion trên silence và đồng thời cung cấp pause/silence information cho feedback.

### Câu 6: Fear/sad có phải stress không?

Không. Emotion chỉ là tín hiệu phụ. Speaking pressure được tính từ pause, silence, energy, pitch, speech rate và filler. Hệ thống không chẩn đoán stress.

### Câu 7: Feedback có được train không?

Không. Feedback được sinh bằng rule-based engine vì không có dataset expert feedback đủ chuẩn. Các rule dựa trên acoustic indicators và transcript metrics.

---

## 33. Cấu trúc báo cáo cuối kỳ

1. Introduction
2. Problem Statement and Motivation
3. Literature Review
4. Dataset Description and Label Mapping
5. System Overview
6. Audio Preprocessing
7. Feature Extraction
8. Baseline Classical ML Models
9. Main SER Model: 1D-CNN
10. Comparison Model: 2D-CNN log-Mel
11. Advanced Temporal Model
12. VAD-guided Chunk-based Inference
13. Smoothing and Emotion Timeline
14. Acoustic Speaking Indicators
15. Speaking Pressure Indicator
16. Transcript Analysis
17. Feedback Engine
18. Dashboard Demo
19. Experiments and Results
20. Ablation Study
21. Discussion
22. Limitations
23. Future Work
24. Conclusion

---

## 34. Slide outline

1. Title and team
2. Motivation: presentation feedback problem
3. Why not stress detection / IELTS scoring
4. Proposed system overview
5. Dataset strategy: 4 datasets, 6 labels
6. Literature map
7. Baseline results
8. Main model decision: 1D-CNN / 2D-CNN
9. Evaluation protocols
10. VAD and sliding window
11. Smoothing and emotion timeline
12. Acoustic indicators
13. Speaking pressure score
14. Feedback engine
15. Dashboard demo
16. Results
17. Limitations
18. Future work
19. Q&A

---

## 35. Checklist cuối cùng

### Dataset

- [ ] 4 dataset load được.
- [ ] Metadata clean.
- [ ] Label mapping 6 lớp.
- [ ] Strict/random split.
- [ ] Per-dataset distribution.

### Baseline

- [x] 04_Baseline_Models hoàn thành.
- [x] Classical ML trained.
- [x] Random/strict results.
- [ ] Export figures clean.
- [ ] Viết report baseline.

### Main model

- [ ] Sequence feature extraction.
- [ ] Train 1D-CNN.
- [ ] Train/evaluate strict/random.
- [ ] Compare với baseline.
- [ ] Save best model.

### Comparison/advanced

- [ ] Train 2D-CNN log-Mel.
- [ ] Optional CNN-GRU.
- [ ] Measure inference time.

### Timeline

- [ ] VAD.
- [ ] Sliding window.
- [ ] Predict per chunk.
- [ ] Smoothing.
- [ ] Emotion timeline figure.

### Feedback

- [ ] RMS.
- [ ] Pitch.
- [ ] Silence ratio.
- [ ] Pause count.
- [ ] Pressure score.
- [ ] Transcript optional.
- [ ] Feedback rules.

### Demo

- [ ] Gradio app.
- [ ] Audio demo set.
- [ ] Screenshots.
- [ ] Backup video.
- [ ] Demo script.

### Report

- [ ] Literature review.
- [ ] Dataset section.
- [ ] Methodology.
- [ ] Results.
- [ ] Ablation.
- [ ] Limitations.
- [ ] Future work.

---

## 36. Kết luận roadmap v3

Roadmap v3 chốt project theo hướng:

```text
Voice-Based Presentation Feedback System
```

Lõi kỹ thuật là:

```text
4-dataset SER
+ strong classical baseline
+ main 1D-CNN / 2D-CNN model
+ VAD-guided chunk inference
+ smoothing
+ emotion timeline
+ acoustic speaking indicators
+ speaking pressure score
+ rule-based feedback
+ Gradio dashboard
```

Đóng góp chính của nhóm không phải là phát minh một model hoàn toàn mới, mà là xây dựng một pipeline ứng dụng đầy đủ từ training SER đến feedback luyện thuyết trình. Đây là hướng vừa có cơ sở học thuật, vừa khả thi trên Colab/Kaggle, vừa có demo rõ ràng để báo cáo.
