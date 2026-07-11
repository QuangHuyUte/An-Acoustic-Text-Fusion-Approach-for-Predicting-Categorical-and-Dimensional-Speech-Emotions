---
title: "Roadmap v1 - Voice-Based Presentation Feedback System"
subtitle: "SER, VAD, Emotion Timeline, Acoustic Indicators, Transcript Metrics and Feedback Engine"
author: "Nhóm: Nguyễn Minh Cường, Nguyễn Tài Huy, Bùi Quang Huy"
date: "11/07/2026"
lang: "vi"
---

# Roadmap v1

## 0. Ghi chú hợp nhất

File này là **Roadmap v1 chính thức** của đề tài, được hợp nhất từ:

```text
RoadMap/roadmapv1.md
RoadMap/roadmapv2.md
RoadMap/roadmapv3.md
```

Phạm vi của Roadmap v1 chỉ hợp nhất roadmap nền từ v1/v2/v3. Chưa tính phần mở rộng trong `bonusv2.md`.

Kết luận sau khi rà soát:

| Bản cũ | Trạng thái | Giữ lại gì | Bỏ gì |
|---|---|---|---|
| `roadmapv1.md` | Lỗi thời phần lớn | problem framing ban đầu, module EDA/preprocessing/feedback | hướng CNN chung chung, stress-related wording quá rộng, timeline 4 tuần cũ |
| `roadmapv2.md` | Còn nhiều ý đúng | VAD, smoothing, chunk-based timeline, pressure indicator, Q&A phản biện | phần lặp lại dài, một số cấu trúc notebook cũ |
| `roadmapv3.md` | Là bản nền tốt nhất | 4-dataset SER, baseline status, 1D-CNN/2D-CNN, strict split, dashboard plan | các đoạn quá dài/lặp lại, path cũ dễ lỗi thời |

Roadmap v1 dùng `roadmapv3` làm xương sống, bổ sung các ý còn đúng từ v1/v2, và loại các phần đã lỗi thời.

---

## 1. Hướng đề tài cuối cùng

Tên đề tài nên dùng:

```text
Voice-Based Presentation Feedback System
```

Tên tiếng Việt:

```text
Hệ thống phản hồi luyện thuyết trình dựa trên phân tích giọng nói
```

Mục tiêu không phải chỉ dự đoán một nhãn cảm xúc cho cả audio. Hệ thống cần:

- nhận audio thuyết trình hoặc luyện nói;
- dự đoán cảm xúc theo từng đoạn ngắn;
- tạo emotion timeline;
- tính các chỉ báo âm học như energy, pitch, silence, pause;
- phân tích speech rate và filler words nếu có transcript;
- sinh feedback giúp người dùng cải thiện giọng nói, tốc độ, độ liền mạch và năng lượng khi trình bày.

Một câu mô tả dùng trong báo cáo:

> Đề tài xây dựng một hệ thống phản hồi luyện thuyết trình dựa trên giọng nói. Hệ thống huấn luyện mô hình Speech Emotion Recognition trên các dataset cảm xúc công khai, sau đó áp dụng mô hình theo từng đoạn audio ngắn để tạo emotion timeline. Kết quả emotion được kết hợp với VAD, smoothing, acoustic indicators, speech rate và filler words để sinh feedback giúp người dùng cải thiện độ tự tin, sự liền mạch, năng lượng và ngữ điệu khi trình bày.

---

## 2. Những hướng không nên gọi là mục tiêu chính

### 2.1. Không gọi là stress detection

Không nên gọi hệ thống là stress detector vì dataset chính là emotion datasets, không có ground truth stress/non-stress. Nếu gọi là stress detection sẽ dễ bị hỏi:

- stress được gán nhãn thế nào;
- có self-report hoặc physiological signal không;
- có rủi ro y khoa/đạo đức không.

Cách gọi an toàn hơn:

```text
Speaking Pressure Indicator
Vocal Stability Indicator
Stress-related Acoustic Indicator
Confidence/Pressure Indicator for Presentation Feedback
```

Emotion chỉ là tín hiệu phụ. Speaking pressure phải dựa trên nhiều feature như pause, silence ratio, pitch variation, energy instability, speech rate và filler words.

### 2.2. Không chuyển hẳn sang IELTS Speaking

IELTS Speaking cần grammar, vocabulary, fluency, coherence và pronunciation. Nếu chuyển sang IELTS, project sẽ phụ thuộc nhiều vào ASR, grammar scoring và LLM, làm phần SER trở thành phụ.

Presentation feedback phù hợp hơn vì emotion, energy, pitch, pause, speech rate và filler words liên quan trực tiếp tới cách trình bày.

---

## 3. Kiến trúc tổng thể

Project gồm hai pha:

```text
PHASE A - Training / Model Development
4 emotion datasets
-> preprocessing
-> feature extraction
-> baseline ML
-> main SER model
-> optional advanced model
-> evaluation
-> save best model

PHASE B - Application / Feedback System
user presentation audio
-> preprocessing
-> VAD / silence detection
-> sliding-window chunking
-> SER prediction per chunk
-> smoothing
-> emotion timeline
-> acoustic indicators
-> optional transcript metrics
-> speaking pressure score
-> feedback engine
-> dashboard
```

Không nên xây một model duy nhất đi từ audio thẳng ra feedback. Nên chia thành module nhỏ:

| Module | Vai trò |
|---|---|
| SER model | Dự đoán cảm xúc của từng audio/chunk |
| VAD | Xác định speech/silence, tránh predict emotion trên im lặng |
| Sliding window | Chia audio dài thành nhiều đoạn ngắn |
| Smoothing | Làm timeline ổn định, giảm nhảy nhãn |
| Acoustic indicators | Tính RMS, pitch, silence, pause, continuity |
| Transcript analysis | Tính WPM và filler words nếu dùng ASR |
| Feedback engine | Biến chỉ số thành nhận xét dễ hiểu |
| Dashboard | Hiển thị waveform, spectrogram, timeline, metrics, feedback |

---

## 4. Dataset strategy

### 4.1. Dataset chính cho SER

Phiên bản nền của project dùng 4 dataset emotion:

```text
CREMA-D
RAVDESS speech-only
TESS
SAVEE
```

Giữ 6 nhãn chung:

```text
neutral, happy, sad, angry, fear, disgust
```

Không dùng `calm` và `surprise` trong bản chính vì không xuất hiện đồng đều ở tất cả dataset.

| Dataset | Vai trò | Điểm mạnh | Hạn chế |
|---|---|---|---|
| CREMA-D | Dataset lõi quan trọng | Nhiều speaker, 6 emotion khớp tốt | Acted speech, khác presentation thật |
| RAVDESS speech-only | Dataset học thuật chuẩn | Phổ biến, dễ cite, audio sạch | Ít hơn CREMA-D nếu chỉ lấy speech |
| TESS | Bổ sung dữ liệu sạch | Nhiều mẫu, dễ train | Chỉ 2 female speakers, dễ làm kết quả ảo cao |
| SAVEE | Benchmark phụ trong 4-dataset setup | Có male British speakers | Chỉ 4 speakers, nhỏ |
| EmoDB | Optional/future benchmark | Phổ biến trong SER | Tiếng Đức, nhỏ, không cần cho bản chính |

### 4.2. Label mapping

| Unified label | CREMA-D | RAVDESS | TESS | SAVEE |
|---|---|---|---|---|
| neutral | neutral | neutral | neutral | neutral / n |
| happy | happy | happy | happy | happiness / h |
| sad | sad | sad | sad | sadness / sa |
| angry | anger | angry | angry | anger / a |
| fear | fear | fearful | fear | fear / f |
| disgust | disgust | disgust | disgust | disgust / d |

### 4.3. Evaluation protocols

Cần báo cáo ít nhất 3 protocol:

| Protocol | Mục đích |
|---|---|
| Paper-comparable random split | So sánh với paper/Kaggle, nhưng dễ hơn thực tế |
| Strict speaker-aware split | Đánh giá nghiêm túc hơn, tránh cùng speaker ở train/test |
| Cross-dataset evaluation | Kiểm tra generalization giữa corpus |

Nếu strict split thấp hơn random split nhiều, đây không phải lỗi. Đó là bằng chứng về speaker/domain shift trong SER.

### 4.4. Audio tự thu

Audio tự thu không dùng để train SER model. Nó dùng cho demo và kiểm thử feedback.

Nên thu 10-30 file, mỗi file 30-90 giây:

| Case | Mục đích kiểm thử |
|---|---|
| nói đều, ít pause | hệ thống báo ổn định |
| nói quá nhanh | speech rate cao |
| nói quá chậm | speech rate thấp |
| nói nhiều pause | silence ratio và pause count cao |
| nói nhỏ dần | energy decreasing |
| monotone | pitch variation thấp |
| nhiều filler | filler count cao |
| mở đầu tự tin, giữa ngập ngừng | timeline và feedback theo đoạn |

---

## 5. Trạng thái nền hiện tại

Theo roadmap v3, project đã có mốc baseline classical ML trên 4 dataset:

| Protocol | Best model | Test accuracy | Test macro-F1 | Ý nghĩa |
|---|---:|---:|---:|---|
| Paper-comparable random split | SVM RBF tuned | 66.61% | 66.64% | Dùng để so với paper/Kaggle, nhưng dễ hơn |
| Strict speaker-aware split | Probability-average ensemble | 47.06% | 46.53% | Khó hơn, thực tế hơn, phản ánh speaker/domain shift |

Cách diễn giải đúng:

> Với random split dùng để so sánh với các paper/Kaggle, SVM RBF đạt 66.61% accuracy và 66.64% macro-F1. Tuy nhiên, khi dùng strict speaker-aware split, kết quả giảm xuống 47.06% accuracy và 46.53% macro-F1. Điều này cho thấy bài toán SER trên nhiều corpus chịu ảnh hưởng mạnh bởi speaker identity, dataset domain, accent và recording condition.

Không nên dùng classical ML baseline làm final model. Nó nên giữ vai trò:

- mốc so sánh truyền thống;
- bằng chứng handcrafted features có hiệu quả cơ bản;
- mô hình dự phòng nếu deep learning gặp lỗi;
- phần giải thích học thuật trong report.

---

## 6. Model strategy

### 6.1. Tầng model

| Vai trò | Model | Input | Mục tiêu |
|---|---|---|---|
| Completed baseline | SVM/RF/ensemble classical ML | handcrafted mean/std features | Mốc so sánh |
| Main model | 1D-CNN feature sequence | MFCC + delta + ZCR/RMS/energy theo thời gian | Học temporal acoustic pattern |
| Comparison model | 2D-CNN log-Mel | log-Mel spectrogram | So sánh representation khác |
| Advanced optional | CNN-GRU / CNN-BiLSTM / attention | sequence feature hoặc log-Mel | Nếu còn thời gian |

### 6.2. Main model nên ưu tiên 1D-CNN feature sequence

1D-CNN không nên nhận vector mean/std 248 chiều như một chuỗi giả. Nó nên nhận chuỗi feature theo thời gian:

```text
audio
-> frame-level feature extraction
-> [time_steps, feature_dim]
-> 1D-CNN along time axis
-> pooling
-> dense classifier
-> emotion posterior
```

Feature gợi ý:

```text
MFCC
delta MFCC
delta-delta MFCC
RMS / energy
ZCR
spectral centroid
spectral rolloff
```

1D-CNN hợp với project vì:

- nhẹ hơn transformer/SSL fine-tuning;
- phù hợp Kaggle/Colab;
- giữ được temporal pattern;
- dễ dùng cho sliding-window inference;
- dễ giải thích hơn model quá lớn.

### 6.3. 2D-CNN log-Mel vẫn nên giữ để so sánh

2D-CNN log-Mel là hướng phổ biến trong SER:

```text
audio
-> log-Mel spectrogram
-> 2D-CNN
-> emotion posterior
```

Dùng 2D-CNN để so sánh:

- handcrafted temporal sequence vs spectrogram image;
- inference time;
- random split và strict split;
- khả năng dùng cho demo.

Nếu 2D-CNN tốt hơn 1D-CNN rõ ràng, có thể chọn 2D-CNN làm demo model và giữ 1D-CNN làm comparison.

### 6.4. Advanced model

Chỉ làm advanced model nếu baseline/main/comparison đã ổn:

```text
CNN-GRU
CNN-BiLSTM
Attention pooling
Wav2Vec2/HuBERT embedding extraction
```

Không nên để advanced model làm chậm dashboard hoặc khiến demo thiếu ổn định.

---

## 7. Preprocessing và feature extraction

### 7.1. Preprocessing

Các bước chuẩn:

```text
load audio
-> convert mono
-> resample 16 kHz
-> normalize volume
-> optional trim leading/trailing silence
-> pad/crop for short-utterance training
-> save metadata and split
```

Lưu ý data leakage:

> Split train/validation/test phải làm trước augmentation và trước khi cắt segment mở rộng dữ liệu.

### 7.2. Feature cho baseline

Baseline classical ML có thể dùng:

```text
MFCC mean/std
delta mean/std
RMS mean/std
ZCR mean/std
spectral centroid/rolloff/bandwidth
```

### 7.3. Feature cho main 1D-CNN

Main 1D-CNN cần sequence feature:

```text
X shape = [num_samples, time_steps, feature_dim]
```

Không dùng toàn bộ feature đã aggregate thành mean/std cho 1D-CNN.

### 7.4. Feature cho feedback

| Feature | Vai trò feedback |
|---|---|
| RMS / energy | Giọng quá nhỏ, quá lớn, giảm dần hoặc thiếu ổn định |
| Pitch/F0 mean | Cao độ trung bình |
| Pitch/F0 std/range | Monotone hoặc dao động quá mạnh |
| Silence ratio | Nhiều khoảng im lặng |
| Pause count/long pause | Ngập ngừng, đứt đoạn |
| Speech rate/WPM | Nói quá nhanh hoặc quá chậm |
| Filler count | Nhiều từ đệm |

---

## 8. VAD, sliding window và smoothing

### 8.1. VAD

VAD dùng để xác định speech/silence. Không có VAD, hệ thống có thể predict emotion trên chunk im lặng, làm timeline sai.

Ưu tiên:

| VAD | Ưu điểm | Nhược điểm |
|---|---|---|
| WebRTC VAD | nhẹ, nhanh, hợp demo | nhạy với frame/noise |
| Silero VAD | robust hơn | nặng hơn, thêm dependency |

Demo có thể bắt đầu bằng WebRTC hoặc silence detection đơn giản; Silero là should-have.

### 8.2. Sliding window

Gợi ý:

```text
window = 3s
hop = 1.5s
```

Với audio dài, mỗi chunk được xem như một utterance ngắn. Đây là chunk-based approximation, không phải frame-level ground truth.

### 8.3. Smoothing

Smoothing cần thiết vì prediction từng chunk dễ nhảy. Có thể dùng:

- moving average probability;
- majority vote theo 3-5 chunks;
- confidence threshold;
- giữ nhãn trước nếu confidence quá thấp.

Nên hiển thị cả raw timeline và smoothed timeline trong report/demo để chứng minh smoothing có tác dụng.

---

## 9. Emotion timeline

Output timeline nên gồm:

```text
start_time
end_time
speech_ratio
predicted_emotion
emotion_confidence
top2_emotion
is_uncertain
```

Timeline không nên được mô tả là ground-truth emotion boundary. Cách gọi đúng:

```text
VAD-guided chunk-based emotion timeline
```

---

## 10. Speaking pressure và acoustic feedback

Speaking pressure không phải stress diagnosis. Nó là điểm tổng hợp từ các chỉ báo âm học.

Feature có thể dùng:

```text
silence_ratio
pause_count
long_pause_count
energy_instability
pitch_variation
speech_rate
filler_count
emotion_uncertainty
```

Rule ví dụ:

| Dấu hiệu | Feedback |
|---|---|
| neutral cao + pitch variance thấp + energy đều | đoạn này có dấu hiệu monotone |
| silence ratio cao hoặc pause dài | đoạn này có nhiều khoảng ngắt |
| energy thấp + speech rate thấp | đoạn này thiếu năng lượng |
| energy cao + high-arousal emotion cao | đoạn này có thể nghe căng/gắt |
| WPM quá cao | đoạn này nói quá nhanh |
| WPM quá thấp | đoạn này nói hơi chậm |
| filler nhiều | nên thay filler bằng pause ngắn có kiểm soát |

---

## 11. Transcript analysis

Transcript là optional/should-have, không phải phần bắt buộc đầu tiên.

Nguồn:

```text
Whisper tiny/base
```

Metrics:

- word count;
- words per minute;
- filler words;
- repeated words/phrases;
- long silence aligned with transcript;
- optional Vietnamese filler words.

Filler words tiếng Anh:

```text
um, uh, er, ah, like, you know, actually, basically, I mean
```

Filler words tiếng Việt có thể thử:

```text
ờ, ừm, à, thì, là, kiểu như, nói chung là, thật ra là
```

---

## 12. Feedback engine

Nên bắt đầu bằng rule-based engine, không dùng LLM làm bộ phân tích chính.

Pipeline:

```text
features + timeline
-> rule detection
-> issue list
-> template feedback
-> optional LLM rewrite for smoother text
```

Feedback nên có 3 tầng:

1. Overall feedback.
2. Timeline feedback theo đoạn.
3. Improvement suggestions cụ thể.

Ví dụ output:

> Bài nói của bạn có nhiều đoạn neutral và pitch variation thấp, đặc biệt ở 00:30-01:00. Đoạn này có thể nghe hơi đều giọng. Bạn nên nhấn mạnh các từ khóa chính và thay đổi cao độ ở phần chuyển ý.

---

## 13. Dashboard demo

Ưu tiên Gradio vì dễ upload/record audio.

Dashboard nên có:

| Khối | Nội dung |
|---|---|
| Audio input | Upload hoặc record |
| Waveform | Tín hiệu audio |
| Spectrogram | Mel/log-Mel |
| Overall emotion | Emotion chính và confidence |
| Emotion distribution | Tỷ lệ emotion toàn bài |
| Raw timeline | Prediction per chunk |
| Smoothed timeline | Timeline sau smoothing |
| VAD plot | Speech/silence |
| Acoustic indicators | RMS, pitch, silence, pause |
| Transcript | Nếu bật ASR |
| Speech metrics | WPM, filler |
| Feedback text | Nhận xét và gợi ý |

Không nên phụ thuộc vào microphone streaming thật khi báo cáo. Upload/record rồi phân tích là ổn định hơn.

---

## 14. Evaluation plan

### 14.1. SER model

Báo cáo:

- accuracy;
- macro-F1;
- weighted-F1;
- precision/recall per class;
- confusion matrix;
- per-dataset metrics;
- random vs strict split;
- optional cross-dataset evaluation.

### 14.2. VAD

Nếu không có ground truth VAD, đánh giá định tính:

- silence chunk có bị predict emotion không;
- pause count có hợp lý không;
- timeline có sạch hơn không.

### 14.3. Smoothing

Đánh giá:

- emotion transition count trước/sau smoothing;
- số uncertain chunks;
- ví dụ timeline raw vs smoothed;
- nhận xét định tính.

### 14.4. Feedback

Dùng audio tự thu theo kịch bản:

| Audio test | Kỳ vọng |
|---|---|
| nhiều pause | báo pause/silence cao |
| nói nhanh | báo WPM cao |
| nói chậm | báo WPM thấp |
| nói nhỏ dần | báo energy decreasing |
| monotone | báo pitch variation thấp |
| nhiều filler | báo filler count cao |

### 14.5. Latency

Đo:

- preprocessing time;
- feature extraction time per chunk;
- model inference time per chunk;
- total processing time for 60s audio;
- dashboard response time.

---

## 15. Notebook roadmap

Tên notebook có thể điều chỉnh theo cấu trúc repo hiện tại, nhưng logic nên giữ như sau:

| Notebook | Nội dung | Output |
|---|---|---|
| 01 - Data Preparation | load datasets, parse labels, map 6 classes, split | metadata, label mapping, split config |
| 02 - EDA | distribution, duration, waveform, MFCC/Mel examples | figures for report |
| 03 - Feature Extraction | baseline features, sequence features, log-Mel | feature files/config |
| 04 - Baseline Models | classical ML, random/strict metrics | baseline metrics, confusion matrix |
| 05 - Main SER Model | 1D-CNN feature sequence hoặc chosen main model | model, metrics |
| 06 - Comparison Model | 2D-CNN log-Mel comparison | comparison table |
| 07 - Advanced Optional | CNN-GRU/BiLSTM/attention nếu còn thời gian | optional metrics |
| 08 - VAD + Timeline | VAD, chunking, smoothing, timeline | timeline figures/csv |
| 09 - Feedback Engine | acoustic indicators, pressure score, transcript optional | feedback report |
| 10 - Dashboard | Gradio app | demo |

---

## 16. Source structure gợi ý

```text
voice-presentation-feedback/
|
|-- data/
|   |-- raw/
|   |-- processed/
|   |-- demo_audio/
|   `-- metadata_clean.csv
|
|-- notebooks/
|   |-- 01_data_preparation.ipynb
|   |-- 02_eda_visualization.ipynb
|   |-- 03_feature_extraction.ipynb
|   |-- 04_baseline_models.ipynb
|   |-- 05_main_ser_model.ipynb
|   |-- 06_comparison_model.ipynb
|   |-- 07_vad_timeline_smoothing.ipynb
|   |-- 08_feedback_engine.ipynb
|   `-- 09_gradio_demo.ipynb
|
|-- src/
|   |-- config.py
|   |-- label_parsers.py
|   |-- audio_io.py
|   |-- preprocessing.py
|   |-- features_baseline.py
|   |-- features_sequence.py
|   |-- features_mel.py
|   |-- models_ser.py
|   |-- train_utils.py
|   |-- evaluation.py
|   |-- vad.py
|   |-- smoothing.py
|   |-- timeline.py
|   |-- acoustic_indicators.py
|   |-- pressure_score.py
|   |-- transcript_analysis.py
|   `-- feedback_engine.py
|
|-- models/
|-- figures/
|-- results/
|-- app/
`-- reports/
```

---

## 17. Timeline triển khai từ hiện tại

### Tuần 1 - Main SER model

- Tạo/load sequence feature.
- Train 1D-CNN hoặc chọn main model đang tốt nhất trong repo.
- Evaluate random và strict split.
- So sánh với classical baseline.
- Lưu model, metrics, confusion matrix.

Deliverables:

```text
main_ser_model
model_metrics.json
confusion_matrix.png
model_comparison_table.csv
```

### Tuần 2 - Comparison và ablation

- Train/check 2D-CNN log-Mel.
- Optional CNN-GRU/BiLSTM nếu còn thời gian.
- Đo inference time.
- Chốt model demo.

### Tuần 3 - VAD, sliding window, smoothing

- Thu audio demo.
- Implement VAD/silence detection.
- Chunking 3s, hop 1.5s.
- Predict per chunk.
- Smooth probabilities.
- Plot raw vs smoothed timeline.

### Tuần 4 - Feedback, dashboard, report

- Compute RMS, pitch, silence, pause.
- Compute speaking pressure score.
- Add transcript/WPM/filler nếu kịp.
- Build Gradio dashboard.
- Chuẩn bị screenshots, demo script, report/slide.

---

## 18. Task assignment

| Thành viên | Vai trò chính | Công việc |
|---|---|---|
| Nguyễn Minh Cường | EDA, acoustic indicators, timeline, dashboard | EDA figures, RMS/pitch/silence/pause, pressure score, timeline visualization, Gradio |
| Nguyễn Tài Huy | Dataset, preprocessing, feature extraction, baseline | metadata, label mapping, split, feature extraction, baseline packaging |
| Bùi Quang Huy | Literature, main model, methodology/report | literature map, main/comparison model, model comparison, methodology, results, limitation |

---

## 19. Must-have, Should-have, Nice-to-have

### Must-have

- 4-dataset metadata clean.
- Label mapping 6 classes.
- EDA figures.
- Baseline classical ML result.
- Main SER model.
- Model comparison table.
- Chunk-based prediction.
- VAD hoặc silence detection cơ bản.
- Smoothing đơn giản.
- Emotion timeline.
- Acoustic indicators: RMS, pitch, silence ratio, pause count.
- Rule-based feedback.
- Gradio upload demo.

### Should-have

- Strict speaker-aware evaluation.
- Cross-dataset test.
- Augmentation noise/shift.
- Raw vs smoothed timeline comparison.
- Speaking pressure score.
- Audio tự thu nhiều case.
- Transcript speech rate.

### Nice-to-have

- Filler words.
- CNN-GRU/CNN-BiLSTM.
- Silero VAD.
- Vietnamese sample.
- Wav2Vec2/HuBERT embeddings.
- Backup demo video.

### Future work

- Full microphone streaming.
- Emotion diarization dataset có boundary.
- Vietnamese public speaking dataset.
- Multimodal video + audio.
- IELTS rubric.
- Clinical stress/depression datasets only if ethical handling is clear.

---

## 20. Rủi ro và cách xử lý

| Rủi ro | Tác động | Cách xử lý |
|---|---|---|
| Strict split thấp | kết quả nhìn không đẹp | giải thích speaker/domain shift |
| TESS quá dễ | random accuracy ảo cao | báo per-dataset metrics |
| SAVEE nhỏ | model bias | dùng per-dataset/cross-dataset analysis |
| Main model không vượt SVM nhiều | khó chọn model | thử 2D-CNN/log-Mel, phân tích latency và strict split |
| Data leakage | kết quả sai | split trước augmentation/segmentation |
| Timeline nhảy | feedback rối | smoothing + threshold |
| Predict trên silence | timeline sai | thêm VAD/silence filtering |
| Feedback chủ quan | khó đánh giá | rule rõ, audio test có kịch bản |
| Stress bị hiểu nhầm | phản biện y khoa | gọi là speaking pressure, không stress diagnosis |
| ASR sai filler | feedback sai | để optional/experimental |

---

## 21. Câu hỏi phản biện và câu trả lời

### Câu 1: Vì sao random split cao hơn strict split?

Random split có thể để cùng speaker hoặc cùng corpus style xuất hiện trong cả train và test. Strict split khó hơn vì kiểm tra khả năng generalize sang speaker/corpus khác. Sự chênh lệch này là hiện tượng thường gặp trong SER.

### Câu 2: Dataset chỉ có label cho toàn audio, sao làm timeline?

Nhóm dùng chunk-based approximation. Audio dài được chia thành các chunk ngắn, mỗi chunk được xem như một utterance ngắn và model dự đoán emotion chủ đạo. Nhóm không khẳng định đây là frame-level emotion ground truth.

### Câu 3: Vì sao cần VAD?

VAD giúp tránh dự đoán emotion trên silence và đồng thời cung cấp pause/silence information cho feedback.

### Câu 4: Fear/sad có phải stress không?

Không. Emotion chỉ là tín hiệu phụ. Speaking pressure được tính từ pause, silence, energy, pitch, speech rate và filler. Hệ thống không chẩn đoán stress.

### Câu 5: Feedback có được train không?

Không ở bản nền. Feedback được sinh bằng rule-based engine vì chưa có dataset expert feedback đủ chuẩn. Các rule dựa trên acoustic indicators và transcript metrics.

---

## 22. Cấu trúc báo cáo

1. Introduction
2. Problem Statement and Motivation
3. Literature Review
4. Dataset Description and Label Mapping
5. System Overview
6. Audio Preprocessing
7. Feature Extraction
8. Baseline Classical ML Models
9. Main SER Model
10. Comparison / Advanced Model
11. VAD-guided Chunk-based Inference
12. Smoothing and Emotion Timeline
13. Acoustic Speaking Indicators
14. Speaking Pressure Indicator
15. Transcript Analysis
16. Feedback Engine
17. Dashboard Demo
18. Experiments and Results
19. Ablation Study
20. Discussion
21. Limitations
22. Future Work
23. Conclusion

---

## 23. Slide outline

1. Title and team
2. Motivation: presentation feedback problem
3. Why not stress detection / IELTS scoring
4. Proposed system overview
5. Dataset strategy: 4 datasets, 6 labels
6. Literature map
7. Baseline results
8. Main model decision
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

## 24. Checklist cuối cùng

### Dataset

- [ ] 4 dataset load được.
- [ ] Metadata clean.
- [ ] Label mapping 6 lớp.
- [ ] Strict/random split.
- [ ] Per-dataset distribution.

### Baseline

- [x] Classical ML baseline đã có theo roadmap v3.
- [ ] Export figures clean.
- [ ] Viết report baseline.

### Main model

- [ ] Sequence feature extraction.
- [ ] Train/evaluate main SER model.
- [ ] Compare với baseline.
- [ ] Save best model.

### Comparison/advanced

- [ ] Train/check 2D-CNN log-Mel.
- [ ] Optional CNN-GRU/BiLSTM.
- [ ] Measure inference time.

### Timeline

- [ ] VAD/silence detection.
- [ ] Sliding window.
- [ ] Predict per chunk.
- [ ] Smoothing.
- [ ] Emotion timeline figure.

### Feedback

- [ ] RMS.
- [ ] Pitch.
- [ ] Silence ratio.
- [ ] Pause count.
- [ ] Speaking pressure score.
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

## 25. Kết luận

Roadmap hợp nhất chốt project theo hướng:

```text
Voice-Based Presentation Feedback System
```

Lõi kỹ thuật:

```text
4-dataset SER
+ classical baseline
+ main SER model
+ VAD-guided chunk inference
+ smoothing
+ emotion timeline
+ acoustic speaking indicators
+ speaking pressure score
+ rule-based feedback
+ Gradio dashboard
```

Đóng góp chính không phải phát minh một model hoàn toàn mới, mà là xây dựng một pipeline ứng dụng đầy đủ từ training SER đến feedback luyện thuyết trình. Đây là hướng có cơ sở học thuật, khả thi trên Colab/Kaggle, và có demo rõ ràng.
