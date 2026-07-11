# Final Project Evaluation - Filled Proposal

## Tên đề tài đề xuất

**Speech Emotion Recognition Toward Presentation Feedback Analysis**

Tên này nối hợp lý từ midterm vì giai đoạn midterm nhóm đã tập trung vào **Speech Emotion Recognition** với mô hình chính 06D. Sang final, hướng mở rộng là dùng SER như một module phân tích sắc thái giọng nói, sau đó kết hợp thêm prosody, fluency, pause, speaking rate và transcript để tiến tới **Presentation Feedback Analysis**.

Tên demo/app có thể dùng:

**Speech Emotion and Prosody-Based Presentation Feedback System**

## Phân công từ Midterm sang Final

Phân công trong midterm đọc được từ `Midterm_Param/Speech_Processing.xlsx`:

| Thành viên | Phần midterm đã đảm nhiệm | Vai trò chính |
|---|---|---|
| Nguyễn Minh Cường | 1.1, 1.2, 1.3, 1.4 | Problem overview, related work, review models, so sánh approaches |
| Bùi Quang Huy | 1.5, 2.1, 3.1, 3.2, 3.3 | Proposed solution, architecture, implementation, experiments |
| Nguyễn Tài Huy | 2.2, 2.3, 2.4, 2.5 | Dataset structure, preprocessing, feature extraction, feature visualization |

Vì vậy ở final, cột D trong `Lasterm_Param/Speech_Processing.csv` được điền theo hai nguyên tắc:

- Mỗi phần lớn có tổng 60 điểm và ghi rõ **Maximum: 20 points per student**, nên trong từng phần 1, 2 và 3, mỗi thành viên phải nhận đúng 20 điểm.
- Trong giới hạn cân bằng điểm đó, các mục liên quan **implementation, model, architecture, experiment chính** ưu tiên giao cho **Bùi Quang Huy**.
- Các mục liên quan **dataset, feature, extended test scenarios, figures/visualization** ưu tiên giao cho **Nguyễn Tài Huy**.
- Các mục liên quan **literature review, report structure, citation, academic writing, UI/demo explanation** ưu tiên giao cho **Nguyễn Minh Cường**.

## Cân bằng điểm theo từng phần

| Phần | Bùi Quang Huy | Nguyễn Minh Cường | Nguyễn Tài Huy |
|---|---:|---:|---:|
| 1. Complete System Implementation | 20 | 20 | 20 |
| 2. Experimental Evaluation | 20 | 20 | 20 |
| 3. Project Report | 20 | 20 | 20 |

## Bảng phân công Final

| Mục | Người phụ trách | Lý do nối từ midterm | Công việc cần làm |
|---|---|---|---|
| 1.1 Complete speech processing pipeline | Nguyễn Tài Huy | Nối từ phần data/preprocessing/feature ở midterm, đồng thời giúp cân bằng phần 1 đủ 20 điểm/người | Mô tả pipeline dữ liệu từ audio -> preprocessing -> feature -> segment timeline -> presentation feedback cues |
| 1.2 Implement selected model | Bùi Quang Huy | Đã phụ trách 06D architecture và model training | Chốt 06D Emotion2Vec Co-Attention SER, clean notebook, package model/output |
| 1.3 Improvements to model/features/system | Nguyễn Tài Huy | Nối từ phần feature extraction/visualization ở midterm; phần improvement không chỉ là model mà còn gồm feature extraction và system architecture | Mô tả các cải tiến về feature, presentation-level cues, segment aggregation và system architecture |
| 1.4 Impact analysis through experiments | Nguyễn Minh Cường | Nối từ phần compare/review approaches ở midterm, đồng thời cân bằng điểm phần 1 | Viết phân tích impact ở mức report: vì sao cải tiến giúp/không giúp, so với references và các protocol |
| 1.5 UI or demo program | Nguyễn Minh Cường | Đã phụ trách overview/related work, phù hợp mô tả demo và user-facing flow | Viết kịch bản demo upload/live, giải thích input-output, giao diện và cách người dùng quan sát kết quả |
| 2.1 Dataset/test/evaluation scenario | Nguyễn Tài Huy | Đã phụ trách dataset structure/preprocessing/features | Mô tả 4 SER datasets và kịch bản mở rộng presentation audio 10-15 phút |
| 2.2 Training/testing/results collection | Bùi Quang Huy | Đã phụ trách implementation/experiment | Thu thập metrics CSV, curves, confusion matrices, prediction files |
| 2.3 Metric analysis | Nguyễn Minh Cường | Nối từ phần compare/evaluation trong related work; mục này thiên về diễn giải kết quả trong report | Giải thích accuracy, macro-F1, weighted-F1, recall/precision, inference time và ý nghĩa từng metric |
| 2.4 Compare with baselines/reference methods | Nguyễn Minh Cường | Đã review models/approaches ở midterm | So sánh với baseline 04, Emonity, CNN_LSTM_CLSTM, PuSQ/readys, SpeechMirror/SOPHIAS |
| 2.5 Extended evaluation | Nguyễn Tài Huy | Đã phụ trách data/feature analysis | Chuẩn bị noisy/microphone tests, single-dataset, strict speaker-aware, long-audio demo scenario |
| 3.1 Full report sections | Nguyễn Minh Cường | Đã phụ trách overview và related work | Đảm bảo report có Introduction, Background, Dataset, Proposed Method, Evaluation, Conclusion, References |
| 3.2 Development and evaluation process | Bùi Quang Huy | Đã phụ trách implementation | Viết quy trình phát triển hệ thống, notebook, training/testing, evaluation |
| 3.3 Figures/tables/diagrams/results | Nguyễn Tài Huy | Đã phụ trách feature visualization | Chuẩn bị hình kiến trúc, feature plots, confusion matrix, result tables, demo screenshots |
| 3.4 Citations and references | Bùi Quang Huy | Đã trực tiếp thu thập paper/code cho SER và presentation feedback, đồng thời cân bằng phần 3 đủ 20 điểm/người | Chuẩn hóa citation cho SER, emotion2vec, PuSQ, readys, SpeechMirror, SOPHIAS, TED Talk prosody |
| 3.5 Format and academic writing | Nguyễn Tài Huy | Nối từ phần figures/feature presentation; phụ trách rà lại caption, bảng biểu và lỗi trình bày, đồng thời cân bằng phần 3 | Rà lỗi chính tả, thuật ngữ, caption, bảng biểu, cấu trúc trình bày học thuật |

## Nội dung final nên bám theo rubric

### 1. Complete System Implementation

Pipeline cuối kỳ nên được mô tả như sau:

```text
Input speech/audio
-> 16 kHz preprocessing
-> segment audio into short windows
-> feature extraction
   - temporal acoustic features
   - log-Mel spectrogram features
   - frozen emotion2vec representation
   - handcrafted statistical features
   - presentation features: pause, silence ratio, pitch/F0, energy, speaking rate
-> 06D SER / affective speech module
-> segment-level emotion probabilities
-> emotion/prosody timeline aggregation
-> presentation feedback analysis layer
-> output: emotion trend, confidence, prosody cues, feedback summary
```

Mô hình chính vẫn là **06D Emotion2Vec Co-Attention SER**:

- Branch A: MFCC + delta + delta-delta + RMS/ZCR/spectral sequence -> 1D-CNN -> BiLSTM/GRU -> attention pooling.
- Branch B: log-Mel + delta log-Mel + delta-delta log-Mel -> 2D CNN / residual CNN -> SE/channel attention.
- Branch C: raw waveform -> frozen emotion2vec -> adapter MLP.
- Branch D: handcrafted statistical vector -> Stats MLP / RBF-SVM.
- Fusion: emotion2vec-guided co-attention + validation-weighted stacking.
- Output: 6 emotion probabilities: neutral, happy, sad, angry, fear, disgust.

Điểm quan trọng khi viết report:

> Emotion is not used as the final judgment of presentation quality. It is used as an affective speech cue, then combined with prosody, fluency and text-based features for presentation feedback.

### 2. Experimental Evaluation

Dataset chính:

- RAVDESS
- CREMA-D
- TESS
- SAVEE

Common labels:

```text
neutral, happy, sad, angry, fear, disgust
```

Các protocol cần báo cáo:

- `combined_random`: so sánh gần hơn với nhiều paper/repo dùng random split.
- `combined_strict_no_tess`: kiểm tra speaker-independent/domain generalization.
- `single_dataset`: train/test riêng CREMA-D, RAVDESS, SAVEE, TESS.
- `long_audio_demo`: audio thuyết trình 10-15 phút, chia segment để tạo timeline.

Metric:

- accuracy
- macro-F1
- weighted-F1
- precision / recall
- confusion matrix
- inference time per segment
- processing time cho audio dài

Khi so với repo accuracy cao, cần nói rõ:

- Emonity có 1D-CNN, 2D-CNN, CNN-BiLSTM, ensemble, nhưng có dấu hiệu augmentation trước split.
- CNN_LSTM_CLSTM báo accuracy rất cao, nhưng cũng augment trước train/test split nên dễ data leakage.
- Vì vậy kết quả strict của nhóm thấp hơn nhưng phản ánh generalization thực tế hơn.

### 3. Project Report

Report cuối kỳ nên có các phần:

1. Introduction
2. Theoretical Background
3. Related Work
4. Dataset
5. Data Processing and Feature Extraction
6. Proposed Method
7. Experimental Evaluation
8. Presentation Feedback Extension / Demo
9. Discussion
10. Conclusion and Future Work
11. References

References nên có:

- emotion2vec
- AST/log-Mel nếu còn dùng spectrogram branch
- CNN-BiLSTM/CNN-GRU/ensemble SER references
- PuSQ: Automatic Assessment of Speaking Skills Using Aural and Textual Information
- readys framework
- SpeechMirror
- SOPHIAS
- Predicting TED Talk Ratings from Language and Prosody
- Automated Presentation Coaching survey

## Ghi chú hướng phát triển cuối kỳ

Hướng cuối kỳ không phải bỏ SER, mà là nâng SER thành một module trong hệ thống lớn hơn:

```text
SER datasets
-> train affective speech model
-> get emotion posterior timeline
-> combine with prosody/fluency/text
-> presentation feedback dashboard/report
```

Điểm cộng học thuật nằm ở chỗ nhóm phân biệt rõ:

- acted SER datasets khác public speaking thật;
- emotion cue không đồng nghĩa presentation quality;
- muốn feedback thuyết trình cần multi-signal system;
- PuSQ/readys chứng minh emotion posterior có thể dùng như feature trung gian cho speaking quality assessment.
