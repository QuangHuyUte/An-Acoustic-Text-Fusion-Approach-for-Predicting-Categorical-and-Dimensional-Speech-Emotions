---
title: "Roadmap v2 - TED Presentation Quality Prediction and Speech Feedback"
subtitle: "TED baselines, 06D affective speech cues, PuSQ-style speaking-quality features, and final-project rubric alignment"
author: "Nhóm: Nguyễn Minh Cường, Nguyễn Tài Huy, Bùi Quang Huy"
date: "11/07/2026"
lang: "vi"
---

# Roadmap v2

## 0. Mục tiêu của Roadmap v2

Roadmap v1 là bản nền cho hệ thống Speech Emotion Recognition và feedback giọng nói. Roadmap v2 mở rộng hướng đề tài sang bài toán chính:

```text
Presentation Quality Prediction and Feedback
```

Nói cách khác, v2 không còn là project xoay quanh emotion-only. Emotion chỉ là một nhóm tín hiệu trung gian trong hệ thống lớn hơn.

Hướng đúng của v2:

```text
TED Talk rating prediction baselines
+ transcript / syntax features
+ prosody / speech features
+ 06D emotion and affective speech cues
+ PuSQ-inspired speaking-quality cues
-> presentation quality prediction
-> localized presentation feedback
```

Mục tiêu cuối không chỉ là dự đoán một điểm/rating, mà là giải thích:

- bài nói có khả năng được đánh giá tốt/yếu ở khía cạnh nào;
- tín hiệu nào góp phần vào đánh giá đó;
- đoạn nào có dấu hiệu monotone, thiếu năng lượng, nói nhanh/chậm, nhiều pause, hoặc thiếu expressiveness.

---

## 1. Liên hệ với rubric cuối kỳ

Roadmap v2 bám theo hai file trong thư mục `Lasterm_Param`:

- `Lasterm_Param/Lasterm_Param_Filled_Proposal.md`
- `Lasterm_Param/Speech_Processing.csv`

Rubric cuối kỳ có ba phần chính:

| Phần | Ý nghĩa trong v2 | Deliverable cần có |
|---|---|---|
| 1. Complete System Implementation | Xây pipeline TED rating + speech feedback | data pipeline, model pipeline, demo upload/audio |
| 2. Experimental Evaluation | Chạy baseline và ablation | AUC/F1, speech-only, text-only, text+speech, processing time |
| 3. Project Report | Viết báo cáo học thuật | related work, model diagrams, tables, citations |

Vì vậy Roadmap v2 phải trả lời được:

- dataset nào dùng;
- baseline TED nào cần reproduce;
- speech/audio đóng góp gì nếu transcript already strong;
- 06D và PuSQ có vai trò gì;
- experiment nào chứng minh được đóng góp;
- demo cuối kỳ hiển thị được gì cho người dùng.

---

## 2. Định hướng đề tài v2

Tên đề tài nên dùng:

```text
Presentation Quality Prediction and Feedback using TED Talk Ratings,
Transcript, Prosody, and Affective Speech Cues
```

Tên tiếng Việt:

```text
Hệ thống dự đoán và phản hồi chất lượng thuyết trình dựa trên TED Talk ratings,
transcript, prosody và tín hiệu cảm xúc từ giọng nói.
```

Mục tiêu nghiên cứu:

```text
Text cho biết người nói nói gì.
Speech cho biết người nói nói như thế nào.
Presentation feedback cần cả hai.
```

Điểm cần nhấn mạnh trong báo cáo:

> TED text baselines can predict audience ratings reasonably well, but transcript-only models cannot explain vocal delivery problems such as monotone speech, weak energy, long pauses, unstable rhythm, or lack of expressiveness. Therefore, speech-based cues are needed for feedback even when text-only prediction is strong.

---

## 3. Dataset chính: TED Talk ratings

Nguồn dataset khả thi:

- GitHub: <https://github.com/xingbow/TED_dataset>
- Paper VoiceCoach: <https://arxiv.org/abs/2001.07876>
- Local: `Papers/Papers for presentation feedback system/code/TED_dataset_xingbow_check`

Dataset này chứa metadata của 2623 TED Talks theo README của repo:

- author;
- datefilmed;
- totalviews;
- comments;
- language;
- downloadlink;
- vidlen;
- AWS transcripts;
- datecrawled;
- datepublished;
- title;
- id;
- url;
- keywords;
- videoname;
- ratings;
- complete JSON field.

Điểm quan trọng cho hướng speech:

```text
Dataset có audio/video download link.
Một số JSON có audioDownload mp3 link.
Ta có thể tải mp3/video rồi convert sang wav 16 kHz mono để chạy speech models.
```

Target labels của TED rating:

```text
beautiful, confusing, courageous, fascinating, funny, informative,
ingenious, inspiring, jaw-dropping, longwinded, obnoxious,
ok, persuasive, unconvincing
```

Chiến lược label nên dùng:

```text
scaled rating count
-> threshold by median
-> binary label per rating
-> multi-label classification with 14 sigmoid outputs
```

Nếu cần giảm scope cho final, nên chọn subset:

```text
positive / presentation quality:
fascinating, inspiring, persuasive, informative

negative / weakness:
confusing, longwinded, unconvincing

control:
ok
```

---

## 4. Paper TED baseline chính

Paper chính:

- Predicting TED Talk Ratings from Language and Prosody: <https://arxiv.org/abs/1906.03940>
- Local PDF: `Papers/Papers for presentation feedback system/papers/Predicting_TED_Talk_Ratings_from_Language_and_Prosody.pdf`
- Local text: `Papers/Papers for presentation feedback system/papers/Predicting_TED_Talk_Ratings_from_Language_and_Prosody.txt`

Paper dùng 2231 talks sau khi lọc:

- bỏ talks published less than 6 months before crawl date;
- bỏ talks có keywords live music, dance, music, performance, entertainment;
- bỏ talks có transcript dưới 450 words;
- tổng 513.49 hours;
- 5,574,444 ratings;
- average ratings per talk 2498.6;
- total word count 5,489,628;
- total sentence count 295,338.

Train/test:

- rating counts được scale theo tổng rating của từng talk;
- scale giúp giảm bias từ total views và age of talks;
- scaled ratings được binarize bằng median threshold;
- 150 randomly sampled TED talks được giữ làm reserved test set;
- phần còn lại split train/development theo tỷ lệ 9:1;
- metric chính: AUC, F-score, Precision, Recall.

Lưu ý quan trọng:

```text
Paper không báo accuracy cho ba neural TED baselines.
Paper báo AUC, F-score, Precision, Recall.
```

---

## 5. Thông số và kết quả các baseline TED

### 5.1. Word Sequence Model

Input:

```text
Transcript -> sentences -> word sequence
```

Feature:

```text
pretrained GloVe 300-dimensional word vectors
```

Architecture:

```text
words in a sentence
-> LSTM
-> sentence embedding
-> average all sentence embeddings in one talk
-> feed-forward layer
-> 14 sigmoid rating outputs
```

Kết quả paper:

| Metric | Value |
|---|---:|
| Avg AUC | 0.83 |
| Avg F-score | 0.76 |
| Avg Precision | 0.76 |
| Avg Recall | 0.76 |

Vai trò trong project:

```text
Text-only neural baseline.
Đây là baseline đầu tiên nên reproduce vì dễ hơn Dependency Tree.
```

### 5.2. Dependency Tree Model

Input:

```text
Transcript -> sentences -> dependency parse tree
```

Parser:

```text
SyntaxNet
```

Feature mỗi node:

```text
GloVe word vector
+ POS embedding
+ dependency type embedding
```

Architecture:

```text
dependency tree
-> Child-Sum TreeLSTM
-> root node hidden state as sentence embedding
-> average all sentence embeddings in one talk
-> feed-forward layer
-> 14 sigmoid rating outputs
```

Kết quả paper:

| Metric | Value |
|---|---:|
| Avg AUC | 0.83 |
| Avg F-score | 0.77 |
| Avg Precision | 0.77 |
| Avg Recall | 0.77 |

Vai trò trong project:

```text
Strong text/syntax baseline.
Paper xem đây là model text tốt hơn Word Sequence theo F-score/Recall.
```

### 5.3. Dependency Tree + Prosody CNN

Đây là model quan trọng nhất cho phần audio vì nó là TED baseline có speech/prosody.

Input text:

```text
Dependency Tree embedding từ TreeLSTM
```

Input speech/audio:

```text
TED audio
-> forced alignment với transcript
-> PRAAT feature extraction
```

Prosody features:

```text
pitch
loudness
formant 1 frequency + bandwidth
formant 2 frequency + bandwidth
formant 3 frequency + bandwidth
```

Tổng cộng mỗi frame là vector 8 chiều. Feature được sample ở 10 Hz, normalize theo whole video, crop theo từng câu, rồi pad theo sentence length dài nhất.

Prosody CNN:

```text
Input: sentence-level prosody sequence, 8-D at 10 Hz

Conv1D layer 1: 16 filters, kernel/receptive field 3
Conv1D layer 2: 16 filters, kernel/receptive field 3 + max pooling window 2
Conv1D layer 3: 32 filters, kernel/receptive field 3 + max pooling window 2
Conv1D layer 4: 64 filters, kernel/receptive field 3 + global max pooling

Output: 64-D prosody vector
```

Fusion:

```text
Dependency Tree sentence embedding
+ 64-D Prosody CNN vector
-> two fully connected layers
-> 14 sigmoid rating outputs
```

Kết quả paper:

| Metric | Value |
|---|---:|
| Avg AUC | 0.83 |
| Avg F-score | 0.72 |
| Avg Precision | 0.75 |
| Avg Recall | 0.73 |

Kết luận từ paper:

```text
Thêm prosody không cải thiện rõ.
AUC vẫn 0.83, nhưng F-score giảm từ 0.77 xuống 0.72 so với Dependency Tree text-only.
```

Tác giả giải thích:

```text
TED Talks là bài nói được luyện tập kỹ, nên prosody có thể mang tính rehearsed/acted
và không thêm nhiều thông tin ngoài transcript.
```

Vai trò trong project:

```text
Đây là audio/prosody baseline quan trọng.
Ta dùng nó làm điểm xuất phát để chứng minh phần audio branch của mình cần mạnh hơn
low-level prosody thông thường.
```

### 5.4. Classical baselines

Feature:

- LIWC language features;
- summary statistics của pitch, loudness, formants;
- pause duration;
- percentage of unvoiced frames;
- jitter;
- shimmer;
- percentage of breaks in speech;
- narrative trajectory features.

Classifiers:

- Linear SVM;
- Ridge;
- LASSO.

Kết quả:

| Model | Avg AUC | Avg F-score | Avg Precision | Avg Recall |
|---|---:|---:|---:|---:|
| Linear SVM | 0.78 | 0.71 | 0.71 | 0.71 |
| Ridge | 0.78 | 0.71 | 0.71 | 0.71 |
| LASSO | 0.77 | 0.70 | 0.70 | 0.70 |

Vai trò trong project:

```text
Classical baseline dễ reproduce.
Nên chạy trước neural baselines để xác nhận dataset/label pipeline đúng.
```

---

## 6. Vấn đề: text-only đã mạnh thì cần speech để làm gì?

Nếu mục tiêu chỉ là predict rating, text-only có thể đã rất mạnh. Tuy nhiên project của nhóm là presentation feedback system, không phải chỉ là TED rating predictor.

Text-only biết:

```text
người nói nói gì
nội dung có informative/persuasive không
cấu trúc câu/ngôn ngữ như thế nào
```

Text-only không biết:

```text
giọng có monotone không
có pause dài không
có thiếu năng lượng không
có nói quá nhanh/chậm không
có biến thiên pitch/energy không
có expressive/enjoyable trong delivery không
```

Vì vậy lý do dùng speech không nên viết là:

```text
speech chắc chắn beat text
```

Mà nên viết là:

```text
Text supports scoring.
Speech supports coaching and explanation.
Text + speech supports both prediction and actionable feedback.
```

Điểm đóng góp của nhóm:

```text
Existing TED baselines rely mainly on transcript/syntax.
The only speech-aware TED baseline uses low-level prosody and does not improve clearly.
We extend the audio branch with affective speech cues from 06D and speaking-quality cues inspired by PuSQ/readys.
```

---

## 7. Liên hệ giữa Dependency Tree + Prosody CNN, 06D và PuSQ

Ba hướng này gặp nhau ở phần audio delivery cues:

```text
Dependency Tree + Prosody CNN
= text/syntax + low-level prosody

06D
= affective speech / emotion representation

PuSQ/readys
= recording-level speaking-quality features + audio/text fusion
```

### 7.1. 06D bổ sung gì cho Prosody CNN?

TED Prosody CNN dùng:

```text
pitch
loudness
formants
```

06D dùng audio sâu hơn:

```text
MFCC / delta / delta-delta / RMS / ZCR / spectral sequence
log-Mel spectrogram
emotion2vec representation
handcrafted statistical acoustic vector
co-attention and stacking
```

06D có thể tạo feature:

```text
emotion probabilities per segment
emotion distribution over full talk
emotion entropy
neutral dominance
emotion variation
emotion transition rate
high-arousal proxy ratio
low-confidence ratio
```

Ý nghĩa:

```text
Prosody CNN = low-level acoustic pattern
06D = affective/emotional delivery pattern
```

### 7.2. PuSQ/readys bổ sung gì?

PuSQ:

- GitHub: <https://github.com/sofiaele/PuSQ>
- Local: `Papers/Papers for presentation feedback system/code/PuSQ`

readys:

- GitHub: <https://github.com/tyiannak/readys>
- Local: `Papers/Papers for presentation feedback system/code/readys`

PuSQ/readys không dùng trực tiếp để gán nhãn TED bằng tay. Ta lấy ý tưởng:

```text
audio/text/recording-level features
-> speaking-quality cues
-> expressiveness / enjoyment / easy-to-follow style indicators
```

Feature structure tham khảo:

```text
LowLevelAudio features
MetaAudio features
Text features
recording-level classifier
late fusion audio/text
```

Ứng dụng vào v2:

```text
PuSQ-inspired speaking-quality features
= pitch variation
+ energy variation
+ pause/silence statistics
+ speaking rhythm
+ speech rate
+ 06D emotion posterior statistics
+ text/audio consistency if transcript is available
```

---

## 8. Kiến trúc đề xuất cho v2

### 8.1. Overall pipeline

```text
TED metadata / TED audio / TED transcript
-> label construction from scaled TED ratings
-> transcript branch
-> speech-prosody branch
-> 06D affective speech branch
-> PuSQ-inspired speaking-quality branch
-> fusion model
-> rating prediction
-> feedback explanation
```

### 8.2. Transcript branch

Baseline options:

```text
TF-IDF + Logistic Regression / Linear SVM
Word Sequence LSTM
Dependency TreeLSTM
BERT / sentence transformer if time allows
```

Minimum viable:

```text
TF-IDF baseline + Word Sequence or BERT-like embedding
```

### 8.3. Speech-prosody branch

Input:

```text
TED mp3/wav
```

Feature:

```text
pitch mean/std/range
energy mean/std/range
pause ratio
silence ratio
speaking rate
voiced ratio
duration
MFCC statistics
spectral centroid/bandwidth
optional formants if stable
```

Model:

```text
Logistic Regression / Random Forest / XGBoost / MLP
```

Optional reproduction:

```text
Prosody CNN similar to TED paper
```

### 8.4. 06D affective speech branch

Input:

```text
TED audio -> segment 3-5 seconds
```

Output:

```text
p(neutral), p(happy), p(sad), p(angry), p(fear), p(disgust)
confidence
embedding if available
```

Aggregation:

```text
mean/std/max of each emotion probability
emotion entropy
neutral dominance
positive emotion ratio
negative emotion ratio
emotion change rate
high-confidence ratio
low-confidence ratio
```

### 8.5. PuSQ-style speaking-quality branch

Feature:

```text
expressiveness proxy
enjoyment proxy
easy-to-follow proxy
fluency proxy
pause/silence statistics
speech-rate stability
pitch variation
energy variation
audio/text consistency
```

Important naming:

```text
Không gọi đây là PuSQ ground-truth score.
Gọi là PuSQ-inspired speaking-quality cues.
```

### 8.6. Fusion

Early fusion:

```text
[text vector ; prosody vector ; 06D vector ; PuSQ-style vector]
-> classifier
-> 14 rating outputs
```

Late fusion:

```text
text model probability
speech model probability
06D model probability
PuSQ-style model probability
-> weighted average / meta-classifier
```

Khuyến nghị:

```text
Start with early fusion using tabular features and simple classifiers.
Then try late fusion if separate branches are stable.
```

---

## 9. Các experiment cần làm

### 9.1. Dataset audit

Việc cần làm:

- count TED talks usable;
- check ratings field;
- parse 14 labels;
- check AWS transcript availability;
- check word timestamp availability;
- check audio download link availability;
- check duration distribution;
- check how many audio files can be downloaded.

Output:

```text
TED metadata summary table
rating distribution plot
duration distribution plot
usable sample count
```

### 9.2. Text-only baseline

Models:

```text
TF-IDF + Logistic Regression / SVM
Word Sequence LSTM or BERT-style baseline
```

Metrics:

```text
per-label AUC
macro AUC
per-label F1
macro F1
```

Purpose:

```text
Establish the transcript-only performance ceiling.
```

### 9.3. Speech-only baseline

Input:

```text
audio-derived prosody/acoustic features only
```

Models:

```text
Logistic Regression
Random Forest
XGBoost
MLP
```

Purpose:

```text
Test whether delivery alone contains rating signal.
```

Expected:

```text
May be lower than text-only, but still useful for feedback and explainability.
```

### 9.4. 06D-only audio features

Input:

```text
06D emotion posterior statistics
```

Purpose:

```text
Check if affective speech cues predict delivery-related ratings.
```

Likely relevant ratings:

```text
fascinating
inspiring
persuasive
funny
longwinded
confusing
unconvincing
ok
```

### 9.5. Prosody + 06D

Compare:

```text
prosody-only
06D-only
prosody + 06D
```

Goal:

```text
Show whether affective cues add information beyond low-level prosody.
```

### 9.6. Text + Speech

Compare:

```text
text-only
text + prosody
text + 06D
text + prosody + 06D
text + prosody + 06D + PuSQ-style cues
```

Goal:

```text
Show whether speech features improve prediction or at least improve feedback interpretability.
```

### 9.7. Ablation table

| Model | Text | Prosody | 06D emotion | PuSQ-style cues | Macro AUC | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Text-only baseline | yes | no | no | no | | |
| Prosody-only | no | yes | no | no | | |
| 06D-only | no | no | yes | no | | |
| Prosody + 06D | no | yes | yes | no | | |
| Text + Prosody | yes | yes | no | no | | |
| Text + 06D | yes | no | yes | no | | |
| Text + Prosody + 06D | yes | yes | yes | no | | |
| Full v2 | yes | yes | yes | yes | | |

### 9.8. Processing time

Rubric có nhắc Processing Time, nên cần đo:

```text
audio download/convert time
feature extraction time per minute
06D inference time per segment
total processing time for 5-min / 10-min TED audio
```

---

## 10. Feedback layer

TED ratings là recording-level labels, không có label từng đoạn. Vì vậy không được nói chắc:

```text
đoạn này bị chấm longwinded
```

Cách nói đúng:

```text
đoạn này có tín hiệu liên quan đến longwinded/confusing/low engagement.
```

Feedback examples:

| Pattern | Interpretation | Feedback |
|---|---|---|
| neutral dominance high + pitch variance low + energy flat | monotone delivery | Cần tăng biến thiên giọng và nhấn ý chính |
| silence ratio high + longest pause long | nhiều khoảng ngắt | Cần luyện mạch nói hoặc chia câu rõ hơn |
| low arousal proxy + low energy | thiếu năng lượng | Cần tăng vocal energy ở đoạn này |
| high energy + angry/high-arousal probability high | có thể nghe căng/gắt | Cần giảm lực giọng hoặc làm mềm nhịp nói |
| speaking rate too high | nói quá nhanh | Cần giảm tốc độ ở đoạn có thông tin quan trọng |
| longwinded predicted high + low speech variation | dễ gây cảm giác dài dòng | Cần thêm điểm nhấn hoặc rút gọn đoạn |
| confusing predicted high + fast rate + complex text | khó theo dõi | Cần chia nhỏ ý và giảm tốc độ |

---

## 11. Demo cuối kỳ

Minimum demo:

```text
Upload audio or TED audio link
-> extract transcript if available or allow transcript input
-> extract prosody/acoustic features
-> run 06D by segments
-> show emotion/prosody timeline
-> output feedback summary
```

Better demo:

```text
Upload TED/user audio + transcript
-> predict selected TED-style labels
-> show score for fascinating / inspiring / persuasive / longwinded / confusing / unconvincing
-> show contribution from text, prosody, 06D emotion, PuSQ-style cues
-> show segment-level feedback
```

Không cần cam kết predict chính xác toàn bộ 14 labels trong demo nếu thời gian không đủ. Nên chọn subset liên quan presentation feedback.

---

## 12. Reference list cần đưa vào report

| Reference | Link | Vai trò |
|---|---|---|
| Predicting TED Talk Ratings from Language and Prosody | <https://arxiv.org/abs/1906.03940> | TED baseline chính: Word Sequence, Dependency Tree, Prosody CNN |
| A Causality-Guided Prediction of TED Talk Ratings | <https://arxiv.org/abs/1905.08392> | transcript/rating prediction reference |
| TED_dataset / VoiceCoach dataset | <https://github.com/xingbow/TED_dataset> | metadata, ratings, transcripts, audio links |
| VoiceCoach | <https://arxiv.org/abs/2001.07876> | voice modulation/public speaking feedback |
| PuSQ | <https://github.com/sofiaele/PuSQ> | speaking-quality dataset and recording-level feature idea |
| readys | <https://github.com/tyiannak/readys> | audio/text/recording-level classifier framework |
| VaryFairyTED / TED_HEM | <https://github.com/racharyy/TED_HEM> | TED rating/fairness reference model |
| VaryFairyTED paper | <https://arxiv.org/abs/2012.06157> | fairness by quality for TED rating prediction |
| emotion2vec | <https://arxiv.org/abs/2312.15185> | pretrained affective speech representation used by 06D direction |

Note:

```text
Chưa tìm thấy repo chính chủ public cho paper 1906.03940 chứa trực tiếp code Word Sequence / Dependency Tree / Prosody CNN.
Trong paper, footnote source code bị blinded for author anonymity.
Vì vậy ta reproduce kiến trúc từ paper và dùng TED_dataset/VoiceCoach làm nguồn data khả thi.
```

---

## 13. Step-by-step plan

### Step 1. Chốt scope v2

Output:

```text
Problem statement mới
Vai trò TED / 06D / PuSQ rõ ràng
Không gọi project là emotion-only
```

### Step 2. Audit TED dataset

Output:

```text
ted_metadata_clean.csv
rating_distribution.csv/png
duration_distribution.csv/png
usable_talks_summary.json
```

### Step 3. Build label pipeline

Output:

```text
scaled ratings
median-threshold binary labels
selected rating subset
train/dev/test split
```

### Step 4. Reproduce simple classical baseline

Output:

```text
LIWC/prosody/statistical if available
or TF-IDF text baseline
macro AUC/F1
```

### Step 5. Reproduce TED text baseline

Output:

```text
Word Sequence or BERT-like transcript baseline
per-rating AUC/F1
```

### Step 6. Build speech-only features

Output:

```text
prosody/acoustic feature table
speech-only prediction result
```

### Step 7. Add 06D emotion features

Output:

```text
segment-level emotion timeline
recording-level 06D feature table
06D-only and prosody+06D result
```

### Step 8. Add PuSQ-inspired cues

Output:

```text
speaking-quality feature table
full audio branch result
```

### Step 9. Fusion experiments

Output:

```text
text-only vs speech-only vs text+speech ablation table
```

### Step 10. Feedback dashboard/demo

Output:

```text
upload audio/transcript
rating subset prediction
timeline visualization
segment-level feedback
summary report
```

### Step 11. Final report

Output:

```text
Introduction
Related Work
Dataset
Method
Experiments
Demo
Discussion
Conclusion
References
```

---

## 14. Kết luận định hướng

Roadmap v2 nên được bảo vệ bằng lập luận:

```text
Text-only TED baselines are strong for rating prediction,
but presentation feedback requires speech delivery analysis.
```

Đóng góp của nhóm:

```text
1. Reproduce or approximate TED rating baselines.
2. Build a speech/audio branch using prosody features.
3. Extend the audio branch with 06D affective speech cues.
4. Add PuSQ-inspired speaking-quality cues.
5. Compare text-only, speech-only and fused models.
6. Convert model outputs into actionable feedback for presentation training.
```

Một câu chốt để đưa vào báo cáo:

> Our system does not treat emotion recognition as the final goal. Instead, 06D is used as an affective speech feature extractor, PuSQ/readys provides inspiration for speaking-quality cues, and TED rating baselines provide the prediction task. This combination allows the system to move from rating prediction toward actionable presentation feedback.
