# Bonus v2 - hướng speech + transcript cho Presentation Feedback

> Cập nhật định hướng: không mở rộng trọng tâm sang multimodal video/body/gaze ở giai đoạn này. Project sẽ tập trung vào **speech audio + transcript**, lấy cảm hứng từ PuSQ và TED Talk Ratings, rồi dùng mô hình **06D Emotion2Vec Co-Attention SER** như affective-speech module trong hệ thống feedback thuyết trình.

---

## 0. Quyết định hướng đi tại giai đoạn này

### 0.1 Không tập trung multimodal video

Các bài như **SpeechMirror** và **E-ffective** rất mạnh về visual analytics, nhưng chúng cần video, face, pose, gaze, body gesture, ordinal regression theo contest level và giao diện phức tạp. Đây là hướng tốt cho future work, nhưng không phải trọng tâm hiện tại vì:

- dữ liệu video/presentation thật khó thu và xử lý hơn audio;
- pipeline nhiều phụ thuộc: face recognition, AffectNet, OpenFace, MMPose, Azure ASR, pose/gesture tracking;
- khó hoàn thiện trong tài nguyên Kaggle/Colab sinh viên;
- dễ làm loãng phần chính là speech emotion recognition.

Vì vậy, project nên định vị là:

```text
Speech Emotion Recognition Toward Presentation Feedback Analysis

= speech audio + transcript
+ affective-speech cues from 06D SER
+ PuSQ-style posterior aggregation
+ TED-style rating/presentation-quality prediction
+ timeline-based feedback for long speech
```

### 0.2 Mục tiêu chính

Mục tiêu không phải thay thế hoàn toàn PuSQ/TED, mà là tạo một hướng nối hợp lý:

```text
06D SER
-> segment-level emotion posterior
-> combine with prosody / fluency / transcript
-> predict or estimate presentation delivery quality
-> show local weak parts in a long presentation
```

Điểm học thuật cần nhấn mạnh:

- emotion không phải nhãn cuối để chấm thuyết trình;
- emotion posterior là **intermediate affective-speech feature**;
- presentation feedback cần thêm prosody, pause, speech rate và transcript;
- với audio dài, không nên chỉ lấy một score toàn bài, mà cần timeline/segment-level analysis.

---

## 0.3 Bài báo và repo cần tham khảo

| Nhóm | Paper / Repo | Vai trò trong project |
|---|---|---|
| SER backbone | `06d_Emotion2Vec_CoAttention_Full_SER.ipynb` | Mô hình chính của nhóm: emotion2vec + temporal/spectral/statistical branch + co-attention + stacking |
| PuSQ | Paper: https://aclanthology.org/2021.icnlsp-1.19/ | Chứng minh emotion/valence/arousal posterior có thể dùng làm feature trung gian cho speaking quality |
| PuSQ repo | https://github.com/sofiaele/PuSQ | Dataset feature-level, annotation, ASR transcript, parser |
| readys repo | https://github.com/tyiannak/readys | Framework train/test segment classifier và recording-level classifier |
| TED Talk Ratings | Paper: https://arxiv.org/abs/1906.03940 | Hướng dataset dài hơn PuSQ, dùng transcript + prosody để dự đoán 14 audience ratings |
| TED causality-guided ratings | Paper: https://arxiv.org/abs/1905.08392 | Cùng hướng TED ratings, tập trung transcript và xử lý bias bằng causal diagram |
| TED alignment tool | https://github.com/JoFrhwld/FAVE/wiki/FAVE-align | Tool được paper TED dùng để align audio với transcript |
| Praat | https://www.fon.hum.uva.nl/praat/ | Tool paper TED dùng để trích pitch/loudness/formants |
| SpeechMirror | https://arxiv.org/abs/2309.05091 | Tham khảo giao diện timeline, speech factor feedback, recommendation, SpeechTwin/SpeechPlayer |
| E-ffective | https://arxiv.org/abs/2110.14908 | Tham khảo emotion shift, E-spiral, E-script, ordinal effectiveness analysis |
| Automated coaching survey | https://arxiv.org/abs/2606.27380 | Related work tổng quan về automated presentation coaching |
| Valence/arousal pretrained | https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | Có thể dùng thử để lấy arousal/valence/dominance cho audio demo |

Ghi chú quan trọng: với hai paper TED (`1906.03940` và `1905.08392`), mình **chưa tìm thấy official dataset/code repo có thể tải trực tiếp**. Paper `1906.03940` ghi source code/dataset sẽ được phát hành, nhưng bản arXiv đang để link code bị ẩn vì anonymous review. Vì vậy, không nên tự đặt một thư mục "code reference" như thể đó là code gốc. Nếu dùng TED trong project, cần viết trung thực là: hiện mới có paper và mô tả dataset/model; chưa có official repo/dataset local.

---

## 0.4 Kế hoạch thực hiện tiếp theo

### Giai đoạn A - Reproduce PuSQ feature-level baseline

Mục tiêu:

- dùng lại feature PuSQ đã tải;
- chạy lại pipeline aggregate annotation -> parse dataset -> train recording-level classifier;
- báo cáo baseline Expressiveness/Enjoyment theo PuSQ.

Input:

```text
PuSQ/features_data/
├── *_MetaAudio.npz      # 20D: pause/silence/speech + emotion/valence/arousal posterior
├── *_LowLevelAudio.npz  # 136D acoustic feature
├── *_Text.npz           # 22D text feature
└── *.asr                # ASR transcript/timestamp
```

Mô hình baseline:

```text
SVM-RBF
Gaussian Naive Bayes
Logistic Regression
early fusion / late fusion
```

Output:

```text
Expressiveness: positive / negative
Enjoyment: positive / negative
ROC-AUC under Leave-One-Speaker-Out
```

### Giai đoạn B - Nâng cấp PuSQ theo ý tưởng 06D

Vì PuSQ public không có raw WAV, không thể đưa nguyên model 06D vào PuSQ. Cách nâng cấp hợp lý là:

```text
MetaAudio branch 20D
LowLevelAudio branch 136D
Text branch 22D
-> branch MLP / gated fusion / stacking
-> Expressiveness / Enjoyment
```

Ý tưởng kế thừa từ 06D:

- multi-branch representation;
- validation-weighted stacking;
- dùng affective posterior như feature điều hướng;
- ablation rõ từng nhóm feature.

### Giai đoạn C - TED Talk Ratings baseline và nâng cấp

TED Talk Ratings phù hợp hơn PuSQ ở mặt độ dài và audience perception:

- 2,231 TED Talks;
- tổng khoảng 513.49 giờ;
- 5.57 triệu viewer ratings;
- 14 rating categories;
- transcript + prosody features;
- reserved test subset 150 talks;
- best average AUC khoảng 0.83.

Tuy nhiên, tại thời điểm rà soát hiện tại:

- chưa tìm thấy link chính thức để tải dataset TED ratings/features;
- chưa tìm thấy official GitHub/source code của tác giả;
- paper `1905.08392` cũng dùng TED transcript + ratings tương tự, nhưng không cung cấp dataset trực tiếp trên trang arXiv;
- vì vậy TED nên được xem là **research direction / model inspiration** trước, chưa phải dataset đã sẵn sàng để reproduce ngay như PuSQ.

Các model chính từ paper TED:

1. **Word Sequence LSTM**
   - sentence = sequence of GloVe word vectors;
   - LSTM encode từng câu;
   - multi-label output 14 ratings.

2. **Dependency TreeLSTM**
   - dùng dependency parse;
   - Child-Sum TreeLSTM;
   - kết hợp GloVe + POS/dependency information;
   - trong paper đạt AUC trung bình khoảng 0.83.

3. **Dependency TreeLSTM + Prosody CNN**
   - forced alignment transcript/audio;
   - Praat trích pitch, loudness, first three formants;
   - prosody signal 8D ở 10 Hz;
   - 1D-CNN 4 lớp tạo vector prosody 64D;
   - concat với TreeLSTM sentence embedding;
   - trong paper cũng khoảng AUC 0.83, không cải thiện rõ so với text-only.

4. **Statistical baseline**
   - LIWC language features;
   - prosody statistics;
   - narrative trajectory;
   - Linear SVM / LASSO/Ridge-style classifier;
   - AUC khoảng 0.78.

Đề xuất cho project nếu sau này tìm được dataset hoặc tự thu thập lại transcript/rating:

```text
Baseline 1: statistical transcript + prosody
Baseline 2: Word Sequence LSTM
Baseline 3: text/prosody fusion
Upgrade: add 06D emotion posterior + PuSQ-style pause/speech features
```

Không nên bắt đầu bằng TreeLSTM vì dependency parsing + tree batching khó tái hiện hơn trong thời gian ngắn. Nên chọn:

```text
Word Sequence LSTM + prosody/statistical baseline
```

rồi thêm:

```text
06D emotion posterior timeline
valence/arousal optional
pause/speech rate
segment/window-level aggregation
```

### Giai đoạn D - Long-audio demo và TED full-talk

Audio TED dài hơn PuSQ rất nhiều. Theo paper TED:

```text
2,231 talks
513.49 total hours
=> trung bình khoảng 13.8 phút / talk
```

Con số **5 phút** chỉ là kịch bản demo/subset ban đầu của project mình, không phải độ dài trung bình của TED. Có thể lấy 5 phút từ:

- một đoạn của TED talk dài hơn;
- một bài thuyết trình tự thu;
- một sample demo để kiểm tra long-audio pipeline.

Nếu xử lý một audio 5 phút:

```text
5 phút = 300 giây
segment 3 giây -> khoảng 100 segment
hop 1.5-2 giây -> khoảng 150-200 segment
```

Về kỹ thuật xử lý được. Nhưng không nên chỉ lấy trung bình toàn bài vì sẽ làm mất lỗi cục bộ. Nên dùng hierarchical aggregation:

```text
segment 3-5 giây
-> block 30-60 giây
-> score từng block
-> aggregate toàn bài
-> highlight đoạn yếu
```

Output demo mong muốn:

```text
Overall expressive proxy
Overall enjoyable/engagement proxy
Timeline:
  00:30-01:00: low pitch variation + neutral dominance -> possible monotone
  02:10-02:40: long pauses + low speech ratio -> fluency issue
  04:00-04:30: low energy + low arousal -> low delivery energy
```

Nếu xử lý một TED talk trung bình 13.8 phút:

```text
13.8 phút = khoảng 828 giây
segment 3 giây -> khoảng 276 segment / talk
```

Nếu xử lý toàn bộ TED raw audio:

```text
513.49 giờ = khoảng 1,848,564 giây
segment 3 giây -> khoảng 616,000 segment
```

Do đó, không nên chạy 06D/emotion2vec trên toàn bộ TED ngay từ đầu. Nên làm subset + cache feature trước.

### 0.5 TED có chỉ ra đoạn hay/dở không?

Paper TED Talk Ratings dự đoán rating ở **cấp toàn talk**, không trực tiếp chỉ ra timestamp nào hay/dở. Input là transcript + prosody được xử lý theo câu, nhưng label cuối là viewer rating toàn bài. Vì vậy:

- TED có thể giúp học **overall audience-rating prediction**;
- TED không có ground truth từng đoạn cho “đoạn này hay/dở”;
- muốn local feedback, project cần tự thiết kế layer diễn giải theo segment/window.

Các bài SpeechMirror và E-ffective làm tốt phần này hơn vì chúng là visual analytics system:

- SpeechMirror có Time Slice Panel, SpeechPlayer, SpeechTwin, recommendation speech tương tự;
- E-ffective có E-spiral, E-script, E-type để xem emotion shift theo thời gian;
- nhưng chúng dùng video/facial/body/modalities nhiều hơn, nên chỉ nên lấy cảm hứng UI/timeline, không lấy toàn bộ multimodal pipeline ở giai đoạn này.

### 0.6 Khả thi về tài nguyên

| Hạng mục | Khả thi | Nhận xét |
|---|---:|---|
| Reproduce PuSQ feature-level | Cao | Dữ liệu đã là feature `.npz`, nhẹ, không cần raw audio |
| Train PuSQ MLP/gated fusion | Cao | Feature nhỏ, train nhanh |
| TED statistical baseline | Trung bình-cao | Cần dataset TED features/transcript; model nhẹ |
| TED Word Sequence LSTM | Trung bình | Cần xử lý transcript dài, padding/batching; có thể chạy Colab/Kaggle |
| TED TreeLSTM | Trung bình-thấp | Khó implementation và batching dependency tree |
| TED prosody CNN | Trung bình | Cần audio/prosody/alignment; xử lý nhiều giờ audio sẽ tốn thời gian |
| 06D + 5 phút demo | Trung bình-cao | Nếu dùng segment-level inference và cache feature |
| Full TED audio training end-to-end | Thấp | 513 giờ audio, không phù hợp GPU/tài nguyên sinh viên |

Chiến lược tiết kiệm thời gian:

```text
Không train end-to-end trên toàn bộ raw TED audio.
Ưu tiên:
1. transcript/statistical baseline
2. prosody features đã trích hoặc trích subset
3. 06D emotion posterior trên subset/demo
4. cache feature, train classifier nhẹ
```

---

# PuSQ và khả năng kế thừa mô hình `06d_Emotion2Vec_CoAttention_Full_SER`

> Tài liệu tổng hợp từ việc kiểm tra paper **Automatic Assessment of Speaking Skills Using Aural and Textual Information**, repo **PuSQ**, repo **readys**, và notebook hiện tại `06d_Emotion2Vec_CoAttention_Full_SER.ipynb`.

---

## 1. Kết luận nhanh

PuSQ là một hướng rất phù hợp để tham khảo khi chuyển từ **Speech Emotion Recognition (SER)** sang **Public Speaking Quality Assessment** vì họ không dùng emotion như output cuối. Thay vào đó:

```text
Audio / Text
      ↓
Emotion + Valence + Arousal ở mức đoạn
      ↓
Lấy posterior trung bình trên toàn bài nói
      ↓
Ghép với pause, silence, speech ratio, word rate,
low-level acoustic feature và text feature
      ↓
Dự đoán:
- Expressiveness
- Enjoyment
```

Tuy nhiên, bản PuSQ công khai hiện tại **không có raw WAV**. Repo chỉ phát hành:

- feature audio đã trích xuất;
- feature text đã trích xuất;
- transcript ASR;
- annotation;
- reference text;
- parser để tổ chức dữ liệu.

Vì vậy, PuSQ public **không thể được đưa trực tiếp vào toàn bộ pipeline hiện tại** của notebook `06d`, vì notebook cần waveform/raw audio để tạo:

- MFCC sequence;
- log-Mel spectrogram;
- emotion2vec embedding;
- handcrafted statistics từ waveform.

Điểm phù hợp giữa hai hệ thống nằm ở **ý tưởng sử dụng emotion representation như feature trung gian**, không nằm ở khả năng dùng trực tiếp cùng tensor đầu vào.

---

# 2. Các link chính

## 2.1 Paper

**Automatic Assessment of Speaking Skills Using Aural and Textual Information**  
Sofia Eleftheriou, Panagiotis Koromilas, Theodoros Giannakopoulos — ICNLSP 2021

- Trang paper:  
  https://aclanthology.org/2021.icnlsp-1.19/
- PDF:  
  https://aclanthology.org/2021.icnlsp-1.19.pdf

## 2.2 Repo 1 — PuSQ dataset

- Repo chính:  
  https://github.com/sofiaele/PuSQ
- Feature dataset trực tiếp:  
  https://github.com/sofiaele/PuSQ/blob/main/features_data.zip
- Annotation:  
  https://github.com/sofiaele/PuSQ/blob/main/annotations_database.txt
- Reference texts:  
  https://github.com/sofiaele/PuSQ/tree/main/reference_texts
- Dataset parser:  
  https://github.com/sofiaele/PuSQ/blob/main/dataset_parser.py
- Issue hỏi xin raw speech/audio:  
  https://github.com/sofiaele/PuSQ/issues/1

## 2.3 Repo 2 — `readys` framework

- Repo chính:  
  https://github.com/tyiannak/readys
- Hướng dẫn train/test model:  
  https://github.com/tyiannak/readys/blob/master/models/readme.md
- Thư mục model:  
  https://github.com/tyiannak/readys/tree/master/models

Hai repo có vai trò khác nhau:

| Repo | Vai trò |
|---|---|
| `sofiaele/PuSQ` | Dataset đã xử lý, annotation, transcript, reference text, parser |
| `tyiannak/readys` | Framework extract feature, train/test segment classifier, recording-level classifier và fusion |

---

# 3. PuSQ có raw WAV không?

## Kết luận

**Không thấy raw WAV được phát hành trong repo public.**

Paper ghi rõ PuSQ được công khai dưới dạng:

> extracted audio and text features and ASR text files

Repo hiện có các file chính:

```text
PuSQ/
├── reference_texts/
├── README.md
├── annotations_database.txt
├── dataset_parser.py
└── features_data.zip
```

Không có:

```text
audio/
wav/
recordings/
raw_audio/
speaker_*.wav
```

`dataset_parser.py` cũng không tìm hoặc copy `.wav`. Với mỗi sample, code tạo tên:

```python
_MetaAudio.npz
_LowLevelAudio.npz
_Text.npz
.asr
```

Do đó, phần public là **feature-level dataset**, không phải raw-audio dataset.

## Hệ quả đối với mô hình pretrained

Không thể làm trực tiếp:

```text
PuSQ WAV
   ↓
emotion2vec
   ↓
768-d embedding
```

Không thể làm:

```text
PuSQ WAV
   ↓
log-Mel
   ↓
2D-CNN
```

Không thể làm:

```text
PuSQ WAV
   ↓
MFCC sequence
   ↓
1D-CNN → BiLSTM
```

vì waveform gốc không có trong bản public.

---

# 4. PuSQ dataset được thu như thế nào?

Paper mô tả dữ liệu được thu bằng web app **RecSurvey**.

Người tham gia:

- dùng microphone/headset cá nhân;
- điền thông tin nhân khẩu học;
- đọc một trong 40 đoạn tiếng Anh có sẵn, mỗi đoạn khoảng 4–5 dòng;
- hoặc trả lời một trong 20 câu hỏi mở.

Quy mô:

| Thuộc tính | Giá trị |
|---|---:|
| Tổng recording | 695 |
| Số người nói | 42 |
| Nữ | 26 |
| Nam | 16 |
| Ngôn ngữ | Tiếng Anh |
| Dạng nội dung | Đọc văn bản + trả lời tự do |

Dữ liệu có nhiều quốc tịch để tăng đa dạng về phát âm và speaking style.

## Nhãn ban đầu

Mỗi recording được đánh giá theo thang 1–5:

| Nhãn | Ý nghĩa |
|---|---|
| Expressiveness | Mức chủ động, cảm xúc, nhiệt tình, giàu biểu cảm của cách nói |
| Ease of following | Độ rõ, độ trôi chảy, tốc độ và mức dễ theo dõi |
| Enjoyment | Mức người nghe cảm thấy thú vị, giải trí hoặc có động lực |

Có:

- 14 annotator;
- 2.687 lượt đánh giá;
- 689/695 recording được gán nhãn.

Sau bước lọc/aggregate, paper chỉ giữ hai task chính:

```text
Expressiveness
Enjoyment
```

`Ease of following` bị loại khỏi thực nghiệm vì mức bất đồng annotator cao.

## Lưu ý về target cuối

PuSQ không giữ nguyên bài toán regression 1–5 trong thí nghiệm cuối. Họ tạo các dataset nhị phân:

```text
Positive
vs
Negative
```

và tách theo giới tính vì paper ghi nhận bias trong annotation giữa giọng nam và nữ.

---

# 5. Kiến trúc tổng thể của PuSQ

PuSQ dùng kiến trúc hai tầng:

```text
                    AUDIO
                      │
             chia đoạn 3 giây
                      │
        ┌─────────────┼─────────────┐
        │             │             │
     Emotion       Valence       Arousal
    classifier     classifier     classifier
        │             │             │
        └─────────────┴─────────────┘
                      │
       aggregate class posteriors
          trên toàn recording
                      │
      + pause/silence/speech features
                      │
                Meta Audio
                      │
                      ├──────────────┐
                      │              │
                  Text feature    Low-level audio
                      │              │
                      └──── Fusion ──┘
                              │
                    Recording-level model
                              │
                Expressiveness / Enjoyment
```

Điểm quan trọng:

> Họ không dùng một model multi-task duy nhất dự đoán đồng thời emotion, valence và arousal. Họ train các classifier riêng, sau đó dùng posterior của các classifier đó như feature trung gian.

---

# 6. Segment-level audio feature

## 6.1 Chia audio

Mỗi recording được chia thành segment 3 giây.

Mỗi segment tiếp tục chia thành frame:

```text
Frame length: 50 ms
Hop/step:     50 ms
Overlap:      không
```

## 6.2 34 low-level feature gốc

| Nhóm | Feature |
|---|---|
| Temporal | Zero-Crossing Rate |
| Energy | Energy |
| Energy | Energy Entropy |
| Spectral | Spectral Centroid |
| Spectral | Spectral Spread |
| Spectral | Spectral Entropy |
| Spectral | Spectral Flux |
| Spectral | Spectral Rolloff |
| Cepstral | 13 MFCC |
| Tonal | 12-dimensional Chroma Vector |
| Tonal | Chroma Standard Deviation |

Tổng:

```text
8 feature cơ bản
+ 13 MFCC
+ 12 Chroma
+ 1 Chroma std
= 34 feature
```

Họ thêm delta cho toàn bộ 34 feature:

```text
34 original
+
34 delta
=
68 feature / frame
```

Sau đó lấy hai thống kê trên toàn segment 3 giây:

```text
Mean
+
Standard deviation
```

Theo phép tính:

```text
68 × 2 = 136
```

## Lỗi không nhất quán trong paper

Một số đoạn của paper ghi vector segment là **134 chiều**, dù chính công thức đi kèm là `68 × 2`. Ở phần recording-level, paper lại ghi Low-Level Audio là **136 chiều**.

Vì vậy hợp lý nhất là xem `134` là lỗi đánh máy và kiểm tra code/dữ liệu thật trước khi hard-code dimension.

---

# 7. Các dataset dùng để train Emotion, Valence và Arousal

Paper dùng 5 dataset open-source và 1 dataset proprietary:

| Dataset | Emotion | Valence/Arousal gốc | Trạng thái |
|---|---:|---:|---|
| EMOVO | Có | Không | Public |
| EmoDB | Có | Không | Public |
| SAVEE | Có | Không | Public |
| RAVDESS | Có | Không | Public |
| IEMOCAP | Có | Có | Cần theo quy trình release chính thức |
| Emotion Speech Movies | Có | Không rõ/public không thấy | Proprietary |

Các link chính:

### EMOVO

- Paper:  
  https://aclanthology.org/L14-1478/

### EmoDB

- Trang chính thức TU Berlin:  
  https://www.tu.berlin/en/kw/research/projects/emotional-speech

### SAVEE

- Trang chính thức:  
  https://kahlan.eps.surrey.ac.uk/savee/

### RAVDESS

- Zenodo chính thức:  
  https://zenodo.org/records/1188976

### IEMOCAP

- Trang chính thức USC SAIL:  
  https://sail.usc.edu/iemocap/

### Emotion Speech Movies

Paper mô tả đây là dataset riêng của nhóm tác giả, lấy audio từ các cảnh phim và chia thành 5 lớp cảm xúc. Không thấy link public chính thức trong paper/repo.

---

# 8. Họ chuẩn hóa label giữa nhiều dataset như thế nào?

## 8.1 Emotion classifier

Họ giữ 4 emotion:

```text
anger
sadness
neutral
happiness
```

`excitement` được gộp vào `happiness`.

Lý do là các dataset có label space khác nhau nên họ chỉ giữ các class chung/phù hợp với mục tiêu.

## 8.2 Valence classifier

Output:

```text
negative
neutral
positive
```

IEMOCAP có nhãn valence liên tục. Paper discretize:

```text
[1, 2.5)   → negative
[2.5, 4)   → neutral
[4, 5.5]   → positive
```

## 8.3 Arousal classifier

Output:

```text
low
neutral
high
```

Paper in các khoảng:

```text
[1, 2.53)  → low
[2.3, 3.6) → neutral
[3.6, 5]   → high
```

### Cảnh báo

Hai khoảng đầu bị overlap từ `2.3` đến `2.53`. Đây có thể là lỗi đánh máy trong paper. Không nên copy nguyên threshold vào implementation mà chưa kiểm tra code/training data thực tế.

## 8.4 Các dataset không có valence/arousal

Paper nói các emotion tag của những dataset không có nhãn valence/arousal được phân phối sang các lớp valence và arousal dựa trên **circumplex model**.

Ý tưởng trực quan:

```text
happy   → positive valence, high arousal
angry   → negative valence, high arousal
sad     → negative valence, low arousal
neutral → neutral valence, neutral arousal
```

Đây là quy đổi từ categorical emotion sang dimensional affect, không phải nhãn valence/arousal được annotator chấm trực tiếp cho từng sample.

---

# 9. Mô hình segment-level họ dùng là gì?

## 9.1 Không phải một model multi-task

Có ba bài toán riêng:

```text
Audio feature
    ├── Emotion classifier
    ├── Valence classifier
    └── Arousal classifier
```

Các model được train/evaluate riêng.

## 9.2 Classical ML chính

Paper thử:

- SVM với RBF kernel;
- XGBoost;
- CNN dùng melgram.

SVM-RBF là hướng chính cho emotion và valence.

Pipeline merged-dataset:

```text
Feature
   ↓
SMOTE-Tomek
   ↓
StandardScaler
   ↓
VarianceThreshold(threshold=0)
   ↓
PCA
   ↓
SVM-RBF / XGBoost
```

Grid search dùng:

```text
RepeatedStratifiedKFold
5 folds
```

Các hyperparameter chính:

- số PCA components;
- `C` của SVM;
- `gamma` của SVM.

## 9.3 CNN baseline

CNN nhận melgram và có:

```text
Conv 5×5, 32 channels
→ MaxPool 2

Conv 5×5, 64 channels
→ MaxPool 2

Conv 5×5, 128 channels
→ MaxPool 2

Conv 5×5, 256 channels
→ MaxPool 2

FC 1024
→ FC 256
→ FC số lớp
```

Không có:

- BiLSTM;
- attention;
- transformer;
- wav2vec2;
- emotion2vec;
- pretrained speech encoder.

## 9.4 Kết quả merged-dataset

Macro-F1:

| Task | XGBoost | CNN | SVM-RBF |
|---|---:|---:|---:|
| Emotion | 60.4 | 60.5 | **64.4 ± 0.9** |
| Valence | 51.7 | 52.6 | **55.2 ± 1.2** |
| Arousal | 64.2 | **69.3** | 66.8 ± 1.1 |

Nhận xét:

- SVM tốt nhất cho emotion;
- SVM tốt nhất cho valence;
- CNN tốt nhất cho arousal;
- paper vẫn xem SVM là lựa chọn đơn giản và ổn định cho feature extractor trung gian.

---

# 10. Họ đánh giá domain shift như thế nào?

Paper làm:

1. inner-dataset evaluation;
2. leave-one-dataset-out cross-dataset evaluation;
3. merged-dataset training.

Cross-dataset macro-F1 trung bình:

| Task | Macro-F1 trung bình |
|---|---:|
| Emotion | 40.8 |
| Valence | 35.9 |
| Arousal | 41.7 |

Điều này cho thấy SER phụ thuộc domain rất mạnh.

Ý nghĩa đối với project hiện tại:

```text
Train SER trên:
RAVDESS / TESS / CREMA-D / SAVEE

không đồng nghĩa

model sẽ hiểu đúng emotion hoặc expressiveness
trong bài thuyết trình thật
```

Vì presentation speech khác acted SER về:

- cường độ cảm xúc;
- microphone;
- độ dài;
- ngữ cảnh;
- accent;
- tốc độ;
- mức tự nhiên;
- tỷ lệ neutral.

---

# 11. Text branch của PuSQ

Audio được chuyển thành transcript bằng Google Speech API.

Paper thử:

- FastText embedding;
- BERT embedding;
- SVM-RBF;
- XGBoost.

Với BERT, embedding từ 4 layer cuối được average.

IEMOCAP được dùng để train/evaluate text emotion, valence và arousal vì trong nhóm dataset họ sử dụng, IEMOCAP có transcript.

Kết quả macro-F1 trên IEMOCAP:

| Task | SVM + FastText | XGBoost + BERT | SVM + BERT |
|---|---:|---:|---:|
| Emotion | 66.5 | 63.9 | **69.5** |
| Valence | 61.5 | 59.4 | **63.8** |
| Arousal | 48.8 | 48.2 | **51.0** |

---

# 12. Từ segment posterior thành feature toàn recording

Với từng segment, ba classifier trả posterior:

```text
Emotion:
P(angry)
P(sad)
P(neutral)
P(happy)

Valence:
P(negative)
P(neutral)
P(positive)

Arousal:
P(low)
P(neutral)
P(high)
```

Tổng:

```text
4 + 3 + 3 = 10 feature
```

Các posterior được aggregate trên toàn recording, ví dụ bằng trung bình xác suất.

Đây là phần quan trọng nhất về mặt ý tưởng:

> SER không phải output cuối. SER posterior trở thành feature mô tả speaking style.

---

# 13. Feature recording-level

## 13.1 Meta Audio — 20 chiều

Gồm:

```text
10 emotion-related posterior
+
10 high-level audio feature
=
20 chiều
```

Năm loại high-level audio feature:

- average silence duration;
- silence segments per minute;
- standard deviation of silence duration;
- speech ratio;
- word rate.

Mỗi loại được tính với hai cấu hình cửa sổ nên:

```text
5 × 2 = 10
```

## 13.2 Text — 22 chiều

Gồm:

```text
10 text emotion/valence/arousal posterior
+
12 high-level text feature
=
22 chiều
```

High-level text feature:

- word rate;
- unique-word rate;
- 10-bin histogram của word frequency.

## 13.3 Low-Level Audio — 136 chiều

Đây là long-term aggregate của low-level acoustic feature.

---

# 14. Model cuối dự đoán Expressiveness/Enjoyment

Input có thể là:

```text
Meta Audio (MA)
Text (T)
Low-Level Audio (LLA)
```

Họ thử:

- từng modality riêng;
- MA + T;
- MA và LLA với early fusion;
- MA và LLA với late fusion;
- MA + T và LLA với early/late fusion.

Classifier cuối:

- SVM-RBF;
- Gaussian Naive Bayes;
- Logistic Regression.

Evaluation:

```text
Leave-One-Speaker-Out
ROC-AUC
```

Paper cho biết Gaussian Naive Bayes tốt nhất ở phần lớn task, ngoại trừ một trường hợp Male Expressiveness/Meta Audio dùng Logistic Regression.

Một kết quả đáng chú ý:

| Task | AUC tốt nhất trong bảng |
|---|---:|
| Free-text Expressiveness | 93 |
| Free-text Enjoyment | 86 |

Các kết quả tốt thường đến từ fusion, đặc biệt khi kết hợp thông tin emotion/speaking-style với low-level audio và text.

---

# 15. Trong repo có model dựng sẵn không?

## 15.1 Repo PuSQ

Repo PuSQ không chứa đầy đủ checkpoint sẵn cho:

```text
emotion classifier
valence classifier
arousal classifier
expressiveness classifier
enjoyment classifier
```

Quickstart yêu cầu người dùng:

1. clone PuSQ;
2. unzip feature;
3. clone `readys`;
4. aggregate annotation;
5. parse feature thành class folder;
6. train recording-level classifier;
7. test model vừa train.

Ví dụ lệnh:

```bash
python3 models/train_recording_level_classifier.py \
  -i annotation_agreement/datasets/expressive_female/ \
  -mn test_model \
  -f
```

Model được tạo ra sau khi train.

## 15.2 Repo `readys`

`readys` có:

- code train text classifier;
- code train audio classifier;
- code train recording-level classifier;
- code test;
- config;
- feature extraction;
- early/late fusion.

README của `readys` có cung cấp link Google Drive tới:

- training data text;
- training data audio;
- training data recording-level;
- trained recording-level models.

Tuy nhiên, cần phân biệt:

> Repo có code và link tải model/data, nhưng không có nghĩa toàn bộ checkpoint dùng trong paper đều nằm trực tiếp trong source tree hoặc chắc chắn chạy ngay với môi trường Python mới.

Một số dependency và code đã cũ, nên khả năng phải sửa compatibility là khá cao.

---

# 16. Notebook `06d_Emotion2Vec_CoAttention_Full_SER` hiện tại

## 16.1 Mục tiêu

Notebook hiện tại dự đoán 6 emotion:

```text
neutral
happy
sad
angry
fear
disgust
```

## 16.2 Kiến trúc

```text
Input audio 16 kHz
   |
   |-- Branch A: Temporal acoustic
   |      MFCC + delta + delta-delta
   |      + RMS/ZCR/spectral sequence
   |      -> 1D-CNN
   |      -> BiLSTM
   |      -> attention pooling
   |      -> z_temporal
   |
   |-- Branch B: Spectrogram
   |      log-Mel
   |      + delta log-Mel
   |      + delta-delta log-Mel
   |      -> residual 2D-CNN
   |      -> SE attention
   |      -> z_spectral
   |
   |-- Branch C: Pretrained emotion
   |      raw waveform
   |      -> frozen emotion2vec
   |      -> adapter MLP
   |      -> z_emotion2vec
   |
   |-- Branch D: Statistics
   |      handcrafted statistical vector
   |      -> Stats MLP
   |      -> z_stats
   |      -> RBF-SVM
   |
   └── Emotion-guided co-attention
          emotion2vec query
          temporal/spectral key-value
                  ↓
              fusion MLP
                  ↓
              deep posterior

Final stacking:
deep posterior
+ stats SVM posterior
+ emotion2vec head posterior
        ↓
final emotion
```

## 16.3 Tensor đã trích trong notebook

Notebook đã tạo:

| Tensor | Shape |
|---|---|
| `X_temporal` | `(11317, 132, 301)` |
| `X_spectral` | `(11317, 3, 96, 301)` |
| `X_stats` | `(11317, 796)` |
| `X_e2v` | `(11317, 768)` |

Feature extraction của 11.317 sample trong lần chạy được lưu trong notebook mất khoảng **17,7 phút** ở môi trường chạy đó. Đây chỉ là thời gian tham khảo, phụ thuộc GPU/CPU, I/O, cache và tốc độ model.

## 16.4 Kết quả strict split trong notebook

Protocol:

```text
combined_strict_no_tess
train:      6215
validation:  749
test:       1953
```

Kết quả test:

| Model | Macro-F1 |
|---|---:|
| Stats RBF-SVM | 0.4748 |
| Emotion2Vec Logistic Regression | 0.6422 |
| Deep Co-Attention Full | 0.6673 |
| Emotion2Vec MLP | 0.6882 |
| Full Stacking | **0.6960** |

Điểm đáng chú ý:

- `emotion2vec + MLP` rất mạnh dù kiến trúc đơn giản;
- full stacking tốt nhất;
- pretrained emotion representation có giá trị rõ trong strict split;
- nhưng target hiện tại vẫn là categorical emotion, chưa phải expressiveness/enjoyment.

---

# 17. So sánh feature PuSQ với notebook hiện tại

| Feature | PuSQ | Notebook `06d` | Mức khớp |
|---|---:|---:|---|
| MFCC | 13 | 40 | Có |
| Delta MFCC | Có | Có | Cao |
| Delta-delta MFCC | Không thấy | Có | Notebook nhiều hơn |
| ZCR | Có | Có | Cao |
| Energy | Có | RMS/energy | Gần |
| Spectral centroid | Có | Có | Cao |
| Spectral rolloff | Có | Có | Cao |
| Spectral spread | Có | Spectral bandwidth | Gần nhưng không đồng nhất |
| Spectral entropy | Có | Không | Thiếu |
| Spectral flux | Có | Không | Thiếu |
| Energy entropy | Có | Không | Thiếu |
| Chroma 12D | Có | Không | Thiếu |
| Chroma std | Có | Không | Thiếu |
| Spectral contrast | Không | Có 7D | Notebook có thêm |
| Log-Mel | Không phát hành | Có | Không dùng lại trực tiếp |
| Delta Log-Mel | Không phát hành | Có | Không dùng lại trực tiếp |
| Emotion2Vec | Không | 768D | Không tương thích trực tiếp |
| Silence duration | Có | Chưa có | Notebook thiếu |
| Speech ratio | Có | Chưa có | Notebook thiếu |
| Word rate | Có | Chưa có | Notebook thiếu |
| Valence | Có | Chưa có head riêng | Notebook thiếu |
| Arousal | Có | Chưa có head riêng | Notebook thiếu |
| Transcript/Text | Có | Chưa có | Notebook thiếu |

---

# 18. Vì sao PuSQ không thể chạy nguyên notebook hiện tại?

## 18.1 Temporal branch

Notebook cần:

```text
[132 feature channels, 301 time frames]
```

PuSQ public chỉ cung cấp feature đã aggregate.

Sau khi chỉ còn:

```text
mean
std
```

thì thứ tự thời gian đã mất.

Ví dụ:

```text
0.1 → 0.2 → 0.8 → 0.7 → 0.2
```

sau aggregate chỉ còn:

```text
mean
std
```

Không thể khôi phục lại chuỗi để đưa vào:

```text
1D-CNN
BiLSTM
temporal attention
```

## 18.2 Spectrogram branch

Notebook cần:

```text
[3, 96, 301]
=
log-Mel
delta log-Mel
delta-delta log-Mel
```

PuSQ public không có spectrogram và không có waveform để tự tạo lại.

## 18.3 Emotion2Vec branch

Notebook cần raw waveform:

```text
WAV
 ↓
emotion2vec
 ↓
768D embedding
```

PuSQ không có WAV và không phát hành emotion2vec embedding.

## 18.4 Co-attention

Co-attention hiện tại phụ thuộc ba embedding:

```text
z_temporal
z_spectral
z_emotion2vec
```

Cả ba đều không thể tái tạo từ feature PuSQ public.

---

# 19. Mức tương thích thực tế

| Khía cạnh | Đánh giá |
|---|---:|
| Ý tưởng dùng SER làm intermediate representation | Rất cao |
| Ý tưởng fusion nhiều loại acoustic representation | Cao |
| Một phần handcrafted feature | Trung bình–cao |
| Statistical branch | Có thể kế thừa một phần |
| Temporal 1D-CNN/BiLSTM | Không dùng trực tiếp với PuSQ public |
| 2D-CNN log-Mel | Không dùng trực tiếp |
| Emotion2Vec | Không dùng trực tiếp |
| Co-attention hiện tại | Không dùng trực tiếp |
| Target emotion | Khác target PuSQ |
| Target expressiveness/enjoyment | Chưa có trong notebook |

Tóm lại:

> Hai hệ thống khớp tốt về **tư duy nghiên cứu**, nhưng không khớp trực tiếp về **dữ liệu đầu vào, tensor và target**.

---

# 20. Có thể dùng PuSQ với phần nào của notebook?

Có thể xây một notebook riêng:

```text
PuSQ released features

├── Meta Audio:      20D
├── Text:            22D
└── Low-Level Audio: 136D

          ↓

StandardScaler

          ↓

SVM / GaussianNB / Logistic Regression
hoặc MLP nhỏ

          ↓

Expressiveness
Enjoyment
```

Có thể tái sử dụng ý tưởng của `StatsBranch`, nhưng không tái sử dụng toàn bộ raw-audio architecture.

Ví dụ kiến trúc hiện đại hóa:

```text
LLA 136D
   ↓
MLP 128D

Meta Audio 20D
   ↓
MLP 64D

Text 22D
   ↓
MLP 64D

Concat / Gated Fusion
   ↓
Classifier
   ↓
Expressive / Not expressive
```

---

# 21. Hướng phát triển hợp lý cho project hiện tại

## Giai đoạn 1 — Reproduce PuSQ

Mục tiêu:

- chạy parser;
- aggregate annotation;
- train baseline;
- xác nhận cách feature fusion hoạt động;
- tái hiện Expressiveness/Enjoyment ở mức feature.

Đầu vào:

```text
MetaAudio.npz
LowLevelAudio.npz
Text.npz
ASR transcript
```

Không cần raw WAV.

## Giai đoạn 2 — Tạo phiên bản modernized

Cần raw audio public-speaking/presentation khác hoặc tự thu.

Pipeline:

```text
Raw presentation audio
        │
        ├── emotion2vec
        ├── MFCC temporal
        ├── log-Mel
        ├── pitch/F0
        ├── energy
        ├── pause
        ├── speech ratio
        ├── speaking rate
        └── transcript
                 ↓
        multimodal / multi-representation fusion
                 ↓
        Expressiveness
        Delivery quality
        Engagement
```

## Giai đoạn 3 — Real-time Presentation Feedback

Tách theo tốc độ xử lý:

```text
FAST PATH — mỗi 0.5–1 giây
volume
pitch
VAD
pause
→ cue real-time

MEDIUM PATH — mỗi 3–5 giây
speech rate
SER / arousal / expressiveness
→ feedback ngắn

POST-SESSION
transcript
content structure
semantic-prosodic alignment
overall scoring
→ báo cáo chi tiết
```

---

# 22. Những feature notebook hiện tại nên bổ sung

Để chuyển từ SER sang Presentation Feedback, nên bổ sung:

## Prosody

- F0/pitch mean;
- F0 standard deviation;
- pitch range;
- pitch contour;
- energy trend;
- loudness stability;
- emphasis detection.

## Fluency

- pause duration;
- silence ratio;
- number of long pauses;
- speech ratio;
- articulation rate;
- words per minute;
- filler words.

## Dimensional affect

- valence;
- arousal;
- expressiveness.

## Text

- ASR transcript;
- word rate;
- unique-word rate;
- repetition;
- filler-word count;
- content structure;
- keyword importance;
- semantic–prosodic alignment.

---

# 23. Đánh giá cuối cùng

PuSQ có ba giá trị lớn đối với project:

1. **Chứng minh SER có thể dùng như feature trung gian**, không cần là output cuối.
2. **Cho một pipeline rõ ràng** từ segment emotion → posterior aggregation → public-speaking quality.
3. **Cho baseline feature-level có thể tái hiện** bằng dữ liệu public.

Nhưng PuSQ cũng có các giới hạn:

- không có raw WAV public;
- dataset nhỏ;
- không phải bài presentation dài có slide;
- target cuối là binary;
- tách theo giới tính;
- model segment-level khá cũ;
- domain shift SER rất rõ;
- một số chi tiết trong paper có lỗi/không nhất quán (`134` vs `136`, khoảng arousal bị overlap).

Do đó, chiến lược hợp lý nhất là:

```text
PuSQ
→ dùng làm baseline và paper nền

Notebook 06d
→ dùng làm backbone hiện đại cho raw audio

Raw presentation dataset mới
→ dùng để train/fine-tune target:
   expressiveness
   delivery quality
   engagement
```

Nói ngắn gọn:

> Không nên cố ép feature PuSQ vào toàn bộ model `06d`. Nên dùng PuSQ để hiểu bài toán, tái hiện baseline và thiết kế target; sau đó dùng raw presentation audio khác để phát huy `emotion2vec + temporal + spectral + co-attention`.

---

# 24. Nguồn đã kiểm tra

1. Paper chính:  
   https://aclanthology.org/2021.icnlsp-1.19/

2. PDF paper:  
   https://aclanthology.org/2021.icnlsp-1.19.pdf

3. PuSQ dataset repo:  
   https://github.com/sofiaele/PuSQ

4. `readys` framework:  
   https://github.com/tyiannak/readys

5. PuSQ parser:  
   https://github.com/sofiaele/PuSQ/blob/main/dataset_parser.py

6. `readys` model guide:  
   https://github.com/tyiannak/readys/blob/master/models/readme.md

7. Notebook được đối chiếu:  
   `06d_Emotion2Vec_CoAttention_Full_SER.ipynb`
