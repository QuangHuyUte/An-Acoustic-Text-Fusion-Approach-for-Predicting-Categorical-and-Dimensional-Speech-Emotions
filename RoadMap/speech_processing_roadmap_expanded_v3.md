
---
title: "Roadmap chi tiết v3 - Hệ thống phản hồi luyện thuyết trình dựa trên giọng nói"
subtitle: "SER, VAD, Sliding Window, Smoothing, Emotion Timeline, Acoustic Indicators, Transcript Analysis và Feedback Engine"
author: "Nhóm: Nguyễn Minh Cường, Nguyễn Tài Huy, Bùi Quang Huy"
date: "22/06/2026"
lang: "vi"
---

# Roadmap chi tiết v3

# Hệ thống phản hồi luyện thuyết trình dựa trên giọng nói

**Tên đề tài đề xuất:** Hệ thống phản hồi luyện thuyết trình dựa trên emotion timeline và chỉ báo âm học từ giọng nói  
**Tên tiếng Anh:** Audio-Based Presentation Feedback System using Speech Emotion Recognition, VAD-guided Emotion Timeline and Acoustic Speaking Indicators  
**Lĩnh vực:** Speech Processing, Speech Emotion Recognition, Audio Feature Analysis, Near Real-Time Inference, Human-Computer Interaction  
**Nền tảng triển khai:** Google Colab/Kaggle Notebook cho training; Gradio hoặc Streamlit cho demo  
**Giới hạn tài nguyên:** không có GPU mạnh, ưu tiên mô hình vừa phải, feature-based module và rule-based feedback  

---

## 0. Vì sao cần bản v3 này?

Bản roadmap trước đã đúng hướng nhưng còn quá gọn. Nó mới liệt kê các module chính như dataset, preprocessing, SER model, emotion timeline, acoustic indicators, transcript, feedback và dashboard. Tuy nhiên, để triển khai thật sự và bảo vệ trước thầy cô, roadmap cần giải thích rõ hơn các câu hỏi sau:

1. Vì sao đề tài nên chốt thành **Public Speaking / Presentation Feedback** thay vì IELTS hoặc stress detection?
2. Dataset nào dùng để train emotion, dataset nào chỉ dùng làm benchmark, dataset nào dùng cho demo?
3. Nếu dataset chỉ có một label cho một audio, tại sao vẫn có thể làm near real-time emotion timeline?
4. VAD dùng để làm gì, không dùng VAD thì lỗi gì xảy ra?
5. Smoothing dùng để làm gì, tại sao prediction từng chunk dễ bị nhảy?
6. Feedback sinh bằng mô hình nào, hay bằng rule-based engine?
7. Stress indicator phải hiểu như thế nào để không bị phản biện là chẩn đoán y khoa?
8. Mỗi notebook cần làm gì, input/output là gì, artifact lưu ra sao?
9. Đánh giá hệ thống như thế nào nếu không có ground truth cho emotion timeline và feedback?
10. Nếu không kịp tất cả tính năng, phần nào bắt buộc làm, phần nào để nâng cao?

Bản v3 này trả lời chi tiết từng điểm trên. Nó không chỉ là danh sách việc cần làm, mà là một roadmap triển khai có thể biến thành notebook, source code, báo cáo và slide.

---

## 1. Chốt lại hướng đề tài cuối cùng

### 1.1. Hướng nên chốt

Hướng tốt nhất của đề tài là:

**Public Speaking / Presentation Feedback System**

Tức là hệ thống nhận một đoạn audio thuyết trình hoặc luyện nói, sau đó phân tích:

- cảm xúc giọng nói theo từng đoạn,
- năng lượng giọng nói,
- cao độ và độ biến thiên cao độ,
- khoảng lặng và pause,
- tốc độ nói,
- filler words nếu có transcript,
- mức độ ổn định khi trình bày,
- đoạn nào nên luyện lại.

Kết quả cuối cùng không chỉ là một nhãn emotion, mà là một báo cáo luyện nói/thuyết trình.

### 1.2. Vì sao không chốt IELTS Speaking làm hướng chính?

IELTS Speaking có 4 tiêu chí chính: Fluency and Coherence, Lexical Resource, Grammatical Range and Accuracy, Pronunciation. Trong đó emotion không phải tiêu chí chính thức. Nếu đề tài chuyển hẳn sang IELTS scoring, nhánh emotion sẽ trở thành phụ, còn trọng tâm sẽ chuyển sang ASR, grammar, lexical resource và LLM scoring. Như vậy đề tài sẽ lệch khỏi Speech Emotion Recognition ban đầu.

IELTS vẫn có thể là future work, nhưng bản chính nên là **presentation feedback** vì emotion, energy, pitch, pause và speaking pressure đều liên quan trực tiếp tới phong thái thuyết trình.

### 1.3. Vì sao không chốt stress detection làm hướng chính?

Các dataset chính như CREMA-D, RAVDESS và TESS có nhãn emotion, không có nhãn stress/non-stress. Nếu nói hệ thống phát hiện stress, sẽ bị hỏi:

- Dataset stress ở đâu?
- Ground truth stress là gì?
- Có dữ liệu physiological stress hoặc self-report không?
- Có rủi ro đạo đức/y khoa không?

Do đó, không gọi là stress detector. Cách gọi an toàn hơn là:

- **Acoustic Speaking Pressure Indicator**
- **Vocal Stability Indicator**
- **Stress-related Acoustic Indicator**
- **Confidence/Pressure Indicator for Presentation Feedback**

Nó chỉ là chỉ báo phục vụ feedback luyện nói, không phải chẩn đoán stress.

### 1.4. Một câu mô tả đề tài nên dùng khi báo cáo

> Đề tài xây dựng một hệ thống phản hồi luyện thuyết trình dựa trên phân tích giọng nói. Hệ thống huấn luyện mô hình Speech Emotion Recognition trên các dataset cảm xúc công khai, sau đó áp dụng mô hình theo các đoạn audio ngắn để tạo emotion timeline. Kết quả emotion được kết hợp với các chỉ báo âm học như RMS energy, pitch/F0, silence ratio, pause count, speech rate và filler words để đưa ra feedback giúp người dùng cải thiện phong thái trình bày.

---

## 2. Bức tranh tổng thể của hệ thống

### 2.1. Pipeline tổng quát

```text
Training phase
CREMA-D + RAVDESS speech-only (+ optional TESS/EmoDB)
        ↓
Preprocessing + Label mapping
        ↓
Feature extraction
        ↓
Baseline model: MFCC + SVM/RF
Main model: Mel-spectrogram + CNN
Advanced model: CNN-GRU/CNN-BiLSTM
        ↓
Saved SER model

Inference / Demo phase
User upload/record presentation audio
        ↓
Preprocessing
        ↓
VAD: speech/silence detection
        ↓
Sliding window segmentation
        ↓
SER prediction per chunk
        ↓
Smoothing and confidence filtering
        ↓
Emotion timeline
        ↓
Acoustic indicators + transcript metrics
        ↓
Feedback engine
        ↓
Dashboard report
```

### 2.2. Tư duy quan trọng

Hệ thống không đi từ raw audio thẳng ra feedback bằng một model lớn. Thay vào đó, hệ thống chia thành nhiều module nhỏ:

1. **SER model** dự đoán emotion.
2. **VAD module** phát hiện speech/silence.
3. **Smoothing module** làm emotion timeline ổn định hơn.
4. **Acoustic module** tính feature vật lý của giọng.
5. **Transcript module** chuyển speech thành text nếu có.
6. **Feedback engine** dùng rule để sinh nhận xét.

Cách chia module này rất phù hợp với sinh viên không có GPU mạnh vì phần nặng nhất chỉ là model SER vừa phải, còn các phần còn lại chủ yếu là feature extraction và rule-based logic.

---

## 3. Các paper nên dùng và vai trò thật sự của từng paper

### 3.1. Không nên đưa paper vào chỉ để làm reference chung chung

Mỗi paper nên được đặt vào một module cụ thể. Khi thầy cô hỏi "paper này giúp gì cho project?", nhóm phải trả lời được:

- Paper đó dùng dataset gì?
- Paper đó dùng mô hình gì?
- Paper đó có điểm nào mình học theo?
- Mình có implement không hay chỉ dùng làm cơ sở học thuật?
- Nếu không implement thì vì sao?

### 3.2. Bảng vai trò paper

| Paper | Vai trò trong project | Dataset của paper | Mô hình/phương pháp | Cách dùng trong đồ án |
|---|---|---|---|---|
| Real-time SER using Deep Learning and Data Augmentation | Backbone cho phần SER, augmentation và model comparison | TESS, EmoDB, RAVDESS | MLP, CNN, CNN+BiLSTM; MFCC, ZCR, Mel, RMS, Chroma; noise + spectrogram shift | Học cách thiết kế pipeline SER và thử CNN/CNN+BiLSTM |
| FLEA / Frame-Level Emotional State Alignment | Cơ sở cho emotion timeline và segment-level SER | IEMOCAP | HuBERT, pseudo frame labels, k-means, attention pooling | Không implement full; dùng để giải thích vì sao không nên chỉ output một emotion cho audio dài |
| Speech Emotion Diarization: Which Emotion Appears When? | Cơ sở học thuật cho bài toán emotion timeline đúng nghĩa | ZED | Emotion diarization, temporal boundaries | Dùng để nói bản mình là chunk-based timeline, chưa phải diarization đầy đủ |
| Multimodal Emotion Diarization | Cơ sở cho smoothing và timeline ổn định | Multimodal diarization setting | WavLM + EmoBERTa + sliding window smoothing | Không implement full; học ý tưởng smoothing theo context |
| VAD + SER paper | Cơ sở cho việc thêm VAD trước SER | IEMOCAP hoặc dataset SER liên quan | SSL features + VAD + SER module | Dùng WebRTC/Silero VAD thay vì joint training |
| Acoustic and Prosodic Stress Features paper | Cơ sở cho speaking pressure/acoustic indicators | Cyberball/MIST speech + physiology/self-report | openSMILE GeMAPS, F0, jitter, speech rate | Dùng để biện minh pitch, pause, speech rate liên quan đến pressure, không suy stress từ emotion |
| TBDM-Net | Cơ sở temporal SER nâng cao | CASIA, EMOVO, EMO-DB, IEMOCAP, RAVDESS, SAVEE | MFCC + temporal dense blocks + gender info | Future work hoặc reference cho temporal modeling, không cần implement full |

### 3.3. Paper nào thực sự nên đi theo khi code?

Paper nên đi theo nhiều nhất khi code là **Real-time SER using Deep Learning and Data Augmentation** vì nó gần với pipeline sinh viên:

- có dataset phổ biến,
- có feature phổ biến,
- có MLP/CNN/CNN+BiLSTM,
- có augmentation dễ làm,
- có mục tiêu real-time SER.

Tuy nhiên, paper này chưa đủ cho project của mình vì mục tiêu của mình là **presentation feedback**, không chỉ real-time SER. Do đó, phải bổ sung:

- VAD để không predict emotion trên silence,
- sliding window để phân tích audio dài,
- smoothing để timeline bớt nhảy,
- acoustic indicators để feedback có cơ sở,
- feedback engine để biến số liệu thành gợi ý.

---

## 4. Chiến lược dataset chi tiết

### 4.1. Nguyên tắc chọn dataset

Không có một dataset duy nhất cho toàn bộ hệ thống. Cần tách dataset theo vai trò:

- Dataset train SER: cần nhãn emotion.
- Dataset timeline thật: cần boundary emotion, nhưng khó kiếm và khó dùng.
- Dataset feedback: thường không có sẵn, nên dùng audio tự thu và đánh giá định tính.
- Dataset stress: không dùng để train chính, chỉ dùng làm cơ sở học thuật.

### 4.2. Dataset lõi nên dùng

**CREMA-D + RAVDESS speech-only** nên là bộ dữ liệu lõi.

Lý do:

- CREMA-D có 6 emotion rất khớp với hệ 6 lớp: anger, disgust, fear, happy, neutral, sad.
- CREMA-D có nhiều actor hơn TESS, giúp giảm bias theo một vài speaker.
- RAVDESS là dataset chuẩn học thuật, dễ đưa vào báo cáo.
- RAVDESS speech-only nhẹ hơn nhiều so với tải toàn bộ audio/video/song.
- Hai dataset này đủ để train baseline và CNN/CNN-GRU trong Colab/Kaggle.

### 4.3. TESS và EmoDB nên dùng như thế nào?

TESS và EmoDB không nên là core dataset cho demo public speaking. Chúng nên dùng theo 2 cách:

1. **Basic academic benchmark:** dùng để tái hiện ý tưởng paper real-time SER.
2. **Optional/add-on:** thêm vào sau nếu cần tăng số mẫu hoặc external test.

TESS có audio sạch nhưng chỉ có 2 speaker nữ, dễ tạo bias. EmoDB nhỏ và tiếng Đức, tốt cho benchmark nhưng không phải lựa chọn tối ưu cho demo tiếng Anh/public speaking.

### 4.4. IEMOCAP có nên dùng không?

IEMOCAP rất mạnh về học thuật, nhưng không nên đặt làm dataset chính nếu nhóm thiếu thời gian. Nó phù hợp cho:

- experiment nâng cao 4-class SER,
- future work với HuBERT/Wav2Vec2,
- thảo luận frame-level/utterance-level emotion.

Nếu dùng IEMOCAP, nên làm experiment riêng, không gộp vội vào 6-class pipeline vì label setting khác.

### 4.5. ZED hoặc emotion diarization dataset có nên dùng không?

ZED phù hợp với emotion diarization, tức là có boundary cảm xúc theo thời gian. Đây là hướng đúng nhất nếu muốn làm emotion timeline chuẩn nghiên cứu. Tuy nhiên, với đồ án sinh viên, việc tải, xử lý và đánh giá boundary có thể vượt scope.

Nên dùng ZED làm reference/future work, không làm dataset chính.

### 4.6. Audio tự thu dùng để làm gì?

Audio tự thu là phần rất quan trọng cho demo presentation feedback. Dataset SER chỉ chứa câu ngắn diễn xuất, không phải bài thuyết trình thật. Vì vậy nhóm nên tự thu khoảng 10-30 file audio:

- mỗi file 30-90 giây,
- có audio nói tự tin,
- có audio nói quá nhanh,
- có audio nói quá chậm,
- có audio nói nhiều pause,
- có audio nói nhiều filler,
- có audio nói nhỏ dần,
- có audio monotone,
- có audio năng lượng tốt.

Audio tự thu không dùng để train emotion model. Nó dùng để kiểm thử demo, feedback engine và dashboard.

### 4.7. Label mapping 6 lớp

| Unified label | CREMA-D | RAVDESS | TESS | Ghi chú |
|---|---|---|---|---|
| neutral | neutral | neutral | neutral | Lớp nền, dễ bị model over-predict |
| happy | happy | happy | happy | Dùng để nhận diện giọng sáng/tích cực |
| sad | sad | sad | sad | Có thể liên quan energy thấp |
| angry | anger | angry | angry | Có thể energy cao, tone mạnh |
| fear | fear | fearful | fear | Có thể liên quan thiếu tự tin nhưng không kết luận stress |
| disgust | disgust | disgust | disgust | Dễ nhầm với angry |

Không dùng calm/surprise ở bản chính vì không đồng nhất giữa dataset.

---

## 5. Module 1 - Literature Review và Problem Framing

### 5.1. Mục tiêu

Module này không phải code, nhưng rất quan trọng cho báo cáo. Mục tiêu là chứng minh đề tài không chỉ là một notebook train CNN đơn giản, mà có cơ sở học thuật rõ.

### 5.2. Cần viết rõ những luận điểm nào?

**Luận điểm 1:** SER truyền thống thường output một label cho toàn bộ audio, nhưng trong presentation feedback cần phân tích theo thời gian.  
**Luận điểm 2:** Real-time SER thường xử lý theo block/window ngắn, không nhất thiết cần frame-level ground truth.  
**Luận điểm 3:** VAD giúp tránh predict emotion trên silence/noise và hỗ trợ tính pause.  
**Luận điểm 4:** Smoothing giúp emotion timeline ổn định hơn vì prediction từng chunk dễ nhảy.  
**Luận điểm 5:** Speaking pressure không phải stress diagnosis, mà là chỉ báo dựa trên pitch, energy, pause, speech rate và filler.  
**Luận điểm 6:** Feedback nên dựa trên nhiều tín hiệu, không dựa vào emotion một mình.

### 5.3. Output của module này

- Bảng tổng quan paper.
- Bảng vai trò của từng paper trong hệ thống.
- Research gap.
- Lý do chọn public speaking/presentation feedback.
- Lý do chọn chunk-based near real-time thay vì frame-level emotion detection.

### 5.4. Cách viết research gap

Một đoạn có thể dùng:

> Many SER systems focus on predicting a single emotion label from a short utterance. However, in presentation practice, the user needs feedback over time: which part sounds unstable, where the speaker pauses too much, and where the voice energy decreases. Therefore, this project extends standard SER into a VAD-guided, sliding-window, smoothed emotion timeline and combines it with acoustic speaking indicators to generate practical presentation feedback.

---

## 6. Module 2 - Data Preparation

### 6.1. Mục tiêu

Chuẩn hóa dữ liệu từ nhiều nguồn thành một metadata thống nhất để train SER 6 lớp.

### 6.2. Input

- Thư mục CREMA-D.
- Thư mục RAVDESS speech-only.
- Tùy chọn TESS/EmoDB.

### 6.3. Output

- `metadata_raw.csv`
- `metadata_clean.csv`
- `label_mapping.json`
- thư mục `processed_audio/`

### 6.4. Các bước chi tiết

**Bước 1: Duyệt file audio.**  
Duyệt tất cả file `.wav`, `.mp3`, `.flac` nếu có. Lưu path tuyệt đối hoặc path tương đối.

**Bước 2: Parse dataset name.**  
Từ folder hoặc filename xác định file thuộc CREMA-D, RAVDESS hay TESS.

**Bước 3: Parse speaker id.**  
Với CREMA-D và RAVDESS có thể parse speaker từ filename. Speaker id rất quan trọng để split theo speaker nếu làm được.

**Bước 4: Parse emotion raw.**  
Mỗi dataset có cách encode emotion khác nhau. Cần viết parser riêng cho từng dataset.

**Bước 5: Map emotion về unified label.**  
Map anger/angry thành angry; fearful/fear thành fear; disgusted/disgust thành disgust.

**Bước 6: Loại bỏ label không dùng.**  
Bỏ calm và surprise trong bản chính.

**Bước 7: Kiểm tra duration.**  
Dùng librosa hoặc soundfile để lấy duration, sample rate, số channel.

**Bước 8: Tạo split.**  
Ưu tiên speaker-independent split. Nếu khó, dùng stratified split theo label nhưng ghi limitation.

### 6.5. Metadata schema đầy đủ

| Cột | Bắt buộc | Ý nghĩa |
|---|---|---|
| file_id | Có | ID duy nhất của audio |
| path | Có | Đường dẫn file |
| dataset | Có | CREMA-D, RAVDESS, TESS, EmoDB |
| speaker_id | Nên có | ID speaker |
| gender | Optional | Nếu dataset có |
| emotion_raw | Có | Nhãn gốc |
| emotion | Có | Nhãn unified 6 lớp |
| duration_sec | Có | Độ dài audio |
| sample_rate | Có | Sample rate gốc |
| num_channels | Có | Mono/stereo |
| split | Có | train/val/test |
| usable | Có | true/false sau khi lọc |
| note | Optional | Ghi chú lỗi, file ngắn, label bỏ |

### 6.6. Lỗi thường gặp

- Parse sai label RAVDESS vì filename encode bằng số.
- Gộp calm vào neutral tùy tiện làm nhiễu nhãn.
- Split sau khi augmentation gây leakage.
- Split sau khi cắt segment khiến cùng audio xuất hiện ở train và test.
- Không kiểm tra class distribution sau split.

### 6.7. Artifact cần lưu

- `metadata_clean.csv`
- `class_distribution.png`
- `dataset_distribution.png`
- `split_distribution.png`
- `label_mapping.json`

---

## 7. Module 3 - Audio Preprocessing

### 7.1. Mục tiêu

Đưa audio về format thống nhất để feature extraction và model training không bị lệch.

### 7.2. Các bước chuẩn hóa

1. Load audio.
2. Convert mono.
3. Resample về 16 kHz.
4. Normalize amplitude.
5. Trim silence đầu/cuối nếu phù hợp.
6. Pad hoặc crop về 3 giây.
7. Lưu processed waveform nếu cần.

### 7.3. Vì sao chọn 16 kHz?

16 kHz đủ cho speech processing cơ bản. Nó giảm dung lượng và tốc độ xử lý so với 44.1 kHz, nhưng vẫn giữ phần lớn thông tin lời nói cần cho MFCC/Mel-spectrogram.

### 7.4. Vì sao chọn 3 giây?

3 giây là độ dài hợp lý cho SER trong project này vì:

- đủ ngữ cảnh hơn 1 giây,
- không quá dài gây chậm realtime,
- phù hợp với chunk-based timeline,
- dễ pad/crop từ các dataset ngắn.

Có thể thử 2 giây hoặc 4 giây nếu có thời gian, nhưng 3 giây nên là mặc định.

### 7.5. Pad/crop như thế nào?

Nếu audio ngắn hơn 3 giây:

- pad bằng silence ở cuối,
- hoặc center pad.

Nếu audio dài hơn 3 giây:

- crop center,
- hoặc random crop trong training,
- hoặc sliding window khi inference.

### 7.6. Lưu ý với trim silence

Trim silence giúp bỏ khoảng lặng đầu/cuối trong dataset, nhưng không nên trim quá mạnh ở audio demo vì silence/pause là tín hiệu quan trọng cho feedback. Do đó:

- training dataset: có thể trim silence đầu/cuối nhẹ,
- demo presentation audio: không trim toàn bộ pause giữa bài, chỉ normalize và dùng VAD để tính pause.

---

## 8. Module 4 - EDA và Visualization

### 8.1. Vì sao EDA quan trọng?

EDA giúp báo cáo có chiều sâu. Nó chứng minh nhóm hiểu dữ liệu trước khi train model.

### 8.2. Biểu đồ bắt buộc

**Samples by emotion:** kiểm tra class imbalance.  
**Samples by dataset:** xem dataset nào chiếm ưu thế.  
**Samples by speaker:** kiểm tra speaker imbalance.  
**Duration distribution:** quyết định pad/crop.  
**Waveform examples:** minh họa tín hiệu raw.  
**Mel-spectrogram examples:** minh họa thời gian-tần số.  
**MFCC heatmap:** minh họa feature truyền thống.  
**RMS by emotion:** xem energy khác nhau theo emotion.  
**Pitch by emotion:** xem F0 khác nhau theo emotion.

### 8.3. Phân tích nên viết trong report

Không nên chỉ chèn hình. Mỗi hình cần có nhận xét:

- Dataset có cân bằng không?
- Emotion nào ít mẫu hơn?
- Duration đa số bao nhiêu giây?
- Có nên pad/crop 3 giây không?
- Angry có energy cao hơn không?
- Sad có energy thấp hơn không?
- Fear và sad có thể dễ nhầm không?

### 8.4. Output cần lưu

- `samples_by_emotion.png`
- `samples_by_dataset.png`
- `duration_distribution.png`
- `waveform_examples.png`
- `mel_spectrogram_examples.png`
- `mfcc_examples.png`
- `rms_by_emotion.png`
- `pitch_by_emotion.png`

---

## 9. Module 5 - Feature Extraction cho SER

### 9.1. MFCC

MFCC là feature truyền thống rất phổ biến trong speech processing. Nó mô phỏng cách tai người cảm nhận tần số. MFCC phù hợp cho baseline vì nhẹ và dễ giải thích.

Dùng cho:

- SVM,
- Random Forest,
- MLP,
- temporal model nếu dùng MFCC sequence.

Cách tạo feature cho baseline:

```text
Audio 3s
→ MFCC matrix: n_mfcc x time_frames
→ mean/std theo thời gian
→ vector cố định
→ SVM/RF
```

### 9.2. Delta MFCC

Delta MFCC biểu diễn sự thay đổi của MFCC theo thời gian. Nó hữu ích vì emotion không chỉ nằm ở phổ tĩnh, mà còn ở cách giọng thay đổi.

Có thể tạo vector:

```text
MFCC mean + MFCC std
Delta mean + Delta std
Delta-delta mean + Delta-delta std
```

### 9.3. Mel-spectrogram

Mel-spectrogram là biểu diễn thời gian-tần số. Nó gần giống một ảnh, nên phù hợp với CNN.

Dùng cho:

- CNN 2D,
- CNN-GRU,
- CNN-BiLSTM,
- visualization.

### 9.4. ZCR, RMS, Chroma

Paper real-time SER dùng MFCC, ZCR, Mel-spectrogram, RMS và Chroma. Với project này:

- RMS dùng nhiều cho acoustic indicator.
- ZCR có thể dùng thêm cho feature vector.
- Chroma không bắt buộc vì speech không giống music, nhưng có thể thử nếu muốn reproduce paper.

### 9.5. Artifact cần lưu

- `features_mfcc_train.npy`
- `features_mfcc_val.npy`
- `features_mfcc_test.npy`
- `mel_train.npy` hoặc extract on-the-fly
- `feature_config.json`

---

## 10. Module 6 - Data Augmentation

### 10.1. Vì sao cần augmentation?

Dataset SER thường nhỏ và acted. Model dễ overfit. Augmentation giúp model robust hơn với noise, timing variation và môi trường thật.

### 10.2. Augmentation nên làm

**Noise addition:** thêm white noise hoặc background noise nhẹ.  
**Time shift / spectrogram shift:** dịch audio hoặc spectrogram theo thời gian.  
**Time stretch:** nói nhanh/chậm nhẹ.  
**Pitch shift:** tăng/giảm pitch nhẹ.  
**Volume gain:** tăng/giảm âm lượng nhẹ.

### 10.3. Augmentation nào nên bắt buộc?

Bắt buộc nên có:

- noise addition,
- time/spectrogram shift.

Vì paper real-time SER bạn thu thập cũng dùng noise addition và spectrogram shift. Hai kỹ thuật này dễ làm và giải thích tốt.

### 10.4. Lưu ý tránh data leakage

Không được augmentation trước rồi mới split. Cách đúng:

```text
Split original files first
→ train files only
→ apply augmentation only on train split
```

Validation/test phải giữ sạch để đánh giá công bằng.

### 10.5. Ablation nên làm

Nếu kịp, so sánh:

| Thử nghiệm | Mục tiêu |
|---|---|
| CNN không augmentation | baseline deep learning |
| CNN + noise | xem noise có giúp không |
| CNN + shift | xem shift có giúp không |
| CNN + noise + shift | cấu hình tốt nhất |

---

## 11. Module 7 - Baseline Model

### 11.1. Mục tiêu

Baseline là mốc so sánh. Nó giúp chứng minh CNN có cải thiện so với ML truyền thống hay không.

### 11.2. Mô hình nên dùng

- MFCC + SVM
- MFCC + Random Forest

Nếu chỉ chọn một, chọn **MFCC + SVM**.

### 11.3. Pipeline

```text
Audio 3s
→ MFCC
→ mean/std pooling
→ feature vector
→ StandardScaler
→ SVM
→ emotion label
```

### 11.4. Hyperparameter cơ bản

| Thành phần | Gợi ý |
|---|---|
| n_mfcc | 40 |
| pooling | mean + std |
| scaler | StandardScaler |
| SVM kernel | RBF |
| C | 1, 10 |
| gamma | scale |

### 11.5. Output cần báo cáo

- Accuracy.
- Macro-F1.
- Classification report.
- Confusion matrix.
- Training time.

### 11.6. Ý nghĩa khi thuyết trình

Có thể nói:

> Baseline MFCC + SVM được dùng để tạo mốc so sánh truyền thống. Nếu CNN tốt hơn baseline, điều đó cho thấy biểu diễn Mel-spectrogram và deep learning học được pattern cảm xúc tốt hơn feature thống kê đơn giản.

---

## 12. Module 8 - Main SER Model: CNN

### 12.1. Vì sao chọn CNN làm main model?

CNN phù hợp vì Mel-spectrogram là ma trận 2D gồm thời gian và tần số. CNN học được pattern cục bộ như vùng năng lượng cao/thấp, formant, biến thiên phổ.

CNN cũng đủ nhẹ để chạy trên Colab/Kaggle và phù hợp demo near real-time.

### 12.2. Input

- Mel-spectrogram shape ví dụ: `1 x 128 x T`
- hoặc log-Mel spectrogram.

### 12.3. Kiến trúc gợi ý

```text
Input log-Mel
→ Conv2D + BatchNorm + ReLU + MaxPool
→ Conv2D + BatchNorm + ReLU + MaxPool
→ Conv2D + BatchNorm + ReLU + MaxPool
→ Global Average Pooling
→ Dropout
→ Dense/Linear
→ Softmax 6 classes
```

### 12.4. Loss và optimizer

- CrossEntropyLoss.
- Adam hoặc AdamW.
- Learning rate: 1e-3 hoặc 3e-4.
- Early stopping theo validation macro-F1.

### 12.5. Metric chính

Macro-F1 quan trọng hơn accuracy vì dataset có thể lệch nhãn.

### 12.6. Cách lưu model

Lưu:

- `cnn_ser_model.pt`
- `label_encoder.pkl`
- `model_config.json`
- `feature_config.json`

### 12.7. Lưu ý overfitting

Nếu train accuracy cao nhưng val F1 thấp:

- thêm dropout,
- thêm augmentation,
- giảm model size,
- dùng class weight,
- kiểm tra split theo speaker.

---

## 13. Module 9 - Advanced Model

### 13.1. Advanced model để làm gì?

Advanced model không bắt buộc, nhưng giúp lấy điểm cộng. Nó chứng minh nhóm có thử nghiệm hướng temporal modeling.

### 13.2. Lựa chọn nên ưu tiên

Thứ tự nên thử:

1. CNN-GRU
2. CNN-LSTM
3. CNN-BiLSTM
4. Wav2Vec2/HuBERT embedding + classifier

### 13.3. Vì sao CNN-GRU có thể hợp hơn CNN-BiLSTM cho demo?

GRU nhẹ hơn LSTM và thường inference nhanh hơn. BiLSTM nhìn cả quá khứ và tương lai trong một chunk, vẫn dùng được cho chunk-based demo, nhưng không tối ưu nếu muốn streaming thật.

### 13.4. CNN-GRU pipeline

```text
Mel-spectrogram
→ CNN feature extractor
→ reshape theo time axis
→ GRU
→ Dense
→ Softmax emotion
```

### 13.5. Khi nào dùng advanced model cho demo?

Chỉ dùng nếu:

- validation macro-F1 tốt hơn CNN rõ ràng,
- inference time vẫn chấp nhận được,
- model không quá khó load trong Gradio.

Nếu advanced model không tốt hơn, vẫn dùng CNN cho demo và trình bày advanced model như experiment.

### 13.6. Vai trò của TBDM-Net và FLEA

Không implement full TBDM-Net/FLEA. Dùng chúng để chứng minh temporal SER là hướng có cơ sở.

- TBDM-Net giúp giải thích temporal dense/multi-scale feature.
- FLEA giúp giải thích frame/segment-level emotion và hạn chế của utterance-level label.

---

## 14. Module 10 - VAD

### 14.1. VAD là gì?

VAD là Voice Activity Detection, dùng để xác định đoạn nào có tiếng nói và đoạn nào là silence/noise.

### 14.2. Vì sao cần VAD trong project này?

Nếu không có VAD, hệ thống có thể predict emotion trên đoạn im lặng. Ví dụ một chunk toàn silence vẫn bị CNN gán thành neutral hoặc sad. Điều này làm emotion timeline sai và feedback không đáng tin.

VAD giúp:

- bỏ hoặc giảm trọng số emotion prediction trên silence,
- tính silence ratio,
- tính pause count,
- phát hiện đoạn ngập ngừng,
- hỗ trợ speaking pressure indicator.

### 14.3. VAD dùng mô hình gì?

Bản demo không cần train VAD. Có thể dùng:

- WebRTC VAD: nhẹ, rule/classical, rất nhanh.
- Silero VAD: neural VAD pretrained, dễ dùng, robust hơn.

### 14.4. Cách tích hợp VAD

```text
Audio chunk
→ VAD frame-level speech probability
→ speech_ratio trong chunk
→ nếu speech_ratio thấp: không tin emotion prediction
→ nếu speech_ratio cao: đưa vào SER bình thường
```

### 14.5. Output của VAD module

| Output | Ý nghĩa |
|---|---|
| speech_ratio | Tỷ lệ speech trong chunk |
| silence_ratio | 1 - speech_ratio |
| voiced_segments | Các đoạn có tiếng nói |
| pause_segments | Các đoạn im lặng đủ dài |
| pause_count | Số lần pause |
| mean_pause_duration | Độ dài pause trung bình |

### 14.6. Ablation nên làm

So sánh:

- timeline không dùng VAD,
- timeline dùng VAD,
- feedback có/không có pause detection.

Mục tiêu không nhất thiết tăng SER accuracy, mà làm timeline và feedback hợp lý hơn.

---

## 15. Module 11 - Sliding Window và Near Real-time

### 15.1. Vấn đề dataset chỉ có 1 label/audio

Dataset SER thường có một nhãn cho một audio. Điều này không ngăn được near real-time vì real-time SER trong thực tế thường xử lý theo window/block ngắn.

Cách hiểu đúng:

> Mỗi chunk 2-3 giây được xem như một utterance ngắn. Model dự đoán emotion chủ đạo cho chunk đó.

### 15.2. Cấu hình đề xuất

| Tham số | Gợi ý |
|---|---|
| window_size | 3 giây |
| hop_size | 1 hoặc 1.5 giây |
| overlap | 50-67% |
| sample_rate | 16 kHz |
| VAD frame | 20-30 ms nếu dùng WebRTC |

### 15.3. Vì sao dùng overlap?

Nếu không overlap, timeline có thể thô. Ví dụ mỗi 3 giây mới có một prediction. Overlap giúp cập nhật mượt hơn và giảm mất thông tin ở biên chunk.

### 15.4. Simulated near real-time

Với Colab/Kaggle, streaming microphone thật dễ lỗi. Do đó demo nên dùng:

1. upload audio dài,
2. chia chunk,
3. xử lý từng chunk theo thứ tự thời gian,
4. hiển thị timeline như đang realtime.

Cách này gọi là **simulated near real-time analysis**.

### 15.5. Record audio trong Gradio

Gradio có thể cho record audio trực tiếp. Sau khi record xong, hệ thống phân tích chunk-by-chunk. Đây là cách demo ổn định hơn streaming thật.

---

## 16. Module 12 - Smoothing

### 16.1. Vì sao cần smoothing?

Nếu predict từng chunk độc lập, emotion timeline dễ nhảy:

```text
neutral → fear → neutral → sad → neutral
```

Điều này có thể do noise, chunk quá ngắn, hoặc confidence thấp. Smoothing giúp timeline ổn định hơn.

### 16.2. Loại smoothing nên dùng

**Moving average probability:** lấy trung bình xác suất của nhiều chunk gần nhất.  
**Majority vote:** lấy nhãn xuất hiện nhiều nhất trong 3 chunk gần nhất.  
**Confidence threshold:** nếu max probability thấp thì giữ nhãn cũ hoặc đánh dấu uncertain.  
**Silence-aware weighting:** chunk có silence nhiều thì giảm trọng số emotion.

### 16.3. Công thức đơn giản

```text
smooth_prob[t] = mean(prob[t-k+1], ..., prob[t])
```

Ví dụ với k = 3:

```text
smooth_prob[t] = (prob[t] + prob[t-1] + prob[t-2]) / 3
```

### 16.4. Rule đổi nhãn

Chỉ đổi current emotion nếu:

- confidence sau smoothing > 0.55 hoặc 0.60,
- hoặc nhãn mới xuất hiện liên tiếp ít nhất 2 chunk.

### 16.5. Output của smoothing

- raw emotion timeline,
- smoothed emotion timeline,
- confidence curve,
- uncertain segments.

### 16.6. Đánh giá smoothing

Không cần ground truth timeline. Có thể đánh giá bằng:

- timeline có bớt nhảy không,
- số lần chuyển emotion giảm bao nhiêu,
- feedback có ít false alarm hơn không,
- người nghe cảm thấy timeline hợp lý hơn không.

---

## 17. Module 13 - Emotion Timeline

### 17.1. Mục tiêu

Emotion timeline trả lời câu hỏi:

> Trong bài thuyết trình, đoạn nào giọng nói mang sắc thái nào?

### 17.2. Output cần có

| Time | Raw emotion | Smoothed emotion | Confidence | Speech ratio | Note |
|---|---|---|---|---|---|
| 0-3s | neutral | neutral | 0.72 | 0.91 | ổn định |
| 1.5-4.5s | neutral | neutral | 0.69 | 0.88 | ổn định |
| 3-6s | fear | neutral | 0.52 | 0.77 | raw thấp, smoothing giữ neutral |
| 4.5-7.5s | sad | sad | 0.64 | 0.86 | energy thấp |

### 17.3. Cách dùng trong feedback

Không dùng emotion một mình. Chỉ dùng emotion khi kết hợp với acoustic indicators:

- neutral quá nhiều + pitch thấp → giọng đều, thiếu biểu cảm.
- fear/sad + pause nhiều + energy thấp → đoạn có speaking pressure cao.
- angry/disgust nhiều + energy cao → tone có thể hơi gắt.
- happy vừa phải + energy ổn → giọng có sức sống.

### 17.4. Hạn chế cần ghi

Do dataset chỉ có utterance-level label, timeline này là chunk-based approximation, không phải ground-truth frame-level emotion detection.

---

## 18. Module 14 - Acoustic Indicators

### 18.1. Mục tiêu

Acoustic indicators giúp feedback có cơ sở rõ hơn emotion softmax. Người dùng sẽ hiểu "vì sao hệ thống nói tôi thiếu ổn định".

### 18.2. RMS Energy

RMS biểu diễn năng lượng/âm lượng của giọng. Dùng để phát hiện:

- giọng quá nhỏ,
- giọng lúc to lúc nhỏ,
- energy giảm về cuối bài,
- đoạn nói thiếu sức sống.

Output:

- mean RMS,
- std RMS,
- energy trend,
- low-energy segments.

### 18.3. Pitch/F0

Pitch/F0 thể hiện cao độ giọng. Dùng để phân tích:

- giọng quá phẳng,
- giọng dao động mạnh,
- intonation chưa tốt,
- voice stability.

Output:

- pitch mean,
- pitch std,
- pitch range,
- voiced frame ratio.

### 18.4. Silence Ratio và Pause Count

Silence ratio là tỷ lệ im lặng trong audio. Pause count là số đoạn im lặng dài hơn một ngưỡng, ví dụ 0.5 giây hoặc 0.8 giây.

Dùng để phát hiện:

- ngập ngừng,
- thiếu liền mạch,
- không chuẩn bị ý tốt,
- chuyển ý bị đứt đoạn.

### 18.5. Speaking Continuity

Có thể tính:

```text
speaking_continuity = 1 - silence_ratio
```

Hoặc phức tạp hơn:

```text
continuity = voiced_duration / total_duration
```

### 18.6. Output acoustic report

| Indicator | Value | Level | Feedback |
|---|---|---|---|
| RMS mean | 0.034 | medium | âm lượng ổn |
| RMS trend | decreasing | warning | energy giảm về cuối |
| Pitch std | low | warning | giọng hơi đều |
| Silence ratio | 0.35 | high | nhiều khoảng lặng |
| Pause count | 12/min | high | ngắt quãng nhiều |

---

## 19. Module 15 - Speaking Pressure Indicator

### 19.1. Định nghĩa

Speaking Pressure Indicator là điểm tổng hợp phản ánh mức độ thiếu ổn định khi nói. Nó không phải stress diagnosis.

### 19.2. Thành phần điểm

```text
pressure_score =
0.25 * pause_score
+ 0.20 * energy_instability_score
+ 0.20 * pitch_instability_score
+ 0.15 * speech_rate_score
+ 0.10 * filler_score
+ 0.10 * emotion_pressure_score
```

Các trọng số này là rule ban đầu, có thể chỉnh sau khi test audio tự thu.

### 19.3. Cách chuẩn hóa từng score

Mỗi score nên đưa về 0-100.

**pause_score:** cao khi silence ratio và pause count cao.  
**energy_instability_score:** cao khi RMS std cao hoặc energy giảm rõ.  
**pitch_instability_score:** cao khi pitch quá phẳng hoặc dao động quá mạnh.  
**speech_rate_score:** cao khi WPM quá nhanh hoặc quá chậm.  
**filler_score:** cao khi filler words per minute cao.  
**emotion_pressure_score:** cao nhẹ khi fear/sad xuất hiện cùng acoustic bất ổn.

### 19.4. Mức đánh giá

| Score | Level | Ý nghĩa |
|---|---|---|
| 0-35 | Low | giọng tương đối ổn định |
| 36-65 | Medium | có một số dấu hiệu thiếu ổn định |
| 66-100 | High | nhiều dấu hiệu ngập ngừng hoặc thiếu kiểm soát |

### 19.5. Cách tránh phản biện

Không nói:

> Người dùng bị stress.

Nói:

> Một số đoạn có speaking pressure cao hơn, thể hiện qua pause dài, energy thấp và pitch không ổn định.

---

## 20. Module 16 - Transcript, Speech Rate và Filler Words

### 20.1. Mục tiêu

Transcript giúp feedback cụ thể hơn. Nếu không có transcript, hệ thống vẫn có emotion và acoustic feedback. Nếu có transcript, có thể thêm speech rate và filler words.

### 20.2. Công cụ

Dùng Whisper tiny hoặc base để giảm tải. Không train ASR.

### 20.3. Speech Rate

```text
speech_rate = number_of_words / duration_minutes
```

Gợi ý:

| WPM | Nhận xét |
|---|---|
| < 90 | hơi chậm, có thể bị ngắt quãng |
| 90-160 | tương đối ổn |
| > 160 | hơi nhanh, người nghe khó theo dõi |

Các ngưỡng này chỉ là rule demo, không phải chuẩn tuyệt đối.

### 20.4. Filler Words

Tiếng Anh:

```text
um, uh, er, ah, like, you know, actually, basically, I mean, so
```

Tiếng Việt:

```text
ờ, ừm, à, thì, là, kiểu như, nói chung là, thật ra là
```

### 20.5. Hạn chế

Whisper có thể bỏ qua filler words hoặc nhận sai, đặc biệt trong audio nhiễu. Do đó filler detection nên ghi là experimental feature.

---

## 21. Module 17 - Feedback Engine

### 21.1. Feedback không nên sinh trực tiếp từ CNN

CNN chỉ dự đoán emotion. Nó không nên tự viết feedback. Feedback nên được tạo bởi rule-based engine dựa trên các chỉ số đã tính.

### 21.2. Input của feedback engine

```json
{
  "emotion_distribution": {},
  "emotion_timeline": [],
  "pressure_timeline": [],
  "rms_stats": {},
  "pitch_stats": {},
  "pause_stats": {},
  "speech_rate": 0,
  "filler_count": 0,
  "high_pressure_segments": []
}
```

### 21.3. Output của feedback engine

1. Overall feedback.
2. Timeline feedback.
3. Improvement suggestions.
4. Segment replay recommendation.

### 21.4. Rule examples

| Điều kiện | Feedback |
|---|---|
| neutral > 70% và pitch std thấp | Giọng khá đều, nên thêm ngữ điệu và nhấn keyword. |
| silence ratio > 0.30 | Có nhiều khoảng lặng, nên chuẩn bị outline rõ hơn. |
| pause count cao ở giữa bài | Đoạn giữa bài bị ngắt quãng, nên luyện phần chuyển ý. |
| energy giảm về cuối | Năng lượng giọng giảm, nên giữ hơi và kết bài chắc hơn. |
| speech rate > 160 WPM | Nói hơi nhanh, nên giảm tốc ở ý quan trọng. |
| filler count cao | Dùng nhiều từ đệm, nên thay bằng pause ngắn có kiểm soát. |
| fear/sad + pause cao + energy thấp | Đoạn này có speaking pressure cao, nên luyện lại. |

### 21.5. Feedback mẫu

> Bài nói của bạn có nội dung tương đối liền mạch, nhưng đoạn 12-18s có nhiều khoảng lặng và energy thấp. Emotion timeline cũng cho thấy đoạn này nghiêng về fear/sad với confidence trung bình. Hệ thống đánh dấu đây là đoạn có speaking pressure cao hơn. Bạn nên luyện lại phần chuyển ý ở đoạn này, giảm pause dài và giữ âm lượng ổn định hơn.

### 21.6. Có nên dùng LLM để viết feedback không?

Có thể dùng LLM ở bước cuối để viết câu mượt hơn, nhưng không nên để LLM tự phân tích. Cách an toàn:

```text
Rule-based engine xác định lỗi chính
→ LLM chỉ rewrite thành đoạn văn tự nhiên
```

Nếu không có API hoặc không muốn phụ thuộc, rule-based template là đủ.

---

## 22. Module 18 - Dashboard Demo

### 22.1. Công nghệ đề xuất

Ưu tiên Gradio vì dễ upload/record audio.

### 22.2. Layout dashboard

Các khối nên có:

1. Audio input.
2. Waveform.
3. Mel-spectrogram.
4. Overall emotion distribution.
5. Raw vs smoothed emotion timeline.
6. Acoustic indicators table.
7. Speaking pressure timeline.
8. Transcript.
9. Speech rate and filler words.
10. Feedback text.

### 22.3. Luồng demo khi thuyết trình

**Demo 1:** Upload audio ngắn từ dataset để chứng minh model predict emotion.  
**Demo 2:** Upload audio tự thu 60 giây để hiển thị timeline.  
**Demo 3:** Bật VAD/smoothing để so sánh timeline raw và smoothed.  
**Demo 4:** Hiển thị feedback cụ thể và giải thích rule.

### 22.4. Không nên demo gì?

Không nên phụ thuộc vào microphone streaming thật nếu chưa test kỹ. Nếu lỗi microphone trong lúc báo cáo, demo sẽ hỏng. Upload/record rồi phân tích sau là an toàn hơn.

---

## 23. Module 19 - Evaluation

### 23.1. Đánh giá SER model

Dùng:

- Accuracy.
- Precision.
- Recall.
- Macro-F1.
- Confusion matrix.
- Per-class F1.
- Training/validation loss.

### 23.2. Đánh giá dataset split

Cần báo cáo:

- class distribution train/val/test,
- speaker distribution nếu có,
- dataset distribution.

### 23.3. Đánh giá augmentation

So sánh:

- không augmentation,
- noise addition,
- spectrogram shift,
- noise + shift.

### 23.4. Đánh giá VAD

Nếu không có ground truth VAD, đánh giá định tính:

- silence chunk có bị predict emotion không,
- pause count có hợp lý không,
- timeline có sạch hơn không.

### 23.5. Đánh giá smoothing

Dùng:

- số lần emotion transition trước/sau smoothing,
- tỷ lệ uncertain segment,
- ví dụ timeline raw vs smoothed,
- nhận xét định tính của nhóm.

### 23.6. Đánh giá feedback

Vì không có expert feedback dataset, dùng audio tự thu có kịch bản:

| Audio test | Kỳ vọng |
|---|---|
| nhiều pause | hệ thống báo pause/silence cao |
| nói nhanh | speech rate cao |
| nói chậm | speech rate thấp |
| nói nhỏ dần | energy decreasing |
| monotone | pitch variation thấp |
| nhiều filler | filler count cao |

### 23.7. Đánh giá latency

Với near real-time, nên đo:

- feature extraction time per chunk,
- model inference time per chunk,
- total processing time per 60s audio,
- dashboard update time.

---

## 24. Cấu trúc notebook chi tiết

### 24.1. Notebook 01 - Data Preparation

Input:

- raw dataset folders.

Tasks:

- parse labels,
- map 6 classes,
- duration check,
- split,
- save metadata.

Output:

- `metadata_clean.csv`,
- `label_mapping.json`,
- distribution figures.

### 24.2. Notebook 02 - EDA

Tasks:

- plot label distribution,
- plot dataset distribution,
- plot duration,
- show waveform,
- show Mel-spectrogram,
- show MFCC,
- show RMS/pitch examples.

Output:

- EDA figures for report.

### 24.3. Notebook 03 - Feature Extraction

Tasks:

- extract MFCC vectors,
- extract Mel-spectrogram,
- compute acoustic indicators,
- save features.

Output:

- `.npy` feature files,
- feature config.

### 24.4. Notebook 04 - Baseline

Tasks:

- train SVM/RF,
- tune basic hyperparameters,
- evaluate,
- plot confusion matrix.

Output:

- `baseline_mfcc_svm.pkl`,
- baseline metrics.

### 24.5. Notebook 05 - CNN SER

Tasks:

- build dataset class,
- train CNN,
- add augmentation,
- early stopping,
- evaluate,
- save model.

Output:

- `cnn_ser_model.pt`,
- CNN metrics,
- confusion matrix.

### 24.6. Notebook 06 - Advanced Model

Tasks:

- train CNN-GRU or CNN-BiLSTM,
- compare with CNN,
- measure inference time.

Output:

- advanced model metrics,
- decision whether to use in demo.

### 24.7. Notebook 07 - VAD + Timeline

Tasks:

- load audio tự thu,
- apply VAD,
- split sliding window,
- predict emotion per chunk,
- apply smoothing,
- plot timeline.

Output:

- raw timeline,
- smoothed timeline,
- VAD/silence plot.

### 24.8. Notebook 08 - Feedback Demo

Tasks:

- compute acoustic indicators,
- run transcript if available,
- compute pressure score,
- generate feedback.

Output:

- full demo report for one audio.

### 24.9. Notebook 09 - Gradio App

Tasks:

- combine trained model,
- upload/record audio,
- dashboard output,
- feedback text.

Output:

- `app.py` or `gradio_app.py`.

---

## 25. Source code structure

```text
speech-feedback-project/
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
│   ├── 04_train_baseline.ipynb
│   ├── 05_train_cnn_ser.ipynb
│   ├── 06_train_advanced_temporal.ipynb
│   ├── 07_vad_timeline_smoothing.ipynb
│   └── 08_feedback_dashboard_demo.ipynb
│
├── src/
│   ├── config.py
│   ├── audio_io.py
│   ├── preprocessing.py
│   ├── label_parsers.py
│   ├── feature_mfcc.py
│   ├── feature_mel.py
│   ├── acoustic_indicators.py
│   ├── vad.py
│   ├── smoothing.py
│   ├── ser_models.py
│   ├── inference.py
│   ├── transcript_analysis.py
│   ├── pressure_score.py
│   └── feedback_engine.py
│
├── models/
│   ├── baseline_mfcc_svm.pkl
│   ├── cnn_ser_model.pt
│   ├── cnn_gru_model.pt
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
└── reports/
    ├── final_report.docx
    ├── slides.pptx
    └── demo_script.md
```

---

## 26. Timeline triển khai chi tiết 4 tuần

### Tuần 1 - Chốt scope, paper, dataset, metadata

**Mục tiêu:** có dataset sạch và định hướng rõ.

Việc cần làm:

- Chốt tên đề tài public speaking/presentation feedback.
- Viết bảng vai trò paper.
- Tải CREMA-D và RAVDESS speech-only.
- Tạo parser label.
- Map 6 emotion.
- Tạo metadata.
- Vẽ distribution ban đầu.

Deliverables:

- `metadata_clean.csv`
- `paper_role_table.md`
- `samples_by_emotion.png`
- section dataset trong report.

### Tuần 2 - EDA, features, baseline

**Mục tiêu:** có baseline và hiểu dữ liệu.

Việc cần làm:

- EDA đầy đủ.
- Extract MFCC.
- Train SVM/RF.
- Extract Mel-spectrogram.
- Tính RMS/pitch/silence thử nghiệm.
- Viết phân tích confusion ban đầu.

Deliverables:

- baseline metrics,
- confusion matrix baseline,
- EDA figures,
- feature extraction notebook.

### Tuần 3 - CNN, augmentation, advanced optional

**Mục tiêu:** có model SER chính.

Việc cần làm:

- Train CNN.
- Add augmentation.
- Compare CNN vs baseline.
- Nếu kịp train CNN-GRU/CNN-BiLSTM.
- Save best model.
- Đo inference time.

Deliverables:

- `cnn_ser_model.pt`
- model comparison table,
- CNN confusion matrix,
- training curves.

### Tuần 4 - VAD, smoothing, timeline, feedback, dashboard

**Mục tiêu:** biến model thành hệ thống demo.

Việc cần làm:

- Tự thu demo audio.
- Implement VAD.
- Implement sliding window.
- Implement smoothing.
- Generate emotion timeline.
- Compute acoustic indicators.
- Add pressure score.
- Add Whisper nếu kịp.
- Build Gradio dashboard.
- Viết báo cáo/slide/script.

Deliverables:

- demo audio set,
- emotion timeline demo,
- pressure timeline demo,
- dashboard running,
- final report and slides.

---

## 27. Timeline mở rộng 6 tuần nếu có thời gian

### Tuần 5

- Tối ưu dashboard.
- So sánh VAD/no VAD.
- So sánh smoothing/no smoothing.
- Thêm transcript/filler words ổn định.
- Thêm audio tự thu nhiều case hơn.

### Tuần 6

- Hoàn thiện report học thuật.
- Viết limitation rõ.
- Chuẩn bị Q&A.
- Quay video demo backup.
- Export model và notebook sạch.

---

## 28. Task assignment chi tiết

### Nguyễn Minh Cường

Vai trò chính: EDA, acoustic indicators, timeline, dashboard.

Công việc:

- Vẽ EDA figures.
- Viết module RMS/pitch/silence/pause.
- Tạo emotion timeline visualization.
- Implement pressure score.
- Làm Gradio dashboard.
- Chuẩn bị demo audio và demo script.

### Nguyễn Tài Huy

Vai trò chính: dataset, preprocessing, baseline.

Công việc:

- Tải và tổ chức dataset.
- Parse label CREMA-D/RAVDESS/TESS.
- Tạo metadata.
- Resample/mono/pad/crop.
- Extract MFCC.
- Train SVM/RF.
- Kiểm tra split và data leakage.

### Bùi Quang Huy

Vai trò chính: literature, main model, methodology/report.

Công việc:

- Tổng hợp paper.
- Viết research gap.
- Train CNN/CNN-GRU.
- So sánh model.
- Viết methodology, experiment, limitation.
- Chuẩn bị phản biện về stress indicator, realtime và dataset.

---

## 29. Rủi ro và cách xử lý

| Rủi ro | Vì sao xảy ra | Cách xử lý |
|---|---|---|
| Dataset tải lâu | RAVDESS đầy đủ rất lớn | chỉ lấy speech-only, ưu tiên CREMA-D trước |
| Label không đồng nhất | calm/surprise không có ở mọi dataset | bản chính dùng 6 lớp, bỏ calm/surprise |
| Data leakage | cắt segment/augmentation trước split | split file gốc trước, augmentation chỉ train |
| Overfitting | dataset nhỏ/acted | dropout, augmentation, early stopping |
| Timeline nhảy lung tung | chunk ngắn, model uncertain | smoothing, threshold, majority vote |
| Predict emotion trên silence | không có VAD | dùng VAD và silence-aware weighting |
| Feedback bị chủ quan | không có ground truth feedback | dùng rule rõ ràng, giải thích từ feature |
| Stress bị hiểu nhầm | dùng từ stress detector | đổi thành speaking pressure/acoustic indicator |
| Whisper nhận sai filler | ASR không hoàn hảo | ghi experimental, ưu tiên tiếng Anh rõ |
| Demo realtime lỗi | Colab/Kaggle không ổn định microphone | dùng upload/record + simulated realtime |

---

## 30. Must-have, Should-have, Nice-to-have

### 30.1. Must-have

- Metadata clean cho CREMA-D + RAVDESS.
- Label mapping 6 emotion.
- EDA figures.
- MFCC + SVM baseline.
- CNN SER model.
- Macro-F1 và confusion matrix.
- Chunk-based inference.
- Emotion timeline.
- RMS, pitch, silence ratio, pause count.
- Rule-based feedback.
- Gradio demo upload audio.

### 30.2. Should-have

- VAD.
- Smoothing.
- Pressure score.
- Audio tự thu nhiều case.
- CNN-GRU/CNN-BiLSTM comparison.
- Augmentation noise + shift.
- Raw timeline vs smoothed timeline comparison.

### 30.3. Nice-to-have

- Whisper transcript.
- Speech rate.
- Filler words.
- TESS/EmoDB benchmark.
- Wav2Vec2/HuBERT embedding.
- Vietnamese demo sample.

### 30.4. Future work

- Emotion diarization dataset có boundary.
- Full streaming microphone.
- Multimodal video + audio feedback.
- Sentence stress/prosody.
- IELTS rubric.
- Vietnamese public speaking dataset tự thu có annotation.

---

## 31. Câu hỏi phản biện và cách trả lời

### Câu 1: Dataset chỉ có 1 label/audio, sao làm emotion timeline?

Trả lời:

> Nhóm không khẳng định có ground truth frame-level emotion. Nhóm dùng chunk-based near real-time analysis: mỗi cửa sổ 3 giây được xem như một utterance ngắn và mô hình dự đoán emotion chủ đạo của cửa sổ đó. Sau đó nhóm dùng smoothing để giảm dao động và tạo emotion timeline phục vụ feedback.

### Câu 2: Fear/sad có phải stress không?

Trả lời:

> Không. Nhóm không suy stress trực tiếp từ emotion. Speaking pressure được tính từ nhiều acoustic indicators như pause, silence ratio, pitch variation, energy instability, speech rate và filler words. Emotion chỉ là tín hiệu phụ.

### Câu 3: Vì sao cần VAD?

Trả lời:

> VAD giúp phân biệt speech và silence. Nếu không có VAD, model có thể dự đoán emotion trên đoạn im lặng. VAD cũng giúp tính pause count và silence ratio, rất quan trọng cho feedback luyện thuyết trình.

### Câu 4: Feedback có được train từ dataset không?

Trả lời:

> Không. Vì không có dataset audio kèm expert feedback đủ chuẩn, nhóm dùng rule-based feedback engine. Các rule dựa trên acoustic indicators và transcript metrics, ví dụ pause cao thì gợi ý luyện nói liền mạch, speech rate cao thì giảm tốc độ.

### Câu 5: Model nào dùng cho demo?

Trả lời:

> Model chính là CNN trên Mel-spectrogram vì cân bằng giữa accuracy và tốc độ inference. CNN-GRU/CNN-BiLSTM là advanced experiment, chỉ dùng cho demo nếu tốt hơn rõ và inference vẫn nhanh.

---

## 32. Cấu trúc báo cáo cuối kỳ đề xuất

1. Introduction
2. Motivation and Problem Statement
3. Literature Review
4. Dataset Description
5. Proposed System Architecture
6. Audio Preprocessing
7. Feature Extraction
8. SER Models
9. VAD-guided Near Real-time Emotion Timeline
10. Acoustic Speaking Indicators
11. Speaking Pressure Indicator
12. Transcript Analysis
13. Feedback Generation
14. Dashboard Demo
15. Experiments and Results
16. Ablation Study
17. Discussion
18. Limitations
19. Future Work
20. Conclusion

---

## 33. Slide thuyết trình đề xuất

1. Title and team
2. Problem: why presentation feedback needs speech analysis
3. Research gap: SER usually predicts one label
4. Proposed system overview
5. Dataset strategy
6. Literature map
7. Preprocessing and feature extraction
8. Baseline vs CNN model
9. VAD + sliding window + smoothing
10. Emotion timeline demo
11. Acoustic indicators
12. Speaking pressure score
13. Feedback engine
14. Dashboard demo
15. Results and evaluation
16. Limitations and future work
17. Conclusion
18. Q&A

---

## 34. Kết luận roadmap

Bản roadmap này chốt đề tài theo hướng **Public Speaking / Presentation Feedback**. Phần lõi vẫn là Speech Emotion Recognition, nhưng được mở rộng thành hệ thống ứng dụng thông qua VAD, sliding window, smoothing, emotion timeline, acoustic indicators, transcript analysis và rule-based feedback.

Đóng góp chính của nhóm không phải là phát minh một model deep learning hoàn toàn mới, mà là xây dựng một pipeline có tính ứng dụng:

```text
SER model
+ VAD-guided near real-time analysis
+ smoothed emotion timeline
+ acoustic speaking indicators
+ transcript metrics
+ feedback engine
= presentation feedback system
```

Đây là hướng phù hợp với Colab/Kaggle, có đủ cơ sở học thuật, có demo rõ ràng, và tránh được rủi ro khi gọi nhầm thành stress detection y khoa.

---

## 35. Checklist triển khai cuối cùng

### Dataset

- [ ] Tải CREMA-D.
- [ ] Tải RAVDESS speech-only.
- [ ] Tùy chọn TESS/EmoDB.
- [ ] Tạo metadata.
- [ ] Map 6 labels.
- [ ] Split train/val/test.

### Model

- [ ] Extract MFCC.
- [ ] Train SVM baseline.
- [ ] Extract Mel-spectrogram.
- [ ] Train CNN.
- [ ] Add augmentation.
- [ ] Optional CNN-GRU/CNN-BiLSTM.
- [ ] Save best model.

### Timeline

- [ ] Implement sliding window.
- [ ] Implement VAD.
- [ ] Predict per chunk.
- [ ] Smooth probabilities.
- [ ] Plot emotion timeline.

### Feedback

- [ ] Compute RMS.
- [ ] Compute pitch.
- [ ] Compute silence ratio.
- [ ] Compute pause count.
- [ ] Compute pressure score.
- [ ] Add transcript if possible.
- [ ] Generate rule-based feedback.

### Demo

- [ ] Thu audio demo.
- [ ] Build Gradio app.
- [ ] Test upload audio.
- [ ] Test record audio.
- [ ] Prepare backup screenshots.
- [ ] Prepare demo script.

### Report

- [ ] Literature review.
- [ ] Dataset section.
- [ ] Methodology.
- [ ] Results.
- [ ] Ablation.
- [ ] Limitations.
- [ ] Future work.

