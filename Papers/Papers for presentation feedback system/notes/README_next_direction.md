# Presentation Feedback System - Next Direction

Folder này lưu paper/code để mở rộng từ bài toán Speech Emotion Recognition (SER) sang Presentation/Public Speaking Feedback.

## 1. Hướng đi tiếp theo nên chọn

Không nên biến emotion thành feedback trực tiếp. Hướng hợp lý là:

```text
SER model hiện tại
-> tạo emotion / valence / arousal / expressiveness cues ở mức segment
-> aggregate theo toàn bài nói
-> kết hợp pause, silence, speech ratio, word rate, prosody, transcript
-> dự đoán hoặc mô tả presentation quality
```

Nói cách khác, emotion là một nhóm feature trung gian, không phải kết luận cuối.

## 2. Paper/code chính đã tải

### PuSQ

- Paper: `papers/PuSQ_Automatic_Assessment_of_Speaking_Skills_Aural_Textual_ICNLSP2021.pdf`
- Code/data repo: `code/PuSQ`
- Framework đi kèm: `code/readys`

Vai trò:

- Paper nền quan trọng nhất để nối SER sang Public Speaking Quality Assessment.
- Chứng minh emotion posterior có thể dùng như feature trung gian cho expressiveness/enjoyment.
- Repo public không có raw WAV, chỉ có feature đã trích xuất, ASR transcript, annotation, reference texts.

### readys

- Repo: `code/readys`

Vai trò:

- Framework speech analytics cho speech quality assessment.
- Có pipeline audio/text/recording-level feature.
- Có code train/test classifier cho recording-level quality.

### SOPHIAS

- Paper: `papers/SOPHIAS_Multimodal_Dataset_Student_Oral_Presentations.pdf`
- Repo: `code/SOPHIAS`

Vai trò:

- Dataset/hướng nghiên cứu rất gần với presentation feedback thật.
- Có multimodal schema: audio, video, gaze, smartwatch, slides, rubric.
- Phù hợp cho future work nếu project mở rộng sang presentation feedback đa tín hiệu.

### SpeechMirror

- Paper: `papers/SpeechMirror_Multimodal_Visual_Analytics_Public_Speaking.pdf`

Vai trò:

- Hệ thống visual analytics cho online public speaking.
- Học cách thiết kế dashboard/timeline/feedback thay vì chỉ báo một emotion label.

### TED Talk Ratings

- Paper: `papers/Predicting_TED_Talk_Ratings_from_Language_and_Prosody.pdf`

Vai trò:

- Chứng minh prosody và language features có liên quan tới perceived speaking quality.
- Hữu ích để justify thêm pitch, energy, speaking rate, transcript features.

### Automated Presentation Coaching Survey

- Paper: `papers/Survey_Automated_Presentation_Coaching.pdf`

Vai trò:

- Dùng làm related work tổng quan cho presentation coaching.
- Giúp phân nhóm các hướng: prosody, pronunciation, pacing, content, visual, realtime feedback.

### E-ffective

- Paper: `papers/E_ffective_Analysis_of_Online_Speaking_Effectiveness.pdf`

Vai trò:

- Liên quan tới phân tích speaking effectiveness.
- Hữu ích để mô tả vì sao presentation feedback cần nhiều tín hiệu hơn emotion.

## 3. Việc nên làm ngay sau folder này

### Step 1 - Reproduce PuSQ baseline ở mức feature

Mục tiêu:

- Unzip `code/PuSQ/features_data.zip`.
- Đọc annotation.
- Chạy parser theo README của PuSQ.
- Train recording-level classifier bằng `readys`.

Output mong muốn:

- Bảng Expressiveness/Enjoyment baseline.
- Hiểu chính xác MetaAudio, LowLevelAudio, Text feature.
- Có một notebook riêng cho presentation-quality feature-level baseline.

### Step 2 - Thiết kế bridge từ 06D sang presentation feedback

Mô hình 06D hiện tại cho:

```text
emotion posterior theo segment
emotion2vec embedding
temporal acoustic embedding
spectrogram embedding
stats embedding
```

Cần bổ sung:

```text
pause duration
silence ratio
speech ratio
word rate
pitch/F0 mean, std, range
energy/loudness trend
ASR transcript
filler-word count
unique-word rate
```

### Step 3 - Demo semi-realtime

Pipeline nên là:

```text
Audio 10-15 phút
-> segment 3-4s, hop 1-2s
-> emotion/prosody per segment
-> timeline
-> post-session ASR/text analysis
-> feedback dashboard/report
```

Không nên chạy feedback cuối chỉ từ emotion. Emotion chỉ nên là một nhánh trong hệ thống.

## 4. Định vị đề tài

Tên hướng phù hợp:

```text
Speech Emotion Recognition Toward Presentation Feedback Analysis
```

hoặc:

```text
Speech Emotion and Prosody-Based Presentation Feedback System
```

Nếu bảo vệ giữa kỳ/cuối kỳ, nên nói:

> Current stage focuses on speech emotion recognition as an affective speech analysis module. The next stage extends it into presentation feedback by combining emotion cues with prosody, fluency, pause, speaking-rate and transcript-based features.

