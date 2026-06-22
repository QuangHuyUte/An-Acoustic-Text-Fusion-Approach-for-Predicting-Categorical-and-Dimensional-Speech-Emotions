---
title: "Project Plan - Hệ thống phản hồi luyện nói dựa trên cảm xúc và chỉ báo âm học"
subtitle: "Speech Emotion Recognition, Emotion Timeline, Acoustic Indicators, Transcript Analysis and Near Real-Time Speaking Feedback"
author: "Nhóm: Nguyễn Minh Cường, Nguyễn Tài Huy, Bùi Quang Huy"
date: "21/06/2026"
lang: "vi"
---

# Hệ thống phản hồi luyện nói dựa trên nhận diện cảm xúc và chỉ báo âm học từ giọng nói

**Tên tiếng Anh:** Audio-Based Speaking Feedback System using Speech Emotion Recognition, Acoustic Indicators and Speaking Feedback  
**Lĩnh vực:** Speech Processing, Speech Emotion Recognition, Acoustic Feature Analysis, Machine Learning  
**Nền tảng triển khai:** Google Colab/Kaggle Notebook, Gradio hoặc Streamlit demo  
**Đầu ra chính:** Notebook huấn luyện/đánh giá, mô hình SER, module emotion timeline, module acoustic indicators, transcript analysis, dashboard demo và báo cáo cuối kỳ.

## 1. Tóm tắt định hướng đề tài

Đề tài xây dựng một hệ thống phân tích giọng nói trong bối cảnh luyện nói, thuyết trình, phỏng vấn hoặc luyện phản xạ ngoại ngữ. Khác với các bài toán Speech Emotion Recognition cơ bản chỉ dự đoán một nhãn cảm xúc cho toàn bộ audio, hệ thống này hướng tới việc phân tích audio theo từng đoạn ngắn, tạo emotion timeline, tính các chỉ báo âm học, phân tích transcript và sinh phản hồi giúp người học cải thiện cách nói.

Điểm quan trọng của đề tài là **không chẩn đoán stress y khoa**. Hệ thống chỉ tạo ra các **stress-related acoustic indicators** hoặc **speaking pressure indicators**, tức là các chỉ báo liên quan đến độ ổn định giọng nói, năng lượng, cao độ, khoảng lặng và tốc độ nói. Các chỉ báo này dùng để hỗ trợ luyện nói, không dùng để kết luận người dùng có vấn đề tâm lý hay bệnh lý.

Định hướng phù hợp nhất cho sinh viên không có GPU mạnh là kết hợp giữa mô hình SER vừa đủ nhẹ và các module feature-based. Phần emotion model có thể train trên CREMA-D, RAVDESS và có thể bổ sung TESS. Phần acoustic indicators, stress-related indicators, speech rate, filler words và feedback có thể triển khai bằng feature extraction, transcript analysis và rule-based scoring, không cần train mô hình lớn.

## 2. Mục tiêu dự án

### 2.1. Mục tiêu tổng quát

Xây dựng một hệ thống Speech Processing có khả năng nhận audio luyện nói, phân tích cảm xúc theo thời gian, đo các đặc trưng âm học quan trọng và tạo phản hồi trực quan giúp người học cải thiện giọng nói, tốc độ nói, độ liền mạch và sự tự tin khi trình bày.

### 2.2. Mục tiêu cụ thể

Hệ thống cần đạt các mục tiêu sau:

- Xây dựng pipeline xử lý audio gồm đọc file, resample về 16 kHz, chuyển mono, trim silence, chuẩn hóa độ dài và chia audio thành segment.
- Tạo metadata thống nhất cho nhiều dataset emotion khác nhau, gồm đường dẫn file, dataset, speaker, emotion, duration và split.
- Huấn luyện mô hình Speech Emotion Recognition với 6 lớp cảm xúc chung: neutral, happy, sad, angry, fear, disgust.
- So sánh baseline truyền thống MFCC + SVM/Random Forest với mô hình deep learning Mel-spectrogram + CNN.
- Phát triển emotion timeline bằng cách dự đoán cảm xúc theo từng segment 3 giây.
- Tính acoustic indicators gồm RMS energy, pitch/F0, pitch variation, silence ratio, pause count và speaking continuity.
- Bổ sung transcript bằng Whisper để phân tích speech rate và filler words nếu còn thời gian.
- Tạo feedback text dựa trên kết quả emotion timeline, acoustic indicators và transcript.
- Xây dựng dashboard demo cho upload/record audio, hiển thị waveform, spectrogram, emotion distribution, emotion timeline, acoustic indicators và gợi ý cải thiện.

## 3. Phạm vi và nguyên tắc triển khai

### 3.1. Phạm vi chính

Project tập trung vào 4 nhánh chính:

1. **Speech Emotion Recognition:** train model nhận diện cảm xúc từ audio ngắn.
2. **Emotion Timeline:** phân tích cảm xúc theo từng đoạn thời gian của bài nói dài.
3. **Acoustic Speaking Indicators:** đo năng lượng, cao độ, khoảng lặng và độ liền mạch.
4. **Speaking Feedback:** tổng hợp kết quả để đưa ra nhận xét cải thiện giọng nói.

### 3.2. Những gì hệ thống làm

Hệ thống có thể:

- Nhận file audio hoặc audio record từ người dùng.
- Chuẩn hóa audio và chia thành các đoạn 2-3 giây.
- Dự đoán cảm xúc cho từng đoạn.
- Vẽ emotion distribution và emotion timeline.
- Tính các chỉ báo như năng lượng giọng nói, độ ổn định pitch, tỷ lệ im lặng, số lần pause và speaking continuity.
- Nếu dùng transcript, tính speech rate và phát hiện filler words.
- Sinh feedback dạng văn bản để người học biết đoạn nào nói chưa ổn và nên cải thiện gì.

### 3.3. Những gì hệ thống không làm

Hệ thống không nên được mô tả là:

- Mô hình chẩn đoán stress.
- Mô hình phát hiện rối loạn tâm lý.
- Mô hình đánh giá sức khỏe tinh thần.
- Mô hình thay thế chuyên gia y tế hoặc tâm lý.

Cách gọi đúng là **stress-related acoustic indicator**, **speaking pressure indicator** hoặc **vocal stability indicator**.

## 4. Câu hỏi nghiên cứu

Project có thể xoay quanh các câu hỏi nghiên cứu sau:

1. Các đặc trưng âm thanh như MFCC, Mel-spectrogram, pitch/F0, RMS energy và silence ratio có hỗ trợ nhận diện cảm xúc giọng nói hiệu quả không?
2. Baseline truyền thống MFCC + SVM/Random Forest khác gì so với mô hình Mel-spectrogram + CNN về accuracy, macro-F1 và confusion matrix?
3. Việc chia audio dài thành các segment ngắn có giúp biểu diễn cảm xúc theo thời gian rõ hơn so với dự đoán một nhãn duy nhất cho toàn bộ audio không?
4. Các acoustic indicators như pitch variation, energy stability, pause ratio và speaking continuity có thể được dùng để sinh speaking feedback như thế nào?
5. Transcript, speech rate và filler words có giúp feedback luyện nói cụ thể và dễ hiểu hơn không?
6. Có thể xây dựng một dashboard demo gần real-time bằng chunk-based analysis trên Colab/local hay không?

## 5. Chiến lược dataset

Không nên tìm một dataset duy nhất cho toàn bộ project. Mỗi module cần một loại dữ liệu khác nhau. Phần emotion cần dataset có nhãn cảm xúc. Phần acoustic/stress indicator có thể tính trực tiếp từ audio người dùng. Phần transcript dùng pretrained ASR như Whisper.

### 5.1. Dataset chính cho Emotion Branch

Dataset chính nên dùng là **CREMA-D + RAVDESS speech-only**. TESS có thể dùng như dữ liệu bổ sung hoặc external test.

| Dataset | Vai trò | Lý do chọn |
|---|---|---|
| CREMA-D | Dataset lõi cho SER | Có 6 emotion phù hợp: anger, disgust, fear, happy, neutral, sad; nhiều speaker; dễ map label. |
| RAVDESS speech-only | Dataset chuẩn học thuật | Dataset SER phổ biến, audio sạch, có nhiều paper sử dụng; nên chỉ lấy speech audio để giảm dung lượng. |
| TESS | Bổ sung hoặc kiểm thử mở rộng | Audio sạch, dễ xử lý, nhưng chỉ có 2 speaker nữ nên không nên làm dataset chính duy nhất. |

### 5.2. Label mapping đề xuất

Nên dùng 6 lớp emotion chung để tránh lệch nhãn giữa dataset:

| Unified label | CREMA-D | RAVDESS | TESS |
|---|---|---|---|
| neutral | neutral | neutral | neutral |
| happy | happy | happy | happy |
| sad | sad | sad | sad |
| angry | anger | angry | angry |
| fear | fear | fearful | fear |
| disgust | disgust | disgust | disgust |

Không nên đưa calm và surprise vào phiên bản chính vì các nhãn này không đồng đều giữa các dataset. Nếu làm tốt phiên bản 6 lớp, nhóm có thể mở rộng lên 8 lớp ở phần future work.

### 5.3. Dataset/nguồn tham khảo cho stress-related indicators

Phần stress indicator **không train trực tiếp bằng RAVDESS/CREMA-D/TESS** vì các dataset này không có nhãn stress. Thay vào đó, project dùng các paper/dataset stress để làm cơ sở học thuật, còn triển khai thực tế bằng feature extraction.

| Nguồn | Vai trò trong project | Cách dùng hợp lý |
|---|---|---|
| MAS Dataset | Tham khảo stress-related multimodal indicators | Không cần train trực tiếp; học cách tách vocal/prosodic indicators. |
| Cyberball/MIST acoustic stress study | Cơ sở học thuật cho F0, speech rate, jitter, voiced segments | Dùng để giải thích vì sao acoustic/prosodic features có thể liên quan đến stress. |
| DAIC-WOZ | Future work cho psychological distress | Không dùng trong bản chính vì chuyển sang mental health/depression detection. |

### 5.4. Dataset/nguồn cho transcript, speech rate và filler words

Không cần train ASR. Dùng pretrained Whisper tiny hoặc base để lấy transcript từ audio người dùng. Sau đó tính:

- Số từ trong transcript.
- Speech rate theo words per minute.
- Filler words như um, uh, er, ah, like, you know, actually, basically, I mean.
- Với tiếng Việt có thể thử các từ như ờ, ừm, à, thì, là, kiểu như, nói chung là, thật ra là.

## 6. Kiến trúc hệ thống đề xuất

Pipeline tổng thể:

```text
Input audio / Record audio
→ Preprocessing: 16 kHz, mono, trim silence, normalize volume
→ Segmentation: chia audio thành các đoạn 2-3 giây
→ Emotion Branch: MFCC/Mel → SER model → emotion per segment
→ Acoustic Branch: RMS, pitch, silence, pause → acoustic indicators
→ Transcript Branch: Whisper → text → speech rate, filler words
→ Fusion / Feedback Rules
→ Dashboard: waveform, spectrogram, timeline, indicators, feedback
```

Có thể triển khai hệ thống theo 2 chế độ:

- **Offline analysis:** upload audio dài rồi phân tích toàn bộ.
- **Near real-time analysis:** chia audio thành chunk ngắn và xử lý từng chunk, mô phỏng real-time.

## 7. Module 1 - Data Preparation và Preprocessing

### 7.1. Mục tiêu

Chuẩn hóa dữ liệu từ nhiều dataset khác nhau để có thể train chung một mô hình SER 6 lớp.

### 7.2. Các bước thực hiện

1. Tải dataset CREMA-D, RAVDESS speech-only và tùy chọn TESS.
2. Parse label từ tên file hoặc cấu trúc folder.
3. Map label về 6 lớp chung.
4. Tạo metadata thống nhất.
5. Resample audio về 16 kHz.
6. Convert audio về mono.
7. Trim silence đầu/cuối nếu cần.
8. Chuẩn hóa duration về 3 giây bằng cut hoặc pad.
9. Chia train/validation/test.
10. Lưu feature ra file để tránh extract lại nhiều lần.

### 7.3. Metadata schema đề xuất

| Cột | Ý nghĩa |
|---|---|
| path | Đường dẫn audio |
| dataset | CREMA-D, RAVDESS hoặc TESS |
| speaker_id | ID người nói nếu parse được |
| emotion_raw | Nhãn gốc trong dataset |
| emotion | Nhãn đã map về 6 lớp |
| duration | Độ dài audio |
| sample_rate | Sample rate sau chuẩn hóa |
| split | train, validation hoặc test |

### 7.4. Lưu ý quan trọng

Nếu có thể, nên split theo speaker để tránh cùng một speaker xuất hiện ở cả train và test. Nếu không làm được speaker-independent split, nhóm cần ghi rõ limitation trong báo cáo.

## 8. Module 2 - EDA và Visualization

EDA giúp chứng minh nhóm hiểu dữ liệu, không chỉ đưa dữ liệu vào model. Đây là phần rất nên đầu tư vì giúp báo cáo có tính học thuật.

### 8.1. Biểu đồ nên có

| Biểu đồ | Mục đích |
|---|---|
| Samples by emotion | Kiểm tra mất cân bằng nhãn |
| Samples by dataset | Xem dataset nào đóng góp nhiều dữ liệu |
| Duration distribution | Kiểm tra độ dài audio |
| Waveform examples | Minh họa tín hiệu âm thanh theo thời gian |
| Mel-spectrogram examples | Minh họa phân bố năng lượng theo thời gian và tần số |
| MFCC heatmap | Minh họa đặc trưng dùng cho model |
| RMS energy by emotion | Quan sát khác biệt năng lượng giữa emotion |
| Pitch/F0 by emotion | Quan sát khác biệt cao độ giữa emotion |

### 8.2. Ý nghĩa học thuật

Phần EDA có thể giúp nhóm giải thích tại sao một số emotion dễ bị nhầm. Ví dụ, fear và sad có thể có vùng acoustic pattern gần nhau trong một số audio; angry và disgust cũng có thể bị nhầm do đều có năng lượng hoặc biểu hiện mạnh. Phân tích này cần kết hợp với confusion matrix sau khi train model.

## 9. Module 3 - Feature Extraction

### 9.1. Feature cho mô hình emotion

| Feature | Dùng cho | Ghi chú |
|---|---|---|
| MFCC | Baseline ML, CNN hoặc temporal model | Dễ giải thích, nhẹ, phổ biến trong SER. |
| Delta MFCC | Bổ sung thay đổi theo thời gian | Optional nếu muốn tăng temporal information. |
| Mel-spectrogram | CNN | Biểu diễn audio như ảnh thời gian-tần số. |
| Chroma | Optional | Có thể thử nhưng không bắt buộc. |
| Spectral centroid | Optional | Feature bổ sung về phổ âm. |
| Zero Crossing Rate | Optional | Feature đơn giản về mức dao động tín hiệu. |

### 9.2. Feature cho acoustic indicators

| Feature | Cách hiểu | Dùng cho feedback |
|---|---|---|
| RMS energy | Mức năng lượng/âm lượng của giọng | Giọng quá nhỏ, quá lớn hoặc không ổn định. |
| Pitch/F0 mean | Cao độ trung bình | Giọng cao/thấp bất thường. |
| Pitch/F0 std | Độ dao động cao độ | Giọng quá đều hoặc dao động mạnh. |
| Silence ratio | Tỷ lệ im lặng | Nói ngập ngừng hoặc nhiều khoảng dừng. |
| Pause count | Số lần ngắt quãng | Độ đứt đoạn khi nói. |
| Speaking continuity | Tỷ lệ đoạn có tiếng nói liên tục | Độ liền mạch của bài nói. |
| Speech rate | Số từ/phút | Nói quá nhanh hoặc quá chậm. |
| Filler count | Số filler words | Mức độ ngập ngừng trong diễn đạt. |

## 10. Module 4 - Speech Emotion Recognition Model

### 10.1. Baseline model

Baseline nên dùng:

```text
MFCC mean/std → SVM hoặc Random Forest → emotion label
```

Mục tiêu của baseline là tạo mốc so sánh đơn giản, nhẹ và dễ giải thích. Baseline phù hợp với Colab/Kaggle và có thể chạy nhanh để kiểm tra pipeline.

### 10.2. Main model

Main model nên dùng:

```text
Mel-spectrogram → CNN → Softmax 6 emotion classes
```

Lý do chọn CNN:

- Phù hợp với biểu diễn Mel-spectrogram như ảnh âm thanh.
- Không quá nặng so với Wav2Vec2/HuBERT fine-tuning.
- Dễ chạy trên GPU miễn phí của Kaggle/Colab.
- Dễ tích hợp vào dashboard demo.

### 10.3. Advanced model nếu còn thời gian

Có thể thử:

```text
MFCC sequence / Mel-spectrogram → CNN-BiGRU hoặc CNN-LSTM
```

Mục tiêu là học sự thay đổi cảm xúc theo thời gian. Tuy nhiên, phần này nên để là advanced experiment, không đặt làm bắt buộc. Nếu không kịp, nhóm vẫn có thể dùng paper TBDM-Net và FLEA/HuBERT làm cơ sở học thuật cho hướng temporal modeling trong future work.

### 10.4. Output của model

Mỗi audio hoặc segment cần trả về:

| Output | Ý nghĩa |
|---|---|
| predicted_emotion | Nhãn cảm xúc dự đoán |
| confidence | Xác suất cao nhất |
| probability_vector | Phân phối xác suất 6 lớp |
| segment_start | Thời điểm bắt đầu segment |
| segment_end | Thời điểm kết thúc segment |

## 11. Module 5 - Emotion Timeline

### 11.1. Mục tiêu

Emotion timeline giúp hệ thống không chỉ nói audio là happy/sad/angry, mà còn cho biết cảm xúc thay đổi ở đoạn nào trong bài nói.

### 11.2. Cách triển khai

1. Nhận audio dài 20-60 giây hoặc dài hơn.
2. Chia audio thành các segment 3 giây.
3. Có thể dùng overlap 1 giây nếu muốn timeline mượt hơn.
4. Với mỗi segment, extract feature giống lúc train.
5. Đưa segment vào SER model.
6. Lưu predicted emotion và confidence.
7. Vẽ timeline theo thời gian.

### 11.3. Output mẫu

| Time range | Emotion | Confidence | Ghi chú |
|---|---|---|---|
| 0-3s | neutral | 0.72 | Mở đầu ổn định |
| 3-6s | happy | 0.64 | Giọng có năng lượng hơn |
| 6-9s | fear | 0.52 | Có thể thiếu tự tin hoặc model chưa chắc |
| 9-12s | neutral | 0.67 | Quay lại ổn định |

### 11.4. Ý nghĩa trong báo cáo

Emotion timeline là điểm cộng lớn vì nó biến bài toán từ classification đơn giản thành phân tích bài nói theo thời gian. Nhóm có thể giải thích rằng trong thực tế, một bài nói dài không nên bị ép thành một nhãn emotion duy nhất.

## 12. Module 6 - Acoustic Indicators

### 12.1. Mục tiêu

Acoustic indicators đo các thuộc tính vật lý và ngôn điệu của giọng nói. Module này không cần nhãn emotion/stress để train, mà tính trực tiếp từ audio.

### 12.2. Các chỉ báo chính

| Indicator | Công thức/cách tính đơn giản | Ý nghĩa |
|---|---|---|
| RMS energy mean | Trung bình RMS theo frame | Mức năng lượng/âm lượng trung bình. |
| RMS energy std | Độ lệch chuẩn RMS | Độ ổn định âm lượng. |
| Pitch mean | Trung bình F0 voiced frames | Cao độ trung bình. |
| Pitch std | Độ lệch chuẩn F0 | Độ biến thiên cao độ. |
| Silence ratio | Thời lượng im lặng / tổng thời lượng | Tỷ lệ ngắt quãng. |
| Pause count | Số đoạn im lặng dài hơn ngưỡng | Số lần dừng. |
| Speaking continuity | 1 - silence ratio | Độ liền mạch. |

### 12.3. Ngưỡng rule-based ban đầu

Các ngưỡng có thể được điều chỉnh sau khi kiểm thử audio mẫu:

| Điều kiện | Nhận xét gợi ý |
|---|---|
| silence ratio > 0.30 | Có nhiều khoảng lặng, nên luyện nói liền mạch hơn. |
| pause count cao | Bài nói bị ngắt đoạn nhiều, nên luyện nối ý trước khi trình bày. |
| RMS energy giảm dần | Âm lượng yếu dần về cuối, cần giữ năng lượng ổn định. |
| pitch std quá thấp | Giọng khá đều, nên thêm ngữ điệu để tự nhiên hơn. |
| pitch std quá cao | Giọng dao động mạnh, có thể cần kiểm soát hơi thở và tốc độ nói. |

Các ngưỡng này không phải chuẩn y khoa. Chúng chỉ là rule ban đầu phục vụ speaking feedback.

## 13. Module 7 - Stress-related Speaking Pressure Indicator

### 13.1. Định nghĩa đúng

Module này không phải stress detector. Nó là module tạo chỉ báo áp lực giọng nói dựa trên các acoustic features.

Tên nên dùng trong báo cáo:

- Acoustic Speaking Pressure Indicator.
- Stress-related Acoustic Indicator.
- Vocal Stability Indicator.
- Confidence/Pressure Indicator for Speaking Feedback.

### 13.2. Cách tính đề xuất

Có thể tạo điểm indicator từ các thành phần:

```text
pressure_score = w1 * normalized_pitch_variation
               + w2 * normalized_energy_instability
               + w3 * normalized_silence_ratio
               + w4 * normalized_pause_count
               + w5 * abnormal_speech_rate_score
```

Trong đó:

- Pitch variation cao có thể gợi ý giọng thiếu ổn định.
- Energy instability cao có thể gợi ý âm lượng không đều.
- Silence ratio cao có thể gợi ý ngập ngừng.
- Pause count cao có thể gợi ý thiếu liền mạch.
- Speech rate quá nhanh hoặc quá chậm có thể gợi ý cần điều chỉnh tốc độ.

### 13.3. Cách hiển thị kết quả

Nên chia thành 3 mức:

| Mức | Ý nghĩa | Feedback |
|---|---|---|
| Low pressure | Giọng tương đối ổn định | Tiếp tục duy trì tốc độ và năng lượng nói. |
| Medium pressure | Có một số dấu hiệu chưa ổn định | Nên giảm pause dài và giữ âm lượng đều hơn. |
| High pressure | Nhiều chỉ báo ngập ngừng/thiếu ổn định | Nên luyện kiểm soát hơi thở, giảm tốc độ và chuẩn bị ý trước khi nói. |

### 13.4. Cách viết trong báo cáo để tránh phản biện

Nên viết:

> The system does not diagnose psychological stress. Instead, it computes stress-related acoustic indicators such as pitch variation, energy instability, silence ratio and pause count to support speaking feedback.

Không nên viết:

> The system detects whether the speaker is stressed.

## 14. Module 8 - Transcript, Speech Rate và Filler Words

### 14.1. Mục tiêu

Transcript giúp hệ thống phân tích không chỉ giọng nói mà còn cách diễn đạt. Với transcript, hệ thống có thể tính speech rate, filler words và đưa ra góp ý cụ thể hơn.

### 14.2. Công cụ đề xuất

Dùng Whisper tiny hoặc base để giảm tải tài nguyên. Không cần train ASR.

### 14.3. Speech rate

Công thức:

```text
speech_rate = number_of_words / audio_duration_minutes
```

Gợi ý ngưỡng ban đầu:

| Speech rate | Nhận xét |
|---|---|
| < 90 WPM | Nói hơi chậm, có thể thiếu tự nhiên hoặc bị ngắt quãng. |
| 90-160 WPM | Tốc độ tương đối ổn. |
| > 160 WPM | Nói hơi nhanh, người nghe có thể khó theo dõi. |

### 14.4. Filler words

Filler words tiếng Anh:

```text
um, uh, er, ah, like, you know, actually, basically, I mean, so
```

Filler words tiếng Việt có thể thử:

```text
ờ, ừm, à, thì, là, kiểu như, nói chung là, thật ra là
```

Lưu ý: Whisper có thể không luôn nhận chính xác filler words, đặc biệt với tiếng Việt hoặc audio nhiễu. Vì vậy phần này nên ghi là experimental feature.

## 15. Module 9 - Feedback Generation

### 15.1. Mục tiêu

Feedback generation chuyển các con số kỹ thuật thành nhận xét dễ hiểu cho người dùng.

### 15.2. Cách triển khai

Không cần train LLM. Dùng rule-based template là đủ cho đồ án.

Pipeline:

```text
emotion timeline + acoustic indicators + transcript metrics
        ↓
rule-based feedback engine
        ↓
summary feedback + segment-level feedback + improvement suggestions
```

### 15.3. Ví dụ rule feedback

| Điều kiện | Feedback |
|---|---|
| Neutral quá nhiều + pitch std thấp | Giọng khá đều, nên thêm ngữ điệu để bài nói tự nhiên hơn. |
| Fear/sad xuất hiện nhiều ở giữa bài | Đoạn giữa bài có thể thể hiện sự thiếu tự tin hoặc năng lượng thấp. |
| Energy giảm dần | Âm lượng yếu dần về cuối, nên giữ hơi và âm lượng ổn định. |
| Silence ratio cao | Có nhiều khoảng lặng, nên luyện nói theo dàn ý để giảm ngập ngừng. |
| Speech rate > 160 WPM | Bạn đang nói hơi nhanh, nên giảm tốc độ ở các ý quan trọng. |
| Speech rate < 90 WPM | Bạn đang nói hơi chậm, nên luyện nối câu để bài nói tự nhiên hơn. |
| Filler words nhiều | Bạn dùng nhiều từ đệm, nên thay bằng pause ngắn có kiểm soát. |

### 15.4. Output feedback đề xuất

Dashboard nên có 3 phần feedback:

1. **Overall feedback:** nhận xét tổng quan toàn bài nói.
2. **Timeline feedback:** đoạn nào có cảm xúc/feature đáng chú ý.
3. **Improvement suggestions:** gợi ý luyện tập cụ thể.

Ví dụ:

> Bài nói của bạn chủ yếu ở trạng thái neutral, tuy nhiên có một số đoạn từ 12-18s có energy giảm và pause dài. Điều này có thể làm bài nói thiếu liền mạch. Bạn nên luyện giữ âm lượng ổn định, chuẩn bị ý chính trước khi nói và giảm filler words như "um" hoặc "uh".

## 16. Module 10 - Near Real-time Analysis

### 16.1. Định nghĩa

Không nên gọi là real-time tuyệt đối nếu chưa xử lý microphone streaming ổn định. Nên gọi là **near real-time chunk-based analysis** hoặc **simulated real-time analysis**.

### 16.2. Cách triển khai khả thi

Có 2 hướng:

| Hướng | Mô tả | Khả thi |
|---|---|---|
| Upload audio rồi giả lập real-time | Chia audio thành chunk và xử lý lần lượt | Rất khả thi |
| Record audio trực tiếp trên Gradio | Người dùng ghi âm rồi phân tích sau khi record | Khả thi |
| Streaming microphone thật | Xử lý liên tục từng chunk từ microphone | Khó hơn, để future work |

### 16.3. Pipeline near real-time

```text
Audio stream / recorded audio
        ↓
Chunk 2-3 giây
        ↓
Preprocess chunk
        ↓
SER prediction + acoustic indicators
        ↓
Update dashboard timeline
```

### 16.4. Lý do phù hợp với Colab/Kaggle

Colab/Kaggle không phải môi trường tối ưu cho microphone streaming thật. Vì vậy, demo upload audio hoặc record audio rồi phân tích chunk-by-chunk là hợp lý và an toàn hơn khi báo cáo.

## 17. Dashboard demo

### 17.1. Công nghệ đề xuất

Nên dùng **Gradio** nếu cần upload/record audio nhanh. Streamlit cũng được, nhưng Gradio đơn giản hơn cho demo audio.

### 17.2. Giao diện nên có

| Thành phần | Nội dung hiển thị |
|---|---|
| Audio input | Upload hoặc record audio |
| Waveform | Hình dạng tín hiệu âm thanh |
| Spectrogram | Mel-spectrogram của audio |
| Emotion prediction | Emotion chính và confidence |
| Emotion distribution | Tỷ lệ emotion trong toàn bài |
| Emotion timeline | Emotion theo từng segment |
| Acoustic indicators | RMS, pitch, silence ratio, pause count |
| Transcript | Nội dung nhận dạng bằng Whisper |
| Speech metrics | WPM, filler count |
| Feedback text | Nhận xét và gợi ý cải thiện |

### 17.3. Luồng demo khi thuyết trình

1. Upload một audio ngắn từ dataset để chứng minh model predict emotion.
2. Upload một audio dài hoặc audio tự thu để hiển thị emotion timeline.
3. Cho dashboard hiện waveform, spectrogram, emotion distribution.
4. Hiển thị acoustic indicators và stress-related pressure indicator.
5. Nếu có transcript, hiển thị speech rate và filler words.
6. Đọc phần feedback text và giải thích vì sao hệ thống đưa ra nhận xét đó.

## 18. Đánh giá mô hình và hệ thống

### 18.1. Đánh giá SER model

Các metric nên báo cáo:

| Metric | Ý nghĩa |
|---|---|
| Accuracy | Tỷ lệ dự đoán đúng tổng thể |
| Precision | Mức độ chính xác khi model dự đoán một lớp |
| Recall | Khả năng tìm đúng mẫu của từng lớp |
| Macro-F1 | Quan trọng khi dataset lệch nhãn |
| Confusion Matrix | Phân tích emotion nào dễ bị nhầm |
| Training/Validation Loss | Theo dõi overfitting |

### 18.2. Đánh giá emotion timeline

Không có ground truth timeline cho audio tự thu, nên đánh giá theo hướng định tính:

- Timeline có hợp lý với cảm nhận người nghe không?
- Các đoạn có energy thấp hoặc pause dài có trùng với đoạn feedback không?
- Model có quá nhạy khi chia segment không?
- Confidence có thấp ở các đoạn nhiễu hoặc im lặng không?

### 18.3. Đánh giá acoustic feedback

Có thể kiểm thử bằng các audio tự tạo:

| Audio test | Kỳ vọng hệ thống phát hiện |
|---|---|
| Nói đều, ít pause | Low pressure, continuity tốt |
| Nói nhiều khoảng lặng | Silence ratio và pause count cao |
| Nói quá nhanh | Speech rate cao |
| Nói nhỏ dần | Energy giảm dần |
| Nói nhiều um/uh | Filler count cao |

## 19. Cấu trúc notebook và source code đề xuất

### 19.1. Cấu trúc thư mục

```text
speech-feedback-project/
│
├── data/
│   ├── raw/
│   │   ├── CREMA-D/
│   │   ├── RAVDESS/
│   │   └── TESS/
│   ├── processed/
│   └── metadata.csv
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_train_baseline_mfcc_ml.ipynb
│   ├── 05_train_cnn_ser.ipynb
│   └── 06_inference_timeline_feedback.ipynb
│
├── src/
│   ├── audio_preprocess.py
│   ├── feature_extraction.py
│   ├── train_baseline.py
│   ├── train_cnn.py
│   ├── inference.py
│   ├── acoustic_indicators.py
│   ├── transcript_analysis.py
│   └── feedback_engine.py
│
├── models/
│   ├── baseline_mfcc_svm.pkl
│   └── cnn_ser_model.pt
│
├── app/
│   └── gradio_app.py
│
├── figures/
│   ├── label_distribution.png
│   ├── duration_distribution.png
│   ├── confusion_matrix.png
│   ├── emotion_timeline_demo.png
│   └── acoustic_indicators_demo.png
│
└── reports/
    ├── final_report.docx
    └── slides.pptx
```

### 19.2. Notebook bắt buộc

| Notebook | Nội dung |
|---|---|
| 01_data_preparation | Tải/đọc dataset, parse label, tạo metadata, split |
| 02_eda_visualization | Vẽ biểu đồ dữ liệu và minh họa waveform/spectrogram/MFCC |
| 03_feature_extraction | Trích xuất MFCC, Mel-spectrogram, acoustic features |
| 04_train_baseline | Train MFCC + SVM/Random Forest |
| 05_train_cnn_ser | Train CNN cho SER |
| 06_inference_timeline_feedback | Emotion timeline, indicators, transcript, feedback demo |

## 20. Task assignment đề xuất

| Thành viên | Vai trò chính | Công việc cụ thể |
|---|---|---|
| Nguyễn Minh Cường | EDA, acoustic indicators, emotion timeline, dashboard | Vẽ biểu đồ dữ liệu, xây module RMS/pitch/pause, tạo timeline, làm visualization và feedback demo. |
| Nguyễn Tài Huy | Dataset, preprocessing, feature extraction, baseline | Tải dataset, parse label, tạo metadata, resample/mono/pad, extract MFCC/Mel, train SVM/Random Forest. |
| Bùi Quang Huy | Literature review, main model, methodology/report | Tổng hợp paper, thiết kế pipeline, train CNN/CNN-BiGRU nếu kịp, phân tích kết quả, viết methodology và limitation. |

Có thể điều chỉnh phân công tùy năng lực từng thành viên, nhưng nên đảm bảo mỗi người có một phần rõ ràng để trình bày khi báo cáo.

## 21. Timeline 4 tuần

### Tuần 1 - Chốt đề tài, dataset và pipeline

- Chốt phạm vi: speaking feedback, không chẩn đoán stress.
- Tải CREMA-D, RAVDESS speech-only, tùy chọn TESS.
- Parse label và map về 6 emotion.
- Tạo metadata.csv.
- Viết phần problem statement, research gap và dataset description.

### Tuần 2 - EDA, feature extraction và baseline

- Vẽ phân bố emotion, dataset, duration.
- Minh họa waveform, Mel-spectrogram, MFCC.
- Extract MFCC, Mel-spectrogram, RMS, pitch, silence ratio, pause count.
- Train baseline MFCC + SVM/Random Forest.
- Báo cáo accuracy, macro-F1 và confusion matrix ban đầu.

### Tuần 3 - CNN model, emotion timeline và indicators

- Train Mel-spectrogram + CNN.
- So sánh CNN với baseline.
- Xây inference cho audio dài bằng segment 3 giây.
- Tạo emotion timeline.
- Xây acoustic indicators và pressure indicator.
- Kiểm thử bằng audio tự thu.

### Tuần 4 - Transcript, dashboard, report và slide

- Thêm Whisper transcript nếu kịp.
- Tính speech rate và filler words.
- Hoàn thiện feedback engine.
- Làm Gradio/Streamlit dashboard.
- Viết báo cáo cuối kỳ, slide và script thuyết trình.
- Chuẩn bị demo và câu trả lời phản biện.

## 22. Rủi ro và hướng xử lý

| Rủi ro | Tác động | Hướng xử lý |
|---|---|---|
| Dataset lớn, tải lâu | Chậm tiến độ | Chỉ dùng CREMA-D + RAVDESS speech-only trước, thêm TESS sau. |
| Label không đồng nhất | Model học nhiễu | Map về 6 emotion chung, bỏ calm/surprise ở bản chính. |
| Overfitting | Validation thấp | Dùng dropout, early stopping, data augmentation nhẹ. |
| Model nhầm emotion gần nhau | Confusion matrix xấu | Phân tích nhầm lẫn như một limitation tự nhiên của SER. |
| Không có nhãn stress | Không train được stress classifier | Chỉ làm stress-related acoustic indicators, không gọi là stress detection. |
| Whisper nhận filler sai | Feedback transcript thiếu chính xác | Ghi là experimental feature, dùng audio tiếng Anh rõ trước. |
| Real-time khó chạy trên Colab | Demo dễ lỗi | Dùng simulated near real-time bằng upload audio và xử lý chunk. |
| Audio tự thu nhiễu | Prediction kém | Thêm normalize volume, trim silence, kiểm thử nhiều audio. |

## 23. Mức độ ưu tiên tính năng

### 23.1. Must-have

- Metadata thống nhất cho CREMA-D và RAVDESS.
- Preprocessing audio 16 kHz, mono, pad/cut 3 giây.
- EDA cơ bản và visualization.
- Baseline MFCC + SVM/Random Forest.
- CNN SER model.
- Confusion matrix và macro-F1.
- Emotion prediction cho audio ngắn.
- Emotion timeline cho audio dài.
- Acoustic indicators: RMS, pitch, silence ratio, pause count.
- Feedback text rule-based.

### 23.2. Should-have

- TESS làm dataset bổ sung hoặc external test.
- Segment overlap để timeline mượt hơn.
- Gradio dashboard đẹp, có waveform/spectrogram/timeline.
- Speaking pressure indicator 3 mức: low, medium, high.
- Audio tự thu để demo thực tế.

### 23.3. Nice-to-have

- Whisper transcript.
- Speech rate và filler words.
- CNN-BiGRU/LSTM.
- Wav2Vec2/HuBERT embedding extraction.
- Simulated near real-time dashboard.
- Vietnamese speaking sample test.

### 23.4. Future work

- Fine-tune Wav2Vec2/HuBERT đầy đủ.
- Dữ liệu tiếng Việt tự thu có nhãn.
- Sentence stress/prosody analysis.
- IELTS Speaking rubric.
- Public speaking scoring.
- Stress/depression dataset như DAIC-WOZ, nhưng cần xử lý đạo đức và phạm vi y tế cẩn thận.

## 24. Kết quả đầu ra mong đợi

Cuối project nên có:

1. Notebook preprocessing và metadata.
2. Notebook EDA và feature visualization.
3. Notebook train baseline MFCC + SVM/Random Forest.
4. Notebook train CNN SER.
5. Notebook hoặc script inference cho emotion timeline.
6. Module acoustic indicators.
7. Module transcript/speech rate/filler words nếu kịp.
8. Dashboard demo bằng Gradio/Streamlit.
9. Saved model và label encoder.
10. Figures cho báo cáo: label distribution, duration distribution, MFCC/spectrogram examples, confusion matrix, emotion timeline, acoustic indicators.
11. Báo cáo cuối kỳ.
12. Slide thuyết trình.

## 25. Nội dung học thuật nên nhấn mạnh khi báo cáo

Nhóm nên nhấn mạnh 5 điểm sau:

1. **SER không chỉ là classification:** project mở rộng sang emotion timeline để phân tích bài nói theo thời gian.
2. **Stress không đồng nghĩa với negative emotion:** hệ thống không suy stress trực tiếp từ sad/fear/angry, mà dùng acoustic indicators.
3. **Feature-based feedback phù hợp với tài nguyên hạn chế:** thay vì fine-tune model lớn, nhóm dùng CNN nhẹ kết hợp rule-based feedback.
4. **Acoustic indicators giúp giải thích kết quả:** RMS, pitch, pause và silence giúp feedback dễ hiểu hơn softmax emotion.
5. **Demo có tính ứng dụng:** người dùng có thể upload/record audio và nhận gợi ý cải thiện cách nói.

## 26. Gợi ý phần kết luận báo cáo

Đề tài đã xây dựng một pipeline phân tích giọng nói phục vụ luyện nói, kết hợp Speech Emotion Recognition, emotion timeline và các chỉ báo âm học. Mô hình SER được huấn luyện trên các dataset cảm xúc công khai như CREMA-D, RAVDESS và có thể bổ sung TESS. Thay vì chẩn đoán stress, hệ thống tính các stress-related acoustic indicators như pitch variation, energy instability, silence ratio và pause count để hỗ trợ phản hồi luyện nói. Kết quả đầu ra được hiển thị qua dashboard gồm waveform, spectrogram, emotion distribution, emotion timeline và feedback text. Hướng phát triển tiếp theo là mở rộng dữ liệu tiếng Việt, cải thiện mô hình bằng pretrained speech models và bổ sung phân tích prosody/sentence stress cho luyện nói nâng cao.

## 27. Tài liệu và nguồn tham khảo chính

### Dataset chính

- RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song.
- CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.
- TESS: Toronto Emotional Speech Set.

### Paper/model tham khảo

1. **TBDM-Net: Bidirectional Dense Networks with Gender Information for Speech Emotion Recognition**  
   Link paper: https://arxiv.org/abs/2409.10056  
   GitHub: https://github.com/adrianastan/tbdm-net  
   Vai trò: tham khảo SER model, MFCC và temporal modeling.

2. **Frame-Level Emotional State Alignment Method for Speech Emotion Recognition**  
   Link paper: https://arxiv.org/abs/2312.16383  
   GitHub: https://github.com/ASolitaryMan/HFLEA  
   Vai trò: tham khảo emotion timeline/frame-level emotion alignment.

3. **Integrating Multimodal Affective Signals for Stress Detection from Audio-Visual Data**  
   Link paper: https://dl.acm.org/doi/10.1145/3678957.3685717  
   GitHub: https://github.com/ScazLab/MAS_dataset  
   Vai trò: tham khảo stress-related multimodal indicators.

4. **Acoustic and Prosodic Speech Features Reflect Physiological Stress but Not Isolated Negative Affect**  
   Link paper: https://www.nature.com/articles/s41598-024-55550-3  
   GitHub: https://github.com/mitchelkappen/stress_cyberball-mist  
   OSF: https://osf.io/qf6ck/  
   Vai trò: cơ sở học thuật cho F0, speech rate, jitter, voiced segments và acoustic stress indicators.

5. **StressTest: Can YOUR Speech LM Handle the Stress?**  
   Link paper: https://arxiv.org/abs/2505.22765  
   GitHub: https://github.com/slp-rl/StressTest  
   Model: https://huggingface.co/slprl/StresSLM  
   Vai trò: future work cho sentence stress/prosody, không dùng làm psychological stress detection.

## 28. Checklist trước khi nộp

- [ ] Có metadata.csv thống nhất.
- [ ] Có EDA và visualization rõ ràng.
- [ ] Có baseline model và main model.
- [ ] Có bảng so sánh accuracy, macro-F1.
- [ ] Có confusion matrix.
- [ ] Có emotion timeline demo.
- [ ] Có acoustic indicators demo.
- [ ] Có feedback text rõ ràng.
- [ ] Có dashboard upload/record audio.
- [ ] Có phần giải thích không chẩn đoán stress.
- [ ] Có phần limitation.
- [ ] Có tài liệu tham khảo paper/dataset/repo.
- [ ] Có slide và script thuyết trình.

