# Speech Emotion Studio - Presentation Feedback Demo

Web demo cho do an Speech Processing cua Group 11. Demo hien tai tap trung vao speech emotion recognition nhu mot tin hieu trung gian cho presentation feedback.

## Chay demo

```bash
cd "D:\UTE\Speech Programming\Speech Project\07_Web_Demo"
npm start
```

Hoac double-click:

```text
start-demo.cmd
```

Sau do mo:

```text
http://localhost:5174
```

Quan trong: microphone va Speech-to-Text can chay qua `http://localhost:5174`. Khong mo truc tiep file HTML bang `file://`.

## Workflow

1. Record
   - Chon 1 trong 4 mission tasks: `neutral`, `happy`, `sad`, `angry`.
   - Moi task dung cung mot script ngan: "I am fine. I can continue the presentation. The result is important to me."
   - Nguoi dung doc cung noi dung nhung thay doi cach bieu cam theo target emotion.
   - Web ghi audio, hien target timestamp va Speech-to-Text transcript neu trinh duyet ho tro.
   - Sau khi ghi xong, bam `Add to queue`.

2. Processing
   - Chon audio trong queue.
   - Bam `Analyze selected audio`.
   - Quan sat flow: load audio, waveform + VAD, feature extraction, representation views, fusion, report.

3. Results
   - Xem emotion probabilities cho 4 emotions.
   - Xem valence, arousal, dominance.
   - Xem target script va recognized transcript kem timestamp.
   - Export JSON report va download audio.

## Feature pipeline trong browser

- waveform
- VAD / voiced ratio / pause ratio
- RMS / energy
- ZCR
- pitch/F0 proxy bang autocorrelation
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flux
- spectral contrast
- log-Mel heatmap
- MFCC approximation

## Ghi chu

Predictor hien tai la heuristic chay trong browser de phuc vu live demo. Sau nay co the thay `predictAffect()` trong `public/js/app.js` bang API goi model that tu notebook 06D/06_w2v.
