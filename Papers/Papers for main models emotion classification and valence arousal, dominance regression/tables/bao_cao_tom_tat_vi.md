# Bao cao tom tat - Emotion / Arousal / Valence / Dominance

Thu muc nay dung cho huong de tai moi: tiep tuc tu notebook 06D, tap trung vao Speech Emotion Recognition va mo rong nhe sang Activation/Arousal, Valence, Dominance neu can.

## 1. IEMOCAP co gi?

| Noi dung | Gia tri |
|---|---|
| Tong so utterance | Khoang 10,039 utterances |
| Dataset pho bien cho SER | 5,531 utterances |
| Emotion setup pho bien | 4 class: angry, sad, neutral, happy + excited |
| Dimensional labels | Activation/Arousal, Valence, Dominance |
| Thang diem AVD | Thuong la 1-5 |
| Transcript | Co |
| Audio | Co |

Luu y:

- Emotion classification do bang WA, UA, Macro-F1, Weighted-F1.
- Arousal/Valence/Dominance la regression thi do bang CCC, MAE, MSE.
- Mot so paper cu roi rac hoa AVD thanh low/mid/high, khi do do bang UAR.

## 2. Cac repo da tai ve

| Repo local | Link goc | Vai tro |
|---|---|---|
| `code/emotion2vec` | https://github.com/ddlBoJack/emotion2vec | Gan nhat voi 06D, dung lam backbone/feature extractor chinh |
| `code/FT-w2v2-ser` | https://github.com/b04901014/FT-w2v2-ser | Baseline Wav2Vec2 fine-tuning manh tren IEMOCAP |
| `code/CA-MSER_co_attention` | https://github.com/Vincent-ZHQ/CA-MSER | Rat sat voi 06D vi cung dung co-attention fusion |
| `code/DST_deformable_speech_transformer` | https://github.com/HappyColor/DST | Transformer baseline manh voi WavLM feature |
| `code/TIM-Net_SER` | https://github.com/Jiaxin-Ye/TIM-Net_SER | Baseline nhe, dung MFCC, de chay hon neu compute yeu |
| `code/SpeechFormer` | https://github.com/HappyColor/SpeechFormer | Kien truc hierarchical transformer cho speech |

## 3. Cac paper da tai ve

| File | Noi dung |
|---|---|
| `2312.15185_emotion2vec.pdf` | emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation |
| `2110.06309_wav2vec2_finetuning_SER.pdf` | Wav2Vec2 fine-tuning, TAPT, P-TAPT |
| `2203.15326_co_attention_multilevel_acoustic.pdf` | Co-attention fusion MFCC + spectrogram + wav2vec2 |
| `2302.13729_DST_deformable_speech_transformer.pdf` | DST: Deformable Speech Transformer |
| `2211.08233_TIM-Net_SER.pdf` | TIM-Net |
| `2203.03812_SpeechFormer.pdf` | SpeechFormer |
| `2111.02735_wav2vec2_HuBERT_benchmark.pdf` | Wav2Vec2/HuBERT fine-tuning benchmark |
| `2410.00390_MSTR_multi_scale_temporal_transformer.pdf` | Multi-Scale Temporal Transformer |
| `1903.12424_attention_augmented_multitask_AVD.pdf` | Multi-task activation/valence/dominance classification |
| `2102.06357_contrastive_unsupervised_AVD_SER.pdf` | CPC/preCPC cho AVD regression |
| `2112.00158_cross_modal_teacher_student_AVD.pdf` | Audio-only student hoc tu audio+text teacher cho AVD |
| `2311.14816_AV_from_categorical_emotion_labels.pdf` | Hoc arousal/valence tu categorical emotion labels |

## 4. Bang model chinh tren IEMOCAP emotion classification

| Model | Co code? | Input | Kien truc | Chia dataset | Label | Ket qua |
|---|---|---|---|---|---|---|
| emotion2vec | Co | Raw speech -> emotion2vec embeddings | Self-supervised pretraining + linear downstream | 5,531 utterances, 4-class, LOSO/LOSpeaker | angry, sad, neutral, happy+excited | WA khoang 71.79-72.94%; mot so protocol starred 74.48-77.64% |
| Wav2Vec2 P-TAPT | Co | Raw waveform | Wav2Vec2 fine-tuning + task adaptive pretraining + pseudo-label TAPT | Leave-one-session-out | 4-class emotion | UA 74.3% |
| CA-MSER Co-Attention | Co | MFCC + spectrogram + wav2vec2 embedding | BiLSTM + spectrogram encoder + wav2vec2 + co-attention | 5,531 utterances; LOSO va LOSpeaker | 4-class emotion | LOSO WA/UA 69.80/71.05%; LOSpeaker WA/UA 71.64/72.70% |
| DST | Co | WavLM features | Deformable Speech Transformer | 4-class, leave-one-session-out | 4-class emotion | WA 71.8%, UA 73.6% |
| TIM-Net | Co | 39-D MFCC | Bidirectional multi-scale temporal dilated CNN | 10-fold CV | 4-class emotion | UAR/WAR 69.00/68.29% hoac 72.50/71.65% tuy protocol |
| SpeechFormer | Co | Spec/Logmel/Wav2vec | Hierarchical speech transformer | 5,531 utterances, LOSO | 4-class emotion | Tot nhat tren IEMOCAP trong paper: WA 62.9%, UA 64.5% |
| Wav2Vec2/HuBERT benchmark | Chua tim thay official repo | Raw waveform | Partial/entire fine-tuning Wav2Vec2/HuBERT | Speaker-dependent va speaker-independent leave-two-speaker-out | 4-class emotion | HuBERT-large PF: SD WA 79.58%, SI WA 73.01% |
| MSTR | Chua tim thay official repo | HuBERT features | Multi-scale temporal transformer | 4-class, LOSO | 4-class emotion | WA 70.60%, UA 71.60% |

## 5. Bang arousal / valence / dominance

| Model | Co code? | Input | Kien truc | Chia dataset | Label | Ket qua |
|---|---|---|---|---|---|---|
| Attention-augmented MTL | Chua tim thay official repo | Raw audio; baseline dung eGeMAPS | End-to-end + attention + multi-task learning | Session 1-3 train, Session 4 dev, Session 5 test | activation/arousal, valence, dominance duoc roi rac thanh low/mid/high | Test UAR: arousal 48.5%, valence 63.8%, dominance 51.6% |
| preCPC AVD | Chua tim thay official repo | Raw audio / CPC features | CPC pretraining + attention + dense regression head | IEMOCAP 5-fold CV | activation, valence, dominance continuous | CCC avg 0.731; act 0.752; val 0.752; dom 0.691 |
| Cross-modal teacher-student | Chua tim thay official repo | HuBERT audio; BERT text chi lam teacher | Audio-only student hoc tu audio+text teacher | 5-fold speaker-independent CV | activation, valence, dominance continuous | Audio-only T/S CCC: act 0.667, val 0.582, dom 0.545 |
| AV from categorical labels | Chua tim thay official repo | WavLM features | Emotion classifier + anchor-based dimensionality reduction | Speaker-independent train/test split | arousal va valence | 4-emotion: val CCC 0.529, arousal CCC 0.632; 9-emotion: val CCC 0.566, arousal CCC 0.672 |

## 6. Model nen uu tien cho de tai hien tai

### Uu tien 1: emotion2vec

Nen dung vi gan nhat voi notebook 06D hien tai. Ta co the:

- giu emotion2vec lam backbone;
- tinh embedding theo utterance hoac theo frame;
- train linear/MLP classifier de co baseline sach;
- so sanh 06D co-attention voi baseline emotion2vec linear.

### Uu tien 2: CA-MSER Co-Attention

Rat hop de tham khao vi no dung:

- MFCC;
- spectrogram;
- wav2vec2 embedding;
- co-attention fusion.

No gan voi y tuong 06D hon DST/TIM-Net, nen nen doc ky kien truc nay truoc.

### Uu tien 3: FT-w2v2-ser P-TAPT

Day la baseline manh va co code. Neu 06D khong vuot duoc ket qua nay thi van co the viet theo huong:

- 06D la emotion2vec/co-attention feature fusion;
- FT-w2v2-ser la SSL fine-tuning baseline;
- so sanh ve accuracy, UA, Macro-F1, do on dinh theo fold.

### Uu tien 4: DST

Dung neu muon nang cap attention block. DST co code, ket qua kha manh, nhung co the ton compute hon.

### Uu tien 5: AVD regression head

Neu con thoi gian, them mot head nho tren embedding cua 06D:

- emotion head: CrossEntropyLoss;
- arousal/valence/dominance head: MSE/CCC loss;
- report CCC/MAE cho EmoAct, EmoVal, EmoDom.

Khong nen bien AVD thanh muc tieu chinh neu deadline gan, vi can xu ly metric va training phuc tap hon emotion classification.

## 7. Ket luan ngan

Huong an toan nhat luc nay:

1. Chay lai IEMOCAP 4-class tren 06D.
2. So sanh voi emotion2vec linear, CA-MSER, FT-w2v2-ser.
3. Bao cao emotion classification bang WA, UA, Macro-F1.
4. Neu con kip, them AVD regression head va bao cao CCC/MAE.
5. Khong mo lai presentation feedback system trong giai doan nay.

## 8. Bang benchmark nen dung cho 06D strict speaker-independent

| Model | Speaker-independent? | Split | Input | Emotion labels | WA | UA | Macro-F1 | Arousal / Valence / Dominance | Metric AVD |
|---|---:|---|---|---|---:|---:|---:|---|---|
| emotion2vec linear | Co | LOSO 5-fold / LOSpeaker | audio | angry, sad, neutral, happy+excited | khoang 71.79-72.94 | khoang 72.69 | thuong bao WF1, khong phai luc nao cung co Macro-F1 | khong report | N/A |
| CA-MSER | Co | LOSO / leave-one-speaker-out | audio | angry, sad, neutral, happy+excited | 69.80 / 71.64 | 71.05 / 72.70 | khong report | khong report | N/A |
| FT-w2v2 P-TAPT | Co | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | khong report | 74.3 | khong report | khong report | N/A |
| DST | Co | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | 71.8 | 73.6 | khong report | khong report | N/A |
| Wav2Vec2/HuBERT benchmark | Co ve protocol, chua co official repo local | leave-two-speaker-out 10-fold | audio | anger, happiness, sadness, neutral | 73.01 | khong report | khong report | khong report | N/A |
| 06D proposed | Co | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | can do | can do | can do | optional: EmoAct, EmoVal, EmoDom | CCC + MAE |

## 9. Arousal, Valence, Dominance duoc do nhu the nao?

Trong IEMOCAP, moi utterance co the co diem dimensional emotion do annotator cham. Ba diem nay la:

| Dimension | Ten trong IEMOCAP/HF | Y nghia | Diem thap | Diem cao |
|---|---|---|---|---|
| Activation / Arousal | `EmoAct` | muc nang luong/kich hoat cua cam xuc | binh tinh, met, nang luong thap | phan khich, tuc gian, nang luong cao |
| Valence | `EmoVal` | muc tich cuc/tieu cuc | tieu cuc, buon, kho chiu | tich cuc, vui, de chiu |
| Dominance | `EmoDom` | muc kiem soat/chu dong/manh me | yeu the, bi dong, thieu kiem soat | manh me, tu tin, chu dong |

Thang diem goc thuong la 1-5. Neu nhieu annotator cham cung mot utterance, gia tri ground truth thuong la diem trung binh cua cac annotator.

Voi 06D, cach lam dung la them mot regression head:

```text
audio
-> 06D / emotion2vec / co-attention encoder
-> shared representation
-> emotion head: angry / sad / neutral / happy
-> AVD head: arousal / valence / dominance
```

Metric dung cho AVD:

| Metric | Dung cho | Y nghia |
|---|---|---|
| CCC | regression A/V/D | do muc du doan khop voi diem human rating ve tuong quan, trung binh, va scale; cang gan 1 cang tot |
| MAE | regression A/V/D | sai so tuyet doi trung binh; cang thap cang tot |
| RMSE/MSE | regression A/V/D | phat nang loi lon; cang thap cang tot |

Khong nen dung WA/UA/Macro-F1 cho AVD neu minh giu diem 1-5 lien tuc. Chi dung accuracy/UAR neu minh bien no thanh 3 class:

```text
low:    [1, 2]
middle: (2, 4)
high:   [4, 5]
```

Khuyen nghi cho project:

- Bat buoc: emotion classification, report WA, UA, Macro-F1.
- Neu con kip: AVD regression, report CCC va MAE cho `EmoAct`, `EmoVal`, `EmoDom`.
