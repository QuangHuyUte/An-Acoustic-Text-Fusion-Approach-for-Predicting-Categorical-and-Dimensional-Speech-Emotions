---
title: "Roadmap - 06D Strict Speaker-Independent Speech Emotion Recognition"
subtitle: "Emotion classification first, optional arousal/valence/dominance regression, aligned with final evaluation rubric"
date: "2026-07-11"
lang: "vi"
---

# Roadmap - 06D Emotion Recognition and Affective Attribute Regression

## 0. Scope update

Huong de tai duoc chot lai:

```text
Update and optimize the 06D Emotion2Vec Co-Attention model for
strict speaker-independent Speech Emotion Recognition, then optionally
extend it to arousal, valence, and dominance regression on IEMOCAP.
```

Khong tiep tuc mo rong thanh full presentation feedback system trong giai doan final, vi huong do can dataset, label va pipeline qua lon. Phan demo van co, nhung demo chi phuc vu minh hoa SER/affective speech analysis:

```text
input audio
-> segment
-> emotion prediction
-> optional arousal/valence/dominance curves
-> simple visualization
```

## 1. Ly do doi huong

Ket qua hien tai cua 06D:

| Protocol | Ket qua hien tai | Nhan xet |
|---|---:|---|
| Random split | around 80% | Kha on, nhung co the phu thuoc vao speaker/data overlap |
| Strict speaker split | around 69% | Thap hon nhieu, cho thay model chua generalize tot sang speaker moi |

Van de can giai quyet:

- 06D co kien truc nang cao nhung feature extraction/fusion chua du manh.
- Random split khong du thuyet phuc vi co nguy co model hoc speaker identity.
- Final report can chung minh model chay tot trong dieu kien strict speaker-independent.
- Cac repo manh tren IEMOCAP dung feature/backbone tot hon: emotion2vec, Wav2Vec2, WavLM, HuBERT, co-attention voi MFCC/spectrogram/wav2vec2.

Muc tieu moi:

1. Doc ky feature pipeline cua cac repo tot.
2. Bat chuoc va nang cap feature extraction cho 06D.
3. Chay lai strict speaker-independent.
4. So sanh voi cac model SER co code va paper.
5. Neu con kip, them regression head cho `EmoAct`, `EmoVal`, `EmoDom`.

## 2. Dataset chinh

### 2.1 IEMOCAP

IEMOCAP la dataset chinh cho giai doan final.

Links:

- Official IEMOCAP: https://sail.usc.edu/iemocap/
- HuggingFace mirror user da dua: https://huggingface.co/datasets/AbstractTTS/IEMOCAP

Thong tin quan trong:

| Noi dung | Gia tri |
|---|---|
| Total utterances | about 10,039 |
| Common SER subset | 5,531 utterances |
| Standard emotion labels | angry, sad, neutral, happy + excited |
| Dimensional labels | activation/arousal, valence, dominance |
| HF columns | audio, transcription, major_emotion, EmoAct, EmoVal, EmoDom |
| Main split | leave-one-session-out 5-fold |
| Stricter split | leave-one-speaker-out 10-fold |

### 2.2 Why IEMOCAP is the right final dataset

- Co raw audio.
- Co transcript neu can dung nhe cho phan demo/report.
- Co categorical emotion labels.
- Co activation/arousal, valence, dominance labels.
- Duoc dung rat rong rai trong SER benchmark.
- Phu hop voi muc tieu strict speaker-independent vi co 5 session, 10 speakers.

## 3. Final evaluation rubric alignment

Roadmap nay bam theo cac nhom muc trong `Lasterm_Param`.

### 3.1 Complete System Implementation

Can co:

| Rubric item | Deliverable |
|---|---|
| Complete speech processing pipeline | audio loading, resampling 16 kHz, segmentation, feature extraction, cached features |
| Implement selected model | cleaned 06D model with strict split support |
| Improvements to model/features/system | improved feature extraction inspired by strong repos |
| Impact analysis through experiments | ablation table: old 06D vs improved features vs co-attention/multi-task |
| UI/demo program | simple audio upload/demo notebook or app showing emotion and optional AVD timeline |

### 3.2 Experimental Evaluation

Can co:

| Rubric item | Deliverable |
|---|---|
| Dataset/test scenario | IEMOCAP 4-class, LOSO 5-fold; optional LOSpeaker 10-fold |
| Training/testing/results | CSV metrics per fold, average/std, confusion matrix |
| Metric analysis | WA, UA, Macro-F1; optional CCC/MAE for AVD |
| Baseline comparison | emotion2vec, CA-MSER, FT-w2v2 P-TAPT, DST |
| Extended evaluation | strict speaker split, optional AVD regression, audio demo |

### 3.3 Project Report

Can co:

1. Introduction and motivation.
2. Dataset and strict speaker-independent protocol.
3. Related work and baseline comparison.
4. Proposed 06D improvement.
5. Feature extraction pipeline.
6. Experiments and ablation.
7. Demo.
8. Discussion: why strict split is harder than random split.
9. Conclusion and future work.

## 4. Reference folder

Papers and code are stored in:

```text
Papers/Papers for main models emotion classification and valence arousal, dominance regression
```

Local contents:

```text
code/
  emotion2vec
  FT-w2v2-ser
  CA-MSER_co_attention
  DST_deformable_speech_transformer
  SpeechFormer
  TIM-Net_SER

papers/
  emotion2vec
  Wav2Vec2 fine-tuning SER
  CA-MSER co-attention
  DST
  TIM-Net
  SpeechFormer
  Wav2Vec2/HuBERT benchmark
  MSTR
  AVD regression references

tables/
  model_comparison.md
  model_comparison.csv
  bao_cao_tom_tat_vi.md
```

## 5. Main papers and repos

| Model | Paper | Repo | Vai tro |
|---|---|---|---|
| emotion2vec | https://arxiv.org/abs/2312.15185 | https://github.com/ddlBoJack/emotion2vec | Main reference for 06D backbone |
| FT-w2v2-ser / P-TAPT | https://arxiv.org/abs/2110.06309 | https://github.com/b04901014/FT-w2v2-ser | Strong Wav2Vec2 audio-only baseline |
| CA-MSER Co-Attention | https://arxiv.org/abs/2203.15326 | https://github.com/Vincent-ZHQ/CA-MSER | Closest reference to 06D co-attention |
| DST | https://arxiv.org/abs/2302.13729 | https://github.com/HappyColor/DST | Strong WavLM/Transformer baseline |
| TIM-Net | https://arxiv.org/abs/2211.08233 | https://github.com/Jiaxin-Ye/TIM-Net_SER | Lightweight temporal baseline |
| SpeechFormer | https://arxiv.org/abs/2203.03812 | https://github.com/HappyColor/SpeechFormer | Hierarchical speech transformer |
| Wav2Vec2/HuBERT benchmark | https://arxiv.org/abs/2111.02735 | official repo not found | SSL fine-tuning reference |
| MSTR | local PDF downloaded | official repo not found | paper-only transformer reference |

AVD references:

| Paper | Link | Task |
|---|---|---|
| Attention-Augmented End-to-End Multi-Task Learning | local PDF downloaded | A/V/D discretized low-mid-high classification |
| Contrastive Unsupervised Learning for SER | local PDF downloaded | activation/valence/dominance CCC regression |
| Cross-modal Conditional Teacher-Student | local PDF downloaded | audio-only A/V/D student from audio+text teacher |
| AV from Categorical Labels | local PDF downloaded | infer arousal/valence from categorical emotion labels |

## 6. Benchmark table for final report

### 6.1 Emotion classification

| Model | Speaker-independent? | Split | Input | WA | UA | Macro-F1 |
|---|---:|---|---|---:|---:|---:|
| emotion2vec linear | yes | LOSO 5-fold / LOSpeaker | audio | around 71.79-72.94 | around 72.69 | WF1 reported |
| CA-MSER | yes | LOSO / LOSpeaker | audio | 69.80 / 71.64 | 71.05 / 72.70 | not reported |
| FT-w2v2 P-TAPT | yes | LOSO 5-fold | audio | not reported | 74.3 | not reported |
| DST | yes | LOSO 5-fold | audio | 71.8 | 73.6 | not reported |
| Wav2Vec2/HuBERT benchmark | yes | leave-two-speaker-out 10-fold | audio | 73.01 | not reported | not reported |
| 06D current | yes | strict split | audio | around 69 | to verify | to verify |
| 06D improved | yes | LOSO 5-fold | audio | target: improve over current | target | target |

### 6.2 Arousal / Valence / Dominance regression

Emotion classification models usually do not report CCC because they predict categories. CCC is used when the output is continuous affect score.

| Model | Input | Task | Split | CCC-Arousal | CCC-Valence | CCC-Dominance |
|---|---|---|---|---:|---:|---:|
| preCPC AVD | audio/CPC | continuous A/V/D regression | IEMOCAP 5-fold | 0.752 | 0.752 | 0.691 |
| Cross-modal teacher-student | HuBERT audio, BERT teacher | continuous A/V/D regression | 5-fold speaker-independent | 0.667 | 0.582 | 0.545 |
| AV from categorical labels | WavLM | arousal/valence only | speaker-independent | 0.632-0.672 | 0.529-0.566 | N/A |
| 06D + AVD head | audio | continuous A/V/D regression | LOSO 5-fold | to measure | to measure | to measure |

## 7. What to copy from strong repos

### 7.1 emotion2vec

Learn:

- use pretrained emotion representation instead of weak handcrafted-only feature;
- cache frame-level and utterance-level embeddings;
- train simple downstream classifier first;
- report strict IEMOCAP 4-class metrics.

Apply to 06D:

```text
audio -> emotion2vec embedding -> adapter/MLP -> classifier
```

### 7.2 CA-MSER

Learn:

- use three feature levels: MFCC, spectrogram, wav2vec2;
- do not simply concatenate features;
- use co-attention to weight SSL embedding with acoustic features;
- run ablation: each feature alone, fusion without attention, fusion with attention.

Apply to 06D:

```text
MFCC branch
LogMel/spectrogram branch
Emotion2Vec branch
-> co-attention / gated fusion
-> classifier
```

### 7.3 FT-w2v2-ser

Learn:

- strict leave-one-session-out generation;
- Wav2Vec2 fine-tuning baseline;
- task adaptive pretraining idea;
- report UA per fold.

Apply to 06D:

- use as benchmark, not necessarily copy entire training;
- compare 06D improved against P-TAPT UA 74.3.

### 7.4 DST

Learn:

- WavLM features can be strong for SER;
- attention window should adapt to relevant emotional regions;
- transformer attention can be upgraded beyond vanilla full attention.

Apply to 06D:

- optional attention upgrade if time remains;
- not first priority because it may increase complexity.

## 8. Proposed 06D optimization plan

### Stage 1 - Reproduce current result

Goal:

```text
Confirm current 06D result:
random split around 80%
strict split around 69%
```

Deliverables:

- exact dataset version;
- exact labels;
- split script;
- metrics per fold;
- confusion matrix.

### Stage 2 - Strict IEMOCAP pipeline

Goal:

```text
IEMOCAP 4-class:
angry, sad, neutral, happy+excited
LOSO 5-fold
```

Deliverables:

- manifest CSV with audio path, speaker, session, label;
- fold assignment;
- no speaker leakage;
- class distribution per fold.

### Stage 3 - Feature extraction upgrade

Fix weak feature extraction first.

Feature groups:

| Feature group | Source idea | Use |
|---|---|---|
| emotion2vec embedding | emotion2vec repo | main affective representation |
| MFCC sequence | CA-MSER/TIM-Net | low-level acoustic branch |
| log-mel/spectrogram | CA-MSER/SpeechFormer | spectral branch |
| wav2vec2/WavLM optional | CA-MSER/DST/FT-w2v2 | baseline or optional branch |
| statistics | current 06D | stable utterance-level auxiliary feature |

Required ablation:

| Experiment | Purpose |
|---|---|
| emotion2vec only | check backbone strength |
| MFCC only | check low-level baseline |
| log-mel only | check spectral branch |
| emotion2vec + MFCC | check acoustic + SSL |
| emotion2vec + log-mel | check spectral + SSL |
| all features concat | check naive fusion |
| all features + co-attention | check 06D proposed gain |

### Stage 4 - Model optimization

Optimize:

- freeze vs unfreeze emotion2vec adapter;
- learning rate;
- dropout;
- class weights or focal loss;
- early stopping by validation UA/Macro-F1;
- batch size and sequence length;
- speaker-independent validation fold;
- seed averaging if time allows.

Do not over-optimize random split. Random split is only sanity check.

### Stage 5 - Compare with reference models

Main comparison:

1. emotion2vec linear.
2. CA-MSER.
3. FT-w2v2 P-TAPT.
4. DST.

The final report should say:

```text
Our previous 06D reached around 80% on random split but only around 69%
on strict speaker split. Therefore, this final stage focuses on strict
speaker-independent generalization rather than random-split accuracy.
```

### Stage 6 - Optional AVD regression

Only after emotion classification is stable.

Architecture:

```text
shared 06D encoder
  -> emotion head: CrossEntropyLoss
  -> AVD head: predicts EmoAct, EmoVal, EmoDom
```

Loss:

```text
total_loss = CE_emotion + lambda * AVD_loss
```

AVD loss options:

- MSE loss first;
- CCC loss if implementation time allows.

Metrics:

- CCC-Arousal;
- CCC-Valence;
- CCC-Dominance;
- MAE-Arousal;
- MAE-Valence;
- MAE-Dominance.

Important:

```text
Do not compare AVD CCC directly with emotion WA/UA.
They are different tasks.
```

## 9. Demo plan

The demo should be small and realistic.

### 9.1 Input

```text
User uploads a WAV/MP3 audio file
or selects one IEMOCAP sample
```

### 9.2 Processing

```text
resample to 16 kHz
split into 3-5 second segments
run 06D improved model
aggregate predictions
optional AVD head for each segment
```

### 9.3 Output

Minimum demo:

- predicted emotion;
- confidence score;
- per-segment emotion timeline;
- confusion/summary visualization for test samples.

Optional demo:

- arousal curve;
- valence curve;
- dominance curve;
- short text explanation:

```text
This segment is predicted as high arousal / negative valence.
This suggests an intense emotional tone.
```

Do not claim presentation-quality scoring in the demo.

## 10. Final report structure

Recommended report:

1. Introduction
   - Why random split is not enough.
   - Why speaker-independent SER matters.
2. Dataset
   - IEMOCAP structure.
   - 4-class subset.
   - AVD labels.
3. Related Work
   - emotion2vec.
   - Wav2Vec2 P-TAPT.
   - CA-MSER.
   - DST.
   - AVD regression papers.
4. Proposed Method
   - 06D current model.
   - Feature extraction weakness.
   - Improved feature pipeline.
   - Co-attention fusion.
5. Experiments
   - random split sanity check.
   - strict LOSO 5-fold.
   - ablation.
   - baseline comparison.
6. AVD Extension
   - optional regression head.
   - CCC/MAE.
7. Demo
   - audio upload/sample demo.
8. Discussion
   - why strict split is harder.
   - why model may still be below some SOTA.
   - feature extraction and speaker leakage discussion.
9. Conclusion

## 11. Work plan

### Week 1 - Data and baseline audit

- Read emotion2vec, CA-MSER, FT-w2v2, DST feature pipelines.
- Build IEMOCAP manifest.
- Implement LOSO 5-fold split.
- Re-run current 06D on same split.

### Week 2 - Feature extraction upgrade

- Cache emotion2vec embeddings.
- Add/clean MFCC and log-mel branches.
- Match CA-MSER-style feature handling where possible.
- Run feature ablation.

### Week 3 - 06D optimization

- Tune LR, dropout, class weight/focal loss.
- Compare concat vs co-attention.
- Report WA, UA, Macro-F1 per fold.
- Compare against reference models.

### Week 4 - Optional AVD and demo

- Add AVD regression head if emotion model is stable.
- Report CCC/MAE.
- Build demo notebook/app.
- Finalize report tables and figures.

## 12. Final decision

Core deliverable:

```text
Improved 06D model for strict speaker-independent IEMOCAP SER.
```

Optional deliverable:

```text
06D + AVD regression head for EmoAct, EmoVal, EmoDom.
```

Removed from current roadmap:

- PuSQ reproduction.
- TED rating prediction.
- Full presentation feedback system.
- Long-audio TED assessment.

Reason:

```text
These directions are useful research references, but they are too broad
for the final deadline and no longer match the current core goal.
```
