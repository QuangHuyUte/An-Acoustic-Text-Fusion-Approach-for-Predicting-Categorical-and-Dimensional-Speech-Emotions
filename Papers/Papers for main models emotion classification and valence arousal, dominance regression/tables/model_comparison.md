# Model Comparison on IEMOCAP and Affective Attributes

## IEMOCAP Dataset Setups

| Setup | Records used | Labels | Split protocol | Typical metric |
|---|---:|---|---|---|
| Full IEMOCAP affective corpus | about 10,039 utterances | categorical emotion plus activation/arousal, valence, dominance | varies by paper | emotion: WA/UA/F1; AVD: CCC/MAE or UAR if discretized |
| Standard 4-class SER subset | 5,531 utterances | angry, sad, neutral, happy+excited | usually leave-one-session-out 5-fold or leave-one-speaker-out 10-fold | WA, UA, Macro-F1, WF1 |
| AVD 3-class discretized setup | 9,949 utterances in one paper | activation/arousal, valence, dominance discretized into low/mid/high | train Session 1-3, dev Session 4, test Session 5 | UAR |
| AVD continuous regression setup | full or filtered IEMOCAP utterances depending paper | activation/arousal, valence, dominance continuous scores | usually 5-fold speaker/session independent CV | CCC, MAE |

## Code-Backed Main Models

| Model | Paper / repo status | Input features | Architecture | IEMOCAP split | Target labels | Reported result | Why it matters for 06D |
|---|---|---|---|---|---|---|---|
| emotion2vec | PDF downloaded; repo cloned: `code/emotion2vec` | raw speech -> emotion2vec frame/utterance embeddings | self-supervised online distillation pretraining, then linear downstream classifier | 5,531 utterances; 4-class; leave-one-session-out 5-fold and leave-one-speaker-out 10-fold reported | emotion category | reported IEMOCAP WA around 71.79%-72.94%; starred protocol variants 74.48%-77.64% | Closest foundation model to the current 06D direction. Best to keep as feature extractor/backbone. |
| FT-w2v2-ser P-TAPT | PDF downloaded; repo cloned: `code/FT-w2v2-ser` | raw waveform | Wav2Vec2 fine-tuning; V-FT, TAPT, and pseudo-label TAPT objective | leave-one-session-out; one session test, other four sessions train; no validation, fixed 15 epochs | 4-class emotion | mean UA: V-FT 69.9%, TAPT 73.5%, P-TAPT 74.3% | Strong reproducible audio-only fine-tuning baseline. Useful comparison against 06D. |
| CA-MSER co-attention | PDF downloaded; repo cloned: `code/CA-MSER_co_attention` | MFCC, spectrogram, wav2vec2 embeddings | BiLSTM for MFCC, AlexNet-like spectrogram encoder, wav2vec2 encoder, co-attention fusion | 5,531 utterances; 4-class; both leave-one-session-out and leave-one-speaker-out | 4-class emotion | LOSO: WA 69.80%, UA 71.05%; LOSpeaker: WA 71.64%, UA 72.70% | Directly related to the current Co-Attention idea in 06D. Good architecture reference. |
| DST | PDF downloaded; repo cloned: `code/DST_deformable_speech_transformer` | WavLM features | Deformable Speech Transformer with adaptive window sizes and offsets | IEMOCAP 4-class; leave-one-session-out | 4-class emotion | WA 71.8%, UA 73.6% | Strong transformer baseline with official code. Good reference if upgrading 06D attention. |
| TIM-Net | PDF downloaded; repo cloned: `code/TIM-Net_SER` | 39-D MFCC | temporal-aware bidirectional multi-scale dilated convolution network | 10-fold CV with 90/10 train/test or 90/10 train+validation/test variants | 4-class emotion on IEMOCAP | TIM-Net*: UAR/WAR 69.00%/68.29%; TIM-Net**: 72.50%/71.65% | Lightweight baseline. Useful if compute is limited. |
| SpeechFormer | PDF downloaded; repo cloned: `code/SpeechFormer` | spectrogram, log-mel, or wav2vec features | hierarchical transformer with speech-aware local attention across frame/phoneme/word/utterance-like scales | 5,531 utterances; leave-one-session-out | 4-class emotion | best IEMOCAP setting in paper: Wav2vec + SpeechFormer-S WA 62.9%, UA 64.5% | Architecturally useful, but not the strongest IEMOCAP emotion result. |

## Paper-Only or Reference Models

| Model | Code status | Input features | Architecture | IEMOCAP split | Target labels | Reported result | Notes |
|---|---|---|---|---|---|---|---|
| Wav2Vec2/HuBERT benchmark | PDF downloaded; official repo not found in this pass | raw waveform | partial or entire fine-tuning of Wav2Vec2/HuBERT plus downstream classifier | speaker-dependent random split and speaker-independent leave-two-speaker-out 10-fold | 4-class emotion | HuBERT-large partial fine-tuning: SD WA 79.58%, SI WA 73.01% | Strong reference for SSL fine-tuning. No local code repo downloaded. |
| MSTR | PDF downloaded; official repo not found in this pass | HuBERT features | multi-scale temporal operator, fractal self-attention, scale mixer | IEMOCAP 4-class; leave-one-session-out | 4-class emotion | WA 70.60%, UA 71.60% | Interesting transformer design, but currently paper-only locally. |

## Arousal / Valence / Dominance References

| Model | Code status | Input features | Architecture | IEMOCAP split | Target labels | Reported result | How to use in our project |
|---|---|---|---|---|---|---|---|
| Attention-augmented end-to-end MTL | PDF downloaded; official repo not found | raw audio; comparison baselines use eGeMAPS/openSMILE | end-to-end model with attention and multi-task heads | train Session 1-3: 6,319; dev Session 4: 1,811; test Session 5: 1,819 | activation/arousal, valence, dominance discretized into low/mid/high | test UAR: arousal 48.5%, valence 63.8%, dominance 51.6% | Good older reference for multi-task AVD classification. |
| Contrastive unsupervised learning / preCPC | PDF downloaded; official repo not found | raw audio for CPC; LFBE baseline | CPC pretraining on LibriSpeech train-clean-100, attention aggregation, dense regression head | IEMOCAP 5-fold CV | continuous activation, valence, dominance | CCC avg 0.731; activation 0.752; valence 0.752; dominance 0.691 | Strong AVD regression reference, but no code found locally. |
| Cross-modal conditional teacher-student | PDF downloaded; official repo not found | HuBERT audio features; BERT text only used in teacher | audio-only student learns from audio+text teacher, GRU summarizer, CCC loss | 5-fold speaker-independent CV; one session eval, one validation, three train | continuous activation, valence, dominance | audio-only T/S CCC: activation 0.667, valence 0.582, dominance 0.545 | Useful if adding transcript lightly, but final student can be audio-only. |
| AV from categorical emotion labels | PDF downloaded; official repo not found | WavLM speech representations | categorical emotion classifier followed by anchor-based dimensionality reduction into AV space | speaker-independent train/test split; 4, 5, and 9 emotion configurations | arousal and valence only | 4-emotion: valence CCC 0.529, arousal CCC 0.632; 9-emotion: valence CCC 0.566, arousal CCC 0.672; with ground-truth class labels, valence rises to 0.693/0.674 | Very relevant if we want to derive AV from an emotion classifier without fully training supervised AV regression. |

## Recommended Reproduction Priority

1. Re-run or clean the current 06D Emotion2Vec + Co-Attention model on IEMOCAP 4-class.
2. Compare against emotion2vec linear downstream and FT-w2v2-ser P-TAPT.
3. Add CA-MSER as the closest co-attention reproduction reference.
4. Use DST as the stronger transformer baseline if time allows.
5. Treat AVD as an extension: first evaluate `EmoAct`, `EmoVal`, `EmoDom` with a small regression head on top of 06D/emotion2vec features; use CCC/MAE, not accuracy.

## Strict Speaker-Independent Benchmark Table for 06D

This table is the recommended comparison table for the updated 06D direction. It separates the core emotion classification task from the optional activation/arousal, valence, and dominance regression task.

| Model | Speaker-independent? | Split | Input | Emotion labels | WA | UA | Macro-F1 | Arousal / Valence / Dominance labels | AVD metric |
|---|---:|---|---|---|---:|---:|---:|---|---|
| emotion2vec linear | Yes | LOSO 5-fold / LOSpeaker reported | audio | angry, sad, neutral, happy+excited | around 71.79-72.94 | around 72.69 | reported as WF1, not always Macro-F1 | not reported in this benchmark | N/A |
| CA-MSER | Yes | LOSO / leave-one-speaker-out | audio | angry, sad, neutral, happy+excited | 69.80 / 71.64 | 71.05 / 72.70 | not reported | not reported | N/A |
| FT-w2v2 P-TAPT | Yes | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | not reported | 74.3 | not reported | not reported | N/A |
| DST | Yes | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | 71.8 | 73.6 | not reported | not reported | N/A |
| Wav2Vec2/HuBERT benchmark | Yes | speaker-independent leave-two-speaker-out 10-fold | audio | anger, happiness, sadness, neutral | 73.01 | not reported | not reported | not reported | N/A |
| 06D proposed | Yes | LOSO 5-fold | audio | angry, sad, neutral, happy+excited | to measure | to measure | to measure | optional: `EmoAct`, `EmoVal`, `EmoDom` | CCC + MAE |

## How Arousal, Valence, and Dominance Are Measured

IEMOCAP contains dimensional emotion annotations for each utterance. Human annotators rate each utterance on three continuous affect dimensions:

| Dimension | IEMOCAP name | Meaning | Low score | High score |
|---|---|---|---|---|
| Activation / Arousal | `EmoAct` | emotional energy or activation level | calm, tired, low-energy | excited, angry, highly activated |
| Valence | `EmoVal` | positive vs negative emotional tone | negative, unpleasant, sad/angry | positive, pleasant, happy |
| Dominance | `EmoDom` | perceived control, strength, or assertiveness | weak, passive, submissive | strong, confident, controlling |

The original dimensional annotations are human ratings, commonly on a 1-5 scale. If multiple annotators rate one utterance, papers commonly average their ratings to obtain the final ground-truth value.

For prediction, activation/arousal, valence, and dominance should normally be treated as regression targets:

```text
audio
-> 06D / emotion2vec / co-attention encoder
-> shared representation
-> emotion classification head: angry/sad/neutral/happy
-> AVD regression head: arousal, valence, dominance
```

Recommended metrics:

| Metric | Use for | Meaning |
|---|---|---|
| CCC | Arousal/Valence/Dominance regression | Measures whether predictions match human scores in correlation, mean, and scale. Best value is 1. |
| MAE | Arousal/Valence/Dominance regression | Average absolute difference between predicted score and human score. Lower is better. |
| RMSE/MSE | Arousal/Valence/Dominance regression | Penalizes large prediction errors. Lower is better. |

Do not report WA/UA/Macro-F1 for continuous AVD regression. Only use accuracy/UAR if the 1-5 ratings are discretized into low/mid/high classes, for example:

```text
low:    [1, 2]
middle: (2, 4)
high:   [4, 5]
```

For the updated 06D project, the safest setup is:

```text
Main task:
IEMOCAP 4-class emotion classification
Metrics: WA, UA, Macro-F1

Optional extension:
EmoAct / EmoVal / EmoDom regression
Metrics: CCC, MAE
```
