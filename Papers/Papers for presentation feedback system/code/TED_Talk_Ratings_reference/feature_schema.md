# Feature schema for TED Talk Ratings experiments

This schema is designed to connect the TED paper with the current 06D + PuSQ project.

## Talk-level target

Each TED talk has 14 binary rating labels after median-threshold binarization.

Example labels:

```text
beautiful
confusing
courageous
fascinating
funny
informative
ingenious
inspiring
jaw-dropping
long-winded
obnoxious
ok
persuasive
unconvincing
```

The exact label names should be checked from the released TED dataset or re-created from TED metadata.

## Text input

For the Word Sequence LSTM:

```text
talk_id
sentences:
  sentence_1: [token_id_1, token_id_2, ...]
  sentence_2: [token_id_1, token_id_2, ...]
labels: 14D binary vector
```

Practical simplification:

```text
word embeddings -> sentence LSTM -> masked mean over sentences -> 14D sigmoid output
```

## Prosody input

The TED paper uses forced alignment and Praat. Each sentence has a prosody sequence:

```text
time_step x 8
```

The 8 dimensions are:

```text
pitch
loudness
formant_1_frequency
formant_1_bandwidth
formant_2_frequency
formant_2_bandwidth
formant_3_frequency
formant_3_bandwidth
```

Each signal is normalized per video.

## 06D/PuSQ extension features

For the project extension, add segment/window features:

```text
emotion posterior from 06D:
  neutral, happy, sad, angry, fear, disgust

prosody/fluency:
  pitch_mean
  pitch_std
  energy_mean
  energy_std
  silence_ratio
  long_pause_count
  speech_ratio
  speaking_rate

optional dimensional affect:
  valence
  arousal
  dominance
```

For long audio:

```text
3-5 second segment
-> 30-60 second block
-> block-level feedback
-> whole-talk aggregate
```

## Why this matters

TED labels are whole-talk ratings. They do not directly label which timestamp is good or bad. To produce useful presentation feedback, the project should preserve local features and generate timeline-based feedback, instead of only predicting one global score.

