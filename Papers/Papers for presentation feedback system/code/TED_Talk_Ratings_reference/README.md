# TED Talk Ratings reference implementation

This folder contains a compact reference implementation based on:

- Paper: "Predicting TED Talk Ratings from Language and Prosody"
- arXiv: https://arxiv.org/abs/1906.03940

Important note: the arXiv paper states that the dataset and source code were made available, but the PDF version in this workspace still shows the source-code link as blinded for anonymous review. Therefore, this folder is not the official author repository. It is a paper-faithful reference scaffold for reproducing the main model ideas.

## Paper setup

Dataset summary reported by the paper:

- 2,231 TED talks
- 513.49 total hours
- 5,574,444 viewer ratings
- 14 rating categories
- manual transcripts, audio/prosody features, and metadata
- 150 randomly sampled TED talks reserved as final test subset

Main models:

1. Word Sequence LSTM
   - sentence as a sequence of GloVe word vectors
   - LSTM encodes each sentence
   - talk-level pooling predicts 14 binary rating labels

2. Dependency TreeLSTM
   - dependency parse tree per sentence
   - Child-Sum TreeLSTM
   - GloVe + POS/dependency information

3. Dependency TreeLSTM + Prosody CNN
   - forced alignment between transcript and audio
   - Praat prosody features at 10 Hz
   - 8D prosody vector: pitch, loudness, first three formants frequency/bandwidth
   - 1D-CNN produces a 64D prosody vector
   - fused with text sentence representation

4. Statistical baselines
   - LIWC language features
   - prosody summary statistics
   - narrative trajectory features
   - Linear SVM and LASSO/Ridge-style linear classifier

## How this folder maps to the project

The final project should not reproduce full TED raw-audio training first. A practical path is:

```text
TED transcript/statistical baseline
-> Word Sequence LSTM
-> add prosody features if available
-> add 06D emotion posterior and PuSQ-style pause/speech features
-> window-level feedback for 5-minute presentation audio
```

Recommended first experiment:

```text
text features + prosody summary + optional 06D emotion posterior
-> multi-label classifier
-> 14 TED rating categories or reduced quality labels
```

## Files

- `models_ted_talk_ratings.py`: PyTorch model components for Word Sequence LSTM, Prosody CNN, and text/prosody fusion.
- `train_ted_reference.py`: training skeleton with expected batch schema.
- `feature_schema.md`: expected feature/data layout.
- `source_links.md`: source papers/tools/repos.

