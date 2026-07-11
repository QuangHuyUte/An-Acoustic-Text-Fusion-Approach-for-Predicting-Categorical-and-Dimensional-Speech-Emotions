# Main SER and Affective Attribute Models

This folder collects papers and code repositories for the revised project scope:

- Core task: speech emotion recognition on IEMOCAP.
- Extended task: activation/arousal, valence, and dominance prediction when labels are available.
- Main local reference: the existing 06D Emotion2Vec + Co-Attention SER notebook.

## Downloaded Code Repositories

| Local folder | Upstream repo | Main use |
|---|---|---|
| `code/emotion2vec` | https://github.com/ddlBoJack/emotion2vec | Emotion2Vec representation and IEMOCAP downstream scripts. |
| `code/FT-w2v2-ser` | https://github.com/b04901014/FT-w2v2-ser | Wav2Vec2 fine-tuning, TAPT, and P-TAPT baseline on IEMOCAP. |
| `code/CA-MSER_co_attention` | https://github.com/Vincent-ZHQ/CA-MSER | Co-attention fusion of MFCC, spectrogram, and wav2vec2 embeddings. |
| `code/DST_deformable_speech_transformer` | https://github.com/HappyColor/DST | Deformable Speech Transformer with WavLM features. |
| `code/SpeechFormer` | https://github.com/HappyColor/SpeechFormer | Hierarchical speech transformer baseline. |
| `code/TIM-Net_SER` | https://github.com/Jiaxin-Ye/TIM-Net_SER | Lightweight temporal-aware bidirectional multi-scale network. |

## Downloaded Papers

| File | Paper |
|---|---|
| `papers/2312.15185_emotion2vec.pdf` | emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation |
| `papers/2110.06309_wav2vec2_finetuning_SER.pdf` | Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition |
| `papers/2203.15326_co_attention_multilevel_acoustic.pdf` | Speech Emotion Recognition with Co-Attention based Multi-level Acoustic Information |
| `papers/2302.13729_DST_deformable_speech_transformer.pdf` | DST: Deformable Speech Transformer for Emotion Recognition |
| `papers/2211.08233_TIM-Net_SER.pdf` | Temporal Modeling Matters: TIM-Net |
| `papers/2203.03812_SpeechFormer.pdf` | SpeechFormer |
| `papers/2111.02735_wav2vec2_HuBERT_benchmark.pdf` | Fine-tuned Wav2Vec2/HuBERT benchmark |
| `papers/2410.00390_MSTR_multi_scale_temporal_transformer.pdf` | Multi-Scale Temporal Transformer for SER |
| `papers/1903.12424_attention_augmented_multitask_AVD.pdf` | Attention-Augmented End-to-End Multi-Task Learning for AVD prediction |
| `papers/2102.06357_contrastive_unsupervised_AVD_SER.pdf` | Contrastive Unsupervised Learning for SER and emotion primitives |
| `papers/2112.00158_cross_modal_teacher_student_AVD.pdf` | Cross-modal conditional teacher-student learning for AVD |
| `papers/2311.14816_AV_from_categorical_emotion_labels.pdf` | Learning arousal-valence from categorical emotion labels |

Extracted `.txt` files are stored beside each PDF to make later search and summarization easier.

## Notes

- IEMOCAP has about 10,039 utterances overall.
- The most common SER benchmark subset uses 5,531 utterances from 4 categories: angry, sad, neutral, and happy, with excited merged into happy.
- IEMOCAP also contains dimensional affect ratings: activation/arousal, valence, and dominance, commonly on a 1-5 scale.
- Emotion classification uses WA/UA/Macro-F1. Arousal/valence/dominance regression usually uses CCC, MAE, or MSE. Some older AVD work discretizes the 1-5 scale into low/mid/high classes and reports UAR.
