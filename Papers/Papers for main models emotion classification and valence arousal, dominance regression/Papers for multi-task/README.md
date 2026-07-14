# Papers for Multi-Task Emotion and Affective Attribute Modeling

This folder groups the papers used to justify and compare two design options:

1. Separate models for emotion classification and AVD regression.
2. One shared encoder with task-specific heads:
   - classification head for categorical emotion;
   - regression head for arousal/valence/dominance or arousal/valence.

These papers are stored inside the main folder:

```text
Papers/Papers for main models emotion classification and valence arousal, dominance regression
```

## Included Papers

| Local file | Paper link | Main idea | Why it matters |
|---|---|---|---|
| `2401.00536_multitask_multimodal_categorical_dimensional_emotions.pdf` | https://arxiv.org/abs/2401.00536 | Multi-task model predicting categorical emotion, valence, and arousal. Uses classifier + regressor heads. | Closest reference for the proposed two-head idea. |
| `1903.12424_attention_augmented_multitask_AVD.pdf` | https://arxiv.org/abs/1903.12424 | Speech-based attention-augmented multi-task learning for arousal, valence, dominance. | Supports the idea that related affective tasks can share representations. |
| `2112.00158_cross_modal_teacher_student_AVD.pdf` | https://arxiv.org/abs/2112.00158 | Shared speech representation predicts activation, valence, dominance using CCC loss. | Useful reference for AVD regression head and CCC-based loss. |
| `2311.14816_AV_from_categorical_emotion_labels.pdf` | https://arxiv.org/abs/2311.14816 | Learns arousal-valence representation from categorical emotion labels. | Supports the link between categorical emotion classification and dimensional affect prediction. |

## Code Status

As of the current check, official runnable repositories were not found for these four papers.

Therefore, these papers should be used mainly for:

- architectural justification;
- loss design;
- metric design;
- report discussion;
- ablation design.

They should not be treated as ready-to-run baselines unless official or reliable third-party code is found later.

## Practical Use for This Project

Recommended ablation:

```text
Emotion-only 06D
AVD-only regressor
06D shared encoder + emotion classification head + AVD regression head
```

Decision rule:

```text
If multi-task keeps or improves emotion performance and obtains acceptable CCC/MAE,
use the two-head model as an extension.

If multi-task lowers emotion performance significantly,
keep emotion-only 06D as the main model and report AVD as a separate experiment.
```

