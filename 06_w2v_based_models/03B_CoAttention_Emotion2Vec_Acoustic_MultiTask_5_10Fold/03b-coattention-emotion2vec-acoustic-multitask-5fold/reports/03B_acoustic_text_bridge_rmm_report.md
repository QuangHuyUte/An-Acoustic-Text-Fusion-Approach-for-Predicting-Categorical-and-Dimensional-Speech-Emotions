# 03B Acoustic + Transcript Bridge Fusion Report

## Config

- `TEXT_MODEL_NAME`: `roberta-base`
- `TEXT_MODEL_PATH`: ``
- `TEXT_MODEL_SOURCE`: `/kaggle/input/datasets/linhlaz225/robert/roberta-base-kaggle`
- `ALLOW_HF_DOWNLOAD`: `False`
- `FUSION_MODE`: `acoustic_text_bridge_rmm`
- `RUN_PROTOCOLS`: `['5fold_session']`
- `MAX_FOLDS`: `0`
- `EPOCHS`: `70`
- `BATCH_SIZE`: `12`
- `HIDDEN_DIM`: `256`
- `NUM_HEADS`: `8`
- `BRIDGE_TOKENS`: `16`
- `LR`: `0.0005`
- `TEXT_LR`: `0.0`
- `USE_RMM`: `True`
- `RMM_START`: `0.2`
- `RMM_TEXT_PROB`: `0.9`
- `RMM_WARMUP_EPOCHS`: `5`
- `USE_BALANCED_SAMPLER`: `False`
- `CE_WEIGHT`: `0.35`
- `CCC_WEIGHT`: `0.45`
- `MSE_WEIGHT`: `0.2`
- `VAD_VAR_WEIGHT`: `0.05`
- `PRIMARY_UAR_WEIGHT`: `0.4`
- `PRIMARY_CCC_WEIGHT`: `0.6`
- `VAD_REPRESENTATION`: `acoustic_heavy`
- `VAD_RAW_MIN`: `1.0`
- `VAD_RAW_MAX`: `5.0`
- `LOG_EVERY_STEPS`: `25`
- `SCALER_CHUNK_SIZE`: `128`
- `OUTPUT_DIR`: `/kaggle/working/output_03b_acoustic_text_bridge_fusion`

## Summary

| protocol      | fusion_mode              |   folds |   WA_mean |    WA_std | WA_paper       |   UAR_mean |   UAR_std | UAR_paper      |   Macro_F1_mean |   Macro_F1_std | Macro_F1_paper   |   CCC_mean_mean |   CCC_mean_std | CCC_mean_paper   |   CCC_valence_mean |   CCC_valence_std | CCC_valence_paper   |   CCC_arousal_mean |   CCC_arousal_std | CCC_arousal_paper   |   CCC_dominance_mean |   CCC_dominance_std | CCC_dominance_paper   |   MAE_mean_mean |   MAE_mean_std | MAE_mean_paper    |
|:--------------|:-------------------------|--------:|----------:|----------:|:---------------|-----------:|----------:|:---------------|----------------:|---------------:|:-----------------|----------------:|---------------:|:-----------------|-------------------:|------------------:|:--------------------|-------------------:|------------------:|:--------------------|---------------------:|--------------------:|:----------------------|----------------:|---------------:|:------------------|
| 5fold_session | acoustic_text_bridge_rmm |       5 |  0.697231 | 0.0261168 | 69.72 +/- 2.61 |   0.703541 |  0.025264 | 70.35 +/- 2.53 |        0.695842 |      0.0252322 | 69.58 +/- 2.52   |         0.63859 |       0.036022 | 0.639 +/- 0.036  |           0.658207 |         0.0526178 | 0.658 +/- 0.053     |           0.701402 |         0.0413074 | 0.701 +/- 0.041     |             0.556161 |           0.0542573 | 0.556 +/- 0.054       |        0.548012 |      0.0578835 | 0.5480 +/- 0.0579 |

## Ablation / Reference

| Model                       | Input                                          | Purpose                                                  | How to run                           |
|:----------------------------|:-----------------------------------------------|:---------------------------------------------------------|:-------------------------------------|
| 03B acoustic-only old logic | 06D acoustic + emotion2vec                     | Ablation baseline                                        | FUSION_MODE=acoustic_only            |
| 03B proposed bridge + RMM   | 06D acoustic + emotion2vec + transcript        | Main proposed lightweight multimodal model               | FUSION_MODE=acoustic_text_bridge_rmm |
| Paper 2024 multimodal       | HuBERT-large + DeBERTaV3-large / RoBERTa-large | Reference architecture, heavier full reproduction in 03D | Use 03D                              |

## Interpretation Guide

- So sánh `acoustic_only` với `acoustic_text_bridge_rmm` để kiểm tra transcript có bổ sung không.
- So sánh `acoustic_text_concat` với `acoustic_text_bridge` để kiểm tra bridge-token fusion có tốt hơn nối vector không.
- So sánh `acoustic_text_bridge` với `acoustic_text_bridge_rmm` để kiểm tra RMM có làm model cân bằng modality hơn không.

## Quick Read

- VAD đã được train ở thang normalized 0-1 và prediction được xuất ngược về thang 1-5.
- `VAD_pred_std_mean` trong results/predictions không còn collapse về 0, nên lỗi scale VAD trước đó đã được xử lý.
