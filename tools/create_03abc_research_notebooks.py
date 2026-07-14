from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
BASE = ROOT / "06_w2v_based_models"

NB_03A = BASE / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
NB_03B = BASE / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
NB_03C = BASE / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def remove_auto_cells(nb):
    markers = ["AUTO-03ABC-REFERENCE", "AUTO-03ABC-PAPER-RESULTS"]
    nb.cells = [
        cell for cell in nb.cells
        if not any(marker in "".join(cell.get("source", "")) for marker in markers)
    ]


REFERENCE_03A = r"""
<!-- AUTO-03ABC-REFERENCE -->
## Nhánh 03A đang kiểm chứng điều gì?

03A là nhánh **pretrained speech / raw-audio backbone**. Khác với 03B, nhánh này không đọc MFCC/log-Mel đã trích sẵn, mà đưa waveform vào một backbone tự giám sát đã học từ lượng lớn audio.

Mục tiêu của nhánh này:

| Câu hỏi | Cách notebook trả lời |
|---|---|
| Pretrained speech model có tốt hơn acoustic thủ công không? | Train 5-fold và 10-fold trên cùng split với 03B/03C |
| Có học được cả emotion class và VAD không? | Dùng 2 head: classification + regression |
| Có thể đưa sang fusion 04 không? | Xuất embedding, emotion probability và VAD prediction cho từng fold |
| Có tránh leakage không? | Mỗi fold train model riêng, test chỉ được dùng sau khi chọn best epoch bằng validation |

### Paper và mô hình tham khảo cho 03A

| Paper / model | Input | Split / task | Kiến trúc sơ bộ | Kết quả tham khảo | Ý nghĩa với 03A |
|---|---|---|---|---|---|
| [Chen & Rudnicky, 2021 - Exploring Wav2Vec2 fine-tuning for SER](https://arxiv.org/abs/2110.06309) | Raw waveform | IEMOCAP 4-class SER, leave-one-session style | wav2vec2 backbone, fine-tuning, TAPT/P-TAPT | P-TAPT báo cáo UA khoảng 74.3 trên IEMOCAP | Cho thấy fine-tune pretrained speech backbone là baseline mạnh hơn handcrafted-only |
| [emotion2vec, 2023](https://arxiv.org/abs/2312.15185) | Raw speech / speech representation | SER downstream | self-supervised emotion representation + downstream classifier | Paper nhấn mạnh representation học cảm xúc dùng được cho nhiều SER benchmarks | Lý do ta muốn có nhánh Emotion2Vec/pretrained speech riêng |
| [Ispas et al., 2024 - Multi-task multi-modal categorical + dimensional emotions](https://arxiv.org/abs/2401.00536) | Audio + transcript | IEMOCAP 10-fold speaker guideline | HuBERT-large + DeBERTaV3-large, self-attention, bridge tokens, multi-task heads | UAR 74.68, WAR 74.69, Valence CCC 0.738, Arousal CCC 0.685 | Cho thấy multi-task categorical + dimensional emotion là hướng hợp lý |

### Kiến trúc 03A

```mermaid
flowchart LR
    A["Raw waveform 16 kHz"] --> B["Pretrained speech backbone<br/>WavLM / wav2vec2 / Emotion2Vec-compatible"]
    B --> C["Frame hidden states"]
    C --> D["Attentive statistics pooling<br/>mean + std có trọng số"]
    D --> E["Shared projection"]
    E --> F["Emotion head<br/>4-class softmax"]
    E --> G["VAD regression head<br/>valence/arousal/dominance"]
    E --> H["Fusion embedding for notebook 04"]
```

### Vì sao dùng attentive statistics pooling?

Mean pooling thường làm mất đoạn cảm xúc ngắn nhưng quan trọng. Attentive statistics pooling học trọng số theo frame, rồi lấy cả trung bình và độ lệch chuẩn có trọng số. Vì vậy embedding giữ được cả mức cảm xúc trung bình và mức biến thiên theo thời gian.
"""


REFERENCE_03B = r"""
<!-- AUTO-03ABC-REFERENCE -->
## Nhánh 03B đang kiểm chứng điều gì?

03B là nhánh **06D full acoustic nâng cấp**. Nhánh này giữ lại phần mạnh của notebook 06D: nhiều nhóm đặc trưng acoustic có thể diễn giải, nhưng tổ chức lại thành nhiều branch thay vì ép tất cả thành một vector phẳng.

### Paper và mô hình tham khảo cho 03B

| Paper / model | Input | Split / task | Kiến trúc sơ bộ | Kết quả tham khảo | Ý nghĩa với 03B |
|---|---|---|---|---|---|
| [TIM-Net, 2022 - Temporal Modeling Matters](https://arxiv.org/abs/2211.08233) / [GitHub](https://github.com/Jiaxin-Ye/TIM-Net_SER) | Acoustic temporal features | SER trên nhiều dataset, gồm IEMOCAP | temporal-aware block, bidirectional temporal modeling, multi-scale dilation | Paper báo cáo TIM-Net cải thiện trung bình UAR/WAR so với model đứng thứ hai trên nhiều corpus | Lý do nâng temporal branch từ CNN/BiLSTM đơn giản sang multi-scale temporal branch |
| [Ispas et al., 2024](https://arxiv.org/abs/2401.00536) | Audio + transcript | IEMOCAP 10-fold speaker | self-attention, cross-attention, bridge tokens, multi-task heads | UAR 74.68, WAR 74.69, Valence CCC 0.738, Arousal CCC 0.685 | Lý do dùng multi-task 2-head và attention/fusion |
| [emotion2vec, 2023](https://arxiv.org/abs/2312.15185) | Speech representation | SER downstream | pretrained emotion representation + simple downstream head | Representation-based SER mạnh trên nhiều benchmark | Dùng `X_e2v` làm query hướng dẫn acoustic branch |

### Kiến trúc 03B

```mermaid
flowchart LR
    A["X_temporal<br/>MFCC/delta/F0/RMS/ZCR/spectral"] --> B["TIM-Net-style temporal branch"]
    C["X_spectral<br/>log-Mel + delta + delta-delta"] --> D["Residual SE Conv2D branch"]
    E["X_stats<br/>functionals/chroma/tonnetz"] --> F["Stats MLP"]
    G["X_e2v<br/>Emotion2Vec embedding"] --> H["Emotion2Vec adapter"]
    B --> I["Acoustic tokens"]
    D --> I
    F --> I
    H --> J["Emotion2Vec-guided co-attention"]
    I --> J
    J --> K["Gated fusion"]
    K --> L["Emotion head"]
    K --> M["VAD regression head"]
    K --> N["Fusion embedding for notebook 04"]
```

### Vì sao 03B vẫn cần dù có 03A?

03A có pretrained representation mạnh nhưng khó giải thích. 03B giữ các tín hiệu acoustic như pitch, energy, speaking dynamics, spectral shape. Nếu 03B riêng thấp hơn 03A nhưng fusion 04 tăng, nghĩa là acoustic thủ công vẫn đóng vai trò bổ sung.
"""


PAPER_RESULTS_CODE = r"""
# AUTO-03ABC-PAPER-RESULTS
def _format_mean_std(df, metric_cols):
    rows = []
    for protocol, group in df.groupby("protocol"):
        row = {"Protocol": protocol, "Folds": int(group["fold"].nunique())}
        for metric in metric_cols:
            if metric in group.columns:
                scale = 100.0 if metric in ["WA", "UAR", "Macro_F1", "Weighted_F1"] else 1.0
                row[metric] = f"{group[metric].mean() * scale:.2f} ± {group[metric].std(ddof=0) * scale:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)

if "results_df" in globals() and len(results_df) > 0:
    metric_cols = ["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]
    paper_style = _format_mean_std(results_df, metric_cols)
    paper_style_path = REPORT_DIR / "paper_style_results.csv"
    paper_style.to_csv(paper_style_path, index=False, encoding="utf-8-sig")
    display(paper_style)

    reference_rows = [
        {
            "Model": "Wav2Vec2 P-TAPT (Chen & Rudnicky, 2021)",
            "Modality": "audio",
            "Split": "IEMOCAP session-level",
            "WA/WAR": "",
            "UAR/UA": "74.30",
            "CCC V": "",
            "CCC A": "",
            "CCC D": "",
            "Note": "Pretrained speech fine-tuning reference for 03A",
        },
        {
            "Model": "Ispas et al. 2024 multi-task multi-modal",
            "Modality": "audio + transcript",
            "Split": "10-fold speaker",
            "WA/WAR": "74.69",
            "UAR/UA": "74.68",
            "CCC V": "0.738",
            "CCC A": "0.685",
            "CCC D": "",
            "Note": "Reference for multi-task + bridge/cross-attention",
        },
        {
            "Model": "Proposed notebook branch",
            "Modality": config.get("notebook", "current branch") if "config" in globals() else "current branch",
            "Split": "5-fold + 10-fold",
            "WA/WAR": "; ".join([f"{r['Protocol']}: {r.get('WA', '')}" for _, r in paper_style.iterrows()]),
            "UAR/UA": "; ".join([f"{r['Protocol']}: {r.get('UAR', '')}" for _, r in paper_style.iterrows()]),
            "CCC V": "; ".join([f"{r['Protocol']}: {r.get('CCC_valence', '')}" for _, r in paper_style.iterrows()]),
            "CCC A": "; ".join([f"{r['Protocol']}: {r.get('CCC_arousal', '')}" for _, r in paper_style.iterrows()]),
            "CCC D": "; ".join([f"{r['Protocol']}: {r.get('CCC_dominance', '')}" for _, r in paper_style.iterrows()]),
            "Note": "Direct result from this notebook; compare carefully because modality/split can differ from papers",
        },
    ]
    comparison_df = pd.DataFrame(reference_rows)
    comparison_path = REPORT_DIR / "paper_reference_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    display(comparison_df)

    if "plt" in globals() and plt is not None:
        plot_metrics = [m for m in ["WA", "UAR", "Macro_F1", "CCC_mean"] if m in results_df.columns]
        melted = results_df.melt(id_vars=["protocol", "fold"], value_vars=plot_metrics, var_name="metric", value_name="value")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        if "sns" in globals() and sns is not None:
            try:
                sns.barplot(data=melted, x="metric", y="value", hue="protocol", errorbar="sd", ax=ax)
            except TypeError:
                sns.barplot(data=melted, x="metric", y="value", hue="protocol", ci="sd", ax=ax)
        else:
            for i, metric in enumerate(plot_metrics):
                vals = [results_df.loc[results_df["protocol"] == p, metric].mean() for p in results_df["protocol"].unique()]
                ax.bar([i + j * 0.25 for j in range(len(vals))], vals, width=0.22)
            ax.set_xticks(range(len(plot_metrics)), plot_metrics)
        ax.set_title("Paper-style metric summary by protocol")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig_path = FIGURE_DIR / "paper_style_metric_summary.png"
        fig.savefig(fig_path, dpi=160)
        plt.show()
        print("Saved:", fig_path)

        history_files = sorted(REPORT_DIR.glob("*_history.csv"))
        if history_files:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for hp in history_files:
                h = pd.read_csv(hp)
                label = hp.stem.replace("_history", "")
                if "epoch" in h.columns:
                    if "train_loss" in h.columns:
                        axes[0].plot(h["epoch"], h["train_loss"], alpha=0.45)
                    if "val_UAR" in h.columns:
                        axes[1].plot(h["epoch"], h["val_UAR"], alpha=0.45)
                    if "val_CCC_mean" in h.columns:
                        axes[2].plot(h["epoch"], h["val_CCC_mean"], alpha=0.45)
            axes[0].set_title("Train loss by fold")
            axes[1].set_title("Validation UAR by fold")
            axes[2].set_title("Validation CCC mean by fold")
            for ax in axes:
                ax.set_xlabel("epoch")
                ax.grid(alpha=0.25)
            fig.tight_layout()
            fig_path = FIGURE_DIR / "training_curves_by_fold.png"
            fig.savefig(fig_path, dpi=160)
            plt.show()
            print("Saved:", fig_path)

    manifest_rows = []
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            relative = path.relative_to(OUTPUT_DIR)
            manifest_rows.append({
                "relative_path": str(relative).replace("\\", "/"),
                "folder": relative.parts[0] if relative.parts else "",
                "bytes": int(path.stat().st_size),
            })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(REPORT_DIR / "output_manifest.csv", index=False, encoding="utf-8-sig")
    display(manifest_df)
else:
    print("Chưa có results_df. Hãy chạy cell train trước khi tạo bảng paper-style.")
"""


def patch_existing_notebook(path: Path, reference_md: str):
    nb = nbf.read(path, as_version=4)
    remove_auto_cells(nb)
    nb.cells.insert(1, md(reference_md))
    insert_at = len(nb.cells)
    for idx, cell in enumerate(nb.cells):
        src = "".join(cell.get("source", ""))
        if cell.cell_type == "code" and src.lstrip().startswith("config = {"):
            insert_at = idx
            break
    nb.cells.insert(insert_at, md("""<!-- AUTO-03ABC-PAPER-RESULTS -->\n## Bảng kết quả, biểu đồ và manifest output\n\nSau khi train xong, cell dưới đây gom kết quả thành bảng kiểu paper, tạo biểu đồ training/validation và ghi manifest các file output để dễ upload sang notebook 04."""))
    nb.cells.insert(insert_at + 1, code(PAPER_RESULTS_CODE))
    nbf.write(nb, path)


def build_03c():
    NB_03C.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    cells.append(md(r"""
# 03C - Transcript Pretrained Text Multi-Task SER 5-Fold + 10-Fold

Notebook này tạo nhánh **transcript pretrained text** cho cùng bài toán IEMOCAP:

```text
Transcript
  -> pretrained text backbone BERT/DeBERTa/RoBERTa
  -> attentive pooling
  -> shared embedding
  -> two heads: emotion classification + VAD regression
  -> fusion_features cho notebook 04
```

03C không thay thế 03A/03B. Nó trả lời câu hỏi: **chỉ dùng transcript thì model học được bao nhiêu tín hiệu cảm xúc**, và liệu text embedding có bổ sung được cho speech/acoustic trong bước fusion hay không.
"""))
    cells.append(md(r"""
<!-- AUTO-03ABC-REFERENCE -->
## Nghiên cứu tham khảo cho nhánh transcript

| Paper / model | Input | Split / task | Kiến trúc sơ bộ | Kết quả tham khảo | Ý nghĩa với 03C |
|---|---|---|---|---|---|
| [Ispas et al., 2024 - Multi-task, Multi-modal categorical + dimensional emotion](https://arxiv.org/abs/2401.00536) | Audio + transcript | IEMOCAP 10-fold speaker guideline | HuBERT-large + DeBERTaV3-large, self-attention, bridge tokens, multi-task heads | UAR 74.68, WAR 74.69, Valence CCC 0.738, Arousal CCC 0.685 | Cho thấy transcript backbone rất đáng dùng trong multi-task/fusion |
| [Zhao et al., 2022 - Multi-level Fusion of Wav2Vec2 and BERT](https://arxiv.org/abs/2207.04697) | Speech + BERT text | IEMOCAP multimodal SER | wav2vec2 + BERT, co-attention/late fusion, multi-granularity speech embeddings | Paper báo cáo vượt baseline UA trên IEMOCAP | Cho thấy text pretrained model thường tăng kết quả khi fusion với speech |
| [Padi et al., 2022 - Transfer learning from speaker recognition and BERT](https://arxiv.org/abs/2202.08974) | Audio + BERT text | IEMOCAP emotion recognition | ResNet audio transfer + fine-tuned BERT text + late fusion | Paper cho thấy audio/text đều cải thiện và fusion tốt hơn | Gợi ý train text branch riêng rồi late fusion là hướng hợp lý |
| [Siriwardhana et al., 2020 - jointly fine-tuning BERT-like SSL models](https://arxiv.org/abs/2008.06682) | Speech + text | IEMOCAP / CMU-MOSEI / CMU-MOSI | modality-specific SSL backbones, joint fine-tuning/fusion | Báo cáo SSL text/speech giúp multimodal SER | Củng cố việc dùng pretrained transcript branch |

Lưu ý: các kết quả paper không luôn so sánh 1-1 vì khác label mapping, số mẫu, split, và có/không dùng visual/audio. Trong project này, 03C phải được so sánh trực tiếp bằng **cùng 5-fold và 10-fold** với 03A/03B.
"""))
    cells.append(md(r"""
## Kiến trúc 03C

```mermaid
flowchart LR
    A["Transcript text"] --> B["Tokenizer"]
    B --> C["Pretrained text backbone<br/>DeBERTa/BERT/RoBERTa"]
    C --> D["CLS pooling"]
    C --> E["Masked mean pooling"]
    C --> F["Attentive pooling"]
    D --> G["Concatenate"]
    E --> G
    F --> G
    G --> H["Shared projection"]
    H --> I["Emotion head<br/>4-class softmax"]
    H --> J["VAD regression head<br/>valence/arousal/dominance"]
    H --> K["Fusion embedding for notebook 04"]
```

Text branch có ưu điểm là bắt nội dung ngữ nghĩa: lời nói có từ ngữ buồn, tức giận, vui, phủ định, cường điệu. Nhưng nó không biết giọng nói có năng lượng cao/thấp, pitch hay pause, nên 03C nên được xem là nhánh bổ sung cho 03A/03B.
"""))
    cells.append(md(r"""
## Metric và loss

Emotion classification dùng:

$$
\operatorname{WA} = \frac{\sum_i \mathbf{1}(\hat{y}_i=y_i)}{N}
$$

$$
\operatorname{UAR} = \frac{1}{C}\sum_{c=1}^{C}\frac{TP_c}{TP_c+FN_c}
$$

VAD regression dùng CCC:

$$
\rho_c = \frac{2\sigma_{xy}}{\sigma_x^2+\sigma_y^2+(\mu_x-\mu_y)^2}
$$

Loss multi-task:

$$
\mathcal{L}
= \lambda_{CE}\mathcal{L}_{CE}
+ \lambda_{MSE}\mathcal{L}_{MSE}
+ \lambda_{CCC}(1-\operatorname{CCC})
$$

Trong notebook, VAD được chuẩn hóa từ thang IEMOCAP 1-5 về 0-1 khi train, rồi đổi lại 1-5 khi tính MAE/RMSE/CCC để dễ đọc.
"""))
    cells.append(md(r"""
## Input cần upload

Notebook 03C cần output từ notebook 01/02:

```text
metadata/iemocap_4class_avd_metadata.csv
splits/iemocap_5fold_session_long.csv
splits/iemocap_10fold_speaker_long.csv
text/text_ready_metadata.csv        # nếu có từ notebook 02
text/text_folds_long.csv            # nếu có từ notebook 02
```

Nếu không có `text_ready_metadata.csv`, notebook sẽ tự dùng `transcript` trong metadata/split. Không cần raw audio.
"""))
    cells.append(code(r"""
import os
import sys
import json
import time
import math
import zipfile
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 160)
pd.set_option("display.width", 180)

INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"
if INSTALL_DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece", "accelerate"])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
print("CUDA devices:", torch.cuda.device_count())
"""))
    cells.append(code(r"""
LOCAL_PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")

def unique_existing(paths):
    out, seen = [], set()
    for item in paths:
        if not item:
            continue
        path = Path(item)
        if path.exists():
            key = str(path.resolve()).lower()
            if key not in seen:
                seen.add(key)
                out.append(path.resolve())
    return out

def search_roots():
    roots = [
        os.getenv("IEMOCAP_DATA_DIR"),
        os.getenv("KAGGLE_INPUT_DIR"),
        Path.cwd(),
        Path.cwd().parent,
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        LOCAL_PROJECT / "06_w2v_based_models" / "data",
        LOCAL_PROJECT / "06_w2v_based_models",
    ]
    return unique_existing(roots)

def find_named_file(filename, env_var=None, description=None, required=True):
    if env_var and os.getenv(env_var):
        candidate = Path(os.getenv(env_var))
        if candidate.exists():
            return candidate.resolve()
    candidates = []
    for root in search_roots():
        for rel in [
            filename,
            Path("data") / filename,
            Path("metadata") / filename,
            Path("splits") / filename,
            Path("text") / filename,
            Path("output") / filename,
            Path("output") / "metadata" / filename,
            Path("output") / "splits" / filename,
            Path("output") / "text" / filename,
        ]:
            candidates.append(root / rel)
        try:
            candidates.extend(root.rglob(filename))
        except Exception:
            pass
    existing = sorted({p.resolve() for p in candidates if p.exists() and p.is_file()}, key=lambda p: (len(p.parts), str(p).lower()))
    if existing:
        return existing[0]
    if required:
        roots_text = "\n".join(f"- {r}" for r in search_roots())
        raise FileNotFoundError(f"Không tìm thấy {description or filename}. Đã quét:\n{roots_text}")
    return None

TEXT_READY_PATH = find_named_file("text_ready_metadata.csv", env_var="IEMOCAP_TEXT_READY_PATH", required=False)
TEXT_FOLDS_PATH = find_named_file("text_folds_long.csv", env_var="IEMOCAP_TEXT_FOLDS_PATH", required=False)
METADATA_PATH = find_named_file("iemocap_4class_avd_metadata.csv", env_var="IEMOCAP_METADATA_PATH", description="metadata IEMOCAP")
SPLIT_5FOLD_PATH = find_named_file("iemocap_5fold_session_long.csv", env_var="IEMOCAP_5FOLD_SPLIT_PATH", description="5-fold split")
SPLIT_10FOLD_PATH = find_named_file("iemocap_10fold_speaker_long.csv", env_var="IEMOCAP_10FOLD_SPLIT_PATH", description="10-fold split")

print("TEXT_READY_PATH:", TEXT_READY_PATH)
print("TEXT_FOLDS_PATH:", TEXT_FOLDS_PATH)
print("METADATA_PATH:", METADATA_PATH)
print("SPLIT_5FOLD_PATH:", SPLIT_5FOLD_PATH)
print("SPLIT_10FOLD_PATH:", SPLIT_10FOLD_PATH)
"""))
    cells.append(code(r"""
RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()
IS_TUNE_MODE = RUN_MODE != "full"

TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "roberta-base")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "96"))
EPOCHS = int(os.getenv("EPOCHS", "6" if not IS_TUNE_MODE else "3"))
PATIENCE = int(os.getenv("PATIENCE", "3"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.001"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "12"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "1"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

LR_BACKBONE = float(os.getenv("LR_BACKBONE", "1.5e-5"))
LR_HEAD = float(os.getenv("LR_HEAD", "3e-4"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.08"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
DROPOUT = float(os.getenv("DROPOUT", "0.25"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
USE_AMP = os.getenv("USE_AMP", "1") == "1"
USE_DATA_PARALLEL = os.getenv("USE_DATA_PARALLEL", "1") == "1" and torch.cuda.device_count() > 1

RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker" if not IS_TUNE_MODE else "5fold_session").split(",") if x.strip()]
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0" if not IS_TUNE_MODE else "1"))
EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "1" if not IS_TUNE_MODE else "0") == "1"
SEED = int(os.getenv("SEED", "42"))

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03c_transcript_text_branch")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print({
    "RUN_MODE": RUN_MODE,
    "TEXT_MODEL_NAME": TEXT_MODEL_NAME,
    "MAX_LENGTH": MAX_LENGTH,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LR_BACKBONE": LR_BACKBONE,
    "LR_HEAD": LR_HEAD,
    "RUN_PROTOCOLS": RUN_PROTOCOLS,
    "MAX_FOLDS": MAX_FOLDS,
    "USE_DATA_PARALLEL": USE_DATA_PARALLEL,
    "OUTPUT_DIR": str(OUTPUT_DIR),
})
"""))
    cells.append(code(r"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_split_names(df):
    df = df.copy()
    df["split"] = df["split"].replace({"validation": "val"})
    return df

metadata = pd.read_csv(TEXT_READY_PATH if TEXT_READY_PATH is not None else METADATA_PATH)
split_5fold = normalize_split_names(pd.read_csv(SPLIT_5FOLD_PATH))
split_10fold = normalize_split_names(pd.read_csv(SPLIT_10FOLD_PATH))

if "transcript" not in metadata.columns:
    raise ValueError("Metadata/text_ready cần có cột transcript để chạy 03C.")

metadata["train_sample_id"] = metadata["train_sample_id"].astype(str)
metadata["utterance_id"] = metadata["utterance_id"].astype(str)
metadata["transcript"] = metadata["transcript"].fillna("").astype(str)
metadata["emotion_id"] = metadata["emotion_id"].astype(int)
for col in ["valence", "arousal", "dominance"]:
    metadata[col] = pd.to_numeric(metadata[col], errors="coerce")

EXPECTED_ROWS = int(os.getenv("EXPECTED_IEMOCAP_4CLASS_ROWS", "5531"))
if len(metadata) != EXPECTED_ROWS:
    raise ValueError(f"Metadata đang có {len(metadata):,} mẫu, cần bản 4-class consensus {EXPECTED_ROWS:,} mẫu.")

def prepare_split(split_df):
    split_df = split_df.copy()
    split_df["train_sample_id"] = split_df["train_sample_id"].astype(str)
    keep_cols = ["train_sample_id", "utterance_id", "transcript", "emotion_4class", "emotion_id", "valence", "arousal", "dominance", "session", "speaker_id"]
    meta_small = metadata[[c for c in keep_cols if c in metadata.columns]].drop_duplicates("train_sample_id")
    if "transcript" not in split_df.columns:
        split_df = split_df.merge(meta_small, on="train_sample_id", how="left", suffixes=("", "_meta"))
    else:
        split_df["transcript"] = split_df["transcript"].fillna("")
    for col in ["utterance_id", "emotion_4class", "emotion_id", "valence", "arousal", "dominance", "session", "speaker_id"]:
        if col not in split_df.columns and col in meta_small.columns:
            split_df = split_df.merge(meta_small[["train_sample_id", col]], on="train_sample_id", how="left")
    split_df["transcript"] = split_df["transcript"].fillna("").astype(str)
    split_df["emotion_id"] = split_df["emotion_id"].astype(int)
    return split_df

split_5fold = prepare_split(split_5fold)
split_10fold = prepare_split(split_10fold)

for name, df, n_folds in [("5fold", split_5fold, 5), ("10fold", split_10fold, 10)]:
    split_values = set(df["split"].unique())
    if split_values != {"train", "val", "test"}:
        raise ValueError(f"{name} split phải là train/val/test, hiện có {split_values}")
    if df["fold"].nunique() != n_folds:
        raise ValueError(f"{name} cần {n_folds} folds, hiện có {df['fold'].nunique()}")

display(metadata[["train_sample_id", "utterance_id", "transcript", "emotion_4class", "valence", "arousal", "dominance"]].head())
display(metadata["emotion_4class"].value_counts().rename_axis("emotion").reset_index(name="count"))
"""))
    cells.append(code(r"""
text_stats = metadata.assign(
    char_len=metadata["transcript"].str.len(),
    word_len=metadata["transcript"].str.split().map(len),
)
stats_table = text_stats.groupby("emotion_4class")[["char_len", "word_len"]].describe().round(2)
stats_table.to_csv(REPORT_DIR / "transcript_length_stats_by_emotion.csv", encoding="utf-8-sig")
display(stats_table)

if plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=metadata, x="emotion_4class", ax=axes[0]) if sns is not None else metadata["emotion_4class"].value_counts().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Emotion distribution")
    if sns is not None:
        sns.boxplot(data=text_stats, x="emotion_4class", y="word_len", ax=axes[1])
    else:
        text_stats.boxplot(column="word_len", by="emotion_4class", ax=axes[1])
    axes[1].set_title("Transcript word length by emotion")
    axes[1].set_xlabel("emotion")
    axes[1].set_ylabel("words")
    fig.tight_layout()
    fig_path = FIGURE_DIR / "transcript_dataset_overview.png"
    fig.savefig(fig_path, dpi=160)
    plt.show()
    print("Saved:", fig_path)
"""))
    cells.append(code(r"""
def vad_to_0_1(values):
    return np.clip((values.astype(np.float32) - 1.0) / 4.0, 0.0, 1.0)

def vad_from_0_1(values):
    return values.astype(np.float32) * 4.0 + 1.0

def concordance_ccc_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_true = y_true.mean(axis=0)
    mean_pred = y_pred.mean(axis=0)
    var_true = y_true.var(axis=0)
    var_pred = y_pred.var(axis=0)
    cov = ((y_true - mean_true) * (y_pred - mean_pred)).mean(axis=0)
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + eps)

def concordance_ccc_torch(pred, target, eps=1e-8):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    return (2 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)

def compute_metrics(y_true, y_pred, vad_true_norm, vad_pred_norm):
    vad_true = vad_from_0_1(np.asarray(vad_true_norm))
    vad_pred = vad_from_0_1(np.asarray(vad_pred_norm))
    ccc = concordance_ccc_np(vad_true, vad_pred)
    return {
        "WA": float(accuracy_score(y_true, y_pred)),
        "UAR": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro")),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted")),
        "CCC_valence": float(ccc[0]),
        "CCC_arousal": float(ccc[1]),
        "CCC_dominance": float(ccc[2]),
        "CCC_mean": float(np.mean(ccc)),
        "MAE_mean": float(np.mean(np.abs(vad_true - vad_pred))),
        "RMSE_mean": float(np.sqrt(np.mean((vad_true - vad_pred) ** 2))),
    }

def primary_score(metrics):
    return 0.35 * metrics["UAR"] + 0.20 * metrics["WA"] + 0.20 * metrics["Macro_F1"] + 0.25 * metrics["CCC_mean"]

def zip_output(output_dir):
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    print("Saved zip:", zip_path)
    return zip_path
"""))
    cells.append(code(r"""
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

class TranscriptDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "text": str(row["transcript"]),
            "emotion_id": int(row["emotion_id"]),
            "vad": vad_to_0_1(row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)),
            "train_sample_id": str(row["train_sample_id"]),
            "utterance_id": str(row["utterance_id"]),
        }

def collate_text(batch):
    texts = [b["text"] if b["text"].strip() else "[EMPTY]" for b in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc["emotion_id"] = torch.tensor([b["emotion_id"] for b in batch], dtype=torch.long)
    enc["vad"] = torch.tensor(np.stack([b["vad"] for b in batch]), dtype=torch.float32)
    enc["train_sample_id"] = np.asarray([b["train_sample_id"] for b in batch])
    enc["utterance_id"] = np.asarray([b["utterance_id"] for b in batch])
    return enc

def make_loader(df, shuffle=False):
    return DataLoader(
        TranscriptDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_text,
    )

sample_batch = next(iter(make_loader(metadata.head(4), shuffle=False)))
print({k: (v.shape if hasattr(v, "shape") else v[:2]) for k, v in sample_batch.items()})
"""))
    cells.append(md(r"""
## Model 03C

Model lấy `last_hidden_state` từ transformer rồi tạo ba representation:

- `CLS`: token đầu, thường dùng cho classification.
- `mean`: trung bình có mask, ổn định với câu ngắn/dài.
- `attentive`: model tự học token nào quan trọng hơn.

Ba vector này được concat để tạo shared embedding. Sau đó tách thành 2 head:

- `emotion_head`: 4-class classification.
- `vad_head`: regression valence/arousal/dominance.
"""))
    cells.append(code(r"""
class MaskedAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden, attention_mask):
        scores = self.score(hidden).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return pooled

class TranscriptMultiTaskSER(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.25):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.att_pool = MaskedAttentionPooling(hidden)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.emotion_head = nn.Linear(hidden, num_classes)
        self.vad_head = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    def forward(self, input_ids, attention_mask, return_embedding=False):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        cls = hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        att = self.att_pool(hidden, attention_mask)
        emb = self.proj(torch.cat([cls, mean, att], dim=-1))
        emb = self.dropout(emb)
        logits = self.emotion_head(emb)
        vad = self.vad_head(emb)
        output = {"logits": logits, "vad": vad, "embedding": emb}
        return output if return_embedding else {"logits": logits, "vad": vad}

preview_model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT)
print("Trainable parameters:", sum(p.numel() for p in preview_model.parameters() if p.requires_grad))
del preview_model
torch.cuda.empty_cache()
"""))
    cells.append(code(r"""
def to_device(batch):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
    return out

def multitask_loss(outputs, y, vad_true, class_weight=None):
    ce = F.cross_entropy(outputs["logits"], y, weight=class_weight, label_smoothing=LABEL_SMOOTHING)
    mse = F.mse_loss(outputs["vad"], vad_true)
    ccc = concordance_ccc_torch(outputs["vad"], vad_true)
    ccc_loss = 1.0 - ccc.mean()
    return 0.45 * ce + 0.25 * mse + 0.30 * ccc_loss

def state_dict_clean(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def load_state_dict_clean(model, state):
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state)

@torch.no_grad()
def evaluate_model(model, loader, class_weight=None, return_features=False):
    model.eval()
    y_true, y_pred = [], []
    vad_true, vad_pred = [], []
    probs_all, emb_all = [], []
    ids_all, utt_all = [], []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        with autocast(device_type="cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
            outputs = model(batch["input_ids"], batch["attention_mask"], return_embedding=return_features)
            loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weight=class_weight)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(probs, dim=-1).detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad"].detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        if return_features:
            emb_all.append(outputs["embedding"].detach().cpu().numpy())
            ids_all.extend(batch["train_sample_id"].tolist())
            utt_all.extend(batch["utterance_id"].tolist())
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError("Loader rỗng, không thể evaluate.")
    vad_true = np.concatenate(vad_true, axis=0)
    vad_pred = np.concatenate(vad_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(1, n_batches)
    if not return_features:
        return metrics, None
    feature_npz = {
        "embedding": np.concatenate(emb_all, axis=0),
        "emotion_prob": np.concatenate(probs_all, axis=0),
        "vad_pred_norm": vad_pred,
        "vad_pred": vad_from_0_1(vad_pred),
        "vad_true": vad_from_0_1(vad_true),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "train_sample_id": np.asarray(ids_all),
        "utterance_id": np.asarray(utt_all),
    }
    return metrics, feature_npz
"""))
    cells.append(code(r"""
PROTOCOLS = []
if "5fold_session" in RUN_PROTOCOLS:
    PROTOCOLS.append(("5fold_session", split_5fold, 5))
if "10fold_speaker" in RUN_PROTOCOLS:
    PROTOCOLS.append(("10fold_speaker", split_10fold, 10))

def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    classes = np.sort(train_df["emotion_id"].unique())
    class_weight = None
    if USE_CLASS_WEIGHTS:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=train_df["emotion_id"].to_numpy())
        full_weights = np.ones(4, dtype=np.float32)
        full_weights[classes] = weights.astype(np.float32)
        class_weight = torch.tensor(full_weights, dtype=torch.float32, device=DEVICE)

    model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)
    if USE_DATA_PARALLEL:
        model = nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    target = model.module if isinstance(model, nn.DataParallel) else model
    optimizer_groups = [
        {
            "params": [p for n, p in target.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
            "lr": LR_BACKBONE,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in target.backbone.named_parameters() if any(nd in n for nd in no_decay)],
            "lr": LR_BACKBONE,
            "weight_decay": 0.0,
        },
        {
            "params": list(target.att_pool.parameters()) + list(target.proj.parameters()) + list(target.emotion_head.parameters()) + list(target.vad_head.parameters()),
            "lr": LR_HEAD,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups)
    total_updates = max(1, math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS)
    warmup_steps = max(1, int(total_updates * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates)
    scaler = GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")

    best_score = -1e9
    best_state = None
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, n_steps = 0.0, 0
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch)
            with autocast(device_type="cuda", enabled=USE_AMP and DEVICE.type == "cuda"):
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weight=class_weight) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM
            n_steps += 1

        val_metrics, _ = evaluate_model(model, val_loader, class_weight=class_weight)
        score = primary_score(val_metrics)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, n_steps), **{f"val_{k}": v for k, v in val_metrics.items()}, "score": score}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}")

        if score > best_score + MIN_DELTA:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in state_dict_clean(model).items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping")
                break

    load_state_dict_clean(model, best_state)
    test_metrics, _ = evaluate_model(model, test_loader, class_weight=class_weight)
    result = {
        "protocol": protocol,
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        **test_metrics,
    }
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})

    torch.save(best_state, MODEL_DIR / f"{protocol}_{fold_name}_best.pt")
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")

    export_splits = [("val", val_loader), ("test", test_loader)]
    if EVAL_TRAIN_SPLIT:
        export_splits.insert(0, ("train", make_loader(train_df, shuffle=False)))
    for split_name, loader in export_splits:
        _, feature_npz = evaluate_model(model, loader, class_weight=class_weight, return_features=True)
        feature_npz["protocol"] = np.asarray([protocol] * len(feature_npz["y_true"]))
        feature_npz["fold"] = np.asarray([fold_name] * len(feature_npz["y_true"]))
        feature_npz["split"] = np.asarray([split_name] * len(feature_npz["y_true"]))
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_text_features.npz", **feature_npz)
        pred_df = pd.DataFrame({
            "train_sample_id": feature_npz["train_sample_id"],
            "utterance_id": feature_npz["utterance_id"],
            "y_true": feature_npz["y_true"],
            "y_pred": feature_npz["y_pred"],
            "vad_true_valence": feature_npz["vad_true"][:, 0],
            "vad_true_arousal": feature_npz["vad_true"][:, 1],
            "vad_true_dominance": feature_npz["vad_true"][:, 2],
            "vad_pred_valence": feature_npz["vad_pred"][:, 0],
            "vad_pred_arousal": feature_npz["vad_pred"][:, 1],
            "vad_pred_dominance": feature_npz["vad_pred"][:, 2],
        })
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    del model
    torch.cuda.empty_cache()
    return result
"""))
    cells.append(code(r"""
start = time.time()
all_results = []
for protocol, split_df, n_folds in PROTOCOLS:
    folds = list(split_df["fold"].drop_duplicates())
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        fold_df = split_df[split_df["fold"] == fold].reset_index(drop=True)
        all_results.append(train_fold(protocol, fold, fold_df, SEED + idx))

results_df = pd.DataFrame(all_results)
results_df.to_csv(REPORT_DIR / "03c_text_branch_results_by_fold.csv", index=False, encoding="utf-8-sig")
display(results_df)

summary = results_df.groupby("protocol")[["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(REPORT_DIR / "03c_text_branch_summary.csv", encoding="utf-8-sig")
display(summary)
print("Total seconds:", round(time.time() - start, 2))
"""))
    cells.append(md("""<!-- AUTO-03ABC-PAPER-RESULTS -->\n## Bảng kết quả, biểu đồ và manifest output\n\nCell dưới đây tạo bảng paper-style, bảng so sánh với paper liên quan, biểu đồ metric/training curve và manifest output."""))
    cells.append(code(PAPER_RESULTS_CODE))
    cells.append(code(r"""
config = {
    "notebook": "03C transcript pretrained text multi-task branch",
    "text_model": TEXT_MODEL_NAME,
    "max_length": MAX_LENGTH,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr_backbone": LR_BACKBONE,
    "lr_head": LR_HEAD,
    "dropout": DROPOUT,
    "run_protocols": RUN_PROTOCOLS,
    "max_folds": MAX_FOLDS,
    "architecture": {
        "backbone": "pretrained text transformer",
        "pooling": "CLS + masked mean + attentive pooling",
        "heads": "emotion classification + VAD regression",
        "output": "fusion_features for notebook 04",
    },
}
(OUTPUT_DIR / "03c_text_branch_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""))

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NB_03C)


def main():
    patch_existing_notebook(NB_03A, REFERENCE_03A)
    patch_existing_notebook(NB_03B, REFERENCE_03B)
    build_03c()
    print("Updated", NB_03A)
    print("Updated", NB_03B)
    print("Created", NB_03C)


if __name__ == "__main__":
    main()
