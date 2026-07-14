import json
from pathlib import Path


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
OUT_DIR = ROOT / "06_w2v_based_models" / "03D_MultiTask_MultiModal_2024_Style_Fusion_5_10Fold"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "03D_MultiTask_MultiModal_2024_Style_Fusion_5_10Fold.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


cells = [
    md(
        r"""
# 03D - Full Reproduction: Multi-Task Multi-Modal Categorical + Dimensional SER

Notebook này là bản **độc lập từ đầu**, không dùng output của `03A`, `03B`, hoặc `03C`.

Mục tiêu là tái hiện pipeline của bài:

> **A Multi-Task, Multi-Modal Approach for Predicting Categorical and Dimensional Emotions** (`arXiv:2401.00536`)

Pipeline đúng theo paper:

1. Load trực tiếp `raw audio` và `transcript` từ IEMOCAP.
2. Speech branch: pretrained `HuBERT-large`.
3. Text branch: pretrained `DeBERTaV3-large`.
4. Freeze backbone pretrained.
5. Tinh chỉnh từng modality bằng self-attention.
6. Fusion bằng dual cross-attention với learnable bridge tokens.
7. Multi-task heads:
   - emotion classification
   - valence regression
   - arousal regression
8. Train/test trên 5-fold và 10-fold speaker/session-independent splits.

Notebook này là bản reproduction nghiêm túc hơn bản fusion trước đó vì nó dùng **sequence-level embeddings** từ backbone, không dùng pooled embeddings đã xuất sẵn.
"""
    ),
    md(
        r"""
## Kiến trúc paper cần tái hiện

Với một utterance \(i\):

- Audio waveform: \(u_i\)
- Transcript: \(t_i\)
- Categorical emotion: \(y_i^{disc}\)
- Dimensional labels: \(y_i^V, y_i^A\)

Backbone:

$$
A = HuBERT(u_i) \in \mathbb{R}^{N_a \times d}
$$

$$
T = DeBERTa(t_i) \in \mathbb{R}^{N_t \times d}
$$

Paper dùng hidden size \(d=1024\) cho HuBERT-large và DeBERTaV3-large. Nếu GPU không đủ, notebook cho phép đổi sang model base bằng biến môi trường, nhưng khi báo cáo reproduction chính nên ghi rõ model đã dùng.
"""
    ),
    md(
        r"""
## Self-Attention, Bridge Tokens và Dual Cross-Attention

Trước fusion, mỗi modality đi qua self-attention riêng:

$$
A' = SelfAttn(A), \quad T' = SelfAttn(T)
$$

Paper dùng hai nhóm learnable bridge tokens:

$$
Q_a \in \mathbb{R}^{L \times d}, \quad Q_t \in \mathbb{R}^{L \times d}
$$

Trong đó \(L=30\) theo thiết lập paper.

Dual cross-attention:

$$
e_a = MultiHead(Q_a, K=A', V=T')
$$

$$
e_t = MultiHead(Q_t, K=T', V=A')
$$

Sau đó average pooling:

$$
z = Mean(Stack(Mean(e_a), Mean(e_t)))
$$

Vector \(z\) đi vào classifier và regressor.
"""
    ),
    md(
        r"""
## Loss đúng theo paper

Emotion classification dùng cross-entropy:

$$
\mathcal{L}_{CE} = - \sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

Valence/arousal dùng CCC loss:

$$
CCC =
\frac{2Cov(\hat{y}, y)}
{\sigma_{\hat{y}}^2 + \sigma_y^2 + (\mu_{\hat{y}}-\mu_y)^2}
$$

$$
\mathcal{L}_{CCC} = 1 - CCC
$$

Multi-task loss:

$$
\mathcal{L}_{total}
= h_1\mathcal{L}_{CE}
+ h_2(1-CCC_V)
+ h_3(1-CCC_A)
$$

Mặc định notebook dùng trọng số cân bằng gần paper:

```text
CE_WEIGHT = 0.34
VALENCE_CCC_WEIGHT = 0.33
AROUSAL_CCC_WEIGHT = 0.33
```

Paper chính không dùng dominance trong head regression, nên notebook này mặc định chỉ train valence và arousal.
"""
    ),
    md(
        r"""
## Random Modality Masking

Paper dùng Random Modality Masking để giảm tình trạng model dựa quá nhiều vào một modality.

- Xác suất mask ban đầu: \(p=0.8\)
- \(p\) giảm theo cosine decay qua epoch
- Nếu \(p < 0.1\), đặt về 0
- Mask text với xác suất 0.6
- Mask audio với xác suất 0.4

Trong notebook này, vì backbone đã freeze, khi mask một modality ta zero-out representation sau projection/self-attention path. Điều này giữ đúng mục tiêu paper: ép fusion học cả hai nguồn tín hiệu.
"""
    ),
    code(
        r"""
import os
import sys
import json
import math
import time
import random
import zipfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

pd.set_option("display.max_columns", 140)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
print("CUDA devices:", torch.cuda.device_count())
"""
    ),
    code(
        r"""
INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"
if INSTALL_DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "sentencepiece", "soundfile", "librosa", "accelerate"])

import soundfile as sf
import librosa
from transformers import AutoModel, AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
from torch.amp import autocast, GradScaler
"""
    ),
    code(
        r"""
LOCAL_PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")

def unique_existing(paths):
    out, seen = [], set()
    for item in paths:
        if not item:
            continue
        p = Path(item)
        key = str(p)
        if key not in seen and p.exists():
            out.append(p)
            seen.add(key)
    return out

def search_roots():
    return unique_existing([
        Path(os.getenv("IEMOCAP_DATA_DIR", "")),
        Path(os.getenv("DATA_DIR", "")),
        Path.cwd(),
        Path("/kaggle/working"),
        Path("/kaggle/input"),
        LOCAL_PROJECT,
        LOCAL_PROJECT / "06_w2v_based_models",
    ])

def find_named_file(filename, required=True):
    direct = Path(filename)
    if direct.exists():
        return direct.resolve()
    for root in search_roots():
        try:
            for p in root.rglob(filename):
                if p.is_file():
                    return p.resolve()
        except Exception:
            continue
    if required:
        roots = "\n".join(f"- {p}" for p in search_roots())
        raise FileNotFoundError(f"Không tìm thấy `{filename}`. Đã quét:\n{roots}")
    return None

def find_iemocap_root(required=False):
    candidates = []
    for root in search_roots():
        candidates.extend([
            root / "IEMOCAP_full_release",
            root / "iemocapfullrelease" / "IEMOCAP_full_release",
            root / "datasets" / "dejolilandry" / "iemocapfullrelease" / "IEMOCAP_full_release",
        ])
    for c in candidates:
        if c.exists():
            return c.resolve()
    for root in search_roots():
        try:
            for p in root.rglob("IEMOCAP_full_release"):
                if p.is_dir():
                    return p.resolve()
        except Exception:
            continue
    if required:
        raise FileNotFoundError("Không tìm thấy thư mục IEMOCAP_full_release.")
    return None

IEMOCAP_ROOT = find_iemocap_root(required=False)
print("IEMOCAP_ROOT:", IEMOCAP_ROOT)
"""
    ),
    code(
        r"""
RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()
IS_TUNE_MODE = RUN_MODE != "full"

SPEECH_MODEL_NAME = os.getenv("SPEECH_MODEL_NAME", "facebook/hubert-large-ls960-ft")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "microsoft/deberta-v3-large")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "8.0" if not IS_TUNE_MODE else "4.0"))
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_AUDIO_SECONDS)
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "96"))

RUN_PROTOCOLS = [x.strip() for x in os.getenv(
    "RUN_PROTOCOLS",
    "5fold_session,10fold_speaker" if not IS_TUNE_MODE else "5fold_session"
).split(",") if x.strip()]
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0" if not IS_TUNE_MODE else "1"))

EPOCHS = int(os.getenv("EPOCHS", "20" if not IS_TUNE_MODE else "3"))
PATIENCE = int(os.getenv("PATIENCE", "5" if not IS_TUNE_MODE else "2"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.001"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "8"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "0"))

FUSION_DIM = int(os.getenv("FUSION_DIM", "1024"))
NUM_HEADS = int(os.getenv("NUM_HEADS", "32"))
BRIDGE_TOKENS = int(os.getenv("BRIDGE_TOKENS", "30"))
DROPOUT = float(os.getenv("DROPOUT", "0.25"))
LR = float(os.getenv("LR", "3e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.08"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
USE_AMP = os.getenv("USE_AMP", "1") == "1"
USE_DATA_PARALLEL = os.getenv("USE_DATA_PARALLEL", "0") == "1" and torch.cuda.device_count() > 1

CE_WEIGHT = float(os.getenv("CE_WEIGHT", "0.34"))
VALENCE_CCC_WEIGHT = float(os.getenv("VALENCE_CCC_WEIGHT", "0.33"))
AROUSAL_CCC_WEIGHT = float(os.getenv("AROUSAL_CCC_WEIGHT", "0.33"))

USE_RMM = os.getenv("USE_RMM", "1") == "1"
RMM_START = float(os.getenv("RMM_START", "0.8"))
RMM_TEXT_PROB = float(os.getenv("RMM_TEXT_PROB", "0.6"))
RMM_MIN_THRESHOLD = float(os.getenv("RMM_MIN_THRESHOLD", "0.1"))

SEED = int(os.getenv("SEED", "42"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03d_full_paper2024_reproduction")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

config = {
    "SPEECH_MODEL_NAME": SPEECH_MODEL_NAME,
    "TEXT_MODEL_NAME": TEXT_MODEL_NAME,
    "MAX_AUDIO_SECONDS": MAX_AUDIO_SECONDS,
    "MAX_TEXT_LENGTH": MAX_TEXT_LENGTH,
    "RUN_PROTOCOLS": RUN_PROTOCOLS,
    "MAX_FOLDS": MAX_FOLDS,
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "GRAD_ACCUM": GRAD_ACCUM,
    "FUSION_DIM": FUSION_DIM,
    "NUM_HEADS": NUM_HEADS,
    "BRIDGE_TOKENS": BRIDGE_TOKENS,
    "LR": LR,
    "USE_RMM": USE_RMM,
    "USE_DATA_PARALLEL": USE_DATA_PARALLEL,
    "OUTPUT_DIR": str(OUTPUT_DIR),
}
print(json.dumps(config, indent=2, ensure_ascii=False))
"""
    ),
    code(
        r"""
SPLIT_FILES = {
    "5fold_session": "iemocap_5fold_session_long.csv",
    "10fold_speaker": "iemocap_10fold_speaker_long.csv",
}

split_tables = {}
for protocol in RUN_PROTOCOLS:
    path = find_named_file(SPLIT_FILES[protocol])
    df = pd.read_csv(path)
    df = df[df["emotion_id"].notna()].copy()
    df["emotion_id"] = df["emotion_id"].astype(int)
    for col in ["train_sample_id", "sample_id", "utterance_id", "fold", "split", "wav_path", "transcript"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    split_tables[protocol] = df
    print(protocol, path, df.shape)
    display(df.groupby(["fold", "split"]).size().unstack(fill_value=0).head())
"""
    ),
    code(
        r"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fold_sort_key(name):
    import re
    m = re.search(r"fold_(\d+)", str(name))
    return int(m.group(1)) if m else str(name)

def resolve_wav_path(wav_path, utterance_id=None):
    p = Path(str(wav_path))
    if p.exists():
        return p
    if IEMOCAP_ROOT is not None:
        parts = list(p.parts)
        if "IEMOCAP_full_release" in parts:
            idx = parts.index("IEMOCAP_full_release")
            candidate = IEMOCAP_ROOT.joinpath(*parts[idx + 1:])
            if candidate.exists():
                return candidate
        if utterance_id:
            try:
                matches = list(IEMOCAP_ROOT.rglob(f"{utterance_id}.wav"))
                if matches:
                    return matches[0]
            except Exception:
                pass
    raise FileNotFoundError(f"Không tìm thấy wav: {wav_path}")

def load_audio_16k(path):
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) > MAX_AUDIO_SAMPLES:
        y = y[:MAX_AUDIO_SAMPLES]
    if np.max(np.abs(y)) > 0:
        y = y / max(1.0, float(np.max(np.abs(y))))
    return y.astype(np.float32)

def vad_norm_from_df(df):
    return df[["valence_norm", "arousal_norm"]].astype(np.float32).to_numpy()

def vad_from_0_1(values):
    return np.asarray(values, dtype=np.float32) * 4.0 + 1.0

def concordance_ccc_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    out = []
    for j in range(y_true.shape[1]):
        x, y = y_pred[:, j], y_true[:, j]
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = ((x - mx) * (y - my)).mean()
        out.append(float((2 * cov) / (vx + vy + (mx - my) ** 2 + eps)))
    return np.asarray(out, dtype=np.float32)

def concordance_ccc_torch(pred, target, eps=1e-8):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    return (2 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + eps)

def compute_metrics(y_true, y_pred, vad_true, vad_pred):
    ccc = concordance_ccc_np(vad_true, vad_pred)
    return {
        "WA": accuracy_score(y_true, y_pred),
        "UAR": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro"),
        "CCC_mean": float(ccc.mean()),
        "CCC_valence": float(ccc[0]),
        "CCC_arousal": float(ccc[1]),
        "MAE_mean": float(np.mean(np.abs(vad_from_0_1(vad_true) - vad_from_0_1(vad_pred)))),
    }

def primary_score(metrics):
    return 0.55 * metrics["UAR"] + 0.45 * metrics["CCC_mean"]
"""
    ),
    code(
        r"""
speech_processor = AutoProcessor.from_pretrained(SPEECH_MODEL_NAME)
text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
print("Loaded processors/tokenizer")
"""
    ),
    code(
        r"""
class IemocapRawTextDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = resolve_wav_path(row["wav_path"], row.get("utterance_id"))
        audio = load_audio_16k(wav)
        return {
            "audio": audio,
            "transcript": str(row.get("transcript", "")),
            "emotion_id": int(row["emotion_id"]),
            "vad": row[["valence_norm", "arousal_norm"]].astype(np.float32).to_numpy(),
            "train_sample_id": str(row["train_sample_id"]),
            "utterance_id": str(row["utterance_id"]),
            "speaker_id": str(row.get("speaker_id", "")),
            "session": str(row.get("session", "")),
        }

def collate_batch(batch):
    audios = [b["audio"] for b in batch]
    texts = [b["transcript"] if b["transcript"].strip() else "[EMPTY]" for b in batch]
    speech = speech_processor(
        audios,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        max_length=MAX_AUDIO_SAMPLES,
        truncation=True,
    )
    text = text_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
    )
    out = {
        "input_values": speech["input_values"],
        "speech_attention_mask": speech.get("attention_mask", torch.ones_like(speech["input_values"], dtype=torch.long)),
        "input_ids": text["input_ids"],
        "text_attention_mask": text["attention_mask"],
        "emotion_id": torch.tensor([b["emotion_id"] for b in batch], dtype=torch.long),
        "vad": torch.tensor(np.stack([b["vad"] for b in batch]), dtype=torch.float32),
        "train_sample_id": [b["train_sample_id"] for b in batch],
        "utterance_id": [b["utterance_id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "session": [b["session"] for b in batch],
    }
    return out

def make_loader(df, shuffle=False):
    return DataLoader(
        IemocapRawTextDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )

def to_device(batch):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v
    return out
"""
    ),
    code(
        r"""
class FullPaper2024MultiModalSER(nn.Module):
    def __init__(self, speech_model_name, text_model_name, fusion_dim=1024, heads=32, bridge_tokens=30, dropout=0.25):
        super().__init__()
        if fusion_dim % heads != 0:
            raise ValueError("FUSION_DIM phải chia hết cho NUM_HEADS.")

        self.speech_backbone = AutoModel.from_pretrained(speech_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)

        for p in self.speech_backbone.parameters():
            p.requires_grad = False
        for p in self.text_backbone.parameters():
            p.requires_grad = False
        self.speech_backbone.eval()
        self.text_backbone.eval()

        speech_hidden = int(getattr(self.speech_backbone.config, "hidden_size", fusion_dim))
        text_hidden = int(getattr(self.text_backbone.config, "hidden_size", fusion_dim))
        self.audio_proj = nn.Linear(speech_hidden, fusion_dim) if speech_hidden != fusion_dim else nn.Identity()
        self.text_proj = nn.Linear(text_hidden, fusion_dim) if text_hidden != fusion_dim else nn.Identity()

        audio_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        text_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=heads,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_self_attention = nn.TransformerEncoder(audio_layer, num_layers=1)
        self.text_self_attention = nn.TransformerEncoder(text_layer, num_layers=1)

        self.bridge_audio_query = nn.Parameter(torch.randn(bridge_tokens, fusion_dim) * 0.02)
        self.bridge_text_query = nn.Parameter(torch.randn(bridge_tokens, fusion_dim) * 0.02)
        self.audio_key_text_value = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)
        self.text_key_audio_value = nn.MultiheadAttention(fusion_dim, heads, dropout=dropout, batch_first=True)

        self.norm_audio = nn.LayerNorm(fusion_dim)
        self.norm_text = nn.LayerNorm(fusion_dim)
        self.shared = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim, 4)
        self.regressor = nn.Sequential(nn.Linear(fusion_dim, 2), nn.Sigmoid())

    def _speech_feature_mask(self, hidden, input_mask):
        if hasattr(self.speech_backbone, "_get_feature_vector_attention_mask"):
            return self.speech_backbone._get_feature_vector_attention_mask(hidden.shape[1], input_mask).bool()
        return torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)

    @staticmethod
    def _pad_to_same_length(a, a_mask, t, t_mask):
        target = max(a.shape[1], t.shape[1])
        def pad(x, mask):
            if x.shape[1] == target:
                return x, mask
            pad_len = target - x.shape[1]
            x = F.pad(x, (0, 0, 0, pad_len), value=0.0)
            mask = F.pad(mask, (0, pad_len), value=False)
            return x, mask
        return (*pad(a, a_mask), *pad(t, t_mask))

    def forward(self, batch, mask_modality=None, return_embedding=False):
        with torch.no_grad():
            speech_out = self.speech_backbone(
                input_values=batch["input_values"],
                attention_mask=batch.get("speech_attention_mask"),
            )
            text_out = self.text_backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["text_attention_mask"],
            )
            audio_hidden = speech_out.last_hidden_state
            text_hidden = text_out.last_hidden_state
            audio_mask = self._speech_feature_mask(audio_hidden, batch.get("speech_attention_mask"))
            text_mask = batch["text_attention_mask"].bool()

        audio = self.audio_proj(audio_hidden)
        text = self.text_proj(text_hidden)
        audio, audio_mask, text, text_mask = self._pad_to_same_length(audio, audio_mask, text, text_mask)

        if mask_modality == "speech":
            audio = torch.zeros_like(audio)
        elif mask_modality == "text":
            text = torch.zeros_like(text)

        audio_refined = self.audio_self_attention(audio, src_key_padding_mask=~audio_mask)
        text_refined = self.text_self_attention(text, src_key_padding_mask=~text_mask)

        qa = self.bridge_audio_query.unsqueeze(0).expand(audio_refined.size(0), -1, -1)
        qt = self.bridge_text_query.unsqueeze(0).expand(audio_refined.size(0), -1, -1)

        # Paper formula:
        # e_a = MultiHead(Q_a, K_audio, V_text)
        # e_t = MultiHead(Q_t, K_text, V_audio)
        e_audio, attn_audio = self.audio_key_text_value(
            query=qa,
            key=audio_refined,
            value=text_refined,
            key_padding_mask=~audio_mask,
            need_weights=return_embedding,
        )
        e_text, attn_text = self.text_key_audio_value(
            query=qt,
            key=text_refined,
            value=audio_refined,
            key_padding_mask=~text_mask,
            need_weights=return_embedding,
        )

        pooled_audio = self.norm_audio(e_audio.mean(dim=1))
        pooled_text = self.norm_text(e_text.mean(dim=1))
        z = 0.5 * (pooled_audio + pooled_text)
        z = self.shared(z)

        out = {
            "emotion_logits": self.classifier(z),
            "vad_pred": self.regressor(z),
        }
        if return_embedding:
            out["embedding"] = z
            out["attn_audio_key_text_value"] = attn_audio
            out["attn_text_key_audio_value"] = attn_text
        return out
"""
    ),
    code(
        r"""
def class_weights_for(df):
    counts = np.bincount(df["emotion_id"].astype(int).to_numpy(), minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def multitask_loss(outputs, y, vad_true, class_weights=None):
    ce = F.cross_entropy(outputs["emotion_logits"], y, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    ccc = concordance_ccc_torch(outputs["vad_pred"], vad_true)
    loss_v = 1.0 - ccc[0]
    loss_a = 1.0 - ccc[1]
    return CE_WEIGHT * ce + VALENCE_CCC_WEIGHT * loss_v + AROUSAL_CCC_WEIGHT * loss_a

def rmm_mask_for_epoch(epoch):
    if not USE_RMM:
        return None
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, EPOCHS - 1)))
    p = RMM_START * 0.5 * (1.0 + math.cos(math.pi * progress))
    if p < RMM_MIN_THRESHOLD:
        return None
    if random.random() >= p:
        return None
    return "text" if random.random() < RMM_TEXT_PROB else "speech"

def clean_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def load_clean_state_dict(model, state):
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state)

def make_grad_scaler():
    return GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
"""
    ),
    code(
        r"""
@torch.no_grad()
def evaluate(model, loader, class_weights=None, return_predictions=False):
    model.eval()
    y_true, y_pred = [], []
    vad_true, vad_pred = [], []
    probs_all = []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
            outputs = model(batch)
            loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights)
        probs = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = probs.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        for i, uid in enumerate(batch["utterance_id"]):
            rows.append({
                "train_sample_id": batch["train_sample_id"][i],
                "utterance_id": uid,
                "speaker_id": batch["speaker_id"][i],
                "session": batch["session"][i],
            })
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    vad_true = np.concatenate(vad_true, axis=0)
    vad_pred = np.concatenate(vad_pred, axis=0)
    probs_all = np.concatenate(probs_all, axis=0)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(1, n_batches)
    if not return_predictions:
        return metrics, None
    pred_df = pd.DataFrame(rows)
    pred_df["true_emotion_id"] = y_true
    pred_df["pred_emotion_id"] = y_pred
    for i, name in enumerate(["neutral", "angry", "sad", "happy"]):
        pred_df[f"prob_{name}"] = probs_all[:, i]
    pred_df["true_valence"] = vad_from_0_1(vad_true)[:, 0]
    pred_df["true_arousal"] = vad_from_0_1(vad_true)[:, 1]
    pred_df["pred_valence"] = vad_from_0_1(vad_pred)[:, 0]
    pred_df["pred_arousal"] = vad_from_0_1(vad_pred)[:, 1]
    return metrics, pred_df
"""
    ),
    code(
        r"""
def train_one_fold(protocol, fold, fold_df, seed):
    set_seed(seed)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\n=== {protocol} | {fold} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = FullPaper2024MultiModalSER(
        SPEECH_MODEL_NAME,
        TEXT_MODEL_NAME,
        fusion_dim=FUSION_DIM,
        heads=NUM_HEADS,
        bridge_tokens=BRIDGE_TOKENS,
        dropout=DROPOUT,
    ).to(DEVICE)
    if USE_DATA_PARALLEL:
        model = nn.DataParallel(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} ({trainable/max(total,1):.2%})")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
    total_updates = max(1, math.ceil(len(train_loader) / GRAD_ACCUM) * EPOCHS)
    warmup_steps = max(1, int(total_updates * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = make_grad_scaler()
    class_weights = class_weights_for(train_df) if USE_CLASS_WEIGHTS else None

    best_score, best_epoch, stale = -1e9, 0, 0
    best_path = MODEL_DIR / protocol / f"{fold}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n_steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        start = time.time()
        mask_modality = rmm_mask_for_epoch(epoch)
        if mask_modality:
            print(f"RMM epoch {epoch}: mask {mask_modality}")
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch)
            with autocast(device_type="cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
                outputs = model(batch, mask_modality=mask_modality)
                loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += float(loss.detach().cpu()) * GRAD_ACCUM
            n_steps += 1

        val_metrics, _ = evaluate(model, val_loader, class_weights=class_weights)
        score = primary_score(val_metrics)
        row = {
            "protocol": protocol,
            "fold": fold,
            "epoch": epoch,
            "train_loss": running / max(1, n_steps),
            "val_primary_score": score,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | "
            f"score={score:.4f} | seconds={row['seconds']:.1f}"
        )
        if score > best_score + MIN_DELTA:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": clean_state_dict(model), "config": config, "best_score": best_score}, best_path)
        else:
            stale += 1
            if stale >= PATIENCE:
                print("Early stopping")
                break

    checkpoint = torch.load(best_path, map_location=DEVICE)
    load_clean_state_dict(model, checkpoint["state_dict"])
    test_metrics, pred_df = evaluate(model, test_loader, class_weights=class_weights, return_predictions=True)
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold}_history.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold}_test_predictions.csv", index=False, encoding="utf-8-sig")
    result = {
        "protocol": protocol,
        "fold": fold,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        **test_metrics,
    }
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "CCC_valence", "CCC_arousal"]})
    del model
    torch.cuda.empty_cache()
    return result
"""
    ),
    code(
        r"""
all_results = []
start_all = time.time()
for protocol in RUN_PROTOCOLS:
    table = split_tables[protocol]
    folds = sorted(table["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        fold_df = table[table["fold"] == fold].reset_index(drop=True)
        all_results.append(train_one_fold(protocol, fold, fold_df, SEED + idx))

results_df = pd.DataFrame(all_results)
results_path = REPORT_DIR / "03D_full_paper2024_results_by_fold.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
print("Total seconds:", round(time.time() - start_all, 2))
display(results_df)
"""
    ),
    code(
        r"""
metric_cols = ["WA", "UAR", "Macro_F1", "CCC_mean", "CCC_valence", "CCC_arousal", "MAE_mean"]
summary_rows = []
for protocol, group in results_df.groupby("protocol"):
    row = {"protocol": protocol, "folds": group["fold"].nunique()}
    for metric in metric_cols:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=0)
        if metric == "MAE_mean":
            row[f"{metric}_paper"] = f"{group[metric].mean():.4f} ± {group[metric].std(ddof=0):.4f}"
        elif metric.startswith("CCC"):
            row[f"{metric}_paper"] = f"{group[metric].mean():.3f} ± {group[metric].std(ddof=0):.3f}"
        else:
            row[f"{metric}_paper"] = f"{group[metric].mean()*100:.2f} ± {group[metric].std(ddof=0)*100:.2f}"
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_path = REPORT_DIR / "03D_full_paper2024_summary_paper_style.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
display(summary_df)
"""
    ),
    code(
        r"""
reference_df = pd.DataFrame([
    {
        "Model": "Paper 2024 best - HuBERT-large + DeBERTaV3-large + self-attention + cross-attention + bridge tokens",
        "Input": "raw speech + transcript",
        "Split": "IEMOCAP 10-fold guideline",
        "UAR": "75.71",
        "WAR": "74.60",
        "CCC_valence": ".748",
        "CCC_arousal": ".677",
    },
    {
        "Model": "03D full reproduction current run",
        "Input": f"{SPEECH_MODEL_NAME} + {TEXT_MODEL_NAME}",
        "Split": ", ".join(RUN_PROTOCOLS),
        "UAR": "",
        "WAR": "",
        "CCC_valence": "",
        "CCC_arousal": "",
    },
])
reference_path = REPORT_DIR / "03D_reference_comparison.csv"
reference_df.to_csv(reference_path, index=False, encoding="utf-8-sig")
display(reference_df)
"""
    ),
    code(
        r"""
if plt is not None and len(results_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = results_df.melt(
        id_vars=["protocol", "fold"],
        value_vars=["WA", "UAR", "Macro_F1", "CCC_mean"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=plot_df, x="metric", y="value", hue="protocol", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("03D full paper-style reproduction metrics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03D_metric_summary.png", dpi=180)
    plt.show()

    history_files = sorted(REPORT_DIR.glob("*_history.csv"))
    if history_files:
        history_df = pd.concat([pd.read_csv(p) for p in history_files], ignore_index=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        sns.lineplot(data=history_df, x="epoch", y="train_loss", hue="protocol", estimator="mean", errorbar="sd", ax=axes[0])
        axes[0].set_title("Training loss")
        sns.lineplot(data=history_df, x="epoch", y="val_primary_score", hue="protocol", estimator="mean", errorbar="sd", ax=axes[1])
        axes[1].set_title("Validation primary score")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "03D_training_curves.png", dpi=180)
        plt.show()
"""
    ),
    code(
        r"""
report_md = REPORT_DIR / "03D_full_paper2024_report.md"
with report_md.open("w", encoding="utf-8") as f:
    f.write("# 03D Full Paper 2024 Reproduction Report\n\n")
    f.write("## Config\n\n")
    for k, v in config.items():
        f.write(f"- `{k}`: `{v}`\n")
    f.write("\n## Summary\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n\n## Reference\n\n")
    f.write(reference_df.to_markdown(index=False))
    f.write("\n\n## Notes\n\n")
    f.write("- Notebook này không dùng output của 03A/03B/03C.\n")
    f.write("- Backbone HuBERT/DeBERTa được freeze, đúng hướng paper.\n")
    f.write("- Model train self-attention, dual cross-attention, bridge tokens, classifier và regressor.\n")
    f.write("- Nếu dùng model base thay vì large, cần ghi rõ trong báo cáo vì không còn là reproduction đầy đủ cấu hình paper.\n")

with (REPORT_DIR / "03D_full_paper2024_config.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

zip_path = OUTPUT_DIR / "03D_full_paper2024_outputs.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for folder in [REPORT_DIR, FIGURE_DIR, MODEL_DIR]:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUTPUT_DIR))
print("Report:", report_md)
print("Zip:", zip_path)
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT_PATH)
