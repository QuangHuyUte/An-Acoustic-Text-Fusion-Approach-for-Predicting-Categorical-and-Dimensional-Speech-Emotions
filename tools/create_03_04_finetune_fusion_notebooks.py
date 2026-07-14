from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
BASE = ROOT / "06_w2v_based_models"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python",
            "version": "3.x",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    }
    nb["cells"] = cells
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(path)


COMMON_SETUP = r'''
import os
import sys
import time
import json
import math
import random
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)

SEED = int(os.getenv("SEED", "42"))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_DATA_PARALLEL = os.getenv("USE_DATA_PARALLEL", "1") == "1" and N_GPUS > 1
USE_AMP = DEVICE.type == "cuda"

EMOTION_ID_TO_NAME = {0: "neutral", 1: "angry", 2: "sad", 3: "happy"}
NUM_CLASSES = 4

print("Python:", sys.version)
print("Device:", DEVICE)
print("GPU count:", N_GPUS)
print("Use DataParallel:", USE_DATA_PARALLEL)
'''


PATH_RESOLVER = r'''
LOCAL_PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")

def unique_existing(paths):
    out = []
    seen = set()
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
        LOCAL_PROJECT / "06_w2v_based_models" / "data",
        LOCAL_PROJECT / "06_w2v_based_models",
        "/kaggle/input",
        "/kaggle/working",
    ]
    return unique_existing(roots)

def find_named_file(filename, env_var=None):
    if env_var and os.getenv(env_var):
        candidate = Path(os.getenv(env_var))
        if candidate.exists():
            return candidate.resolve()
    candidates = []
    for root in search_roots():
        candidates.extend([
            root / filename,
            root / "data" / filename,
            root / "splits" / filename,
            root / "features" / filename,
            root / "metadata" / filename,
            root / "output" / filename,
            root / "output" / "splits" / filename,
            root / "output" / "features" / filename,
            root / "output" / "metadata" / filename,
        ])
        try:
            candidates.extend(root.rglob(filename))
        except Exception:
            pass
    existing = [p.resolve() for p in candidates if p.exists() and p.is_file()]
    if existing:
        return sorted(set(existing), key=lambda p: (len(p.parts), str(p).lower()))[0]
    roots_text = "\n".join(f"- {root}" for root in search_roots())
    raise FileNotFoundError(f"Không tìm thấy {filename}. Notebook đã quét:\n{roots_text}")

def find_audio_dir():
    if os.getenv("IEMOCAP_AUDIO_DIR"):
        candidate = Path(os.getenv("IEMOCAP_AUDIO_DIR"))
        if candidate.exists():
            return candidate.resolve()
    for root in search_roots():
        for candidate in [
            root / "audio_wav",
            root / "data" / "audio_wav",
            root / "output" / "audio_wav",
            root / "datasets" / "AbstractTTS_IEMOCAP" / "audio_wav",
        ]:
            if candidate.exists() and any(candidate.glob("*.wav")):
                return candidate.resolve()
        try:
            for candidate in root.rglob("audio_wav"):
                if candidate.is_dir() and any(candidate.glob("*.wav")):
                    return candidate.resolve()
        except Exception:
            pass
    roots_text = "\n".join(f"- {root}" for root in search_roots())
    raise FileNotFoundError(f"Không tìm thấy thư mục audio_wav. Notebook đã quét:\n{roots_text}")

SPLIT_5FOLD_PATH = find_named_file("iemocap_5fold_session_long.csv", env_var="IEMOCAP_5FOLD_SPLIT_PATH")
SPLIT_10FOLD_PATH = find_named_file("iemocap_10fold_speaker_long.csv", env_var="IEMOCAP_10FOLD_SPLIT_PATH")

def fold_sort_key(name):
    import re
    match = re.search(r"fold_(\d+)", str(name))
    return int(match.group(1)) if match else str(name)

def load_split_table(path, protocol):
    df = pd.read_csv(path)
    required = {"utterance_id", "speaker_id", "session", "emotion_4class", "emotion_id", "valence", "arousal", "dominance", "wav_path", "fold", "split", "train_sample_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Thiếu cột trong split {protocol}: {missing}")
    df = df.copy()
    df["protocol"] = protocol
    split_map = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "testing": "test",
    }
    original_split_values = sorted(df["split"].astype(str).str.lower().str.strip().unique().tolist())
    df["split_original"] = df["split"].astype(str)
    df["split"] = df["split"].astype(str).str.lower().str.strip().map(split_map).fillna(df["split"].astype(str).str.lower().str.strip())
    normalized_split_values = sorted(df["split"].unique().tolist())
    print(f"{protocol} split labels:", original_split_values, "->", normalized_split_values)
    df["emotion_id"] = df["emotion_id"].astype(int)
    for col in ["valence", "arousal", "dominance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["valence", "arousal", "dominance"]).reset_index(drop=True)

def assert_fold_has_required_splits(protocol, fold_name, fold_df):
    counts = fold_df["split"].value_counts().to_dict()
    missing = [name for name in ["train", "val", "test"] if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(
            f"Fold {protocol}/{fold_name} thiếu split {missing}. "
            f"Số lượng hiện có: {counts}. Hãy kiểm tra cột split trong file CSV."
        )

SPLIT_TABLES = {
    "5fold_session": load_split_table(SPLIT_5FOLD_PATH, "5fold_session"),
    "10fold_speaker": load_split_table(SPLIT_10FOLD_PATH, "10fold_speaker"),
}

for protocol, df in SPLIT_TABLES.items():
    print(protocol, "rows:", len(df), "folds:", df["fold"].nunique())
    display(df.groupby(["fold", "split"]).size().unstack(fill_value=0).head(20))
'''


METRICS_AND_ZIP = r'''
def vad_to_0_1(values):
    return np.clip((values.astype(np.float32) - 1.0) / 4.0, 0.0, 1.0)

def vad_from_0_1(values):
    return values.astype(np.float32) * 4.0 + 1.0

def concordance_ccc_torch(pred, target, eps=1e-8):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    return (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + eps)

def concordance_ccc_np(pred, true, eps=1e-8):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pred_mean = pred.mean(axis=0)
    true_mean = true.mean(axis=0)
    pred_var = pred.var(axis=0)
    true_var = true.var(axis=0)
    cov = ((pred - pred_mean) * (true - true_mean)).mean(axis=0)
    return (2.0 * cov) / (pred_var + true_var + (pred_mean - true_mean) ** 2 + eps)

def compute_metrics(y_true, y_pred, vad_true_01, vad_pred_01):
    vad_true_raw = vad_from_0_1(np.asarray(vad_true_01))
    vad_pred_raw = vad_from_0_1(np.asarray(vad_pred_01))
    ccc = concordance_ccc_np(vad_pred_raw, vad_true_raw)
    mae = np.abs(vad_pred_raw - vad_true_raw).mean(axis=0)
    rmse = np.sqrt(((vad_pred_raw - vad_true_raw) ** 2).mean(axis=0))
    return {
        "WA": float(accuracy_score(y_true, y_pred)),
        "UAR": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "CCC_valence": float(ccc[0]),
        "CCC_arousal": float(ccc[1]),
        "CCC_dominance": float(ccc[2]),
        "CCC_mean": float(np.mean(ccc)),
        "MAE_mean": float(np.mean(mae)),
        "RMSE_mean": float(np.mean(rmse)),
    }

def primary_score(metrics):
    return 0.35 * metrics["UAR"] + 0.20 * metrics["WA"] + 0.20 * metrics["Macro_F1"] + 0.25 * metrics["CCC_mean"]

def multitask_loss(outputs, emotion_true, vad_true, ce_weight=1.0, mse_weight=0.35, ccc_weight=0.50):
    ce = F.cross_entropy(outputs["emotion_logits"], emotion_true)
    mse = F.mse_loss(outputs["vad_pred"], vad_true)
    ccc_loss = (1.0 - concordance_ccc_torch(outputs["vad_pred"], vad_true)).mean()
    return ce_weight * ce + mse_weight * mse + ccc_weight * ccc_loss

def module_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def load_module_state_dict(model, state_dict):
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)

def zip_output(output_dir):
    output_dir = Path(output_dir)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    print("Saved zip:", zip_path)
    return zip_path
'''


def build_raw_audio_notebook():
    cells = [
        md("""
# 03A - Emotion2Vec/Pretrained Raw-Audio Backbone Fine-tuning 5-Fold + 10-Fold

Notebook này train pretrained speech backbone từ raw audio theo đúng protocol speaker-independent của IEMOCAP.

Mục tiêu:

- Chạy cả **5-fold theo session** và **10-fold theo speaker**.
- Chỉ fine-tune trên `train`, chọn checkpoint bằng `validation`, báo cáo trên `test`.
- Tự dùng **2 GPU T4x2** bằng `torch.nn.DataParallel` nếu Kaggle cấp 2 GPU.
- Lưu checkpoint, prediction, embedding/fusion feature theo từng fold.
- Đóng gói toàn bộ output thành file `.zip` sau khi chạy xong.

Ghi chú: nếu có checkpoint Emotion2Vec tương thích HuggingFace, set `PRETRAINED_MODEL_NAME` sang checkpoint đó. Mặc định notebook dùng `microsoft/wavlm-base-plus` vì HuggingFace hỗ trợ fine-tune raw waveform ổn định.
"""),
        md("""
## Dữ liệu cần upload

Notebook cần raw audio, không chỉ feature cache:

```text
data/
  audio_wav/*.wav
  splits/iemocap_5fold_session_long.csv
  splits/iemocap_10fold_speaker_long.csv
  metadata/iemocap_metadata_full.csv    # tùy chọn
```

Output quan trọng cho bước 04:

```text
output_03a_pretrained_backbone/
  models/
  reports/
  fusion_features/
```

`fusion_features` chứa embedding/prediction của train/val/test theo từng fold để ghép với co-attention model mà không làm rò rỉ test.
"""),
        md("""
## Quy trình train-only theo từng fold

Với mỗi fold, notebook làm đúng ba bước:

1. Train backbone/head bằng các mẫu `train` của fold đó.
2. Đánh giá `validation` sau mỗi epoch để chọn checkpoint tốt nhất.
3. Load lại checkpoint tốt nhất rồi mới chạy `train`, `val`, `test` để xuất prediction và fusion feature.

Điểm cần chú ý là `test` không tham gia optimizer, không dùng để chọn epoch, và không dùng để chỉnh tham số. Nhờ vậy output của 03A có thể dùng tiếp trong notebook 04 mà vẫn giữ chuẩn speaker-independent.
"""),
        md("""
## Cách dùng T4x2 trên Kaggle

Notebook tự bật `torch.nn.DataParallel` khi Kaggle cấp 2 GPU và `USE_DATA_PARALLEL=1`.

Các biến nên kiểm soát:

- `BATCH_SIZE`: nếu T4x2 còn dư VRAM có thể tăng từ 4 lên 6 hoặc 8.
- `GRAD_ACCUM`: giữ batch hiệu dụng ổn định khi batch vật lý nhỏ.
- `UNFREEZE_LAST_N`: số tầng cuối của backbone được fine-tune. Giá trị 4 là mức cân bằng; 2 nhanh hơn, `-1` là full fine-tune nhưng nặng hơn.
- `MAX_SECONDS`: audio dài hơn tốn VRAM và thời gian hơn.
"""),
        code(COMMON_SETUP),
        code("""
INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"
if INSTALL_DEPS:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate", "soundfile", "librosa", "scikit-learn"])

try:
    import soundfile as sf
except Exception as exc:
    raise ImportError("Thiếu soundfile. Trên Kaggle hãy bật Internet và set INSTALL_DEPS=1.") from exc

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception as exc:
    raise ImportError("Thiếu transformers. Trên Kaggle hãy bật Internet và set INSTALL_DEPS=1.") from exc
"""),
        code(PATH_RESOLVER),
        code("""
AUDIO_DIR = find_audio_dir()
print("AUDIO_DIR:", AUDIO_DIR)
print("SPLIT_5FOLD_PATH:", SPLIT_5FOLD_PATH)
print("SPLIT_10FOLD_PATH:", SPLIT_10FOLD_PATH)

PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "microsoft/wavlm-base-plus")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "6.0"))
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)

EPOCHS = int(os.getenv("EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
LR_BACKBONE = float(os.getenv("LR_BACKBONE", "1e-5"))
LR_HEAD = float(os.getenv("LR_HEAD", "3e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.25"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
UNFREEZE_LAST_N = int(os.getenv("UNFREEZE_LAST_N", "4"))

MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03a_pretrained_backbone")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Pretrained model:", PRETRAINED_MODEL_NAME)
print("Output:", OUTPUT_DIR)
"""),
        code(METRICS_AND_ZIP),
        code("""
def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)

def autocast_context(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)

FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)
_AUDIO_INDEX = None

def audio_index():
    global _AUDIO_INDEX
    if _AUDIO_INDEX is None:
        _AUDIO_INDEX = {p.name: p.resolve() for p in AUDIO_DIR.rglob("*.wav")}
    return _AUDIO_INDEX

def resolve_wav_path(row):
    name = Path(str(row.get("wav_path", "")).replace("\\\\", "/")).name
    direct = AUDIO_DIR / name
    if direct.exists():
        return direct.resolve()
    idx = audio_index()
    if name in idx:
        return idx[name]
    utterance = str(row.get("utterance_id", ""))
    direct = AUDIO_DIR / f"{utterance}.wav"
    if direct.exists():
        return direct.resolve()
    raise FileNotFoundError(f"Không tìm thấy wav cho {utterance} ({name})")

def load_audio_16k(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav
"""),
        code("""
class RawAudioDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vad = row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
        return {
            "utterance_id": str(row["utterance_id"]),
            "train_sample_id": str(row["train_sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
            "split": str(row["split"]),
            "wav_path": str(resolve_wav_path(row)),
            "emotion_id": int(row["emotion_id"]),
            "vad": vad_to_0_1(vad),
        }

def collate_raw(batch):
    arrays = [load_audio_16k(Path(item["wav_path"])) for item in batch]
    encoded = FEATURE_EXTRACTOR(arrays, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SAMPLES)
    out = {
        "input_values": encoded["input_values"],
        "attention_mask": encoded.get("attention_mask"),
        "emotion_id": torch.tensor([x["emotion_id"] for x in batch], dtype=torch.long),
        "vad": torch.tensor(np.stack([x["vad"] for x in batch]), dtype=torch.float32),
        "utterance_id": [x["utterance_id"] for x in batch],
        "train_sample_id": [x["train_sample_id"] for x in batch],
        "speaker_id": [x["speaker_id"] for x in batch],
        "session": [x["session"] for x in batch],
        "split": [x["split"] for x in batch],
    }
    return out

def make_loader(df, shuffle=False):
    return DataLoader(RawAudioDataset(df), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type=="cuda", collate_fn=collate_raw)

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def forward_model(model, input_values, attention_mask=None, return_embedding=False):
    # DataParallel can fail on the final small batch when one GPU receives no samples.
    # For that case, fall back to the underlying module on the main device.
    if isinstance(model, nn.DataParallel) and input_values.size(0) < len(model.device_ids):
        return model.module(input_values, attention_mask, return_embedding=return_embedding)
    return model(input_values, attention_mask, return_embedding=return_embedding)
"""),
        md("""
## Kiến trúc

Backbone pretrained nhận waveform và xuất chuỗi hidden states. Notebook dùng masked mean pooling để tạo utterance embedding, sau đó đi qua shared MLP và hai head:

- `emotion_head`: phân loại 4 cảm xúc.
- `vad_head`: hồi quy valence/arousal/dominance.

Khi Kaggle cấp T4x2, `DataParallel` chia batch qua 2 GPU. Vì batch audio dài khá nặng, có thể tăng `BATCH_SIZE` nếu VRAM cho phép.
"""),
        code("""
class RawBackboneMultiTaskSER(nn.Module):
    def __init__(self, model_name, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = int(getattr(self.backbone.config, "hidden_size", 768))
        self.shared = nn.Sequential(nn.LayerNorm(h), nn.Dropout(dropout), nn.Linear(h, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.emotion_head = nn.Linear(hidden_dim, NUM_CLASSES)
        self.vad_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

    def masked_mean_pool(self, hidden, attention_mask=None):
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask
        if mask.shape[1] != hidden.shape[1]:
            mask = F.interpolate(mask.float().unsqueeze(1), size=hidden.shape[1], mode="nearest").squeeze(1)
        mask = mask.to(hidden.device).float().unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    def forward(self, input_values, attention_mask=None, return_embedding=False):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        pooled = self.masked_mean_pool(outputs.last_hidden_state, attention_mask)
        embedding = self.shared(pooled)
        out = {"emotion_logits": self.emotion_head(embedding), "vad_pred": self.vad_head(embedding)}
        if return_embedding:
            out["embedding"] = embedding
        return out

def freeze_backbone(model, unfreeze_last_n):
    target = model.module if isinstance(model, nn.DataParallel) else model
    if unfreeze_last_n < 0:
        for p in target.backbone.parameters():
            p.requires_grad = True
        return "Full backbone fine-tuning"
    for p in target.backbone.parameters():
        p.requires_grad = False
    layers = getattr(getattr(target.backbone, "encoder", None), "layers", None)
    if layers is not None and unfreeze_last_n > 0:
        for layer in layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        return f"Frozen backbone except last {unfreeze_last_n} encoder layers"
    return "Backbone frozen"

def build_model():
    model = RawBackboneMultiTaskSER(PRETRAINED_MODEL_NAME, HIDDEN_DIM, DROPOUT).to(DEVICE)
    note = freeze_backbone(model, UNFREEZE_LAST_N)
    if USE_DATA_PARALLEL:
        model = nn.DataParallel(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(note)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({trainable/max(total,1):.2%})")
    return model

def build_optimizer(model):
    backbone, heads = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace("module.", "")
        if clean.startswith("backbone."):
            backbone.append(param)
        else:
            heads.append(param)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": LR_BACKBONE})
    if heads:
        groups.append({"params": heads, "lr": LR_HEAD})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
"""),
        code("""
@torch.no_grad()
def evaluate(model, loader, split_name):
    if len(loader.dataset) == 0:
        raise ValueError(f"Split `{split_name}` rỗng, không thể evaluate. Hãy kiểm tra mapping train/val/test của fold.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, embeddings, probs = [], [], [], [], [], []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = forward_model(model, batch["input_values"], batch.get("attention_mask"), return_embedding=True)
        loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
        prob = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        embeddings.append(outputs["embedding"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        for i, uid in enumerate(batch["utterance_id"]):
            rows.append({
                "utterance_id": uid,
                "train_sample_id": batch["train_sample_id"][i],
                "speaker_id": batch["speaker_id"][i],
                "session": batch["session"][i],
                "split": split_name,
            })
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError(f"Không có batch nào trong split `{split_name}`. DataLoader đang rỗng.")
    vad_true = np.concatenate(vad_true)
    vad_pred = np.concatenate(vad_pred)
    embeddings = np.concatenate(embeddings)
    probs = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame(rows)
    pred_df["true_emotion_id"] = y_true
    pred_df["pred_emotion_id"] = y_pred
    for i, name in EMOTION_ID_TO_NAME.items():
        pred_df[f"prob_{name}"] = probs[:, i]
    for j, name in enumerate(["valence", "arousal", "dominance"]):
        pred_df[f"true_{name}"] = vad_from_0_1(vad_true)[:, j]
        pred_df[f"pred_{name}"] = vad_from_0_1(vad_pred)[:, j]
    feature_npz = {
        "utterance_id": pred_df["utterance_id"].to_numpy(),
        "train_sample_id": pred_df["train_sample_id"].to_numpy(),
        "embedding": embeddings.astype(np.float32),
        "emotion_probs": probs.astype(np.float32),
        "vad_pred": vad_pred.astype(np.float32),
        "emotion_true": np.asarray(y_true, dtype=np.int64),
        "vad_true": vad_true.astype(np.float32),
    }
    return metrics, pred_df, feature_npz
"""),
        code("""
def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = build_model()
    optimizer = build_optimizer(model)
    scaler = make_grad_scaler(USE_AMP)
    best_score, best_epoch = -1e9, -1
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch)
            with autocast_context(USE_AMP):
                outputs = forward_model(model, batch["input_values"], batch.get("attention_mask"), return_embedding=False)
                loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"]) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * GRAD_ACCUM
        val_metrics, _, _ = evaluate(model, val_loader, "val")
        score = primary_score(val_metrics)
        row = {"protocol": protocol, "fold": fold_name, "epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_primary_score": score, **{f"val_{k}": v for k, v in val_metrics.items()}, "seconds": time.time() - start}
        history.append(row)
        print(f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model_state_dict": module_state_dict(model), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)

    checkpoint = torch.load(best_path, map_location=DEVICE)
    load_module_state_dict(model, checkpoint["model_state_dict"])
    split_outputs = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, pred_df, feature_npz = evaluate(model, loader, split_name)
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_pretrained_features.npz", **feature_npz)
        split_outputs[split_name] = metrics
    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")
    result = {"protocol": protocol, "fold": fold_name, "best_epoch": best_epoch, "best_val_score": best_score, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df), **split_outputs["test"]}
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})
    return result
"""),
        code("""
all_results = []
start_all = time.time()
for protocol in RUN_PROTOCOLS:
    df = SPLIT_TABLES[protocol]
    folds = sorted(df["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        all_results.append(train_fold(protocol, fold, df[df["fold"] == fold].reset_index(drop=True), SEED + idx))

results_df = pd.DataFrame(all_results)
results_df.to_csv(REPORT_DIR / "03a_pretrained_backbone_results_by_fold.csv", index=False, encoding="utf-8-sig")
display(results_df)
summary = results_df.groupby("protocol")[["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(REPORT_DIR / "03a_pretrained_backbone_summary.csv", encoding="utf-8-sig")
display(summary)
print("Total seconds:", round(time.time() - start_all, 2))
"""),
        code("""
config = {
    "notebook": "03A pretrained raw-audio backbone fine-tuning",
    "pretrained_model_name": PRETRAINED_MODEL_NAME,
    "sample_rate": SAMPLE_RATE,
    "max_seconds": MAX_SECONDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr_backbone": LR_BACKBONE,
    "lr_head": LR_HEAD,
    "unfreeze_last_n": UNFREEZE_LAST_N,
    "use_data_parallel": USE_DATA_PARALLEL,
    "n_gpus": N_GPUS,
    "run_protocols": RUN_PROTOCOLS,
}
(OUTPUT_DIR / "03a_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""),
    ]
    path = BASE / "03_Emotion2Vec RawAudio Backbone Finetune 5Fold 10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
    write_notebook(path, cells)


def build_coattention_notebook():
    cells = [
        md("""
# 03B - Co-Attention Model riêng 5-Fold + 10-Fold

Notebook này train mô hình riêng của project: **Emotion2Vec feature + acoustic feature + cross-attention/co-attention + multi-task 2 head**.

Khác với 03A, notebook này không fine-tune raw-audio backbone. Nó dùng feature cache đã trích từ notebook 02:

- `emotion2vec`: embedding pretrained 768 chiều.
- `acoustic`: đặc trưng thủ công/statistical đã chuẩn hóa theo train fold.

Output của notebook cũng lưu `fusion_features` để bước 04 ghép với pretrained model 03A.
"""),
        md("""
## Data input của 03B

03B **không cần raw audio** như 03A. Notebook này chỉ cần các file đã upload trong folder `data`:

```text
data/
  features/
    iemocap_full_emotion2vec_acoustic_cache.npz
  splits/
    iemocap_5fold_session_long.csv
    iemocap_10fold_speaker_long.csv
  metadata/
    iemocap_metadata_full.csv    # tùy chọn
```

File `.npz` đã chứa sẵn:

- `sample_ids`: mã mẫu để nối với split CSV.
- `emotion2vec`: embedding pretrained.
- `acoustic`: vector đặc trưng acoustic đã trích từ audio.
- `acoustic_names`: tên từng chiều trong vector acoustic.

Vì vậy 03B chạy nhanh hơn 03A và phù hợp để test kiến trúc co-attention/fusion feature. Nếu muốn đổi hoặc thêm feature acoustic, phải quay lại notebook 02 để trích cache mới.
"""),
        md("""
## Vì sao vẫn cần 03B?

03A mạnh ở pretrained representation. 03B giữ ý tưởng chính của 06D: bổ sung các tín hiệu acoustic có diễn giải được, rồi dùng attention/gated fusion để kết hợp với emotion2vec.

Khi so sánh 03A, 03B và 04 fusion, ta sẽ biết:

- Pretrained backbone tự nó đã đủ mạnh chưa.
- Acoustic/co-attention có bổ sung tín hiệu hữu ích không.
- Fusion cuối có tăng WA/UAR/Macro-F1/CCC không.
"""),
        md(r"""
## Kiến trúc sơ bộ của 03B

03B là mô hình **feature-level co-attention** của project. Nó không đọc waveform trực tiếp như 03A, mà nhận hai nhóm đặc trưng đã trích sẵn:

| Nhánh | Input | Ý nghĩa |
|---|---|---|
| Emotion2Vec branch | `emotion2vec` embedding 768 chiều | Biểu diễn pretrained học từ dữ liệu cảm xúc lớn, mạnh về đặc trưng cảm xúc high-level |
| Acoustic branch | `acoustic` vector 184 chiều trong cache hiện tại | RMS, ZCR, spectral centroid/bandwidth/rolloff, MFCC, delta MFCC, log-Mel statistical functionals |

Pipeline:

```text
emotion2vec vector ── Linear + LayerNorm + GELU ─┐
                                                  ├─ two-token Multi-Head Attention ─ gated fusion ─ shared embedding ─ emotion head
acoustic vector ──── Linear + LayerNorm + GELU ──┘                                             └─ VAD head
```

Ta dùng `StandardScaler` riêng cho từng fold và chỉ fit scaler trên `train`. Đây là điểm quan trọng vì nếu fit scaler trên toàn bộ dữ liệu thì thống kê của test set đã bị lộ vào train.
"""),
        md(r"""
## Nhánh acoustic đang dùng những gì?

Trong cache hiện tại, nhánh acoustic không phải một feature đơn lẻ. Nó là vector 184 chiều được gom từ nhiều nhóm đặc trưng và các thống kê:

| Nhóm feature | Số chiều dự kiến trong cache hiện tại | Ý nghĩa |
|---|---:|---|
| RMS energy | 4 | năng lượng/độ lớn tín hiệu |
| Zero-crossing rate | 4 | mức thay đổi dấu, liên quan nhiễu/voicing/độ sắc tín hiệu |
| Spectral centroid | 4 | “trọng tâm” phổ, âm sáng/tối |
| Spectral bandwidth | 4 | độ rộng phổ |
| Spectral rolloff | 4 | tần số dưới đó tập trung phần lớn năng lượng phổ |
| MFCC 1-20 | 80 | đặc trưng dạng bao phổ, phổ biến trong speech/audio |
| Delta MFCC 1-20 | 80 | biến thiên theo thời gian của MFCC |
| log-Mel summary | 4 | tóm tắt năng lượng Mel-scale |

Mỗi nhóm thường được gom bằng statistical functionals:

$$
\text{mean},\quad \text{std},\quad \text{min},\quad \text{max}
$$

Vì vậy nhánh acoustic trong 03B là nhánh đặc trưng có thể diễn giải, còn emotion2vec là nhánh pretrained representation. Hai nhánh này đi qua projection riêng rồi mới tương tác bằng co-attention.

Lưu ý: nếu muốn đúng bản acoustic “rộng” hơn gồm `chroma`, `spectral contrast`, `tonnetz`, `pitch/F0`, `delta-delta MFCC`, thì cache hiện tại chưa có các nhóm đó. Khi đó cần cập nhật lại notebook 02 để trích thêm feature và tạo cache mới.
"""),
        md(r"""
## Co-attention trong 03B là gì?

Attention chuẩn có dạng:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

Trong 03B, ta tạo hai token:

$$
e = f_e(x_{\text{emotion2vec}}), \quad
a = f_a(x_{\text{acoustic}})
$$

Sau đó xếp thành:

$$
T = [e, a] \in \mathbb{R}^{2 \times d}
$$

và đưa qua multi-head attention:

$$
T' = \operatorname{MHA}(T,T,T)
$$

Vì chỉ có hai token, attention ở đây học quan hệ giữa hai nguồn feature:

- `emotion2vec` nên nghe theo acoustic bao nhiêu.
- acoustic nên bổ sung cho `emotion2vec` ở mức nào.
- nguồn nào đang mang tín hiệu rõ hơn cho emotion/VAD.

Đây là bản gọn của ý tưởng cross-attention/co-attention trong các paper multimodal. Điểm khác là paper multimodal thường dùng audio + text sequence, còn notebook này dùng audio-only nhưng hai kiểu representation khác nhau.
"""),
        md(r"""
## Gated fusion và hai head

Sau attention, ta có:

$$
e', a' = \operatorname{MHA}([e,a])
$$

Gate được học bằng:

$$
g = \sigma(W_g[e,a,e',a'] + b_g)
$$

Fusion từng nhánh:

$$
\tilde{e} = g \odot e' + (1-g)\odot e
$$

$$
\tilde{a} = g \odot a' + (1-g)\odot a
$$

Embedding cuối:

$$
z = f_{\text{fusion}}([\tilde{e},\tilde{a}])
$$

Từ embedding chung \(z\), model tách ra hai head:

**Emotion head**

$$
\ell = W_cz+b_c,\quad
\hat{p}=\operatorname{softmax}(\ell)
$$

**VAD regression head**

$$
\hat{y}_{vad}=\sigma(W_rz+b_r)
$$

VAD được chuẩn hóa từ thang 1-5 về [0,1]:

$$
y_{\text{norm}}=\frac{y-1}{4}
$$
"""),
        md("""
## Nguyên tắc xử lý feature

Notebook này dùng feature cache đã trích sẵn, nhưng việc chuẩn hóa vẫn phải theo từng fold:

- `StandardScaler` chỉ fit trên `train`.
- `validation` và `test` chỉ transform bằng scaler của train.
- Model chỉ học trên `train`, chọn checkpoint bằng `validation`, báo cáo bằng `test`.

Cách này tránh lỗi rất hay gặp: fit chuẩn hóa trên toàn bộ dataset làm test set bị rò rỉ thống kê.
"""),
        md("""
## Output cho bước fusion

Sau khi train xong mỗi fold, notebook lưu:

- `reports/*_predictions.csv`: prediction có thể đọc bằng Excel/pandas.
- `models/*_best.pt`: checkpoint tốt nhất theo validation.
- `fusion_features/*_coattention_features.npz`: embedding, probability, VAD prediction, label thật cho train/val/test.

Notebook 04 sẽ đọc các file `.npz` này để ghép với output của 03A.
"""),
        code(COMMON_SETUP),
        code(PATH_RESOLVER),
        code("""
FEATURE_CACHE_PATH = find_named_file("iemocap_full_emotion2vec_acoustic_cache.npz", env_var="FEATURE_CACHE_PATH")
cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
print("FEATURE_CACHE_PATH:", FEATURE_CACHE_PATH)
print("Keys:", cache.files)

sample_ids = cache["sample_ids"].astype(str)
emotion2vec_features = cache["emotion2vec"].astype(np.float32)
acoustic_features = cache["acoustic"].astype(np.float32)
feature_index = {sid: i for i, sid in enumerate(sample_ids)}

E2V_DIM = emotion2vec_features.shape[1]
ACOUSTIC_DIM = acoustic_features.shape[1]
print("Emotion2Vec dim:", E2V_DIM)
print("Acoustic dim:", ACOUSTIC_DIM)

EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LR = float(os.getenv("LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.35"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
PATIENCE = int(os.getenv("PATIENCE", "12"))
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03b_coattention")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR]:
    p.mkdir(parents=True, exist_ok=True)
print("Output:", OUTPUT_DIR)
"""),
        md("""
## Kiểm tra acoustic feature cache

Cell dưới đây đọc `acoustic_names` để xác nhận trong cache hiện tại thật sự có những nhóm đặc trưng nào. Đây là phần giúp tránh hiểu nhầm rằng nhánh acoustic chỉ là một vector không rõ nguồn gốc.
"""),
        code("""
def infer_acoustic_group(name):
    name = str(name).lower()
    if name.startswith("rms_"):
        return "RMS energy"
    if name.startswith("zcr_"):
        return "Zero-crossing rate"
    if name.startswith("centroid_"):
        return "Spectral centroid"
    if name.startswith("bandwidth_"):
        return "Spectral bandwidth"
    if name.startswith("rolloff_"):
        return "Spectral rolloff"
    if name.startswith("delta_mfcc_"):
        return "Delta MFCC"
    if name.startswith("mfcc_"):
        return "MFCC"
    if name.startswith("logmel_"):
        return "log-Mel summary"
    if name.startswith("chroma_"):
        return "Chroma"
    if name.startswith("contrast_") or name.startswith("spectral_contrast_"):
        return "Spectral contrast"
    if name.startswith("tonnetz_"):
        return "Tonnetz"
    if name.startswith("pitch_") or name.startswith("f0_"):
        return "Pitch/F0"
    if name.startswith("delta_delta") or name.startswith("ddelta"):
        return "Delta-delta"
    return "Other"

acoustic_names = pd.Series(cache["acoustic_names"].astype(str), name="feature_name")
acoustic_feature_table = pd.DataFrame({
    "feature_name": acoustic_names,
    "group": acoustic_names.map(infer_acoustic_group),
})
group_summary = (
    acoustic_feature_table
    .groupby("group")
    .size()
    .reset_index(name="n_dimensions")
    .sort_values(["group"])
)
display(group_summary)
display(acoustic_feature_table.head(40))

expected_groups = [
    "RMS energy",
    "Zero-crossing rate",
    "Spectral centroid",
    "Spectral bandwidth",
    "Spectral rolloff",
    "MFCC",
    "Delta MFCC",
    "Delta-delta",
    "log-Mel summary",
    "Chroma",
    "Spectral contrast",
    "Tonnetz",
    "Pitch/F0",
]
present_groups = set(group_summary["group"])
feature_presence = pd.DataFrame({
    "expected_group": expected_groups,
    "present_in_current_cache": [group in present_groups for group in expected_groups],
})
display(feature_presence)

feature_table_path = REPORT_DIR / "03b_acoustic_feature_names.csv"
feature_summary_path = REPORT_DIR / "03b_acoustic_feature_group_summary.csv"
acoustic_feature_table.to_csv(feature_table_path, index=False, encoding="utf-8-sig")
group_summary.to_csv(feature_summary_path, index=False, encoding="utf-8-sig")
print("Saved:", feature_table_path)
print("Saved:", feature_summary_path)
"""),
        code(METRICS_AND_ZIP),
        code("""
class FeatureDataset(Dataset):
    def __init__(self, df, scaler_e2v=None, scaler_acoustic=None, fit_scaler=False):
        self.df = df.reset_index(drop=True).copy()
        idx = [feature_index[str(sid)] for sid in self.df["train_sample_id"].astype(str)]
        e2v = emotion2vec_features[idx]
        acoustic = acoustic_features[idx]
        if fit_scaler:
            self.scaler_e2v = StandardScaler().fit(e2v)
            self.scaler_acoustic = StandardScaler().fit(acoustic)
        else:
            self.scaler_e2v = scaler_e2v
            self.scaler_acoustic = scaler_acoustic
        self.e2v = self.scaler_e2v.transform(e2v).astype(np.float32)
        self.acoustic = self.scaler_acoustic.transform(acoustic).astype(np.float32)
        self.y = self.df["emotion_id"].astype(int).to_numpy()
        self.vad = vad_to_0_1(self.df[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "e2v": torch.tensor(self.e2v[idx], dtype=torch.float32),
            "acoustic": torch.tensor(self.acoustic[idx], dtype=torch.float32),
            "emotion_id": torch.tensor(self.y[idx], dtype=torch.long),
            "vad": torch.tensor(self.vad[idx], dtype=torch.float32),
            "utterance_id": str(row["utterance_id"]),
            "train_sample_id": str(row["train_sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
        }

def collate_features(batch):
    return {
        "e2v": torch.stack([x["e2v"] for x in batch]),
        "acoustic": torch.stack([x["acoustic"] for x in batch]),
        "emotion_id": torch.stack([x["emotion_id"] for x in batch]),
        "vad": torch.stack([x["vad"] for x in batch]),
        "utterance_id": [x["utterance_id"] for x in batch],
        "train_sample_id": [x["train_sample_id"] for x in batch],
        "speaker_id": [x["speaker_id"] for x in batch],
        "session": [x["session"] for x in batch],
    }

def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_features)

def to_device(batch):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
"""),
        md("""
## Kiến trúc co-attention

Mô hình có hai nhánh:

- Nhánh `emotion2vec`: biểu diễn pretrained.
- Nhánh `acoustic`: đặc trưng thủ công/statistical.

Sau khi chiếu về cùng `hidden_dim`, notebook tạo hai token và dùng `MultiheadAttention` để hai nguồn thông tin nhìn lẫn nhau. Sau đó dùng gated fusion để tạo embedding cuối cho hai head:

- Head emotion classification.
- Head VAD regression.
"""),
        code("""
class CoAttentionMultiTaskSER(nn.Module):
    def __init__(self, e2v_dim, acoustic_dim, hidden_dim=256, dropout=0.35):
        super().__init__()
        self.e2v_proj = nn.Sequential(nn.Linear(e2v_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.acoustic_proj = nn.Sequential(nn.Linear(acoustic_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.emotion_head = nn.Linear(hidden_dim, NUM_CLASSES)
        self.vad_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

    def forward(self, e2v, acoustic, return_embedding=False):
        e = self.e2v_proj(e2v)
        a = self.acoustic_proj(acoustic)
        tokens = torch.stack([e, a], dim=1)
        attended, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
        e_att, a_att = attended[:, 0], attended[:, 1]
        gate = self.gate(torch.cat([e, a, e_att, a_att], dim=-1))
        fused_pair = torch.cat([gate * e_att + (1.0 - gate) * e, gate * a_att + (1.0 - gate) * a], dim=-1)
        embedding = self.fusion(fused_pair)
        out = {"emotion_logits": self.emotion_head(embedding), "vad_pred": self.vad_head(embedding)}
        if return_embedding:
            out["embedding"] = embedding
        return out
"""),
        code("""
@torch.no_grad()
def evaluate(model, loader, split_name):
    if len(loader.dataset) == 0:
        raise ValueError(f"Split `{split_name}` rỗng, không thể evaluate. Hãy kiểm tra mapping train/val/test của fold.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, embeddings, probs = [], [], [], [], [], []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = model(batch["e2v"], batch["acoustic"], return_embedding=True)
        loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
        prob = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        embeddings.append(outputs["embedding"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        for i, uid in enumerate(batch["utterance_id"]):
            rows.append({"utterance_id": uid, "train_sample_id": batch["train_sample_id"][i], "speaker_id": batch["speaker_id"][i], "session": batch["session"][i], "split": split_name})
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError(f"Không có batch nào trong split `{split_name}`. DataLoader đang rỗng.")
    vad_true = np.concatenate(vad_true)
    vad_pred = np.concatenate(vad_pred)
    embeddings = np.concatenate(embeddings)
    probs = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame(rows)
    pred_df["true_emotion_id"] = y_true
    pred_df["pred_emotion_id"] = y_pred
    for i, name in EMOTION_ID_TO_NAME.items():
        pred_df[f"prob_{name}"] = probs[:, i]
    for j, name in enumerate(["valence", "arousal", "dominance"]):
        pred_df[f"true_{name}"] = vad_from_0_1(vad_true)[:, j]
        pred_df[f"pred_{name}"] = vad_from_0_1(vad_pred)[:, j]
    feature_npz = {
        "utterance_id": pred_df["utterance_id"].to_numpy(),
        "train_sample_id": pred_df["train_sample_id"].to_numpy(),
        "embedding": embeddings.astype(np.float32),
        "emotion_probs": probs.astype(np.float32),
        "vad_pred": vad_pred.astype(np.float32),
        "emotion_true": np.asarray(y_true, dtype=np.int64),
        "vad_true": vad_true.astype(np.float32),
    }
    return metrics, pred_df, feature_npz
"""),
        code("""
def class_weights_for_sampler(df):
    counts = df["emotion_id"].value_counts().to_dict()
    weights = df["emotion_id"].map(lambda x: 1.0 / counts[int(x)]).to_numpy(dtype=np.float32)
    return torch.tensor(weights, dtype=torch.double)

def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_ds = FeatureDataset(train_df, fit_scaler=True)
    val_ds = FeatureDataset(val_df, scaler_e2v=train_ds.scaler_e2v, scaler_acoustic=train_ds.scaler_acoustic)
    test_ds = FeatureDataset(test_df, scaler_e2v=train_ds.scaler_e2v, scaler_acoustic=train_ds.scaler_acoustic)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = CoAttentionMultiTaskSER(E2V_DIM, ACOUSTIC_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, best_epoch, bad_epochs = -1e9, -1, 0
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["e2v"], batch["acoustic"])
            loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        val_metrics, _, _ = evaluate(model, val_loader, "val")
        score = primary_score(val_metrics)
        row = {"protocol": protocol, "fold": fold_name, "epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_primary_score": score, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}")
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])
    split_outputs = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, pred_df, feature_npz = evaluate(model, loader, split_name)
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_coattention_features.npz", **feature_npz)
        split_outputs[split_name] = metrics
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")
    result = {"protocol": protocol, "fold": fold_name, "best_epoch": best_epoch, "best_val_score": best_score, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df), **split_outputs["test"]}
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})
    return result
"""),
        code("""
all_results = []
start_all = time.time()
for protocol in RUN_PROTOCOLS:
    df = SPLIT_TABLES[protocol]
    folds = sorted(df["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        all_results.append(train_fold(protocol, fold, df[df["fold"] == fold].reset_index(drop=True), SEED + idx))

results_df = pd.DataFrame(all_results)
results_df.to_csv(REPORT_DIR / "03b_coattention_results_by_fold.csv", index=False, encoding="utf-8-sig")
display(results_df)
summary = results_df.groupby("protocol")[["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(REPORT_DIR / "03b_coattention_summary.csv", encoding="utf-8-sig")
display(summary)
print("Total seconds:", round(time.time() - start_all, 2))
"""),
        code("""
config = {
    "notebook": "03B co-attention model",
    "feature_cache_path": str(FEATURE_CACHE_PATH),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "run_protocols": RUN_PROTOCOLS,
}
(OUTPUT_DIR / "03b_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""),
    ]
    path = BASE / "03_Emotion2Vec Downstream Finetune 5Fold 10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
    write_notebook(path, cells)


def build_fusion_notebook():
    cells = [
        md("""
# 04 - Fusion hai mô hình: Pretrained Backbone + Co-Attention 5-Fold + 10-Fold

Notebook này ghép output của hai notebook 03:

- **03A pretrained/raw-audio backbone**: lưu `*_pretrained_features.npz`.
- **03B co-attention model riêng**: lưu `*_coattention_features.npz`.

Fusion vẫn chạy theo từng fold. Với mỗi fold:

- Train fusion trên feature `train` của đúng fold.
- Chọn checkpoint bằng `val`.
- Báo cáo cuối trên `test`.

Như vậy ta không lấy checkpoint đã nhìn thấy test speaker/session để tạo tín hiệu cho test.
"""),
        md(r"""
## Mục tiêu của notebook 04

Notebook 04 là bước cuối để trả lời câu hỏi nghiên cứu:

> Nếu một pretrained raw-audio model đã fine-tune tốt và một co-attention acoustic model học tín hiệu bổ sung khác nhau, việc fusion hai nguồn này có cải thiện emotion classification và VAD regression không?

Ta không fusion bằng cách train một model trên toàn bộ dataset rồi dự đoán lại toàn bộ. Cách đó dễ làm rò rỉ test. Ở đây fusion vẫn theo fold:

```text
fold k:
  03A train trên train_k -> xuất feature train/val/test_k
  03B train trên train_k -> xuất feature train/val/test_k
  04 train fusion trên train_k
  04 chọn checkpoint bằng val_k
  04 báo cáo test_k
```

Vì vậy, mỗi mẫu test chỉ nhận tín hiệu từ các model chưa từng train trên speaker/session test đó.
"""),
        md(r"""
## Ký hiệu của fusion model

Với một utterance \(i\):

| Ký hiệu | Ý nghĩa |
|---|---|
| \(z_i^{pre}\) | embedding từ pretrained/raw-audio model 03A |
| \(p_i^{pre}\) | xác suất emotion từ 03A |
| \(\hat{v}_i^{pre}\) | dự đoán VAD từ 03A |
| \(z_i^{co}\) | embedding từ co-attention model 03B |
| \(p_i^{co}\) | xác suất emotion từ 03B |
| \(\hat{v}_i^{co}\) | dự đoán VAD từ 03B |

Vector fusion đầu vào:

$$
x_i^{fusion}
=
[z_i^{pre}, p_i^{pre}, \hat{v}_i^{pre}, z_i^{co}, p_i^{co}, \hat{v}_i^{co}]
$$

Fusion MLP:

$$
h_i=f_{\theta}(x_i^{fusion})
$$

Hai head:

$$
\hat{p}_i=\operatorname{softmax}(W_ch_i+b_c)
$$

$$
\hat{y}_{i,VAD}=\sigma(W_rh_i+b_r)
$$
"""),
        md("""
## Dữ liệu cần có

Upload hoặc đặt hai output folder/zip từ 03A và 03B vào Kaggle:

```text
output_03a_pretrained_backbone/fusion_features/*.npz
output_03b_coattention/fusion_features/*.npz
```

Notebook sẽ tự tìm các file `.npz` trong `/kaggle/input`, `/kaggle/working` và project local.
Nếu bạn upload output ở dạng `.zip`, notebook sẽ tự giải nén các file zip liên quan vào thư mục working trước khi tìm feature.
"""),
        md("""
## Vì sao fusion phải chạy theo fold?

Nếu lấy một model đã train toàn bộ rồi sinh feature cho toàn bộ dataset, điểm fusion có thể bị cao giả vì test speaker/session đã gián tiếp xuất hiện trong mô hình tạo feature.

Vì vậy notebook 04 chỉ ghép các feature được tạo theo đúng fold:

- Feature `train` của fold dùng để train fusion.
- Feature `val` của fold dùng để chọn checkpoint fusion.
- Feature `test` của fold chỉ dùng để báo cáo kết quả cuối.

Đây là cách đúng nếu muốn nói mô hình vẫn speaker-independent.
"""),
        md("""
## Fusion đang ghép những gì?

Với mỗi utterance, notebook nối các tín hiệu sau:

- Embedding của pretrained/raw-audio backbone 03A.
- Emotion probability của 03A.
- VAD prediction của 03A.
- Embedding của co-attention model 03B.
- Emotion probability của 03B.
- VAD prediction của 03B.

Sau đó một MLP fusion học hai head giống các notebook trước: emotion classification và VAD regression.
"""),
        md(r"""
## Các chỉ số emotion classification

Giả sử có \(K\) lớp emotion và ma trận nhầm lẫn \(C\), trong đó:

$$
C_{ij}
=
\text{số mẫu thuộc lớp thật } i \text{ nhưng dự đoán là lớp } j
$$

### WA / Accuracy / WAR

Trong single-label classification, **WA** thường là accuracy tổng thể:

$$
\operatorname{WA}
=
\frac{\sum_{k=1}^{K} C_{kk}}
{\sum_{i=1}^{K}\sum_{j=1}^{K}C_{ij}}
$$

Một số paper gọi **WAR** là weighted average recall. Với single-label classification, weighted recall theo support thường bằng accuracy:

$$
\operatorname{WAR}
=
\sum_{k=1}^{K}
\frac{n_k}{N}
\operatorname{Recall}_k
$$

với:

$$
n_k=\sum_{j=1}^{K}C_{kj}
$$

### UAR / UA

**UAR** là unweighted average recall. Một số paper viết là **UA**.
Nếu thấy ghi `AUR` trong ghi chú hoặc log, thường đó là viết nhầm của `UAR`; trong SER/IEMOCAP, thuật ngữ chuẩn thường là `UA` hoặc `UAR`.

Recall của lớp \(k\):

$$
\operatorname{Recall}_k
=
\frac{C_{kk}}
{\sum_{j=1}^{K}C_{kj}}
$$

UAR:

$$
\operatorname{UAR}
=
\frac{1}{K}
\sum_{k=1}^{K}
\operatorname{Recall}_k
$$

UAR quan trọng với IEMOCAP vì dataset lệch lớp. Nếu model đoán tốt lớp đông nhưng bỏ qua lớp ít, WA có thể nhìn ổn nhưng UAR sẽ thấp.

### Macro-F1

Precision của lớp \(k\):

$$
\operatorname{Precision}_k
=
\frac{C_{kk}}
{\sum_{i=1}^{K}C_{ik}}
$$

F1 của lớp \(k\):

$$
F1_k
=
\frac{2\operatorname{Precision}_k\operatorname{Recall}_k}
{\operatorname{Precision}_k+\operatorname{Recall}_k}
$$

Macro-F1:

$$
\operatorname{MacroF1}
=
\frac{1}{K}
\sum_{k=1}^{K}F1_k
$$
"""),
        md(r"""
## Các chỉ số VAD regression

IEMOCAP có dimensional labels gồm:

| Dimension | Ý nghĩa |
|---|---|
| Valence | mức tích cực/tiêu cực của cảm xúc |
| Arousal | mức kích hoạt/năng lượng/cường độ cảm xúc |
| Dominance | mức kiểm soát/tự chủ/áp đảo trong biểu đạt |

### CCC

**CCC** là Concordance Correlation Coefficient. Đây là chỉ số phổ biến cho valence/arousal/dominance vì nó kiểm tra cả tương quan lẫn độ lệch thang đo.

Với ground truth \(y\) và prediction \(\hat{y}\):

$$
\rho_c
=
\frac{2\sigma_{y\hat{y}}}
{\sigma_y^2+\sigma_{\hat{y}}^2+(\mu_y-\mu_{\hat{y}})^2}
$$

Trong đó:

- \(\mu_y\): trung bình ground truth.
- \(\mu_{\hat{y}}\): trung bình prediction.
- \(\sigma_y^2\): phương sai ground truth.
- \(\sigma_{\hat{y}}^2\): phương sai prediction.
- \(\sigma_{y\hat{y}}\): covariance giữa ground truth và prediction.

CCC cao khi prediction vừa tăng/giảm cùng ground truth, vừa không lệch trung bình và độ phân tán quá nhiều.

Loss thường dùng:

$$
\mathcal{L}_{CCC}
=
1-\rho_c
$$

### MAE và RMSE

$$
\operatorname{MAE}
=
\frac{1}{N}
\sum_{i=1}^{N}|y_i-\hat{y}_i|
$$

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
}
$$

MAE/RMSE dễ hiểu theo thang 1-5, còn CCC dễ so sánh với các paper regression.
"""),
        md(r"""
## Loss của mô hình 2-head

Emotion classification dùng cross-entropy:

$$
\mathcal{L}_{CE}
=
-
\sum_{k=1}^{K}
y_k\log(\hat{p}_k)
$$

VAD dùng MSE và CCC loss:

$$
\mathcal{L}_{MSE}
=
\frac{1}{3}
\sum_{d\in\{V,A,D\}}
(y_d-\hat{y}_d)^2
$$

$$
\mathcal{L}_{CCC}
=
\frac{1}{3}
\sum_{d\in\{V,A,D\}}
(1-\rho_{c,d})
$$

Tổng loss:

$$
\mathcal{L}
=
\lambda_{CE}\mathcal{L}_{CE}
+
\lambda_{MSE}\mathcal{L}_{MSE}
+
\lambda_{CCC}\mathcal{L}_{CCC}
$$

Trong notebook, các hệ số nằm trong hàm `multitask_loss`. Nếu emotion tốt nhưng CCC thấp, có thể tăng trọng số CCC. Nếu CCC khá nhưng UAR thấp, có thể tăng trọng số CE hoặc xử lý class imbalance.
"""),
        md(r"""
## Co-attention, cross-attention và bridge tokens

Attention chuẩn:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

**Self-attention**: \(Q,K,V\) đến từ cùng một nguồn.

**Cross-attention**: \(Q\) đến từ nguồn A, còn \(K,V\) đến từ nguồn B. Ví dụ audio query nhìn vào text key/value.

**Co-attention**: hai nguồn feature tương tác hai chiều hoặc cùng được đưa vào một attention block để học mức liên quan giữa chúng.

**Bridge tokens** trong paper multi-task/multimodal là các vector học được đóng vai trò trung gian giữa hai modality. Thay vì để audio query text trực tiếp, bridge tokens học cách gom thông tin cảm xúc từ audio và text.

Notebook 04 không train bridge-token transformer từ đầu. Nó là bước fusion nhẹ hơn: lấy embedding/prediction đã học từ 03A và 03B, rồi học một MLP 2-head trên các tín hiệu đó.
"""),
        md(r"""
## Bảng tham chiếu paper trên IEMOCAP

Các kết quả dưới đây không luôn so sánh trực tiếp 1-1 vì mỗi paper có thể khác số mẫu, split, modality và label mapping. Ta dùng bảng này để biết mức điểm kỳ vọng và cách đọc metric.

| Paper | Input/modality | Split | Kiến trúc chính | Kết quả được báo cáo | Ghi chú cho project |
|---|---|---|---|---|---|
| [Antoniou et al., 2023 - Reality check with IEMOCAP](https://arxiv.org/abs/2304.00860) | Review/evaluation guideline | Khuyến nghị 10-fold: 8 speakers train, 1 validation, 1 test | Không phải model mới; phân tích reproducibility | Khuyến nghị baseline tối thiểu dùng 4 lớp `neutral/sad/angry/happy+excited`, khoảng 5531 samples | Dùng để giải thích vì sao 10-fold speaker-independent là setup khó và đáng tin |
| [Chen & Rudnicky, 2021 - Wav2Vec2 fine-tuning](https://arxiv.org/abs/2110.06309) | Audio only/raw waveform | 5-fold leave-one-session-out | wav2vec2 + average pooling + linear classifier; V-FT/TAPT/P-TAPT | UA: V-FT 69.9, TAPT 73.5, P-TAPT 74.3 | Tham chiếu trực tiếp cho 03A: pretrained raw-audio fine-tuning |
| [Ispas et al., 2024 - Multi-task, Multi-modal categorical + dimensional emotion](https://arxiv.org/abs/2401.00536) | Audio + transcript | 10-fold speaker guideline | HuBERT-large + DeBERTaV3-large, self-attention, cross-attention, learnable bridge tokens, multi-task heads | Best table: UAR 74.68, WAR 74.69, Valence CCC .738, Arousal CCC .685 | Tham chiếu chính cho 04: multi-task + cross-attention + 2-head evaluation |
| [emotion2vec, 2023](https://arxiv.org/abs/2312.15185) | Audio representation | IEMOCAP SER | self-supervised emotion representation; downstream linear layers | Paper nhấn mạnh chỉ cần train linear layers đã vượt nhiều pretrained/emotion specialist models | Tham chiếu cho nhánh representation emotion2vec/03B |
| [Sun et al., 2023 - wav2vec2 + BERT auxiliary tasks](https://arxiv.org/abs/2302.13661) | Audio + text | IEMOCAP | wav2vec2 + BERT, fusion bằng multi-head attention, auxiliary tasks | WA 78.42, UA 79.71 | Mốc multimodal cao; project của mình không dùng text chính nên không lấy làm baseline trực tiếp |
| [DST - Deformable Speech Transformer](https://arxiv.org/abs/2302.13729) | Audio/acoustic | IEMOCAP/MELD | transformer với deformable/window attention để học cue đa mức thời gian | Paper báo DST vượt nhiều mô hình transformer SER | Gợi ý cải tiến tương lai nếu muốn thay co-attention bằng temporal attention mạnh hơn |

Khi so sánh, cần ghi rõ:

- 5-fold session hay 10-fold speaker.
- Audio-only hay multimodal audio+text.
- Có dùng transcript/gold text không.
- Có gộp excited vào happy không.
- Dùng WA/UAR/Macro-F1 hay CCC/MAE/RMSE.
"""),
        md(r"""
## Cách đọc kết quả của notebook 04

Sau khi chạy xong, đọc theo thứ tự:

1. `04_fusion_results_by_fold.csv`: kết quả từng fold.
2. `04_fusion_summary.csv`: mean/std theo `5fold_session` và `10fold_speaker`.
3. So sánh với output của 03A và 03B.

Diễn giải nhanh:

```text
Nếu 04 > 03A và 04 > 03B:
    fusion thật sự có lợi.

Nếu 04 gần 03A nhưng > 03B:
    pretrained backbone là nguồn chính, co-attention chỉ bổ sung nhẹ.

Nếu 04 < 03A:
    fusion đang thêm nhiễu hoặc overfit, cần giảm hidden_dim/dropout hoặc chỉ dùng embedding thay vì prediction.

Nếu UAR tăng nhưng WA giảm:
    model đang công bằng hơn giữa các lớp nhỏ, nhưng mất một ít accuracy tổng.

Nếu CCC_valence thấp hơn CCC_arousal/dominance:
    đây là hiện tượng thường gặp trong speech-only VAD vì valence hay cần lexical/text context hơn arousal.
```
"""),
        code(COMMON_SETUP),
        code(PATH_RESOLVER),
        code(METRICS_AND_ZIP),
        code("""
EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LR = float(os.getenv("LR", "1e-3"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.35"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
PATIENCE = int(os.getenv("PATIENCE", "12"))
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_04_fusion")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
EXTRACT_DIR = OUTPUT_DIR / "extracted_inputs"
for p in [MODEL_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)
print("Output:", OUTPUT_DIR)
"""),
        code("""
def maybe_extract_feature_zips():
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    keywords = ["03a", "03b", "pretrained", "coattention", "output_03"]
    extracted = []
    for root in search_roots():
        try:
            zip_files = list(root.rglob("*.zip"))
        except Exception:
            zip_files = []
        for zip_path in zip_files:
            name = zip_path.name.lower()
            if not any(key in name for key in keywords):
                continue
            target = EXTRACT_DIR / zip_path.stem
            marker = target / ".extracted"
            if marker.exists():
                extracted.append(target)
                continue
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(target)
            marker.write_text("ok", encoding="utf-8")
            extracted.append(target)
            print("Extracted:", zip_path, "->", target)
    if not extracted:
        print("Không thấy zip output 03A/03B cần giải nén. Notebook sẽ tìm thư mục đã giải nén sẵn.")

maybe_extract_feature_zips()
"""),
        code("""
def find_feature_file(protocol, fold, split, suffix):
    filename = f"{protocol}_{fold}_{split}_{suffix}.npz"
    return find_named_file(filename)

def load_feature_npz(path):
    data = np.load(path, allow_pickle=True)
    return {
        "utterance_id": data["utterance_id"].astype(str),
        "train_sample_id": data["train_sample_id"].astype(str),
        "embedding": data["embedding"].astype(np.float32),
        "emotion_probs": data["emotion_probs"].astype(np.float32),
        "vad_pred": data["vad_pred"].astype(np.float32),
        "emotion_true": data["emotion_true"].astype(np.int64),
        "vad_true": data["vad_true"].astype(np.float32),
    }

def merge_source_features(pretrained, coattention):
    left = pd.DataFrame({"utterance_id": pretrained["utterance_id"], "idx_pre": np.arange(len(pretrained["utterance_id"]))})
    right = pd.DataFrame({"utterance_id": coattention["utterance_id"], "idx_co": np.arange(len(coattention["utterance_id"]))})
    merged = left.merge(right, on="utterance_id", how="inner")
    if len(merged) != len(left) or len(merged) != len(right):
        print("Warning: số mẫu sau khi merge khác nguồn gốc:", len(left), len(right), len(merged))
    ip = merged["idx_pre"].to_numpy()
    ic = merged["idx_co"].to_numpy()
    features = np.concatenate([
        pretrained["embedding"][ip],
        pretrained["emotion_probs"][ip],
        pretrained["vad_pred"][ip],
        coattention["embedding"][ic],
        coattention["emotion_probs"][ic],
        coattention["vad_pred"][ic],
    ], axis=1).astype(np.float32)
    return {
        "utterance_id": merged["utterance_id"].to_numpy(),
        "features": features,
        "emotion_true": pretrained["emotion_true"][ip],
        "vad_true": pretrained["vad_true"][ip],
    }
"""),
        code("""
class FusionDataset(Dataset):
    def __init__(self, data, scaler=None, fit_scaler=False):
        self.utterance_id = data["utterance_id"]
        x = data["features"]
        if fit_scaler:
            self.scaler = StandardScaler().fit(x)
        else:
            self.scaler = scaler
        self.x = self.scaler.transform(x).astype(np.float32)
        self.y = data["emotion_true"].astype(np.int64)
        self.vad = data["vad_true"].astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "emotion_id": torch.tensor(self.y[idx], dtype=torch.long),
            "vad": torch.tensor(self.vad[idx], dtype=torch.float32),
            "utterance_id": str(self.utterance_id[idx]),
        }

def collate_fusion(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "emotion_id": torch.stack([b["emotion_id"] for b in batch]),
        "vad": torch.stack([b["vad"] for b in batch]),
        "utterance_id": [b["utterance_id"] for b in batch],
    }

def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, collate_fn=collate_fusion)

def to_device(batch):
    return {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

class FusionMultiTaskSER(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.35):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_dim, NUM_CLASSES)
        self.vad_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

    def forward(self, x, return_embedding=False):
        emb = self.encoder(x)
        out = {"emotion_logits": self.emotion_head(emb), "vad_pred": self.vad_head(emb)}
        if return_embedding:
            out["embedding"] = emb
        return out
"""),
        code("""
@torch.no_grad()
def evaluate(model, loader):
    if len(loader.dataset) == 0:
        raise ValueError("DataLoader rỗng, không thể evaluate. Hãy kiểm tra feature train/val/test của fold.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, probs = [], [], [], [], []
    utterances = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = model(batch["x"])
        loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
        prob = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        utterances.extend(batch["utterance_id"])
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError("Không có batch nào trong DataLoader. Dataset đang rỗng.")
    vad_true = np.concatenate(vad_true)
    vad_pred = np.concatenate(vad_pred)
    probs = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame({"utterance_id": utterances, "true_emotion_id": y_true, "pred_emotion_id": y_pred})
    for i, name in EMOTION_ID_TO_NAME.items():
        pred_df[f"prob_{name}"] = probs[:, i]
    for j, name in enumerate(["valence", "arousal", "dominance"]):
        pred_df[f"true_{name}"] = vad_from_0_1(vad_true)[:, j]
        pred_df[f"pred_{name}"] = vad_from_0_1(vad_pred)[:, j]
    return metrics, pred_df
"""),
        code("""
def load_split_features(protocol, fold, split):
    pre = load_feature_npz(find_feature_file(protocol, fold, split, "pretrained_features"))
    co = load_feature_npz(find_feature_file(protocol, fold, split, "coattention_features"))
    return merge_source_features(pre, co)

def train_fold(protocol, fold, seed):
    set_seed(seed)
    fold_df = SPLIT_TABLES[protocol][SPLIT_TABLES[protocol]["fold"] == fold].reset_index(drop=True)
    assert_fold_has_required_splits(protocol, fold, fold_df)
    train_data = load_split_features(protocol, fold, "train")
    val_data = load_split_features(protocol, fold, "val")
    test_data = load_split_features(protocol, fold, "test")
    train_ds = FusionDataset(train_data, fit_scaler=True)
    val_ds = FusionDataset(val_data, scaler=train_ds.scaler)
    test_ds = FusionDataset(test_data, scaler=train_ds.scaler)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = FusionMultiTaskSER(train_ds.x.shape[1], HIDDEN_DIM, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, best_epoch, bad_epochs = -1e9, -1, 0
    best_path = MODEL_DIR / protocol / f"{fold}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["x"])
            loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        val_metrics, _ = evaluate(model, val_loader)
        score = primary_score(val_metrics)
        history.append({"protocol": protocol, "fold": fold, "epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_primary_score": score, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"{protocol} | {fold} | epoch {epoch:03d} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}")
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": train_ds.x.shape[1], "best_epoch": best_epoch, "best_val_score": best_score}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])
    test_metrics, pred_df = evaluate(model, test_loader)
    pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold}_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold}_history.csv", index=False, encoding="utf-8-sig")
    result = {"protocol": protocol, "fold": fold, "best_epoch": best_epoch, "best_val_score": best_score, "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds), **test_metrics}
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})
    return result
"""),
        code("""
all_results = []
start_all = time.time()
for protocol in RUN_PROTOCOLS:
    df = SPLIT_TABLES[protocol]
    folds = sorted(df["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        all_results.append(train_fold(protocol, fold, SEED + idx))

results_df = pd.DataFrame(all_results)
results_df.to_csv(REPORT_DIR / "04_fusion_results_by_fold.csv", index=False, encoding="utf-8-sig")
display(results_df)
summary = results_df.groupby("protocol")[["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(REPORT_DIR / "04_fusion_summary.csv", encoding="utf-8-sig")
display(summary)
print("Total seconds:", round(time.time() - start_all, 2))
"""),
        md(r"""
## Bảng kết quả gọn kiểu paper

Cell này tạo bảng `mean ± std` để đưa trực tiếp vào báo cáo hoặc slide. Bảng tách theo protocol:

- `5fold_session`: đánh giá theo session.
- `10fold_speaker`: đánh giá theo speaker.

Các metric chính:

- Emotion: WA, UAR, Macro-F1.
- Regression: CCC theo Valence/Arousal/Dominance, CCC trung bình, MAE, RMSE.
"""),
        code("""
def format_mean_std(mean_value, std_value, scale=100.0, digits=2):
    if pd.isna(std_value):
        return f"{mean_value * scale:.{digits}f}"
    return f"{mean_value * scale:.{digits}f} ± {std_value * scale:.{digits}f}"

def build_paper_style_table(results_df):
    rows = []
    metric_cols = ["WA", "UAR", "Macro_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]
    for protocol, group in results_df.groupby("protocol"):
        row = {"Protocol": protocol, "Folds": group["fold"].nunique(), "N test total": int(group["n_test"].sum())}
        for metric in metric_cols:
            scale = 100.0 if metric in ["WA", "UAR", "Macro_F1"] else 1.0
            digits = 2 if scale == 100.0 else 4
            row[metric] = format_mean_std(group[metric].mean(), group[metric].std(), scale=scale, digits=digits)
        rows.append(row)
    return pd.DataFrame(rows)

paper_table = build_paper_style_table(results_df)
paper_table_path = REPORT_DIR / "04_fusion_paper_style_results.csv"
paper_table.to_csv(paper_table_path, index=False, encoding="utf-8-sig")
display(paper_table)
print("Saved:", paper_table_path)
"""),
        md(r"""
## Bảng so sánh với các mô hình tham chiếu

Bảng này đặt kết quả 04 cạnh các paper thường được dùng để tham chiếu trên IEMOCAP. Khi đưa vào báo cáo, cần ghi rõ là các kết quả không hoàn toàn 1-1 vì khác modality, split, số mẫu và cách gộp label.
"""),
        code("""
reference_rows = [
    {
        "Model": "Chen & Rudnicky 2021 - wav2vec2 V-FT",
        "Input": "audio waveform",
        "Split": "5-fold session",
        "WA/WAR": "",
        "UAR/UA": "69.90",
        "Macro-F1": "",
        "CCC V": "",
        "CCC A": "",
        "CCC D": "",
        "Notes": "Audio-only fine-tuning baseline",
        "Link": "https://arxiv.org/abs/2110.06309",
    },
    {
        "Model": "Chen & Rudnicky 2021 - wav2vec2 P-TAPT",
        "Input": "audio waveform",
        "Split": "5-fold session",
        "WA/WAR": "",
        "UAR/UA": "74.30",
        "Macro-F1": "",
        "CCC V": "",
        "CCC A": "",
        "CCC D": "",
        "Notes": "Task-adaptive pretraining; strong audio-only reference",
        "Link": "https://arxiv.org/abs/2110.06309",
    },
    {
        "Model": "Ispas et al. 2024 - HuBERT + DeBERTaV3 MTL",
        "Input": "audio + transcript",
        "Split": "10-fold speaker",
        "WA/WAR": "74.69",
        "UAR/UA": "74.68",
        "Macro-F1": "",
        "CCC V": "0.738",
        "CCC A": "0.685",
        "CCC D": "",
        "Notes": "Cross-attention, bridge tokens, categorical + dimensional emotion",
        "Link": "https://arxiv.org/abs/2401.00536",
    },
    {
        "Model": "Sun et al. 2023 - wav2vec2 + BERT + auxiliary tasks",
        "Input": "audio + text",
        "Split": "IEMOCAP",
        "WA/WAR": "78.42",
        "UAR/UA": "79.71",
        "Macro-F1": "",
        "CCC V": "",
        "CCC A": "",
        "CCC D": "",
        "Notes": "Multimodal high reference; not directly audio-only comparable",
        "Link": "https://arxiv.org/abs/2302.13661",
    },
]

our_rows = []
for _, row in paper_table.iterrows():
    our_rows.append({
        "Model": "Proposed 04 Fusion - 03A pretrained + 03B co-attention",
        "Input": "audio-only fusion features",
        "Split": row["Protocol"],
        "WA/WAR": row["WA"],
        "UAR/UA": row["UAR"],
        "Macro-F1": row["Macro_F1"],
        "CCC V": row["CCC_valence"],
        "CCC A": row["CCC_arousal"],
        "CCC D": row["CCC_dominance"],
        "Notes": "Same local fold protocol; train-only fusion",
        "Link": "",
    })

comparison_df = pd.DataFrame(reference_rows + our_rows)
comparison_path = REPORT_DIR / "04_fusion_reference_comparison_table.csv"
comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
display(comparison_df)
print("Saved:", comparison_path)
"""),
        md(r"""
## Thống kê train/validation/test theo fold

Bảng này giúp kiểm tra lại protocol khi viết báo cáo. Nếu một fold có validation/test bằng 0 thì kết quả không hợp lệ.
"""),
        code("""
split_stat_rows = []
for protocol in RUN_PROTOCOLS:
    df = SPLIT_TABLES[protocol]
    for fold, group in df.groupby("fold"):
        counts = group["split"].value_counts().to_dict()
        split_stat_rows.append({
            "protocol": protocol,
            "fold": fold,
            "train": counts.get("train", 0),
            "val": counts.get("val", 0),
            "test": counts.get("test", 0),
            "n_speakers_train": group[group["split"] == "train"]["speaker_id"].nunique(),
            "n_speakers_val": group[group["split"] == "val"]["speaker_id"].nunique(),
            "n_speakers_test": group[group["split"] == "test"]["speaker_id"].nunique(),
        })
split_stats_df = pd.DataFrame(split_stat_rows).sort_values(["protocol", "fold"])
split_stats_path = REPORT_DIR / "04_fusion_split_statistics.csv"
split_stats_df.to_csv(split_stats_path, index=False, encoding="utf-8-sig")
display(split_stats_df)
print("Saved:", split_stats_path)
"""),
        md(r"""
## Biểu đồ train/validation

Các biểu đồ này mô phỏng cách paper thường trình bày quá trình huấn luyện:

- Training loss theo epoch.
- Validation primary score theo epoch.
- Validation UAR và CCC mean theo epoch.

Nếu validation score tăng rồi giảm, model có dấu hiệu overfit. Nếu train loss giảm nhưng validation không tăng, fusion feature có thể chưa bổ sung thông tin hữu ích hoặc MLP quá lớn.
"""),
        code("""
def load_history_tables():
    history_files = sorted(REPORT_DIR.glob("*_history.csv"))
    frames = []
    for path in history_files:
        df = pd.read_csv(path)
        df["history_file"] = path.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

history_all = load_history_tables()
history_path = REPORT_DIR / "04_fusion_all_history.csv"
history_all.to_csv(history_path, index=False, encoding="utf-8-sig")
display(history_all.head())
print("Saved:", history_path)

if plt is not None and len(history_all):
    for protocol, group in history_all.groupby("protocol"):
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for fold, fg in group.groupby("fold"):
            axes[0].plot(fg["epoch"], fg["train_loss"], alpha=0.55, label=fold)
            axes[1].plot(fg["epoch"], fg["val_primary_score"], alpha=0.55)
            axes[2].plot(fg["epoch"], fg["val_UAR"], alpha=0.55, linestyle="-")
            axes[2].plot(fg["epoch"], fg["val_CCC_mean"], alpha=0.55, linestyle="--")
        axes[0].set_title(f"{protocol} - train loss")
        axes[1].set_title(f"{protocol} - validation primary score")
        axes[2].set_title(f"{protocol} - validation UAR / CCC mean")
        axes[0].set_xlabel("Epoch")
        axes[1].set_xlabel("Epoch")
        axes[2].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[1].set_ylabel("Score")
        axes[2].set_ylabel("Metric")
        axes[0].grid(alpha=0.25)
        axes[1].grid(alpha=0.25)
        axes[2].grid(alpha=0.25)
        axes[0].legend(fontsize=7, ncol=1)
        fig.tight_layout()
        fig_path = REPORT_DIR / f"04_fusion_{protocol}_training_curves.png"
        fig.savefig(fig_path, dpi=180)
        plt.show()
        print("Saved:", fig_path)
else:
    print("Không thể vẽ training curves vì thiếu matplotlib hoặc history.")
"""),
        md(r"""
## Biểu đồ tổng hợp metric và confusion matrix

Cell này tạo:

- Bar chart WA/UAR/Macro-F1/CCC mean theo protocol.
- Confusion matrix gộp tất cả fold theo từng protocol.
- Bar chart CCC cho Valence/Arousal/Dominance.
"""),
        code("""
if plt is not None and len(results_df):
    plot_metrics = ["WA", "UAR", "Macro_F1", "CCC_mean"]
    metric_summary = results_df.groupby("protocol")[plot_metrics].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_metrics))
    width = 0.35
    protocols = list(metric_summary.index)
    for idx, protocol in enumerate(protocols):
        means = [metric_summary.loc[protocol, (m, "mean")] for m in plot_metrics]
        stds = [metric_summary.loc[protocol, (m, "std")] for m in plot_metrics]
        offset = (idx - (len(protocols) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_metrics)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("04 Fusion - paper-style metric summary")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig_path = REPORT_DIR / "04_fusion_metric_summary_bar.png"
    fig.savefig(fig_path, dpi=180)
    plt.show()
    print("Saved:", fig_path)

    ccc_metrics = ["CCC_valence", "CCC_arousal", "CCC_dominance"]
    ccc_summary = results_df.groupby("protocol")[ccc_metrics].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ccc_metrics))
    for idx, protocol in enumerate(protocols):
        means = [ccc_summary.loc[protocol, (m, "mean")] for m in ccc_metrics]
        stds = [ccc_summary.loc[protocol, (m, "std")] for m in ccc_metrics]
        offset = (idx - (len(protocols) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, capsize=4, label=protocol)
    ax.set_xticks(x)
    ax.set_xticklabels(["Valence", "Arousal", "Dominance"])
    ax.set_ylim(-0.1, 1)
    ax.set_ylabel("CCC")
    ax.set_title("04 Fusion - CCC by VAD dimension")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig_path = REPORT_DIR / "04_fusion_ccc_by_dimension.png"
    fig.savefig(fig_path, dpi=180)
    plt.show()
    print("Saved:", fig_path)
else:
    print("Không thể vẽ metric summary vì thiếu matplotlib hoặc results_df rỗng.")

def plot_confusion_for_protocol(protocol):
    if plt is None:
        return
    files = sorted(REPORT_DIR.glob(f"{protocol}_*_test_predictions.csv"))
    if not files:
        print("Không có prediction file cho", protocol)
        return
    pred = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    cm = confusion_matrix(pred["true_emotion_id"], pred["pred_emotion_id"], labels=[0, 1, 2, 3])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=[EMOTION_ID_TO_NAME[i] for i in range(4)], yticklabels=[EMOTION_ID_TO_NAME[i] for i in range(4)], ax=ax)
    else:
        im = ax.imshow(cm_norm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(4), [EMOTION_ID_TO_NAME[i] for i in range(4)])
        ax.set_yticks(range(4), [EMOTION_ID_TO_NAME[i] for i in range(4)])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"04 Fusion normalized confusion matrix - {protocol}")
    fig.tight_layout()
    fig_path = REPORT_DIR / f"04_fusion_{protocol}_confusion_matrix.png"
    fig.savefig(fig_path, dpi=180)
    plt.show()
    print("Saved:", fig_path)

for protocol in RUN_PROTOCOLS:
    plot_confusion_for_protocol(protocol)
"""),
        md(r"""
## Xuất báo cáo markdown tóm tắt

Cell này tạo một file markdown gom bảng kết quả, bảng so sánh và danh sách hình đã sinh ra. File này tiện để đưa vào báo cáo cuối kỳ hoặc copy sang README.
"""),
        code("""
report_md = REPORT_DIR / "04_fusion_paper_style_report.md"
figure_files = sorted(REPORT_DIR.glob("04_fusion_*.png"))
with report_md.open("w", encoding="utf-8") as f:
    f.write("# 04 Fusion Paper-Style Report\\n\\n")
    f.write("## Result table\\n\\n")
    f.write(paper_table.to_markdown(index=False))
    f.write("\\n\\n## Reference comparison\\n\\n")
    f.write(comparison_df.to_markdown(index=False))
    f.write("\\n\\n## Split statistics\\n\\n")
    f.write(split_stats_df.to_markdown(index=False))
    f.write("\\n\\n## Figures\\n\\n")
    for fig_path in figure_files:
        f.write(f"- {fig_path.name}\\n")
print("Saved:", report_md)
"""),
        code("""
config = {
    "notebook": "04 fusion pretrained backbone + co-attention",
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "hidden_dim": HIDDEN_DIM,
    "dropout": DROPOUT,
    "run_protocols": RUN_PROTOCOLS,
}
(OUTPUT_DIR / "04_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""),
    ]
    path = BASE / "04_MultiTask_Emotion2Vec_CoAttention_5_10Fold" / "04_Fusion_Pretrained_Backbone_CoAttention_5_10Fold.ipynb"
    write_notebook(path, cells)


def build_coattention_full_06d_notebook():
    cells = [
        md(r"""
# 03B - Full 06D Multi-Branch Co-Attention Multi-Task SER 5-Fold + 10-Fold

Notebook này là bản sửa lại của 03B để **khai thác đúng phần mạnh của 06D**, thay vì chỉ dùng một vector acoustic đã pooling.

03B-full dùng lại ý tưởng chính từ notebook 06D:

```text
X_temporal -> 1D CNN -> BiLSTM -> attention pooling
X_spectral -> 2D CNN residual/SE blocks
X_stats    -> MLP
X_e2v      -> Emotion2Vec adapter MLP
Emotion2Vec-guided co-attention
Fusion MLP
Two heads: emotion classification + VAD regression
```

Điểm mới so với 06D cũ:

- Dữ liệu là IEMOCAP.
- Chạy cả `5fold_session` và `10fold_speaker`.
- Output không chỉ emotion class mà còn có VAD regression.
- Mỗi fold vẫn train-only, validation-only model selection, test-only reporting.
- Xuất `fusion_features` để notebook 04 ghép với pretrained model 03A.
"""),
        md(r"""
## Vì sao bản 03B-light trước đó chưa đủ?

Bản 03B-light chỉ nhận:

```text
emotion2vec: (N, 768)
acoustic:    (N, 184)
```

Tức là acoustic đã bị nén thành một vector thống kê ở mức utterance. Khi đó không thể dùng đúng các nhánh của 06D:

- Không có chuỗi thời gian cho 1D CNN/BiLSTM.
- Không có spectrogram 2D cho CNN 2D.
- Không có delta-delta/log-Mel multi-channel theo frame.

Vì vậy bản này yêu cầu cache kiểu 06D:

```text
X_temporal: (N, C_temporal, T)
X_spectral: (N, C_spectral, F, T)
X_stats:    (N, D_stats)
X_e2v:      (N, D_e2v)
```

Nếu cache này chưa có cho IEMOCAP, cần quay lại notebook 02 để trích lại feature theo định dạng full 06D.
"""),
        md(r"""
## Kiến trúc kế thừa từ 06D

### Temporal branch

Input:

```text
[batch, C_temporal, time]
```

Trong 06D cũ, nhánh này gồm các đặc trưng theo thời gian như MFCC, delta MFCC, delta-delta MFCC, RMS, ZCR, centroid, bandwidth, rolloff, spectral contrast. Kiến trúc:

```text
Conv1D -> BatchNorm -> GELU -> Conv1D -> BatchNorm -> GELU -> MaxPool
     -> BiLSTM -> attention pooling -> projection
```

Ý nghĩa: 1D CNN học pattern cục bộ theo thời gian, BiLSTM học ngữ cảnh trước-sau, attention pooling chọn frame quan trọng cho cảm xúc.

### Spectral branch

Input:

```text
[batch, C_spectral, n_mels, time]
```

Trong 06D cũ, nhánh này dùng log-Mel, delta log-Mel, delta-delta log-Mel. Kiến trúc:

```text
Residual SE Conv2D blocks -> AdaptiveAvgPool2D -> projection
```

Ý nghĩa: 2D CNN học vùng thời gian-tần số; SE block học trọng số channel/filter quan trọng.

### Stats branch

Input:

```text
[batch, D_stats]
```

MLP học từ các đặc trưng thống kê toàn utterance.

### Emotion2Vec branch

Input:

```text
[batch, D_e2v]
```

MLP adapter đưa embedding pretrained về cùng không gian ẩn với các nhánh acoustic.
"""),
        md(r"""
## Emotion-guided co-attention

06D dùng emotion2vec làm query để hỏi hai nhánh acoustic temporal/spectral:

$$
Q = z_{e2v}
$$

$$
K,V = [z_{temporal}, z_{spectral}]
$$

$$
z_{context}
=
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V
$$

Sau đó:

$$
\tilde{z}_{e2v}
=
\operatorname{LayerNorm}(z_{e2v}+z_{context})
$$

Fusion:

$$
z
=
f([z_t,z_s,z_e,\tilde{z}_e,z_{stats}])
$$

Từ shared representation \(z\), notebook tách ra hai head:

$$
\hat{p}=\operatorname{softmax}(W_cz+b_c)
$$

$$
\hat{y}_{VAD}=\sigma(W_rz+b_r)
$$
"""),
        md("""
## Data input cần có

03B-full **không cần raw audio nếu đã có cache full 06D**. Nếu chưa có cache, notebook có thể tự trích cache từ raw audio.

Tối thiểu cần:

```text
data/
  audio_wav/*.wav                         # cần khi phải tự build cache
  features/
    iemocap_full_06d_multibranch_cache.npz # nếu đã có thì dùng luôn
    iemocap_full_emotion2vec_acoustic_cache.npz # tùy chọn, dùng để lấy sẵn X_e2v
  splits/
    iemocap_5fold_session_long.csv
    iemocap_10fold_speaker_long.csv
```

Notebook cũng hỗ trợ tìm cache qua biến môi trường:

```text
FEATURE_CACHE_PATH=/kaggle/input/.../iemocap_full_06d_multibranch_cache.npz
```

Nếu chỉ có `iemocap_full_emotion2vec_acoustic_cache.npz`, notebook sẽ dùng phần `emotion2vec` trong đó làm `X_e2v`, rồi tự trích `X_temporal/X_spectral/X_stats` từ `audio_wav`.

Nếu không có cache emotion2vec light, notebook có thể tự gọi FunASR để trích emotion2vec khi `EXTRACT_EMOTION2VEC=1`; nếu không, đặt `ALLOW_ZERO_E2V=1` chỉ để debug.
"""),
        code(COMMON_SETUP),
        code(PATH_RESOLVER),
        code(METRICS_AND_ZIP),
        code(r"""
INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"
if INSTALL_DEPS:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile", "librosa", "scikit-learn"])

try:
    import soundfile as sf
    import librosa
except Exception as exc:
    raise ImportError("03B-full cần soundfile và librosa để tự trích full 06D cache. Trên Kaggle hãy bật Internet và set INSTALL_DEPS=1.") from exc

def find_optional_file(filename):
    try:
        return find_named_file(filename)
    except FileNotFoundError:
        return None

def find_existing_full_06d_cache():
    candidates = [
        "iemocap_full_06d_multibranch_cache.npz",
        "iemocap_06d_multibranch_cache.npz",
        "06d_iemocap_multibranch_cache.npz",
        "iemocap_full_06d_cache.npz",
    ]
    for name in candidates:
        found = find_optional_file(name)
        if found is not None:
            return found
    return None

SAMPLE_RATE = int(os.getenv("FULL06D_SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("FULL06D_MAX_SECONDS", "6.0"))
N_MFCC = int(os.getenv("FULL06D_N_MFCC", "40"))
N_MELS = int(os.getenv("FULL06D_N_MELS", "96"))
N_FFT = int(os.getenv("FULL06D_N_FFT", "640"))
HOP_LENGTH = int(os.getenv("FULL06D_HOP_LENGTH", "320"))
WIN_LENGTH = int(os.getenv("FULL06D_WIN_LENGTH", "640"))
MAX_AUDIO_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)

def fix_feature_time(x, target_frames):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == target_frames:
        return x
    if x.shape[-1] > target_frames:
        return x[..., :target_frames]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_frames - x.shape[-1])
    return np.pad(x, pad_width, mode="constant")

_AUDIO_INDEX = None

def audio_index(audio_dir):
    global _AUDIO_INDEX
    if _AUDIO_INDEX is None:
        _AUDIO_INDEX = {p.name: p.resolve() for p in audio_dir.rglob("*.wav")}
    return _AUDIO_INDEX

def resolve_wav_path_from_row(row, audio_dir):
    name = Path(str(row.get("wav_path", "")).replace("\\\\", "/")).name
    direct = audio_dir / name
    if direct.exists():
        return direct.resolve()
    idx = audio_index(audio_dir)
    if name in idx:
        return idx[name]
    utterance = str(row.get("utterance_id", ""))
    direct = audio_dir / f"{utterance}.wav"
    if direct.exists():
        return direct.resolve()
    raise FileNotFoundError(f"Không tìm thấy wav cho {utterance} ({name}) trong {audio_dir}")

def load_audio_for_06d(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        wav = np.zeros(MAX_AUDIO_SAMPLES, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    if len(wav) > MAX_AUDIO_SAMPLES:
        wav = wav[:MAX_AUDIO_SAMPLES]
    elif len(wav) < MAX_AUDIO_SAMPLES:
        wav = np.pad(wav, (0, MAX_AUDIO_SAMPLES - len(wav)), mode="constant")
    return wav.astype(np.float32)

def extract_06d_features_from_waveform(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    target_frames = mfcc.shape[-1]

    rms = fix_feature_time(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH), target_frames)
    zcr = fix_feature_time(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH), target_frames)
    centroid = fix_feature_time(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH), target_frames)
    bandwidth = fix_feature_time(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH), target_frames)
    rolloff = fix_feature_time(librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH), target_frames)
    try:
        contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    except Exception:
        contrast = np.zeros((7, target_frames), dtype=np.float32)
    contrast = fix_feature_time(contrast, target_frames)

    temporal = np.concatenate([
        fix_feature_time(mfcc, target_frames),
        fix_feature_time(delta, target_frames),
        fix_feature_time(delta2, target_frames),
        rms,
        zcr,
        centroid,
        bandwidth,
        rolloff,
        contrast,
    ], axis=0).astype(np.float32)

    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    logmel = fix_feature_time(logmel, target_frames)
    d_logmel = librosa.feature.delta(logmel)
    dd_logmel = librosa.feature.delta(logmel, order=2)
    spectral = np.stack([logmel, d_logmel, dd_logmel], axis=0).astype(np.float32)

    functionals = [
        temporal.mean(axis=1),
        temporal.std(axis=1),
        temporal.min(axis=1),
        temporal.max(axis=1),
        np.percentile(temporal, 25, axis=1),
        np.percentile(temporal, 75, axis=1),
    ]
    stats = np.concatenate(functionals + [
        np.asarray([
            MAX_SECONDS,
            float(np.mean(rms)),
            float(np.mean(zcr)),
            float(np.max(np.abs(y))) if y.size else 0.0,
        ], dtype=np.float32)
    ]).astype(np.float32)

    return temporal, spectral, stats

def load_light_emotion2vec_cache():
    light = find_optional_file("iemocap_full_emotion2vec_acoustic_cache.npz")
    if light is None:
        return None, {}
    data = np.load(light, allow_pickle=True)
    if "emotion2vec" not in data.files:
        return light, {}
    ids = None
    for key in ["sample_ids", "train_sample_id", "sample_id", "utterance_id"]:
        if key in data.files:
            ids = data[key].astype(str)
            break
    if ids is None:
        return light, {}
    mapping = {sid: data["emotion2vec"][i].astype(np.float32) for i, sid in enumerate(ids)}
    print("Loaded light emotion2vec cache:", light)
    return light, mapping

def build_emotion2vec_extractor_if_needed():
    if os.getenv("EXTRACT_EMOTION2VEC", "0") != "1":
        return None
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise ImportError("Muốn tự trích emotion2vec thì cần funasr. Bật Internet và cài funasr/modelscope, hoặc dùng light cache có sẵn.") from exc
    model_name = os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base")
    print("Loading emotion2vec extractor:", model_name)
    return AutoModel(model=model_name, disable_update=True)

def find_numeric_embedding(obj):
    candidates = []
    def visit(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            candidates.append(x.reshape(-1))
        elif isinstance(x, torch.Tensor):
            candidates.append(x.detach().cpu().numpy().reshape(-1))
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
    visit(obj)
    if not candidates:
        raise ValueError(f"Không tìm thấy embedding trong output emotion2vec type={type(obj)}")
    return sorted(candidates, key=lambda a: a.size, reverse=True)[0].astype(np.float32)

def extract_e2v_for_row(row, wav_path, light_map, extractor):
    sid = str(row["train_sample_id"])
    if sid in light_map:
        return light_map[sid]
    if extractor is not None:
        result = extractor.generate(input=str(wav_path), granularity="utterance", extract_embedding=True)
        return find_numeric_embedding(result)
    if os.getenv("ALLOW_ZERO_E2V", "0") == "1":
        return np.zeros(768, dtype=np.float32)
    raise ValueError(
        "Không có X_e2v cho sample và chưa bật EXTRACT_EMOTION2VEC=1. "
        "Cách nhanh nhất: upload thêm iemocap_full_emotion2vec_acoustic_cache.npz để lấy sẵn emotion2vec."
    )

def build_full_06d_cache_from_audio(output_path):
    audio_dir = find_audio_dir()
    print("AUDIO_DIR:", audio_dir)
    _, light_map = load_light_emotion2vec_cache()
    extractor = build_emotion2vec_extractor_if_needed()

    all_rows = pd.concat([df for df in SPLIT_TABLES.values()], ignore_index=True)
    all_rows = all_rows.drop_duplicates("train_sample_id").sort_values("train_sample_id").reset_index(drop=True)
    print("Unique samples to build:", len(all_rows))

    first_row = all_rows.iloc[0]
    first_wav = resolve_wav_path_from_row(first_row, audio_dir)
    temporal0, spectral0, stats0 = extract_06d_features_from_waveform(load_audio_for_06d(first_wav))
    e2v0 = extract_e2v_for_row(first_row, first_wav, light_map, extractor)

    n = len(all_rows)
    X_temporal_build = np.empty((n, *temporal0.shape), dtype=np.float32)
    X_spectral_build = np.empty((n, *spectral0.shape), dtype=np.float32)
    X_stats_build = np.empty((n, stats0.shape[0]), dtype=np.float32)
    X_e2v_build = np.zeros((n, e2v0.shape[0]), dtype=np.float32)
    sample_ids_build = all_rows["train_sample_id"].astype(str).to_numpy()
    utterance_ids_build = all_rows["utterance_id"].astype(str).to_numpy()

    X_temporal_build[0] = temporal0
    X_spectral_build[0] = spectral0
    X_stats_build[0] = stats0
    X_e2v_build[0, :e2v0.shape[0]] = e2v0

    failures = []
    for row_idx in range(1, n):
        row = all_rows.iloc[row_idx]
        try:
            wav_path = resolve_wav_path_from_row(row, audio_dir)
            y = load_audio_for_06d(wav_path)
            temporal, spectral, stats = extract_06d_features_from_waveform(y)
            e2v = extract_e2v_for_row(row, wav_path, light_map, extractor)
            X_temporal_build[row_idx] = temporal
            X_spectral_build[row_idx] = spectral
            X_stats_build[row_idx] = stats
            if e2v.shape[0] != X_e2v_build.shape[1]:
                fixed = np.zeros(X_e2v_build.shape[1], dtype=np.float32)
                fixed[:min(len(fixed), len(e2v))] = e2v[:min(len(fixed), len(e2v))]
                e2v = fixed
            X_e2v_build[row_idx] = e2v
        except Exception as exc:
            failures.append({"train_sample_id": str(row["train_sample_id"]), "utterance_id": str(row.get("utterance_id", "")), "error": repr(exc)})
        if (row_idx + 1) % 200 == 0 or row_idx == n - 1:
            print(f"Built {row_idx + 1}/{n} samples | failures={len(failures)}")

    if failures:
        failure_df = pd.DataFrame(failures)
        failure_path = output_path.with_name("iemocap_full_06d_multibranch_cache_failures.csv")
        failure_df.to_csv(failure_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Có {len(failures)} mẫu lỗi khi trích full 06D cache. Xem {failure_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_temporal=X_temporal_build,
        X_spectral=X_spectral_build,
        X_stats=X_stats_build,
        X_e2v=X_e2v_build,
        train_sample_id=sample_ids_build,
        sample_ids=sample_ids_build,
        utterance_id=utterance_ids_build,
        temporal_channels=np.asarray([f"temporal_{i:03d}" for i in range(X_temporal_build.shape[1])]),
        spectral_channels=np.asarray(["logmel", "delta_logmel", "delta_delta_logmel"]),
        config=np.asarray([json.dumps({
            "sample_rate": SAMPLE_RATE,
            "max_seconds": MAX_SECONDS,
            "n_mfcc": N_MFCC,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "stats_dim": int(X_stats_build.shape[1]),
        }, ensure_ascii=False)], dtype=object),
    )
    print("Saved full 06D cache:", output_path)
    return output_path

def resolve_full_06d_cache():
    env = os.getenv("FEATURE_CACHE_PATH", "").strip()
    if env and Path(env).exists():
        return Path(env).resolve()

    existing = find_existing_full_06d_cache()
    if existing is not None:
        return existing

    if os.getenv("BUILD_FULL06D_CACHE", "1") != "1":
        raise FileNotFoundError("Không tìm thấy full 06D cache và BUILD_FULL06D_CACHE=0.")

    output_dir = Path(os.getenv("OUTPUT_DIR", "output_03b_full_06d_coattention")).resolve() / "features"
    output_path = output_dir / "iemocap_full_06d_multibranch_cache.npz"
    if output_path.exists():
        return output_path.resolve()
    return build_full_06d_cache_from_audio(output_path).resolve()

FEATURE_CACHE_PATH = resolve_full_06d_cache()
cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
print("FEATURE_CACHE_PATH:", FEATURE_CACHE_PATH)
print("Cache keys:", cache.files)

required_keys = ["X_temporal", "X_spectral", "X_stats", "X_e2v"]
missing = [k for k in required_keys if k not in cache.files]
if missing:
    raise ValueError(f"Cache không đủ key cho 03B-full: thiếu {missing}. Keys hiện có: {cache.files}")

X_temporal = cache["X_temporal"].astype(np.float32)
X_spectral = cache["X_spectral"].astype(np.float32)
X_stats = cache["X_stats"].astype(np.float32)
X_e2v = cache["X_e2v"].astype(np.float32)

id_key = None
for key in ["train_sample_id", "sample_ids", "sample_id", "utterance_id"]:
    if key in cache.files:
        id_key = key
        break
if id_key is None:
    raise ValueError("Cache cần có một id key: train_sample_id, sample_ids, sample_id hoặc utterance_id.")

cache_ids = cache[id_key].astype(str)
feature_index = {sid: i for i, sid in enumerate(cache_ids)}

print("id_key:", id_key)
print("X_temporal:", X_temporal.shape)
print("X_spectral:", X_spectral.shape)
print("X_stats:", X_stats.shape)
print("X_e2v:", X_e2v.shape)
"""),
        code("""
EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "8e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.35"))
PATIENCE = int(os.getenv("PATIENCE", "12"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.03"))
USE_AUGMENTATION = os.getenv("USE_AUGMENTATION", "1") == "1"
TEMPORAL_AUG_PROB = float(os.getenv("TEMPORAL_AUG_PROB", "0.35"))
USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03b_full_06d_coattention")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:
    p.mkdir(parents=True, exist_ok=True)
print("Output:", OUTPUT_DIR)
"""),
        md("""
## Kiểm tra split và khả năng nối cache

Cell này kiểm tra `train_sample_id` trong split CSV có nối được với id trong cache không. Nếu mismatch nhiều, cần kiểm tra notebook 02 khi tạo cache.
"""),
        code("""
for protocol, df in SPLIT_TABLES.items():
    matched = df["train_sample_id"].astype(str).isin(feature_index).mean()
    print(protocol, "matched ratio:", round(float(matched), 4), "rows:", len(df))
    if matched < 0.99:
        missing_ids = df.loc[~df["train_sample_id"].astype(str).isin(feature_index), "train_sample_id"].astype(str).head(10).tolist()
        raise ValueError(f"{protocol}: nhiều sample không tìm thấy trong full cache. Ví dụ missing ids: {missing_ids}")
"""),
        code(r"""
def compute_scalers(indices):
    scalers = {}
    scalers["temporal_mean"] = X_temporal[indices].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    scalers["temporal_std"] = (X_temporal[indices].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    scalers["spectral_mean"] = X_spectral[indices].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    scalers["spectral_std"] = (X_spectral[indices].std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
    scalers["stats_scaler"] = StandardScaler().fit(X_stats[indices])
    scalers["e2v_scaler"] = StandardScaler().fit(X_e2v[indices])
    return scalers

def augment_temporal(x):
    if random.random() > TEMPORAL_AUG_PROB:
        return x
    x = x.copy()
    frames = x.shape[-1]
    if frames > 8 and random.random() < 0.5:
        width = random.randint(4, min(32, frames))
        start = random.randint(0, max(0, frames - width))
        x[:, start:start + width] = 0
    if random.random() < 0.35:
        x += np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
    return x

def augment_spectral(x):
    if random.random() > TEMPORAL_AUG_PROB:
        return x
    x = x.copy()
    _, n_mels, frames = x.shape
    if frames > 8 and random.random() < 0.6:
        width = random.randint(4, min(24, frames))
        start = random.randint(0, max(0, frames - width))
        x[:, :, start:start + width] = 0
    if n_mels > 8 and random.random() < 0.6:
        width = random.randint(4, min(20, n_mels))
        start = random.randint(0, max(0, n_mels - width))
        x[:, start:start + width, :] = 0
    return x

class Full06DDataset(Dataset):
    def __init__(self, df, scalers, train=False):
        self.df = df.reset_index(drop=True).copy()
        self.indices = np.asarray([feature_index[str(sid)] for sid in self.df["train_sample_id"].astype(str)], dtype=np.int64)
        self.scalers = scalers
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        i = self.indices[item]
        temporal = (X_temporal[i] - self.scalers["temporal_mean"][0]) / self.scalers["temporal_std"][0]
        spectral = (X_spectral[i] - self.scalers["spectral_mean"][0]) / self.scalers["spectral_std"][0]
        if self.train and USE_AUGMENTATION:
            temporal = augment_temporal(temporal)
            spectral = augment_spectral(spectral)
        stats = self.scalers["stats_scaler"].transform(X_stats[i:i+1]).astype(np.float32)[0]
        e2v = self.scalers["e2v_scaler"].transform(X_e2v[i:i+1]).astype(np.float32)[0]
        vad = vad_to_0_1(row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32))
        return {
            "temporal": torch.tensor(temporal, dtype=torch.float32),
            "spectral": torch.tensor(spectral, dtype=torch.float32),
            "stats": torch.tensor(stats, dtype=torch.float32),
            "e2v": torch.tensor(e2v, dtype=torch.float32),
            "emotion_id": torch.tensor(int(row["emotion_id"]), dtype=torch.long),
            "vad": torch.tensor(vad, dtype=torch.float32),
            "utterance_id": str(row["utterance_id"]),
            "train_sample_id": str(row["train_sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
        }

def collate_full06d(batch):
    return {
        "temporal": torch.stack([b["temporal"] for b in batch]),
        "spectral": torch.stack([b["spectral"] for b in batch]),
        "stats": torch.stack([b["stats"] for b in batch]),
        "e2v": torch.stack([b["e2v"] for b in batch]),
        "emotion_id": torch.stack([b["emotion_id"] for b in batch]),
        "vad": torch.stack([b["vad"] for b in batch]),
        "utterance_id": [b["utterance_id"] for b in batch],
        "train_sample_id": [b["train_sample_id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "session": [b["session"] for b in batch],
    }

def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=DEVICE.type == "cuda", collate_fn=collate_full06d)

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
"""),
        md("""
## Model blocks từ 06D, chuyển sang multi-task 2-head

Phần dưới giữ tinh thần kiến trúc 06D cũ, nhưng thay classifier đơn bằng hai head:

- `emotion_head`: 4-class emotion.
- `vad_head`: valence/arousal/dominance.
"""),
        code(r"""
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))
    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1), weights

class SE2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SE2D(out_ch),
        )
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class TemporalBranch(nn.Module):
    def __init__(self, in_channels, out_dim=192):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.5),
            nn.Conv1d(128, 160, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(160),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Dropout(DROPOUT * 0.5),
        )
        self.rnn = nn.LSTM(input_size=160, hidden_size=96, num_layers=1, batch_first=True, bidirectional=True)
        self.pool = AttentionPooling(192)
        self.proj = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, out_dim), nn.GELU(), nn.Dropout(DROPOUT))
    def forward(self, x):
        x = self.cnn(x).transpose(1, 2)
        x, _ = self.rnn(x)
        pooled, attn = self.pool(x)
        return self.proj(pooled), attn

class SpectralBranch(nn.Module):
    def __init__(self, in_channels, out_dim=192):
        super().__init__()
        self.net = nn.Sequential(
            ResidualSEBlock(in_channels, 32),
            nn.MaxPool2d(2),
            ResidualSEBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualSEBlock(64, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, out_dim), nn.GELU(), nn.Dropout(DROPOUT))
    def forward(self, x):
        return self.proj(self.net(x))

class MLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(hidden, out_dim), nn.GELU(), nn.Dropout(DROPOUT))
    def forward(self, x):
        return self.net(x)

class EmotionGuidedCoAttention(nn.Module):
    def __init__(self, dim=192, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=DROPOUT * 0.5)
        self.norm = nn.LayerNorm(dim)
    def forward(self, z_e2v, z_temporal, z_spectral):
        q = z_e2v.unsqueeze(1)
        kv = torch.stack([z_temporal, z_spectral], dim=1)
        context, weights = self.attn(q, kv, kv, need_weights=True)
        context = context.squeeze(1)
        return self.norm(z_e2v + context), weights

class Full06DCoAttentionMultiTaskSER(nn.Module):
    def __init__(self, temporal_dim, spectral_channels, stats_dim, e2v_dim, num_classes=4):
        super().__init__()
        self.temporal_branch = TemporalBranch(temporal_dim, out_dim=192)
        self.spectral_branch = SpectralBranch(spectral_channels, out_dim=192)
        self.e2v_branch = MLPBranch(e2v_dim, out_dim=192, hidden=384)
        self.stats_branch = MLPBranch(stats_dim, out_dim=128, hidden=256)
        self.co_attention = EmotionGuidedCoAttention(dim=192, heads=4)
        self.fusion = nn.Sequential(
            nn.Linear(192 + 192 + 192 + 192 + 128, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(384, 192),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.emotion_head = nn.Linear(192, num_classes)
        self.vad_head = nn.Sequential(nn.Linear(192, 3), nn.Sigmoid())

    def forward(self, temporal, spectral, stats, e2v, return_embedding=False):
        z_t, temporal_attn = self.temporal_branch(temporal)
        z_s = self.spectral_branch(spectral)
        z_e = self.e2v_branch(e2v)
        z_stats = self.stats_branch(stats)
        z_context, co_weights = self.co_attention(z_e, z_t, z_s)
        z = torch.cat([z_t, z_s, z_e, z_context, z_stats], dim=1)
        fused = self.fusion(z)
        out = {"emotion_logits": self.emotion_head(fused), "vad_pred": self.vad_head(fused)}
        if return_embedding:
            out["embedding"] = fused
            out["z_temporal"] = z_t
            out["z_spectral"] = z_s
            out["z_e2v"] = z_e
            out["z_stats"] = z_stats
            out["co_weights"] = co_weights
            out["temporal_attn"] = temporal_attn
        return out

model_preview = Full06DCoAttentionMultiTaskSER(X_temporal.shape[1], X_spectral.shape[1], X_stats.shape[1], X_e2v.shape[1], NUM_CLASSES)
print("Parameters:", sum(p.numel() for p in model_preview.parameters()))
del model_preview
"""),
        code(r"""
@torch.no_grad()
def evaluate(model, loader, split_name):
    if len(loader.dataset) == 0:
        raise ValueError(f"Split {split_name} rỗng.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, embeddings, probs = [], [], [], [], [], []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = model(batch["temporal"], batch["spectral"], batch["stats"], batch["e2v"], return_embedding=True)
        loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
        prob = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        embeddings.append(outputs["embedding"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        for i, uid in enumerate(batch["utterance_id"]):
            rows.append({"utterance_id": uid, "train_sample_id": batch["train_sample_id"][i], "speaker_id": batch["speaker_id"][i], "session": batch["session"][i], "split": split_name})
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError(f"Không có batch nào trong split {split_name}.")
    vad_true = np.concatenate(vad_true)
    vad_pred = np.concatenate(vad_pred)
    embeddings = np.concatenate(embeddings)
    probs = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame(rows)
    pred_df["true_emotion_id"] = y_true
    pred_df["pred_emotion_id"] = y_pred
    for i, name in EMOTION_ID_TO_NAME.items():
        pred_df[f"prob_{name}"] = probs[:, i]
    for j, name in enumerate(["valence", "arousal", "dominance"]):
        pred_df[f"true_{name}"] = vad_from_0_1(vad_true)[:, j]
        pred_df[f"pred_{name}"] = vad_from_0_1(vad_pred)[:, j]
    feature_npz = {
        "utterance_id": pred_df["utterance_id"].to_numpy(),
        "train_sample_id": pred_df["train_sample_id"].to_numpy(),
        "embedding": embeddings.astype(np.float32),
        "emotion_probs": probs.astype(np.float32),
        "vad_pred": vad_pred.astype(np.float32),
        "emotion_true": np.asarray(y_true, dtype=np.int64),
        "vad_true": vad_true.astype(np.float32),
    }
    return metrics, pred_df, feature_npz

def class_weights_for(df):
    labels = df["emotion_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_indices = np.asarray([feature_index[str(sid)] for sid in train_df["train_sample_id"].astype(str)], dtype=np.int64)
    scalers = compute_scalers(train_indices)
    train_ds = Full06DDataset(train_df, scalers, train=True)
    val_ds = Full06DDataset(val_df, scalers, train=False)
    test_ds = Full06DDataset(test_df, scalers, train=False)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = Full06DCoAttentionMultiTaskSER(X_temporal.shape[1], X_spectral.shape[1], X_stats.shape[1], X_e2v.shape[1], NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_score, best_epoch, bad_epochs = -1e9, -1, 0
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["temporal"], batch["spectral"], batch["stats"], batch["e2v"])
            loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
        val_metrics, _, _ = evaluate(model, val_loader, "val")
        score = primary_score(val_metrics)
        row = {"protocol": protocol, "fold": fold_name, "epoch": epoch, "train_loss": running / max(len(train_loader), 1), "val_primary_score": score, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}")
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])
    split_outputs = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, pred_df, feature_npz = evaluate(model, loader, split_name)
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_coattention_features.npz", **feature_npz)
        split_outputs[split_name] = metrics
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")
    result = {"protocol": protocol, "fold": fold_name, "best_epoch": best_epoch, "best_val_score": best_score, "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df), **split_outputs["test"]}
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})
    return result
"""),
        code("""
all_results = []
start_all = time.time()
for protocol in RUN_PROTOCOLS:
    df = SPLIT_TABLES[protocol]
    folds = sorted(df["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS > 0:
        folds = folds[:MAX_FOLDS]
    for idx, fold in enumerate(folds, start=1):
        all_results.append(train_fold(protocol, fold, df[df["fold"] == fold].reset_index(drop=True), SEED + idx))

results_df = pd.DataFrame(all_results)
results_df.to_csv(REPORT_DIR / "03b_full_06d_coattention_results_by_fold.csv", index=False, encoding="utf-8-sig")
display(results_df)
summary = results_df.groupby("protocol")[["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]].agg(["mean", "std"]).round(4)
summary.to_csv(REPORT_DIR / "03b_full_06d_coattention_summary.csv", encoding="utf-8-sig")
display(summary)
print("Total seconds:", round(time.time() - start_all, 2))
"""),
        code("""
config = {
    "notebook": "03B full 06D multi-branch co-attention multi-task model",
    "feature_cache_path": str(FEATURE_CACHE_PATH),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "dropout": DROPOUT,
    "patience": PATIENCE,
    "use_augmentation": USE_AUGMENTATION,
    "run_protocols": RUN_PROTOCOLS,
    "architecture": {
        "temporal": "Conv1D + BiLSTM + attention pooling",
        "spectral": "Residual SE Conv2D blocks",
        "stats": "MLP",
        "emotion2vec": "MLP adapter",
        "fusion": "Emotion2Vec-guided co-attention + fusion MLP",
        "heads": "emotion classification + VAD regression",
    },
}
(OUTPUT_DIR / "03b_full_06d_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""),
    ]
    path = BASE / "03_Emotion2Vec Downstream Finetune 5Fold 10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
    write_notebook(path, cells)


def main():
    build_raw_audio_notebook()
    build_coattention_notebook()
    build_fusion_notebook()
    build_coattention_full_06d_notebook()


if __name__ == "__main__":
    main()
