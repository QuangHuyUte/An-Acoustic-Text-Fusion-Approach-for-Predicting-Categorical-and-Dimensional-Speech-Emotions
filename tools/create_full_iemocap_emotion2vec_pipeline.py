from pathlib import Path, PureWindowsPath

import nbformat as nbf


BASE = Path(r"D:/UTE/Speech Programming/Speech Project")
ROOT = BASE / "06_w2v_based_models"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


COMMON_IMPORTS = r'''
from pathlib import Path
import json
import math
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import display, Markdown
except Exception:
    display = print
    Markdown = lambda x: x

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)
'''


ZIP_HELPERS = r'''
def zip_folder(source_dir, zip_path, exclude_dir_names=None, exclude_suffixes=None):
    source_dir = Path(source_dir)
    zip_path = Path(zip_path)
    exclude_dir_names = set(exclude_dir_names or [])
    exclude_suffixes = set(exclude_suffixes or [])
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if any(part in exclude_dir_names for part in file_path.relative_to(source_dir).parts):
                continue
            if file_path.suffix.lower() in exclude_suffixes:
                continue
            zf.write(file_path, file_path.relative_to(source_dir))
    return zip_path
'''


PATH_RESOLVER = r'''
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name != "Speech Project" and PROJECT_ROOT.parent.name == "Speech Project":
    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()

BASE_DIR = PROJECT_ROOT / "06_w2v_based_models"
if not BASE_DIR.exists() and Path("/kaggle/working").exists():
    BASE_DIR = Path("/kaggle/working") / "06_w2v_based_models"

LOCAL_DATA_DIR = PROJECT_ROOT / "06_w2v_based_models" / "data"

def has_train_tables(path):
    return (
        path is not None
        and (path / "metadata" / "iemocap_metadata_full.csv").exists()
        and (path / "splits").exists()
    )

def candidate_data_roots(base):
    if base is None:
        return []
    base = Path(base)
    roots = [base, base / "data"]
    if base.exists():
        for meta_path in base.rglob("iemocap_metadata_full.csv"):
            roots.append(meta_path.parent.parent)
    return roots

def resolve_data_dir():
    candidates = []
    env_dir = os.environ.get("IEMOCAP_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([
        LOCAL_DATA_DIR,
        Path("/kaggle/input/iemocap-full-multitask-data"),
        Path("/kaggle/input/iemocap_full_multitask_data"),
        Path("/kaggle/input/iemocap-multitask-train-data"),
        Path("/kaggle/input/iemocap_multitask_train_data"),
    ])

    for candidate in candidates:
        for root in candidate_data_roots(candidate):
            if has_train_tables(root):
                return root.resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root in candidate_data_roots(kaggle_input):
            if has_train_tables(root):
                return root.resolve()

    raise FileNotFoundError(
        "Không tìm thấy dữ liệu IEMOCAP. Notebook sẽ tự quét /kaggle/input, nhưng folder dữ liệu cần có "
        "metadata/iemocap_metadata_full.csv và splits/. Nếu bạn đặt ở chỗ khác, set IEMOCAP_DATA_DIR."
    )

DATA_DIR = resolve_data_dir()
METADATA_DIR = DATA_DIR / "metadata"
SPLIT_DIR = DATA_DIR / "splits"
AUDIO_DIR = DATA_DIR / "audio_wav"

FULL_METADATA_PATH = METADATA_DIR / "iemocap_metadata_full.csv"

WORKING_DATA_DIR = Path("/kaggle/working/iemocap_full_multitask_data") if Path("/kaggle/working").exists() else DATA_DIR
WORKING_FEATURE_DIR = WORKING_DATA_DIR / "features"
WORKING_REPORT_DIR = WORKING_DATA_DIR / "feature_reports"
WORKING_FIGURE_DIR = WORKING_DATA_DIR / "feature_figures"
INPUT_FEATURE_DIR = DATA_DIR / "features"
FEATURE_CACHE_NAME = "iemocap_full_emotion2vec_acoustic_cache.npz"
WORKING_FEATURE_CACHE_PATH = WORKING_FEATURE_DIR / FEATURE_CACHE_NAME
INPUT_FEATURE_CACHE_PATH = INPUT_FEATURE_DIR / FEATURE_CACHE_NAME

def resolve_feature_cache_path(require_exists=False):
    env_cache = os.environ.get("IEMOCAP_FEATURE_CACHE", "").strip()
    candidates = []
    if env_cache:
        candidates.append(Path(env_cache))
    candidates.extend([WORKING_FEATURE_CACHE_PATH, INPUT_FEATURE_CACHE_PATH])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if require_exists:
        raise FileNotFoundError(
            "Không tìm thấy full feature cache. Hãy chạy notebook 02 trước. Trên Kaggle, notebook 02 ghi cache vào "
            f"{WORKING_FEATURE_CACHE_PATH}, không ghi vào /kaggle/input vì Kaggle input dataset là read-only."
        )
    return WORKING_FEATURE_CACHE_PATH

print("DATA_DIR:", DATA_DIR)
print("AUDIO_DIR:", AUDIO_DIR, AUDIO_DIR.exists())
print("INPUT_FEATURE_CACHE_PATH:", INPUT_FEATURE_CACHE_PATH, INPUT_FEATURE_CACHE_PATH.exists())
print("WORKING_FEATURE_CACHE_PATH:", WORKING_FEATURE_CACHE_PATH, WORKING_FEATURE_CACHE_PATH.exists())
print("WORKING_REPORT_DIR:", WORKING_REPORT_DIR)
print("WORKING_FIGURE_DIR:", WORKING_FIGURE_DIR)
'''


FEATURE_EXTRACTION_HELPERS = r'''
TARGET_SR = 16000
MAX_FILES = os.getenv("MAX_FILES", "").strip()
MAX_FILES = int(MAX_FILES) if MAX_FILES else None
EMOTION2VEC_MODEL_NAME = os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base")
IS_KAGGLE = Path("/kaggle/working").exists()
INSTALL_DEPS = os.getenv("INSTALL_EMOTION2VEC_DEPS", "1" if IS_KAGGLE else "0") == "1"

def maybe_install_deps():
    if not INSTALL_DEPS:
        return
    import importlib.util
    import subprocess
    import sys
    missing = [pkg for pkg in ["funasr", "modelscope", "librosa", "soundfile"] if importlib.util.find_spec(pkg) is None]
    if missing:
        print("Đang cài package còn thiếu cho Emotion2Vec:", missing)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "funasr", "modelscope", "librosa", "soundfile",
            "addict", "simplejson", "sortedcontainers",
        ])
    else:
        print("Các package Emotion2Vec đã có sẵn.")

def require_audio_dir():
    if not AUDIO_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy audio directory: {AUDIO_DIR}. Với mô hình full, Kaggle Dataset cần có audio_wav/, "
            "hoặc bạn phải upload sẵn feature cache vào features/."
        )

def load_audio_16k(path):
    import librosa
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if len(y) == 0:
        y = np.zeros(TARGET_SR, dtype=np.float32)
    peak = np.max(np.abs(y)) + 1e-8
    y = (y / peak).astype(np.float32)
    return y, TARGET_SR

def acoustic_vector(y, sr):
    import librosa
    feats = []
    names = []

    def add(name, arr):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        feats.extend([float(np.nanmean(arr)), float(np.nanstd(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))])
        names.extend([f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max"])

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)

    add("rms", rms)
    add("zcr", zcr)
    add("centroid", centroid)
    add("bandwidth", bandwidth)
    add("rolloff", rolloff)
    for i in range(mfcc.shape[0]):
        add(f"mfcc_{i+1:02d}", mfcc[i])
        add(f"delta_mfcc_{i+1:02d}", delta[i])
    add("logmel", mel)

    return np.asarray(feats, dtype=np.float32), names

def load_emotion2vec_model():
    maybe_install_deps()
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise ImportError(
            "Không import được funasr. Trên Kaggle hãy bật Internet. Notebook đã mặc định tự cài khi ở Kaggle; "
            "nếu bạn tắt cơ chế này thì set INSTALL_EMOTION2VEC_DEPS=1 rồi chạy lại cell."
        ) from exc
    print("Loading emotion2vec:", EMOTION2VEC_MODEL_NAME)
    return AutoModel(model=EMOTION2VEC_MODEL_NAME, hub="ms")

def first_numeric_array(obj):
    if isinstance(obj, np.ndarray) and np.issubdtype(obj.dtype, np.number):
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        preferred = ["feats", "embedding", "embeddings", "hidden_states", "xvector", "scores"]
        for key in preferred:
            if key in obj:
                arr = first_numeric_array(obj[key])
                if arr is not None:
                    return arr
        for value in obj.values():
            arr = first_numeric_array(value)
            if arr is not None:
                return arr
    if isinstance(obj, (list, tuple)):
        for value in obj:
            arr = first_numeric_array(value)
            if arr is not None:
                return arr
    return None

def extract_emotion2vec_embedding(model, wav_path):
    result = model.generate(str(wav_path), granularity="utterance", extract_embedding=True)
    arr = first_numeric_array(result)
    if arr is None:
        raise ValueError(f"No numeric emotion2vec embedding returned for {wav_path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    return arr.astype(np.float32)

_AUDIO_INDEX = None

def build_audio_index():
    global _AUDIO_INDEX
    if _AUDIO_INDEX is None:
        _AUDIO_INDEX = {p.name: p for p in AUDIO_DIR.rglob("*.wav")} if AUDIO_DIR.exists() else {}
        print(f"Audio index: {len(_AUDIO_INDEX)} wav files under {AUDIO_DIR}")
    return _AUDIO_INDEX

def path_basename_any_os(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    # Metadata was created on Windows, while Kaggle runs Linux. Normalize both styles.
    return Path(text.replace("\\", "/")).name or PureWindowsPath(text).name

def wav_name_candidates(row):
    candidates = []
    for key in ["source_filename", "wav_path", "utterance_id"]:
        if key in row and pd.notna(row.get(key)):
            value = str(row.get(key))
            if key == "utterance_id" and not value.lower().endswith(".wav"):
                value = value + ".wav"
            name = path_basename_any_os(value)
            if name:
                candidates.append(name)
    # Keep order but remove duplicates.
    return list(dict.fromkeys(candidates))

def resolve_wav_path(row):
    original = Path(str(row.get("wav_path", "")))
    if original.exists():
        return original

    audio_index = build_audio_index()
    candidates = wav_name_candidates(row)
    for name in candidates:
        direct = AUDIO_DIR / name
        if direct.exists():
            return direct
        if name in audio_index:
            return audio_index[name]

    raise FileNotFoundError(
        "Không tìm thấy WAV cho sample "
        f"{row.get('train_sample_id')}. Candidates={candidates}. "
        f"AUDIO_DIR={AUDIO_DIR}. Số file wav đã index={len(audio_index)}."
    )
'''


MODEL_HELPERS = r'''
EMOTIONS = ["neutral", "angry", "sad", "happy"]
ID_TO_EMOTION = {0: "neutral", 1: "angry", 2: "sad", 3: "happy"}

EPOCHS = int(os.getenv("EPOCHS", "25"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
LR = float(os.getenv("LR", "8e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
PATIENCE = int(os.getenv("PATIENCE", "7"))
LOSS_EMOTION_WEIGHT = float(os.getenv("LOSS_EMOTION_WEIGHT", "1.0"))
LOSS_AVD_WEIGHT = float(os.getenv("LOSS_AVD_WEIGHT", "0.35"))
LOSS_CCC_WEIGHT = float(os.getenv("LOSS_CCC_WEIGHT", "0.65"))

def normalize_avd(df):
    out = df.copy()
    for src, dst in [("valence", "valence_norm"), ("arousal", "arousal_norm"), ("dominance", "dominance_norm")]:
        if dst not in out.columns:
            out[dst] = (pd.to_numeric(out[src], errors="coerce") - 1.0) / 4.0
        out[dst] = out[dst].clip(0.0, 1.0)
    return out

def load_feature_cache(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy full feature cache: {path}. Hãy chạy notebook 02 trước, hoặc upload file .npz này vào data/features/."
        )
    cache = np.load(path, allow_pickle=True)
    return {
        "sample_ids": cache["sample_ids"].astype(str),
        "emotion2vec": cache["emotion2vec"].astype(np.float32),
        "acoustic": cache["acoustic"].astype(np.float32),
        "acoustic_names": cache["acoustic_names"].astype(str) if "acoustic_names" in cache else np.array([]),
    }

def load_and_align_split(split_path, metadata_path, cache):
    split_df = pd.read_csv(split_path)
    meta = pd.read_csv(metadata_path)
    extra_cols = ["train_sample_id", "transcription", "source_filename", "duration", "sample_rate", "channels"]
    extra_cols = [c for c in extra_cols if c in meta.columns]
    split_df = split_df.merge(meta[extra_cols].drop_duplicates("train_sample_id"), on="train_sample_id", how="left")
    split_df = normalize_avd(split_df)

    id_to_idx = {sid: i for i, sid in enumerate(cache["sample_ids"])}
    split_df["feature_idx"] = split_df["train_sample_id"].astype(str).map(id_to_idx)
    missing = split_df["feature_idx"].isna().sum()
    if missing:
        missing_ids = split_df.loc[split_df["feature_idx"].isna(), "train_sample_id"].head(10).tolist()
        raise ValueError(f"{missing} rows have no feature cache. First missing ids: {missing_ids}")
    split_df["feature_idx"] = split_df["feature_idx"].astype(int)
    return split_df

class FullFeatureDataset(Dataset):
    def __init__(self, df, cache, e_scaler=None, a_scaler=None, fit_scalers=False):
        self.df = df.reset_index(drop=True).copy()
        idx = self.df["feature_idx"].values
        e = cache["emotion2vec"][idx]
        a = cache["acoustic"][idx]
        if fit_scalers:
            self.e_scaler = StandardScaler().fit(e)
            self.a_scaler = StandardScaler().fit(a)
        else:
            self.e_scaler = e_scaler
            self.a_scaler = a_scaler
        self.e = torch.tensor(self.e_scaler.transform(e), dtype=torch.float32)
        self.a = torch.tensor(self.a_scaler.transform(a), dtype=torch.float32)
        self.y_cls = torch.tensor(self.df["emotion_id"].values, dtype=torch.long)
        self.y_reg = torch.tensor(self.df[["valence_norm", "arousal_norm", "dominance_norm"]].values, dtype=torch.float32)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return {"e2v": self.e[idx], "acoustic": self.a[idx], "emotion": self.y_cls[idx], "avd": self.y_reg[idx], "idx": idx}

class MLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim=192, hidden=384, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class Emotion2VecAcousticCoAttentionMultiTask(nn.Module):
    """
    Full proposed model:
      Branch A: frozen emotion2vec embedding -> adapter MLP -> z_e
      Branch B: handcrafted acoustic vector -> acoustic MLP -> z_a
      Cross attention: acoustic query attends to emotion2vec and emotion2vec query attends to acoustic
      Fusion: gated fusion -> shared representation
      Head 1: 4-class emotion classification
      Head 2: valence/arousal/dominance regression
    """
    def __init__(self, e2v_dim, acoustic_dim, hidden=192, heads=4, dropout=0.25, n_classes=4):
        super().__init__()
        self.e2v_branch = MLPBranch(e2v_dim, out_dim=hidden, hidden=hidden * 2, dropout=dropout)
        self.acoustic_branch = MLPBranch(acoustic_dim, out_dim=hidden, hidden=hidden * 2, dropout=dropout)
        self.a_queries_e = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.e_queries_a = nn.MultiheadAttention(hidden, heads, dropout=0.1, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden, n_classes)
        self.avd_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 3), nn.Sigmoid())
    def forward(self, e2v, acoustic):
        z_e = self.e2v_branch(e2v)
        z_a = self.acoustic_branch(acoustic)
        e_tok = z_e.unsqueeze(1)
        a_tok = z_a.unsqueeze(1)
        a_ctx, _ = self.a_queries_e(query=a_tok, key=e_tok, value=e_tok)
        e_ctx, _ = self.e_queries_a(query=e_tok, key=a_tok, value=a_tok)
        parts = torch.cat([z_e, z_a, e_ctx.squeeze(1), a_ctx.squeeze(1)], dim=-1)
        g = self.gate(parts)
        fused = self.fusion(parts)
        z = g * fused + (1.0 - g) * 0.5 * (z_e + z_a)
        return self.emotion_head(z), self.avd_head(z)

def ccc_torch(pred, target, eps=1e-8):
    pred_mean = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)
    pred_var = torch.var(pred, dim=0, unbiased=False)
    target_var = torch.var(target, dim=0, unbiased=False)
    cov = torch.mean((pred - pred_mean) * (target - target_mean), dim=0)
    return (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + eps)

def ccc_loss(pred, target):
    return 1.0 - ccc_torch(pred, target).mean()

def ccc_np(pred, target, eps=1e-8):
    vals = []
    for i in range(pred.shape[1]):
        x, y = pred[:, i], target[:, i]
        mx, my = x.mean(), y.mean()
        vx, vy = x.var(), y.var()
        cov = np.mean((x - mx) * (y - my))
        vals.append((2 * cov) / (vx + vy + (mx - my) ** 2 + eps))
    return np.asarray(vals)

def class_weights(train_df):
    counts = train_df["emotion_id"].value_counts().sort_index()
    total = counts.sum()
    return torch.tensor([total / (4.0 * counts.get(i, 1)) for i in range(4)], dtype=torch.float32)

def run_epoch(model, loader, optimizer, ce):
    model.train()
    losses = []
    for batch in loader:
        e = batch["e2v"].to(DEVICE)
        a = batch["acoustic"].to(DEVICE)
        yc = batch["emotion"].to(DEVICE)
        yr = batch["avd"].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits, pred = model(e, a)
        loss = LOSS_EMOTION_WEIGHT * ce(logits, yc) + LOSS_AVD_WEIGHT * nn.functional.smooth_l1_loss(pred, yr) + LOSS_CCC_WEIGHT * ccc_loss(pred, yr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))

@torch.no_grad()
def predict(model, loader):
    model.eval()
    logits, reg, yc, yr = [], [], [], []
    for batch in loader:
        lo, pr = model(batch["e2v"].to(DEVICE), batch["acoustic"].to(DEVICE))
        logits.append(lo.cpu().numpy())
        reg.append(pr.cpu().numpy())
        yc.append(batch["emotion"].numpy())
        yr.append(batch["avd"].numpy())
    return {"logits": np.concatenate(logits), "reg": np.concatenate(reg), "y_cls": np.concatenate(yc), "y_reg": np.concatenate(yr)}

def compute_metrics(pred):
    y_true = pred["y_cls"]
    y_pred = pred["logits"].argmax(axis=1)
    ccc = ccc_np(pred["reg"], pred["y_reg"])
    mae = np.mean(np.abs(pred["reg"] - pred["y_reg"]), axis=0)
    rmse = np.sqrt(np.mean((pred["reg"] - pred["y_reg"]) ** 2, axis=0))
    return {
        "WA": accuracy_score(y_true, y_pred),
        "UAR": balanced_accuracy_score(y_true, y_pred),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "CCC_valence": ccc[0],
        "CCC_arousal": ccc[1],
        "CCC_dominance": ccc[2],
        "CCC_mean": float(ccc.mean()),
        "MAE_mean": float(mae.mean()),
        "RMSE_mean": float(rmse.mean()),
    }

def primary_score(m):
    return float(m["UAR"] + m["CCC_mean"])
'''


def feature_notebook():
    return [
        md(
            """
# 02 - Trích Xuất Đặc Trưng IEMOCAP Full: Emotion2Vec + Acoustic

Notebook này tạo **full feature cache** cho model 03/04.

Output bắt buộc:

```text
data/features/iemocap_full_emotion2vec_acoustic_cache.npz
  sample_ids
  emotion2vec      # Branch A: frozen Emotion2Vec utterance embedding
  acoustic         # Branch B: handcrafted acoustic vector
  acoustic_names
```

Khác với bản baseline trước, notebook này không coi metadata là feature chính. Muốn train model full thì phải có cache này.
"""
        ),
        code(COMMON_IMPORTS),
        code(ZIP_HELPERS),
        code(PATH_RESOLVER),
        code(FEATURE_EXTRACTION_HELPERS),
        code(
            r'''
require_audio_dir()
WORKING_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
WORKING_REPORT_DIR.mkdir(parents=True, exist_ok=True)
WORKING_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
metadata = pd.read_csv(FULL_METADATA_PATH)
metadata = metadata[metadata["emotion_4class"].isin(["neutral", "angry", "sad", "happy"])].copy()
metadata = metadata.drop_duplicates("train_sample_id").reset_index(drop=True)
if MAX_FILES is not None:
    metadata = metadata.head(MAX_FILES).copy()
print("Rows to extract:", len(metadata))
display(metadata[["train_sample_id", "emotion_4class", "wav_path"]].head())
'''
        ),
        md(
            """
## 1. Bảng đặc trưng và vai trò trong mô hình

Phần này ghi rõ mỗi nhóm đặc trưng được trích như thế nào, phục vụ nhánh nào trong mô hình, và vì sao cần nó.

Điểm quan trọng của mô hình full:

- `Emotion2Vec` là **nhánh riêng**, nhận waveform 16 kHz và tạo embedding cảm xúc mức cao.
- Acoustic handcrafted feature là **nhánh riêng**, dùng để bổ sung các tín hiệu mà embedding pretrained có thể không biểu diễn rõ: năng lượng, texture phổ, MFCC dynamics.
- Hai nhánh không thay thế nhau; chúng được fusion bằng cross-attention trong notebook 03/04.
"""
        ),
        code(
            r'''
feature_reference_rows = [
    {
        "feature_group": "Emotion2Vec embedding",
        "branch": "Branch A - pretrained emotion representation",
        "how_to_extract": "waveform 16 kHz -> frozen emotion2vec_base -> utterance embedding",
        "why_it_matters": "Cung cấp high-level emotion cue đã học trước từ speech emotion representation.",
        "used_by_model": "Emotion2Vec adapter MLP, cross-attention fusion, emotion/AVD heads",
        "reference": "emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation; ModelScope iic/emotion2vec_base",
    },
    {
        "feature_group": "MFCC",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "waveform -> Mel spectrum -> cepstral coefficients",
        "why_it_matters": "Mô tả timbre/màu giọng và spectral envelope.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "SER feature engineering; librosa.feature.mfcc",
    },
    {
        "feature_group": "Delta MFCC",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "đạo hàm bậc 1 của MFCC theo thời gian",
        "why_it_matters": "Cho biết màu giọng thay đổi nhanh hay chậm.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "Delta features for speech dynamics; librosa.feature.delta",
    },
    {
        "feature_group": "RMS energy",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "root mean square energy theo frame",
        "why_it_matters": "Đo cường độ/loudness; hữu ích cho angry, happy, low-energy sad/neutral.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "librosa.feature.rms",
    },
    {
        "feature_group": "ZCR",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "zero-crossing rate theo frame",
        "why_it_matters": "Bổ sung cue về voiced/unvoiced, độ sắc, texture high-frequency.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "librosa.feature.zero_crossing_rate",
    },
    {
        "feature_group": "Spectral centroid / bandwidth / rolloff",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "thống kê hình dạng phổ theo frame",
        "why_it_matters": "Mô tả độ sáng, độ rộng phổ và mức năng lượng vùng tần số cao.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "librosa spectral features; acoustic feature fusion SER papers",
    },
    {
        "feature_group": "Log-Mel spectrogram summary",
        "branch": "Branch B - handcrafted acoustic",
        "how_to_extract": "Mel spectrogram -> log scale -> mean/std/min/max",
        "why_it_matters": "Giữ thông tin time-frequency tổng quát, bổ sung cho MFCC.",
        "used_by_model": "Acoustic vector statistics",
        "reference": "log-Mel is standard for CNN/audio emotion recognition",
    },
]

feature_reference = pd.DataFrame(feature_reference_rows)
feature_reference.to_csv(WORKING_REPORT_DIR / "feature_reference_table.csv", index=False, encoding="utf-8-sig")
display(feature_reference)
'''
        ),
        md(
            """
## 2. Thống kê dữ liệu trước khi trích đặc trưng

Mục tiêu của phần này là kiểm tra lại dataset sau khi lọc 4 emotion chính. Nếu dữ liệu lệch lớp hoặc thiếu audio, phần train phía sau sẽ bị ảnh hưởng trực tiếp.
"""
        ),
        code(
            r'''
overview = {
    "n_rows": len(metadata),
    "n_speakers": metadata["speaker_id"].nunique() if "speaker_id" in metadata.columns else np.nan,
    "n_sessions": metadata["session"].nunique() if "session" in metadata.columns else np.nan,
    "audio_dir": str(AUDIO_DIR),
    "audio_file_count": len(list(AUDIO_DIR.glob("*.wav"))) if AUDIO_DIR.exists() else 0,
}
display(pd.DataFrame([overview]))

emotion_counts = metadata["emotion_4class"].value_counts().rename_axis("emotion").reset_index(name="n")
emotion_counts["percent"] = emotion_counts["n"] / emotion_counts["n"].sum() * 100
emotion_counts.to_csv(WORKING_REPORT_DIR / "emotion_distribution.csv", index=False, encoding="utf-8-sig")
display(emotion_counts)

if "duration" in metadata.columns:
    duration_summary = metadata.groupby("emotion_4class")["duration"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    duration_summary.to_csv(WORKING_REPORT_DIR / "duration_by_emotion.csv", index=False, encoding="utf-8-sig")
    display(duration_summary)

if "session" in metadata.columns:
    session_emotion = pd.crosstab(metadata["session"], metadata["emotion_4class"])
    session_emotion.to_csv(WORKING_REPORT_DIR / "session_emotion_crosstab.csv", encoding="utf-8-sig")
    display(session_emotion)
'''
        ),
        md(
            """
## 3. Chuẩn hóa waveform trước khi trích đặc trưng

Mỗi audio được đưa về:

- mono
- 16 kHz
- biên độ chuẩn hóa theo peak

Lý do: Emotion2Vec, wav2vec2, HuBERT, WavLM đều nhận waveform/raw speech. MFCC/log-Mel/RMS/ZCR cũng phụ thuộc trực tiếp vào sample rate và biên độ, nên nếu không chuẩn hóa trước thì feature giữa các file sẽ khó so sánh.
"""
        ),
        code(
            r'''
maybe_install_deps()

selected_samples = (
    metadata.sort_values(["emotion_4class", "train_sample_id"])
    .groupby("emotion_4class", group_keys=False)
    .head(1)
    .reset_index(drop=True)
)
display_cols = [c for c in ["train_sample_id", "emotion_4class", "speaker_id", "session", "duration", "transcription"] if c in selected_samples.columns]
display(selected_samples[display_cols])

def visualize_sample(row, axes):
    import librosa
    wav_path = resolve_wav_path(row)
    y, sr = load_audio_16k(wav_path)
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64), ref=np.max)
    t = np.arange(len(y)) / sr
    frame_t = np.linspace(0, len(y) / sr, len(rms))

    axes[0].plot(t, y, linewidth=0.6)
    axes[0].set_title(f"{row['emotion_4class']} waveform")
    axes[0].set_xlabel("time (s)")

    axes[1].plot(frame_t, rms, label="RMS", linewidth=1.0)
    axes[1].plot(frame_t, zcr, label="ZCR", linewidth=1.0)
    axes[1].set_title("RMS / ZCR")
    axes[1].legend(fontsize=8)

    axes[2].imshow(mfcc, aspect="auto", origin="lower", interpolation="nearest")
    axes[2].set_title("MFCC")

    axes[3].imshow(mel, aspect="auto", origin="lower", interpolation="nearest")
    axes[3].set_title("log-Mel")

fig, axes = plt.subplots(len(selected_samples), 4, figsize=(18, 3.2 * len(selected_samples)))
if len(selected_samples) == 1:
    axes = np.expand_dims(axes, 0)
for row_idx, (_, row) in enumerate(selected_samples.iterrows()):
    visualize_sample(row, axes[row_idx])
plt.tight_layout()
fig_path = WORKING_FIGURE_DIR / "feature_visualization_one_sample_per_emotion.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print("Saved:", fig_path)
'''
        ),
        md(
            """
## 4. So sánh cùng câu nói hoặc các câu khác nhau

Nếu metadata có nhiều mẫu trùng transcript, notebook sẽ tìm một câu xuất hiện ở nhiều emotion/speaker để so sánh. Nếu không có transcript trùng, notebook vẫn tạo bảng so sánh các mẫu đại diện khác emotion.
"""
        ),
        code(
            r'''
comparison_rows = []
if "transcription" in metadata.columns:
    tmp = metadata.dropna(subset=["transcription"]).copy()
    tmp["transcription_norm"] = tmp["transcription"].astype(str).str.lower().str.strip()
    candidate = (
        tmp.groupby("transcription_norm")
        .agg(n=("train_sample_id", "count"), n_emotions=("emotion_4class", "nunique"))
        .query("n >= 2")
        .sort_values(["n_emotions", "n"], ascending=False)
        .head(1)
    )
    if len(candidate):
        key = candidate.index[0]
        comparison_rows = tmp[tmp["transcription_norm"].eq(key)].groupby("emotion_4class", group_keys=False).head(1).head(4)
        display(Markdown(f"Transcript được chọn: `{key}`"))

if len(comparison_rows) == 0:
    comparison_rows = selected_samples.copy()
    display(Markdown("Không tìm thấy transcript trùng đủ tốt; dùng mỗi emotion một mẫu đại diện để so sánh."))

summary_rows = []
for _, row in comparison_rows.iterrows():
    wav_path = resolve_wav_path(row)
    y, sr = load_audio_16k(wav_path)
    a_vec, a_names = acoustic_vector(y, sr)
    vals = dict(zip(a_names, a_vec))
    summary_rows.append({
        "train_sample_id": row["train_sample_id"],
        "emotion": row["emotion_4class"],
        "speaker_id": row.get("speaker_id", ""),
        "duration": row.get("duration", np.nan),
        "rms_mean": vals.get("rms_mean", np.nan),
        "zcr_mean": vals.get("zcr_mean", np.nan),
        "centroid_mean": vals.get("centroid_mean", np.nan),
        "mfcc_01_mean": vals.get("mfcc_01_mean", np.nan),
        "logmel_mean": vals.get("logmel_mean", np.nan),
    })

comparison_df = pd.DataFrame(summary_rows)
comparison_df.to_csv(WORKING_REPORT_DIR / "selected_sample_acoustic_comparison.csv", index=False, encoding="utf-8-sig")
display(comparison_df)
'''
        ),
        md(
            """
## 5. Thống kê acoustic trên một subset trước khi extract full

Phần này không thay thế training feature cache. Nó chỉ giúp kiểm tra xem feature có giá trị hợp lý không, có bị toàn 0/NaN không, và các emotion có phân bố acoustic khác nhau ở mức mô tả hay không.
"""
        ),
        code(
            r'''
ANALYSIS_SAMPLE_N = int(os.getenv("ACOUSTIC_ANALYSIS_SAMPLE_N", "160"))
per_emotion_n = max(1, ANALYSIS_SAMPLE_N // max(1, metadata["emotion_4class"].nunique()))
analysis_subset = (
    metadata.groupby("emotion_4class", group_keys=False)
    .apply(lambda x: x.sample(min(len(x), per_emotion_n), random_state=SEED))
    .reset_index(drop=True)
)

rows = []
feature_names = None
for _, row in analysis_subset.iterrows():
    y, sr = load_audio_16k(resolve_wav_path(row))
    a_vec, a_names = acoustic_vector(y, sr)
    feature_names = a_names
    d = {"train_sample_id": row["train_sample_id"], "emotion": row["emotion_4class"]}
    d.update(dict(zip(a_names, a_vec)))
    rows.append(d)

acoustic_demo = pd.DataFrame(rows)
acoustic_demo.to_csv(WORKING_REPORT_DIR / "acoustic_demo_subset_features.csv", index=False, encoding="utf-8-sig")

focus_cols = [c for c in ["rms_mean", "zcr_mean", "centroid_mean", "bandwidth_mean", "rolloff_mean", "mfcc_01_mean", "logmel_mean"] if c in acoustic_demo.columns]
demo_summary = acoustic_demo.groupby("emotion")[focus_cols].agg(["mean", "std"]).round(5)
demo_summary.to_csv(WORKING_REPORT_DIR / "acoustic_demo_summary_by_emotion.csv", encoding="utf-8-sig")
display(demo_summary)

fig, axes = plt.subplots(1, len(focus_cols[:4]), figsize=(4.2 * len(focus_cols[:4]), 4))
if len(focus_cols[:4]) == 1:
    axes = [axes]
for ax, col in zip(axes, focus_cols[:4]):
    grouped = acoustic_demo.groupby("emotion")[col].mean().reindex(["neutral", "angry", "sad", "happy"])
    ax.bar(grouped.index, grouped.values)
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
fig_path = WORKING_FIGURE_DIR / "acoustic_demo_feature_means_by_emotion.png"
plt.savefig(fig_path, dpi=160)
plt.show()
print("Saved:", fig_path)
'''
        ),
        md(
            """
## 6. Extract full cache cho notebook 03/04

Cell này sẽ load `funasr.AutoModel(model="iic/emotion2vec_base")`.

Trên Kaggle:

1. Bật Internet.
2. Nếu chưa có package, notebook sẽ tự cài trên Kaggle. Có thể ép cài bằng `INSTALL_EMOTION2VEC_DEPS=1`.
3. Add Kaggle Dataset có `audio_wav/`, `metadata/`, `splits/`.

Nếu chỉ muốn test nhanh, set `MAX_FILES=50`, nhưng để train 03/04 nghiêm túc thì phải extract full 6,877 mẫu.
"""
        ),
        code(
            r'''
model = load_emotion2vec_model()
sample_ids = []
e2v_rows = []
acoustic_rows = []
failures = []
acoustic_names = None

start = time.time()
for i, row in metadata.iterrows():
    sid = str(row["train_sample_id"])
    try:
        wav_path = resolve_wav_path(row)
        y, sr = load_audio_16k(wav_path)
        a_vec, a_names = acoustic_vector(y, sr)
        e_vec = extract_emotion2vec_embedding(model, wav_path)
        if acoustic_names is None:
            acoustic_names = a_names
        sample_ids.append(sid)
        acoustic_rows.append(a_vec)
        e2v_rows.append(e_vec)
    except Exception as exc:
        failures.append({"train_sample_id": sid, "error": repr(exc)})
        print("FAILED", sid, repr(exc))
        break
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(metadata)} done, elapsed={time.time()-start:.1f}s")

if failures:
    pd.DataFrame(failures).to_csv(WORKING_FEATURE_DIR / "feature_extraction_failures.csv", index=False, encoding="utf-8-sig")
    raise RuntimeError("Feature extraction stopped because at least one sample failed. See feature_extraction_failures.csv")

max_e_dim = max(len(x) for x in e2v_rows)
e2v = np.zeros((len(e2v_rows), max_e_dim), dtype=np.float32)
for i, x in enumerate(e2v_rows):
    e2v[i, : len(x)] = x
acoustic = np.vstack(acoustic_rows).astype(np.float32)

np.savez_compressed(
    WORKING_FEATURE_CACHE_PATH,
    sample_ids=np.asarray(sample_ids, dtype=str),
    emotion2vec=e2v,
    acoustic=acoustic,
    acoustic_names=np.asarray(acoustic_names, dtype=str),
    model_name=np.asarray([EMOTION2VEC_MODEL_NAME], dtype=str),
)

print("Saved:", WORKING_FEATURE_CACHE_PATH)
print("emotion2vec:", e2v.shape)
print("acoustic:", acoustic.shape)
'''
        ),
        md(
            """
## 7. Kiểm tra cache sau khi extract

Cell cuối xác nhận file `.npz` đã được tạo đúng schema để notebook 03/04 có thể đọc ngay.
"""
        ),
        code(
            r'''
if not WORKING_FEATURE_CACHE_PATH.exists():
    raise FileNotFoundError(f"Chưa thấy cache sau extraction: {WORKING_FEATURE_CACHE_PATH}")

cache_check = np.load(WORKING_FEATURE_CACHE_PATH, allow_pickle=True)
cache_summary = {
    "cache_path": str(WORKING_FEATURE_CACHE_PATH),
    "n_sample_ids": int(len(cache_check["sample_ids"])),
    "emotion2vec_shape": tuple(cache_check["emotion2vec"].shape),
    "acoustic_shape": tuple(cache_check["acoustic"].shape),
    "n_acoustic_names": int(len(cache_check["acoustic_names"])) if "acoustic_names" in cache_check else 0,
    "model_name": str(cache_check["model_name"][0]) if "model_name" in cache_check else "",
}
cache_summary_df = pd.DataFrame([cache_summary])
cache_summary_df.to_csv(WORKING_REPORT_DIR / "feature_cache_summary.csv", index=False, encoding="utf-8-sig")
display(cache_summary_df)

report_lines = [
    "# IEMOCAP full feature extraction report",
    "",
    f"- Cache path: `{WORKING_FEATURE_CACHE_PATH}`",
    f"- Samples: **{cache_summary['n_sample_ids']}**",
    f"- Emotion2Vec shape: **{cache_summary['emotion2vec_shape']}**",
    f"- Acoustic shape: **{cache_summary['acoustic_shape']}**",
    f"- Acoustic feature count: **{cache_summary['n_acoustic_names']}**",
    "",
    "Notebook 03/04 sẽ dùng cache này để train mô hình full hai nhánh: Emotion2Vec branch + acoustic branch.",
]
report_path = WORKING_REPORT_DIR / "feature_extraction_report.md"
report_path.write_text("\n".join(report_lines), encoding="utf-8")
display(Markdown("\n".join(report_lines)))
print("Saved report:", report_path)
'''
        ),
        md(
            """
## 8. Zip toàn bộ output của notebook 02

Cell này gom các file do notebook 02 sinh ra để bạn tải về từ Kaggle:

- `features/iemocap_full_emotion2vec_acoustic_cache.npz`
- `feature_reports/`
- `feature_figures/`

Không zip `audio_wav/` vì đó là input raw audio rất lớn, không phải output của notebook.
"""
        ),
        code(
            r'''
zip_path = Path("/kaggle/working/iemocap_full_feature_outputs.zip") if Path("/kaggle/working").exists() else WORKING_DATA_DIR.parent / "iemocap_full_feature_outputs.zip"
zip_folder(
    WORKING_DATA_DIR,
    zip_path,
    exclude_dir_names={"audio_wav"},
    exclude_suffixes=set(),
)
print("ZIP_OUTPUT:", zip_path)
print("ZIP_SIZE_MB:", round(zip_path.stat().st_size / (1024 * 1024), 2))
'''
        ),
    ]


def train_notebook(title, protocol, split_file, n_folds):
    return [
        md(
            f"""
# {title}

Notebook này train **full proposed model**, không còn là metadata baseline.

Input bắt buộc:

```text
data/features/iemocap_full_emotion2vec_acoustic_cache.npz
```

Kiến trúc:

```text
Branch A: Emotion2Vec frozen embedding -> adapter MLP -> z_e
Branch B: handcrafted acoustic vector -> acoustic MLP -> z_a
Cross-attention: z_a queries z_e, z_e queries z_a
Gated fusion -> shared representation
Head 1: emotion classification
Head 2: valence/arousal/dominance regression
```
"""
        ),
        code(COMMON_IMPORTS),
        code(ZIP_HELPERS),
        code(PATH_RESOLVER),
        code(
            f'''
NOTEBOOK_DIR = BASE_DIR / "{'03MultiTask Emotion2Vec CoAttention 5Fold' if n_folds == 5 else '04MultiTask Emotion2Vec CoAttention 10Fold'}"
REPORT_DIR = NOTEBOOK_DIR / "reports"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
PREDICTION_DIR = NOTEBOOK_DIR / "predictions"
MODEL_DIR = NOTEBOOK_DIR / "models"
for folder in [REPORT_DIR, FIGURE_DIR, PREDICTION_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

PROTOCOL = "{protocol}"
SPLIT_PATH = SPLIT_DIR / "{split_file}"
N_EXPECTED_FOLDS = {n_folds}
FEATURE_CACHE_PATH = resolve_feature_cache_path(require_exists=True)
print("SPLIT_PATH:", SPLIT_PATH, SPLIT_PATH.exists())
print("FEATURE_CACHE_PATH:", FEATURE_CACHE_PATH, FEATURE_CACHE_PATH.exists())
'''
        ),
        code(MODEL_HELPERS),
        code(
            r'''
cache = load_feature_cache(FEATURE_CACHE_PATH)
split_df = load_and_align_split(SPLIT_PATH, FULL_METADATA_PATH, cache)
print("Split rows:", split_df.shape)
print("Feature shapes:", cache["emotion2vec"].shape, cache["acoustic"].shape)
print("Folds:", split_df["fold"].nunique())
if split_df["fold"].nunique() != N_EXPECTED_FOLDS:
    raise ValueError(f"Expected {N_EXPECTED_FOLDS} folds")
display(split_df.head())
'''
        ),
        code(
            r'''
def run_fold(fold):
    train_df = split_df[(split_df["fold"].eq(fold)) & (split_df["split"].eq("train"))].copy()
    val_df = split_df[(split_df["fold"].eq(fold)) & (split_df["split"].eq("validation"))].copy()
    test_df = split_df[(split_df["fold"].eq(fold)) & (split_df["split"].eq("test"))].copy()

    train_ds = FullFeatureDataset(train_df, cache, fit_scalers=True)
    val_ds = FullFeatureDataset(val_df, cache, e_scaler=train_ds.e_scaler, a_scaler=train_ds.a_scaler)
    test_ds = FullFeatureDataset(test_df, cache, e_scaler=train_ds.e_scaler, a_scaler=train_ds.a_scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = Emotion2VecAcousticCoAttentionMultiTask(
        e2v_dim=cache["emotion2vec"].shape[1],
        acoustic_dim=cache["acoustic"].shape[1],
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss(weight=class_weights(train_df).to(DEVICE), label_smoothing=0.05)

    best_state = None
    best_score = -1e9
    best_epoch = 0
    patience_left = PATIENCE
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = run_epoch(model, train_loader, optimizer, ce)
        val_pred = predict(model, val_loader)
        val_metrics = compute_metrics(val_pred)
        score = primary_score(val_metrics)
        history.append({"fold": fold, "epoch": epoch, "train_loss": loss, "primary_score": score, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    test_pred = predict(model, test_loader)
    metrics = compute_metrics(test_pred)
    pred_cls = test_pred["logits"].argmax(axis=1)
    probs = torch.softmax(torch.tensor(test_pred["logits"]), dim=1).numpy()

    out = test_df.reset_index(drop=True).copy()
    out["pred_emotion_id"] = pred_cls
    out["pred_emotion"] = [ID_TO_EMOTION[int(x)] for x in pred_cls]
    for i, name in ID_TO_EMOTION.items():
        out[f"prob_{name}"] = probs[:, i]
    out["pred_valence"] = test_pred["reg"][:, 0] * 4.0 + 1.0
    out["pred_arousal"] = test_pred["reg"][:, 1] * 4.0 + 1.0
    out["pred_dominance"] = test_pred["reg"][:, 2] * 4.0 + 1.0

    safe = str(fold).replace(" ", "_").replace("/", "_")
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{safe}_history.csv", index=False, encoding="utf-8-sig")
    out.to_csv(PREDICTION_DIR / f"{safe}_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save({"model_state_dict": model.state_dict(), "fold": fold, "best_epoch": best_epoch, "metrics": metrics}, MODEL_DIR / f"{safe}_best.pt")

    return {"protocol": PROTOCOL, "fold": fold, "best_epoch": best_epoch, "n_train": len(train_df), "n_validation": len(val_df), "n_test": len(test_df), **metrics}, out

rows = []
preds = []
start = time.time()
for fold in sorted(split_df["fold"].unique()):
    print("Running fold:", fold)
    row, pred = run_fold(fold)
    rows.append(row)
    preds.append(pred)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k in ["fold", "WA", "UAR", "Macro_F1", "CCC_mean"]})

metrics_df = pd.DataFrame(rows)
all_predictions = pd.concat(preds, ignore_index=True)
metrics_df.to_csv(REPORT_DIR / f"{PROTOCOL}_full_model_metrics.csv", index=False, encoding="utf-8-sig")
all_predictions.to_csv(PREDICTION_DIR / f"{PROTOCOL}_full_model_predictions.csv", index=False, encoding="utf-8-sig")
print("Seconds:", round(time.time() - start, 2))
display(metrics_df)
'''
        ),
        code(
            r'''
summary = metrics_df.drop(columns=["protocol", "fold"], errors="ignore").select_dtypes(include=[np.number]).agg(["mean", "std"]).T.reset_index()
summary.columns = ["metric", "mean", "std"]
summary.to_csv(REPORT_DIR / f"{PROTOCOL}_full_model_summary.csv", index=False, encoding="utf-8-sig")
display(summary)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plot_df = metrics_df.melt(id_vars=["fold"], value_vars=["WA", "UAR", "Macro_F1", "CCC_mean"], var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x="fold", y="value", hue="metric")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{PROTOCOL}_full_model_fold_metrics.png", dpi=180)
    plt.show()
except Exception as exc:
    print("Figure skipped:", exc)

report = [
    f"# {PROTOCOL} Full Emotion2Vec + Acoustic Multi-Task Report",
    "",
    "- Feature cache: `data/features/iemocap_full_emotion2vec_acoustic_cache.npz`.",
    "- Branch A: frozen emotion2vec embedding + adapter.",
    "- Branch B: handcrafted acoustic vector + adapter.",
    "- Fusion: bidirectional cross-attention + gated fusion.",
    "- Heads: emotion classification and AVD regression.",
    "",
    "## Mean +/- std",
]
for _, row in summary.iterrows():
    report.append(f"- {row['metric']}: {row['mean']:.4f} +/- {row['std']:.4f}")
report_path = REPORT_DIR / f"{PROTOCOL}_full_model_report.md"
report_path.write_text("\n".join(report), encoding="utf-8")
display(Markdown("\n".join(report)))
'''
        ),
        md(
            """
## Zip toàn bộ output của notebook train

Cell cuối gom các file sinh ra sau khi train để tải về từ Kaggle:

- `reports/`
- `predictions/`
- `models/`
- `figures/`
"""
        ),
        code(
            r'''
zip_path = Path("/kaggle/working") / f"{PROTOCOL}_full_model_outputs.zip" if Path("/kaggle/working").exists() else NOTEBOOK_DIR / f"{PROTOCOL}_full_model_outputs.zip"
zip_folder(NOTEBOOK_DIR, zip_path)
print("ZIP_OUTPUT:", zip_path)
print("ZIP_SIZE_MB:", round(zip_path.stat().st_size / (1024 * 1024), 2))
'''
        ),
    ]


def main():
    nb2_dir = ROOT / "02IEMOCAP Feature Extraction Emotion2Vec Acoustic"
    nb2_dir.mkdir(parents=True, exist_ok=True)
    nb2 = nbf.v4.new_notebook()
    nb2["cells"] = feature_notebook()
    nbf.write(nb2, nb2_dir / "02_IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb")
    print("Updated", nb2_dir / "02_IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb")

    configs = [
        ("03 - Full MultiTask Emotion2Vec CoAttention 5-Fold", "5fold_session", "iemocap_5fold_session_long.csv", 5, "03MultiTask Emotion2Vec CoAttention 5Fold", "03_MultiTask_Emotion2Vec_CoAttention_5Fold.ipynb"),
        ("04 - Full MultiTask Emotion2Vec CoAttention 10-Fold", "10fold_speaker", "iemocap_10fold_speaker_long.csv", 10, "04MultiTask Emotion2Vec CoAttention 10Fold", "04_MultiTask_Emotion2Vec_CoAttention_10Fold.ipynb"),
    ]
    for title, protocol, split_file, n_folds, folder, filename in configs:
        nb_dir = ROOT / folder
        nb_dir.mkdir(parents=True, exist_ok=True)
        nb = nbf.v4.new_notebook()
        nb["cells"] = train_notebook(title, protocol, split_file, n_folds)
        nbf.write(nb, nb_dir / filename)
        print("Updated", nb_dir / filename)


if __name__ == "__main__":
    main()
