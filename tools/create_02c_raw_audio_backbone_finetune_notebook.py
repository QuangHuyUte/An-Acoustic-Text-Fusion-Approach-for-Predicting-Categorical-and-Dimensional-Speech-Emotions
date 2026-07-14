from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
OUT_DIR = ROOT / "06_w2v_based_models" / "03_Emotion2Vec RawAudio Backbone Finetune 5Fold 10Fold"
OUT_PATH = OUT_DIR / "02C_Emotion2Vec_RawAudio_Backbone_Finetune_5Fold_10Fold.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
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

cells = []

cells.append(md("""
# 02C - Fine-tune backbone từ raw audio cho Multi-task SER trên IEMOCAP

Notebook này được tạo để kiểm tra một hướng mạnh hơn 02B: thay vì chỉ huấn luyện head trên embedding đã trích sẵn, mô hình sẽ nhận trực tiếp waveform và fine-tune một phần backbone speech pretrained.

Mục tiêu chính:

- Chạy cùng một mô hình multi-task trên cả hai giao thức: **5-fold theo session** và **10-fold theo speaker**.
- Huấn luyện đồng thời hai đầu ra: **emotion classification** và **valence/arousal/dominance regression**.
- Giữ đúng chuẩn speaker-independent: mỗi fold chỉ học trên `train`, dùng `validation` để chọn checkpoint, và chỉ đánh giá `test` sau khi đã chọn checkpoint tốt nhất.
- Lưu lại bảng kết quả, dự đoán từng mẫu, checkpoint từng fold và file zip output để tải về.

> Lưu ý quan trọng: 02B/03/04 có thể chạy bằng feature cache, nhưng 02C là raw-audio backbone fine-tuning nên cần có thư mục `audio_wav/*.wav`.
"""))

cells.append(md("""
## Vì sao cần 02C?

Kết quả hiện tại của 02B vẫn còn thấp vì 02B chỉ học trên representation đã đóng băng. Nếu embedding chưa đủ khớp với IEMOCAP hoặc chưa tối ưu cho hai task emotion + VAD, downstream head rất khó tự sửa lỗi sâu.

02C thử một hướng gần hơn với các bài dùng pretrained speech model đạt kết quả cao:

- Chuẩn hóa audio về mono, 16 kHz.
- Đưa waveform vào speech backbone pretrained.
- Fine-tune một phần các tầng cuối của backbone để representation thích nghi với IEMOCAP.
- Dùng head phân loại cho emotion và head hồi quy cho valence/arousal/dominance.

Trong các bài SER hiện đại, pretrained backbone như wav2vec 2.0, HuBERT, WavLM hoặc emotion2vec thường đóng vai trò **feature learner**. Phần fine-tune quyết định backbone chỉ đứng yên như bộ trích đặc trưng, hay được cập nhật để học tốt hơn theo dataset đích.
"""))

cells.append(md("""
## Ghi chú về emotion2vec và fine-tuning

Paper emotion2vec gốc chủ yếu chứng minh sức mạnh của representation bằng cách **đóng băng emotion2vec và huấn luyện downstream linear layer** trên IEMOCAP. Vì vậy 02B gần với protocol gốc hơn: dùng feature đã trích rồi huấn luyện downstream model.

02C là nhánh thử nghiệm nâng cấp: fine-tune backbone từ raw audio để xem representation có thích nghi tốt hơn với IEMOCAP và multi-task emotion + VAD hay không. Notebook dùng API HuggingFace `AutoModel` để fine-tune ổn định. Mặc định là `microsoft/wavlm-base-plus`; nếu có checkpoint Emotion2Vec tương thích HuggingFace, chỉ cần đổi `PRETRAINED_MODEL_NAME`.

Tài liệu tham khảo chính:

- [emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation](https://arxiv.org/abs/2312.15185)
- [Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition](https://arxiv.org/abs/2110.06309)
- [Emotion Recognition from Speech Using Wav2vec 2.0 Embeddings](https://arxiv.org/abs/2104.03502)
"""))

cells.append(md("""
## Fine-tune train-only có đúng không?

Đúng. Với mỗi fold, quy trình chuẩn là:

| Phần dữ liệu | Có dùng để cập nhật trọng số không? | Vai trò |
|---|---:|---|
| Train | Có | Học trọng số backbone/head |
| Validation | Không | Chọn checkpoint, early stopping, chỉnh quyết định huấn luyện |
| Test | Không | Báo cáo kết quả cuối cùng sau khi đã chọn checkpoint |

Nếu dùng validation hoặc test để backpropagation thì kết quả sẽ bị rò rỉ dữ liệu. Vì vậy trong notebook này optimizer chỉ nhìn thấy `train_loader`.
"""))

cells.append(md("""
## Dữ liệu cần upload lên Kaggle

Với 02C, Kaggle dataset nên có cấu trúc tối thiểu như sau:

```text
data/
  audio_wav/
    Ses01F_impro01_F000.wav
    ...
  splits/
    iemocap_5fold_session_long.csv
    iemocap_10fold_speaker_long.csv
  metadata/
    iemocap_metadata_full.csv          # tùy chọn
```

Notebook sẽ tự quét `/kaggle/input`, `/kaggle/working` và thư mục project local để tìm `audio_wav` và hai file split. Nếu bạn đặt dữ liệu ở vị trí khác, có thể set:

- `IEMOCAP_AUDIO_DIR`
- `IEMOCAP_SPLIT_DIR`
- `IEMOCAP_DATA_DIR`

Vì đây là fine-tune raw audio, chỉ upload thư mục `output` của notebook 02 là chưa đủ nếu trong đó không có `audio_wav`.
"""))

cells.append(code("""
import os
import sys
import time
import json
import math
import random
import shutil
import zipfile
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

print("Python:", sys.version)
"""))

cells.append(md("""
## Cài thư viện khi chạy trên Kaggle

Nếu Kaggle chưa có `transformers`, `soundfile` hoặc `librosa`, bật Internet và set:

```python
INSTALL_DEPS = 1
```

Mặc định notebook không tự cài để tránh thay đổi môi trường ngoài ý muốn.
"""))

cells.append(code("""
INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"

if INSTALL_DEPS:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "transformers",
        "accelerate",
        "soundfile",
        "librosa",
        "scikit-learn",
    ])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings(
    "ignore",
    message="Support for mismatched key_padding_mask and attn_mask is deprecated.*",
    category=UserWarning,
)

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, classification_report

try:
    import soundfile as sf
except Exception as exc:
    raise ImportError("Missing soundfile. On Kaggle, set INSTALL_DEPS=1 or install soundfile manually.") from exc

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception as exc:
    raise ImportError("Missing transformers. On Kaggle, set INSTALL_DEPS=1 or install transformers manually.") from exc

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None
"""))

cells.append(md("""
## Cấu hình thí nghiệm

`PRETRAINED_MODEL_NAME` mặc định dùng `microsoft/wavlm-base-plus` vì đây là backbone HuggingFace nhận raw waveform ổn định cho fine-tuning speech. Nếu có checkpoint Emotion2Vec tương thích HuggingFace, đổi biến này thành checkpoint đó.

Các tham số nên chỉnh trên Kaggle:

- `MAX_SECONDS`: độ dài tối đa mỗi utterance sau padding/truncation.
- `UNFREEZE_LAST_N`: số tầng cuối của backbone được mở để fine-tune. Giá trị nhỏ sẽ nhẹ GPU hơn.
- `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM`: cân bằng giữa chất lượng và thời gian chạy.
- `MAX_FOLDS`: đặt `1` để debug nhanh, đặt `0` để chạy toàn bộ.
"""))

cells.append(code("""
SEED = int(os.getenv("SEED", "42"))
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
LOSS_EMOTION_WEIGHT = float(os.getenv("LOSS_EMOTION_WEIGHT", "1.0"))
LOSS_REG_MSE_WEIGHT = float(os.getenv("LOSS_REG_MSE_WEIGHT", "0.35"))
LOSS_REG_CCC_WEIGHT = float(os.getenv("LOSS_REG_CCC_WEIGHT", "0.50"))

MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "0"))
MAX_TEST_SAMPLES = int(os.getenv("MAX_TEST_SAMPLES", "0"))

RUN_PROTOCOLS = [
    item.strip()
    for item in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",")
    if item.strip()
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_02c_raw_audio_backbone_finetune")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
for path in [OUTPUT_DIR, MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

EMOTION_ID_TO_NAME = {0: "neutral", 1: "angry", 2: "sad", 3: "happy"}
EMOTION_NAME_TO_ID = {v: k for k, v in EMOTION_ID_TO_NAME.items()}
NUM_CLASSES = 4

print("Device:", DEVICE)
print("Pretrained model:", PRETRAINED_MODEL_NAME)
print("Protocols:", RUN_PROTOCOLS)
print("Output:", OUTPUT_DIR)
"""))

cells.append(code("""
def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)
"""))

cells.append(code("""
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)
"""))

cells.append(md("""
## Tự dò dữ liệu

File split có thể giữ `wav_path` dạng đường dẫn Windows từ máy local. Notebook không phụ thuộc tuyệt đối vào đường dẫn đó; nó lấy tên file `.wav` rồi tìm trong thư mục `audio_wav` đang có trên Kaggle.
"""))

cells.append(code("""
LOCAL_PROJECT = Path(r"D:\\UTE\\Speech Programming\\Speech Project")


def existing_paths(paths):
    return [Path(p).resolve() for p in paths if p and Path(p).exists()]


def search_roots():
    roots = []
    env_candidates = [
        os.getenv("IEMOCAP_DATA_DIR"),
        os.getenv("IEMOCAP_SPLIT_DIR"),
        os.getenv("IEMOCAP_AUDIO_DIR"),
        os.getenv("KAGGLE_INPUT_DIR"),
    ]
    roots.extend(existing_paths(env_candidates))
    roots.extend(existing_paths([
        Path.cwd(),
        Path.cwd().parent,
        LOCAL_PROJECT / "06_w2v_based_models" / "data",
        LOCAL_PROJECT / "06_w2v_based_models",
        "/kaggle/input",
        "/kaggle/working",
    ]))
    unique = []
    seen = set()
    for root in roots:
        key = str(root).lower()
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def find_named_file(filename: str, env_var: str | None = None):
    if env_var and os.getenv(env_var):
        candidate = Path(os.getenv(env_var))
        if candidate.exists():
            return candidate.resolve()
    for root in search_roots():
        direct_candidates = [
            root / filename,
            root / "splits" / filename,
            root / "data" / "splits" / filename,
            root / "output" / "splits" / filename,
        ]
        for candidate in direct_candidates:
            if candidate.exists():
                return candidate.resolve()
        for candidate in root.rglob(filename):
            if candidate.exists():
                return candidate.resolve()
    checked = "\\n".join(f"- {root}" for root in search_roots())
    raise FileNotFoundError(f"Cannot find {filename}. Checked roots:\\n{checked}")


def find_audio_dir():
    if os.getenv("IEMOCAP_AUDIO_DIR"):
        candidate = Path(os.getenv("IEMOCAP_AUDIO_DIR"))
        if candidate.exists():
            return candidate.resolve()
    for root in search_roots():
        direct_candidates = [
            root / "audio_wav",
            root / "data" / "audio_wav",
            root / "output" / "audio_wav",
            root / "datasets" / "AbstractTTS_IEMOCAP" / "audio_wav",
        ]
        for candidate in direct_candidates:
            if candidate.exists() and any(candidate.glob("*.wav")):
                return candidate.resolve()
        for candidate in root.rglob("audio_wav"):
            if candidate.is_dir() and any(candidate.glob("*.wav")):
                return candidate.resolve()
    checked = "\\n".join(f"- {root}" for root in search_roots())
    raise FileNotFoundError(f"Cannot find audio_wav directory. Checked roots:\\n{checked}")


AUDIO_DIR = find_audio_dir()
SPLIT_5FOLD_PATH = find_named_file("iemocap_5fold_session_long.csv", env_var="IEMOCAP_5FOLD_SPLIT_PATH")
SPLIT_10FOLD_PATH = find_named_file("iemocap_10fold_speaker_long.csv", env_var="IEMOCAP_10FOLD_SPLIT_PATH")

print("AUDIO_DIR:", AUDIO_DIR)
print("SPLIT_5FOLD_PATH:", SPLIT_5FOLD_PATH)
print("SPLIT_10FOLD_PATH:", SPLIT_10FOLD_PATH)
print("Number of wav files:", len(list(AUDIO_DIR.glob("*.wav"))))
"""))

cells.append(md("""
## Kiểm tra split trước khi train

Ở 5-fold, mỗi fold test một session và validation một session khác. Ở 10-fold, mỗi fold test một speaker và validation một speaker khác. Đây là hai cách đánh giá khó hơn random split vì speaker/session trong test không được dùng để học.
"""))

cells.append(code("""
def load_split_table(path: Path, protocol: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "utterance_id",
        "speaker_id",
        "session",
        "emotion_4class",
        "emotion_id",
        "valence",
        "arousal",
        "dominance",
        "wav_path",
        "fold",
        "split",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{protocol} split file is missing columns: {missing}")
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
    print(f"{protocol} split labels:", original_split_values, "->", sorted(df["split"].unique().tolist()))
    df["emotion_id"] = df["emotion_id"].astype(int)
    for col in ["valence", "arousal", "dominance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["valence", "arousal", "dominance", "emotion_id"])
    return df


def assert_fold_has_required_splits(protocol: str, fold_name: str, fold_df: pd.DataFrame):
    counts = fold_df["split"].value_counts().to_dict()
    missing = [name for name in ["train", "val", "test"] if counts.get(name, 0) == 0]
    if missing:
        raise ValueError(
            f"Fold {protocol}/{fold_name} is missing split(s) {missing}. "
            f"Current split counts: {counts}. Check the split column in the CSV file."
        )


split_tables = {
    "5fold_session": load_split_table(SPLIT_5FOLD_PATH, "5fold_session"),
    "10fold_speaker": load_split_table(SPLIT_10FOLD_PATH, "10fold_speaker"),
}

for protocol, df in split_tables.items():
    print("\\n", protocol)
    print("Rows:", len(df), "unique utterances:", df["utterance_id"].nunique())
    display(df.groupby(["fold", "split"]).size().unstack(fill_value=0).head(20))
    display(pd.crosstab(df["emotion_4class"], df["split"]))
"""))

cells.append(md("""
## Audio preprocessing

Trước khi đưa vào pretrained model, notebook thực hiện các bước cơ bản:

1. Đọc waveform từ `.wav`.
2. Chuyển stereo sang mono nếu cần.
3. Resample về 16 kHz nếu file không đúng sample rate.
4. Chuẩn hóa biên độ nhẹ để tránh audio quá nhỏ/quá lớn.
5. Padding/truncation theo `MAX_SECONDS`.

Các bước này không phải đặc trưng thủ công như MFCC/log-Mel. Đây là chuẩn hóa đầu vào để pretrained backbone nhận waveform cùng định dạng.
"""))

cells.append(code("""
_AUDIO_FILE_INDEX = None


def build_audio_file_index():
    global _AUDIO_FILE_INDEX
    if _AUDIO_FILE_INDEX is None:
        _AUDIO_FILE_INDEX = {path.name: path.resolve() for path in AUDIO_DIR.rglob("*.wav")}
    return _AUDIO_FILE_INDEX


def resolve_wav_path(row) -> Path:
    raw = str(row.get("wav_path", ""))
    name = Path(raw.replace("\\\\", "/")).name
    direct = AUDIO_DIR / name
    if direct.exists():
        return direct.resolve()
    index = build_audio_file_index()
    if name in index:
        return index[name]
    utterance_id = str(row.get("utterance_id", ""))
    by_utterance = AUDIO_DIR / f"{utterance_id}.wav"
    if by_utterance.exists():
        return by_utterance.resolve()
    raise FileNotFoundError(f"Cannot find wav for utterance={utterance_id}, name={name}")


def load_audio_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        try:
            import librosa
        except Exception as exc:
            raise ImportError("Audio needs resampling but librosa is missing. Set INSTALL_DEPS=1.") from exc
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1.0:
        wav = wav / peak
    return wav


example_df = split_tables["5fold_session"].head(5).copy()
example_df["resolved_wav"] = [str(resolve_wav_path(row)) for _, row in example_df.iterrows()]
display(example_df[["utterance_id", "emotion_4class", "wav_path", "resolved_wav"]])
"""))

cells.append(md("""
## Dataset và collate function

`Dataset` chỉ giữ metadata. Audio được đọc trong `collate_fn` để tạo batch waveform, sau đó `AutoFeatureExtractor` xử lý padding/truncation đúng chuẩn của backbone.
"""))

cells.append(code("""
FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)


def vad_to_0_1(values: np.ndarray) -> np.ndarray:
    return np.clip((values.astype(np.float32) - 1.0) / 4.0, 0.0, 1.0)


def vad_from_0_1(values: np.ndarray) -> np.ndarray:
    return values.astype(np.float32) * 4.0 + 1.0


class IemocapRawAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True).copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vad = row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
        return {
            "row_index": idx,
            "utterance_id": str(row["utterance_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
            "fold": str(row["fold"]),
            "split": str(row["split"]),
            "wav_path": str(resolve_wav_path(row)),
            "emotion_id": int(row["emotion_id"]),
            "vad": vad_to_0_1(vad),
            "vad_raw": vad,
        }


def collate_raw_audio(batch):
    arrays = [load_audio_16k(Path(item["wav_path"])) for item in batch]
    encoded = FEATURE_EXTRACTOR(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SAMPLES,
    )
    result = {
        "input_values": encoded["input_values"],
        "emotion_id": torch.tensor([item["emotion_id"] for item in batch], dtype=torch.long),
        "vad": torch.tensor(np.stack([item["vad"] for item in batch]), dtype=torch.float32),
        "vad_raw": torch.tensor(np.stack([item["vad_raw"] for item in batch]), dtype=torch.float32),
        "utterance_id": [item["utterance_id"] for item in batch],
        "speaker_id": [item["speaker_id"] for item in batch],
        "session": [item["session"] for item in batch],
    }
    if "attention_mask" in encoded:
        result["attention_mask"] = encoded["attention_mask"]
    else:
        result["attention_mask"] = None
    return result
"""))

cells.append(md("""
## Kiến trúc mô hình 02C

Mô hình gồm ba phần:

1. **Speech backbone pretrained** nhận raw waveform và xuất frame-level hidden states.
2. **Masked mean pooling** gom chuỗi frame thành một vector utterance-level.
3. **Hai head multi-task**:
   - `emotion_head`: phân loại 4 lớp `neutral/angry/sad/happy`.
   - `vad_head`: hồi quy 3 giá trị `valence/arousal/dominance` đã chuẩn hóa về [0, 1].

Nếu `UNFREEZE_LAST_N = 4`, notebook đóng băng phần lớn backbone và chỉ mở 4 tầng encoder cuối. Đây là cách tiết kiệm GPU hơn full fine-tuning, đồng thời vẫn cho backbone thích nghi với IEMOCAP.
"""))

cells.append(code("""
class RawAudioBackboneMultiTaskSER(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 4, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_hidden = int(getattr(self.backbone.config, "hidden_size", 768))
        self.shared = nn.Sequential(
            nn.LayerNorm(backbone_hidden),
            nn.Dropout(dropout),
            nn.Linear(backbone_hidden, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden_dim, num_classes)
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid(),
        )

    def masked_mean_pool(self, hidden_states, attention_mask=None):
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask
        if mask.shape[1] != hidden_states.shape[1]:
            mask = F.interpolate(
                mask.float().unsqueeze(1),
                size=hidden_states.shape[1],
                mode="nearest",
            ).squeeze(1)
        mask = mask.to(hidden_states.device).float().unsqueeze(-1)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        pooled = self.masked_mean_pool(outputs.last_hidden_state, attention_mask)
        shared = self.shared(pooled)
        return {
            "emotion_logits": self.emotion_head(shared),
            "vad_pred": self.vad_head(shared),
        }


def freeze_backbone_for_partial_finetune(model: RawAudioBackboneMultiTaskSER, unfreeze_last_n: int):
    if unfreeze_last_n < 0:
        for param in model.backbone.parameters():
            param.requires_grad = True
        return "Full backbone fine-tuning"

    for param in model.backbone.parameters():
        param.requires_grad = False

    encoder = getattr(model.backbone, "encoder", None)
    layers = getattr(encoder, "layers", None) if encoder is not None else None
    if layers is not None and unfreeze_last_n > 0:
        for layer in layers[-unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True
        return f"Frozen backbone except last {unfreeze_last_n} encoder layers"

    return "Backbone frozen because encoder layers were not found"


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, trainable / max(total, 1)
"""))

cells.append(md("""
## Loss cho hai task

Emotion dùng `CrossEntropyLoss`. VAD dùng kết hợp:

- `MSE`: phạt sai số tuyệt đối theo từng chiều.
- `1 - CCC`: khuyến khích dự đoán có tương quan và cùng thang đo với ground truth.

Tổng loss:

```text
loss = emotion_weight * CE
     + mse_weight * MSE(VAD)
     + ccc_weight * mean(1 - CCC_valence/arousal/dominance)
```
"""))

cells.append(code("""
def concordance_ccc_torch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    ccc = (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + eps)
    return ccc


def multitask_loss(outputs, emotion_true, vad_true):
    ce = F.cross_entropy(outputs["emotion_logits"], emotion_true)
    mse = F.mse_loss(outputs["vad_pred"], vad_true)
    ccc = concordance_ccc_torch(outputs["vad_pred"], vad_true)
    ccc_loss = (1.0 - ccc).mean()
    total = (
        LOSS_EMOTION_WEIGHT * ce
        + LOSS_REG_MSE_WEIGHT * mse
        + LOSS_REG_CCC_WEIGHT * ccc_loss
    )
    return total, {"ce": ce.detach(), "mse": mse.detach(), "ccc_loss": ccc_loss.detach()}
"""))

cells.append(md("""
## Chỉ số đánh giá

Emotion classification:

- **WA**: weighted accuracy, tương đương accuracy tổng thể.
- **UAR**: unweighted average recall, tức trung bình recall của từng lớp; chỉ số này quan trọng khi dữ liệu lệch lớp.
- **Macro-F1**: trung bình F1 của từng lớp.

VAD regression:

- **CCC**: concordance correlation coefficient. CCC cao khi dự đoán vừa tương quan tốt, vừa không lệch trung bình/thang đo quá nhiều.
- **MAE/RMSE**: sai số trên thang gốc 1-5 của IEMOCAP.
"""))

cells.append(code("""
def concordance_ccc_np(pred: np.ndarray, true: np.ndarray, eps: float = 1e-8):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pred_mean = pred.mean(axis=0)
    true_mean = true.mean(axis=0)
    pred_var = pred.var(axis=0)
    true_var = true.var(axis=0)
    cov = ((pred - pred_mean) * (true - true_mean)).mean(axis=0)
    return (2.0 * cov) / (pred_var + true_var + (pred_mean - true_mean) ** 2 + eps)


def compute_metrics(y_true, y_pred, vad_true_01, vad_pred_01):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
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
    return (
        0.35 * metrics["UAR"]
        + 0.20 * metrics["WA"]
        + 0.20 * metrics["Macro_F1"]
        + 0.25 * metrics["CCC_mean"]
    )
"""))

cells.append(code("""
def make_loader(df, shuffle: bool):
    dataset = IemocapRawAudioDataset(df)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=collate_raw_audio,
    )


def move_batch_to_device(batch):
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(DEVICE, non_blocking=True)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate_model(model, loader):
    if len(loader.dataset) == 0:
        raise ValueError("Evaluation dataset is empty. Check train/val/test split mapping.")
    model.eval()
    y_true, y_pred = [], []
    vad_true, vad_pred = [], []
    utterances, speakers, sessions = [], [], []
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch)
        outputs = model(batch["input_values"], batch.get("attention_mask"))
        loss, _ = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
        probs = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = probs.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        utterances.extend(batch["utterance_id"])
        speakers.extend(batch["speaker_id"])
        sessions.extend(batch["session"])
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError("Evaluation DataLoader produced no batches. Check whether the dataset is empty.")
    vad_true = np.concatenate(vad_true, axis=0)
    vad_pred = np.concatenate(vad_pred, axis=0)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame({
        "utterance_id": utterances,
        "speaker_id": speakers,
        "session": sessions,
        "true_emotion_id": y_true,
        "pred_emotion_id": y_pred,
        "true_emotion": [EMOTION_ID_TO_NAME.get(int(x), str(x)) for x in y_true],
        "pred_emotion": [EMOTION_ID_TO_NAME.get(int(x), str(x)) for x in y_pred],
        "true_valence": vad_from_0_1(vad_true)[:, 0],
        "true_arousal": vad_from_0_1(vad_true)[:, 1],
        "true_dominance": vad_from_0_1(vad_true)[:, 2],
        "pred_valence": vad_from_0_1(vad_pred)[:, 0],
        "pred_arousal": vad_from_0_1(vad_pred)[:, 1],
        "pred_dominance": vad_from_0_1(vad_pred)[:, 2],
    })
    return metrics, pred_df
"""))

cells.append(md("""
## Hàm train một fold

Trong hàm này:

- `train_df` là phần duy nhất được đưa vào optimizer.
- `val_df` chỉ dùng để chọn checkpoint tốt nhất.
- `test_df` chỉ chạy sau khi load lại checkpoint tốt nhất.
"""))

cells.append(code("""
def maybe_limit_df(df, limit: int, seed: int):
    if limit and limit > 0 and len(df) > limit:
        return df.sample(n=limit, random_state=seed).reset_index(drop=True)
    return df.reset_index(drop=True)


def build_optimizer(model):
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": LR_BACKBONE})
    if head_params:
        groups.append({"params": head_params, "lr": LR_HEAD})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def train_one_fold(protocol: str, fold_name: str, fold_df: pd.DataFrame, seed: int):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = maybe_limit_df(fold_df[fold_df["split"] == "train"], MAX_TRAIN_SAMPLES, seed)
    val_df = maybe_limit_df(fold_df[fold_df["split"] == "val"], MAX_VAL_SAMPLES, seed)
    test_df = maybe_limit_df(fold_df[fold_df["split"] == "test"], MAX_TEST_SAMPLES, seed)

    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = RawAudioBackboneMultiTaskSER(
        PRETRAINED_MODEL_NAME,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    freeze_note = freeze_backbone_for_partial_finetune(model, UNFREEZE_LAST_N)
    trainable, total, ratio = count_trainable_parameters(model)
    print(freeze_note)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({ratio:.2%})")

    optimizer = build_optimizer(model)
    scaler = make_grad_scaler(enabled=USE_AMP)

    best_score = -1e9
    best_epoch = -1
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_steps = 0
        start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch)
            with autocast_context(enabled=USE_AMP):
                outputs = model(batch["input_values"], batch.get("attention_mask"))
                loss, parts = multitask_loss(outputs, batch["emotion_id"], batch["vad"])
                loss_for_backward = loss / GRAD_ACCUM
            scaler.scale(loss_for_backward).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())
            n_steps += 1

        val_metrics, _ = evaluate_model(model, val_loader)
        score = primary_score(val_metrics)
        row = {
            "protocol": protocol,
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": running_loss / max(n_steps, 1),
            "val_primary_score": score,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} "
            f"| val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} "
            f"| score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "pretrained_model_name": PRETRAINED_MODEL_NAME,
                        "hidden_dim": HIDDEN_DIM,
                        "dropout": DROPOUT,
                        "unfreeze_last_n": UNFREEZE_LAST_N,
                        "max_seconds": MAX_SECONDS,
                        "sample_rate": SAMPLE_RATE,
                    },
                    "best_epoch": best_epoch,
                    "best_val_score": best_score,
                },
                best_path,
            )

    history_df = pd.DataFrame(history)
    history_path = REPORT_DIR / f"{protocol}_{fold_name}_history.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")

    checkpoint = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, pred_df = evaluate_model(model, test_loader)
    pred_path = REPORT_DIR / f"{protocol}_{fold_name}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

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
    return result, history_df, pred_df
"""))

cells.append(md("""
## Chạy 5-fold và 10-fold

Mặc định notebook chạy cả hai protocol. Để debug nhanh trên Kaggle trước khi chạy full, có thể đặt:

```python
MAX_FOLDS = 1
MAX_TRAIN_SAMPLES = 256
MAX_VAL_SAMPLES = 128
MAX_TEST_SAMPLES = 128
EPOCHS = 1
```

Khi chạy thật, đặt `MAX_FOLDS = 0` và bỏ giới hạn sample.
"""))

cells.append(code("""
PROTOCOL_TO_SPLIT = {
    "5fold_session": ("iemocap_5fold_session_long.csv", split_tables["5fold_session"]),
    "10fold_speaker": ("iemocap_10fold_speaker_long.csv", split_tables["10fold_speaker"]),
}


def fold_sort_key(name: str):
    import re
    match = re.search(r"fold_(\\d+)", str(name))
    if match:
        return int(match.group(1))
    return str(name)


all_results = []
all_start = time.time()

for protocol in RUN_PROTOCOLS:
    if protocol not in PROTOCOL_TO_SPLIT:
        raise ValueError(f"Unknown protocol: {protocol}. Valid values: {list(PROTOCOL_TO_SPLIT)}")
    _, df = PROTOCOL_TO_SPLIT[protocol]
    fold_names = sorted(df["fold"].unique().tolist(), key=fold_sort_key)
    if MAX_FOLDS and MAX_FOLDS > 0:
        fold_names = fold_names[:MAX_FOLDS]
    for fold_idx, fold_name in enumerate(fold_names, start=1):
        fold_df = df[df["fold"] == fold_name].reset_index(drop=True)
        result, history_df, pred_df = train_one_fold(protocol, fold_name, fold_df, seed=SEED + fold_idx)
        all_results.append(result)

results_df = pd.DataFrame(all_results)
results_path = REPORT_DIR / "02c_raw_audio_backbone_finetune_results_by_fold.csv"
results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
print("\\nSaved:", results_path)
print("Total seconds:", round(time.time() - all_start, 2))
display(results_df)
"""))

cells.append(md("""
## Tổng hợp kết quả

Bảng dưới đây lấy trung bình và độ lệch chuẩn theo từng protocol. Khi so sánh với paper khác, cần kiểm tra kỹ:

- Họ dùng 4-class hay nhiều hơn.
- Họ dùng 5-fold session hay 10-fold speaker.
- Họ có dùng transcript/text hay chỉ audio.
- Họ báo WA/UAR/Macro-F1 cho emotion hay CCC/MAE cho VAD.
"""))

cells.append(code("""
if "results_df" in globals() and len(results_df):
    metric_cols = ["WA", "UAR", "Macro_F1", "Weighted_F1", "CCC_valence", "CCC_arousal", "CCC_dominance", "CCC_mean", "MAE_mean", "RMSE_mean"]
    summary = results_df.groupby("protocol")[metric_cols].agg(["mean", "std"]).round(4)
    summary_path = REPORT_DIR / "02c_raw_audio_backbone_finetune_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    display(summary)
    print("Saved:", summary_path)
else:
    print("No results yet.")
"""))

cells.append(md("""
## Ma trận nhầm lẫn

Cell này vẽ confusion matrix cho từng protocol bằng cách gộp tất cả file prediction của các fold đã chạy.
"""))

cells.append(code("""
def plot_confusion_for_protocol(protocol: str):
    if plt is None:
        print("matplotlib/seaborn is not available.")
        return
    pred_files = sorted(REPORT_DIR.glob(f"{protocol}_*_test_predictions.csv"))
    if not pred_files:
        print("No prediction files for", protocol)
        return
    pred_df = pd.concat([pd.read_csv(path) for path in pred_files], ignore_index=True)
    cm = confusion_matrix(pred_df["true_emotion_id"], pred_df["pred_emotion_id"], labels=[0, 1, 2, 3])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=[EMOTION_ID_TO_NAME[i] for i in range(4)],
        yticklabels=[EMOTION_ID_TO_NAME[i] for i in range(4)],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized confusion matrix - {protocol}")
    fig_path = FIGURE_DIR / f"{protocol}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.show()
    print("Saved:", fig_path)


for protocol in RUN_PROTOCOLS:
    plot_confusion_for_protocol(protocol)
"""))

cells.append(md("""
## Lưu cấu hình và nén output

File zip cuối notebook giúp tải toàn bộ kết quả gồm checkpoint, prediction, history và hình minh họa.
"""))

cells.append(code("""
run_config = {
    "seed": SEED,
    "pretrained_model_name": PRETRAINED_MODEL_NAME,
    "sample_rate": SAMPLE_RATE,
    "max_seconds": MAX_SECONDS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr_backbone": LR_BACKBONE,
    "lr_head": LR_HEAD,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "hidden_dim": HIDDEN_DIM,
    "unfreeze_last_n": UNFREEZE_LAST_N,
    "loss_emotion_weight": LOSS_EMOTION_WEIGHT,
    "loss_reg_mse_weight": LOSS_REG_MSE_WEIGHT,
    "loss_reg_ccc_weight": LOSS_REG_CCC_WEIGHT,
    "run_protocols": RUN_PROTOCOLS,
    "audio_dir": str(AUDIO_DIR),
    "split_5fold_path": str(SPLIT_5FOLD_PATH),
    "split_10fold_path": str(SPLIT_10FOLD_PATH),
}

config_path = OUTPUT_DIR / "02c_run_config.json"
config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

zip_path = OUTPUT_DIR.with_suffix(".zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(OUTPUT_DIR.parent))

print("Saved config:", config_path)
print("Saved zip:", zip_path)
"""))

cells.append(md("""
## Ghi chú đọc kết quả

Nếu 02C vẫn không tăng mạnh, các hướng cần kiểm tra tiếp là:

- `MAX_SECONDS` có cắt mất phần quan trọng của utterance không.
- `UNFREEZE_LAST_N` quá ít làm backbone chưa thích nghi đủ, hoặc quá nhiều làm overfit.
- Loss VAD đang kéo shared representation lệch khỏi emotion classification hay không.
- Happy/excited đang được gộp như thế nào và độ lệch lớp ảnh hưởng UAR ra sao.
- Có cần tách hai head sâu hơn, hoặc dùng adapter/attention pooling thay vì mean pooling không.

Điểm quan trọng nhất của 02C là protocol đã đúng: fine-tune trên train-only, chọn bằng validation, báo cáo trên test.
"""))

nb["cells"] = cells

OUT_DIR.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT_PATH)
print(OUT_PATH)
