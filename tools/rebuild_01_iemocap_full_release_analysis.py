from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = ROOT / "06_w2v_based_models" / "01_IEMOCAP Dataset Analysis and Speaker-Independent Splits" / "01_IEMOCAP_Dataset_Analysis_and_Splits.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = []

cells.append(md(r"""
# 01 - IEMOCAP Dataset Analysis and Speaker-Independent Splits

Notebook này là bước nền tảng cho toàn bộ pipeline emotion classification + valence/arousal/dominance regression.

Mục tiêu:

1. Đọc **IEMOCAP full release** theo cấu trúc gốc.
2. Parse annotation từ `Session*/dialog/EmoEvaluation/*.txt`.
3. Gắn từng utterance với audio trong `Session*/sentences/wav/**/*.wav`.
4. Gắn transcript từ `Session*/dialog/transcriptions/*.txt`.
5. Tạo metadata đầy đủ và subset 4-class thường dùng trong paper IEMOCAP SER.
6. Tạo hai protocol speaker/session-independent:
   - `5fold_session`
   - `10fold_speaker`
7. Xuất bảng thống kê, hình minh họa, leakage checks, metadata/split CSV và file zip output.

Notebook này **không train model**. Nó chỉ chuẩn hóa dataset và split để các notebook 02/03/04 dùng chung một nguồn dữ liệu.
"""))

cells.append(md(r"""
## Dataset Source

Với Kaggle dataset [dejolilandry/iemocapfullrelease](https://www.kaggle.com/datasets/dejolilandry/iemocapfullrelease/data), cấu trúc thường là:

```text
/kaggle/input/iemocapfullrelease/IEMOCAP_full_release/
```

Một số notebook/environment có thể mount dataset sâu hơn:

```text
/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release/
```

Cấu trúc bên trong:

```text
IEMOCAP_full_release/
  Documentation/
  Session1/
    dialog/
      EmoEvaluation/*.txt
      transcriptions/*.txt
      wav/*.wav
    sentences/
      wav/**/*.wav
  Session2/
  Session3/
  Session4/
  Session5/
```

Trong project này ta dùng:

| Thành phần | Vai trò |
|---|---|
| `dialog/EmoEvaluation/*.txt` | Nguồn label emotion và VAD gốc |
| `sentences/wav/**/*.wav` | Audio utterance-level dùng để train/test |
| `dialog/transcriptions/*.txt` | Transcript theo utterance, dùng cho phân tích hoặc nhánh text sau này |
| `dialog/wav/*.wav` | Audio dialog dài, không dùng trực tiếp cho SER utterance-level |
| `MOCAP_*`, `avi`, `ForcedAlignment` | Dữ liệu multimodal/align nâng cao, chưa dùng trong hướng speech-first hiện tại |

Lưu ý về license: IEMOCAP là corpus có điều kiện sử dụng riêng. Kaggle dataset là mirror thuận tiện cho thực nghiệm. Khi viết báo cáo, nên ghi rõ nguồn sử dụng và protocol lọc nhãn.
"""))

cells.append(md(r"""
## 4-Class IEMOCAP Setting

IEMOCAP full release có nhiều nhãn categorical, ví dụ:

```text
neu, ang, sad, hap, exc, fru, xxx, oth, ...
```

Nhiều paper SER trên IEMOCAP dùng subset 4-class:

```text
neutral, angry, sad, happy
```

Quy tắc gộp:

```text
neu -> neutral
ang -> angry
sad -> sad
hap -> happy
exc -> happy
```

Khi parse từ annotation gốc, subset này thường quanh **5,531 utterances**. Đây là con số hay xuất hiện trong các paper vì họ bỏ các nhãn như `fru`, `xxx`, `oth`, `fea`, `sur`, `dis`, rồi gộp `hap + exc`.
"""))

cells.append(md(r"""
## Split Protocols

Notebook tạo hai protocol:

| Protocol | Cách chia | Ý nghĩa |
|---|---|---|
| `5fold_session` | Mỗi fold test một session, validation là session kế tiếp, train là các session còn lại | Khó vì test khác session, khác cặp speaker và khác context |
| `10fold_speaker` | Mỗi fold test một speaker, validation là speaker kế tiếp, train 8 speakers còn lại | Speaker-independent theo guideline 8 train / 1 val / 1 test |

Trong cả hai protocol, scaler/feature extractor/model ở các notebook sau phải fit/train **chỉ trên split train** của từng fold.
"""))

cells.append(code(r"""
import os
import re
import json
import wave
import zipfile
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from IPython.display import display, Markdown

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    plt = None
    sns = None

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name != "Speech Project":
    for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if parent.name == "Speech Project":
            PROJECT_ROOT = parent
            break

NOTEBOOK_DIR = Path(r"D:\UTE\Speech Programming\Speech Project\06_w2v_based_models\01_IEMOCAP Dataset Analysis and Speaker-Independent Splits")
if not NOTEBOOK_DIR.exists():
    NOTEBOOK_DIR = PROJECT_ROOT / "06_w2v_based_models" / "01_IEMOCAP Dataset Analysis and Speaker-Independent Splits"

OUTPUT_DIR = NOTEBOOK_DIR / "output"
METADATA_DIR = OUTPUT_DIR / "metadata"
SPLIT_DIR = OUTPUT_DIR / "splits"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
CENTRAL_DATA_DIR = PROJECT_ROOT / "06_w2v_based_models" / "data"
CENTRAL_METADATA_DIR = CENTRAL_DATA_DIR / "metadata"
CENTRAL_SPLIT_DIR = CENTRAL_DATA_DIR / "splits"

for path in [OUTPUT_DIR, METADATA_DIR, SPLIT_DIR, REPORT_DIR, FIGURE_DIR, CENTRAL_METADATA_DIR, CENTRAL_SPLIT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("CENTRAL_DATA_DIR:", CENTRAL_DATA_DIR)
"""))

cells.append(md(r"""
## Locate IEMOCAP Full Release

Notebook sẽ tự tìm IEMOCAP full release trong các vị trí phổ biến:

```text
/kaggle/input/iemocapfullrelease/IEMOCAP_full_release
/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release
./IEMOCAP_full_release
./data/IEMOCAP_full_release
./datasets/IEMOCAP_full_release
```

Nếu cần chỉ định thủ công trên Kaggle:

```python
os.environ["IEMOCAP_ROOT"] = "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release"
```

Nếu Kaggle hiển thị path dạng `kaggle/input/datasets/dejolilandry/iemocapfullrelease`, hãy dùng:

```python
os.environ["IEMOCAP_ROOT"] = "/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release"
```
"""))

cells.append(code(r"""
def looks_like_iemocap_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    sessions = [path / f"Session{i}" for i in range(1, 6)]
    has_sessions = sum(session.exists() for session in sessions) == 5
    has_eval = all((session / "dialog" / "EmoEvaluation").exists() for session in sessions)
    has_sentence_wav = all((session / "sentences" / "wav").exists() for session in sessions)
    return has_sessions and has_eval and has_sentence_wav

def candidate_iemocap_roots():
    env_root = os.environ.get("IEMOCAP_ROOT", "").strip()
    seeds = [
        Path(env_root) if env_root else None,
        PROJECT_ROOT / "IEMOCAP_full_release",
        PROJECT_ROOT / "IEMOCAP",
        PROJECT_ROOT / "data" / "IEMOCAP_full_release",
        PROJECT_ROOT / "datasets" / "IEMOCAP_full_release",
        PROJECT_ROOT / "06_w2v_based_models" / "datasets" / "IEMOCAP_full_release",
        Path("/kaggle/input/iemocapfullrelease/IEMOCAP_full_release"),
        Path("/kaggle/input/datasets/dejolilandry/iemocapfullrelease/IEMOCAP_full_release"),
        Path("/kaggle/input/datasets/dejolilandry/iemocapfullrelease"),
        Path("/kaggle/input"),
        Path("/kaggle/input/datasets"),
        Path("/kaggle/working"),
    ]
    seen = set()
    for seed in seeds:
        if seed is None or not seed.exists():
            continue
        direct_candidates = [
            seed,
            seed / "IEMOCAP_full_release",
            seed / "iemocapfullrelease" / "IEMOCAP_full_release",
            seed / "data" / "IEMOCAP_full_release",
        ]
        for candidate in direct_candidates:
            key = str(candidate).lower()
            if key not in seen:
                seen.add(key)
                yield candidate
        try:
            for candidate in seed.rglob("IEMOCAP_full_release"):
                key = str(candidate).lower()
                if key not in seen:
                    seen.add(key)
                    yield candidate
            for session1 in seed.rglob("Session1"):
                candidate = session1.parent
                key = str(candidate).lower()
                if key not in seen:
                    seen.add(key)
                    yield candidate
        except Exception:
            pass

IEMOCAP_ROOT = None
checked_roots = []
for candidate in candidate_iemocap_roots():
    checked_roots.append(str(candidate))
    if looks_like_iemocap_root(candidate):
        IEMOCAP_ROOT = candidate.resolve()
        break

if IEMOCAP_ROOT is None:
    display(Markdown(
        "**Chưa tìm thấy IEMOCAP full release.** Hãy add Kaggle dataset `dejolilandry/iemocapfullrelease` "
        "hoặc set `IEMOCAP_ROOT` tới thư mục `IEMOCAP_full_release`."
    ))
    print("Checked candidates:")
    for item in checked_roots[:50]:
        print("-", item)
else:
    print("Found IEMOCAP_ROOT:", IEMOCAP_ROOT)
"""))

cells.append(md(r"""
## Full Release File Inventory

Trước khi parse label, notebook kiểm kê nhanh số file chính trong từng session. Bảng này giúp xác nhận dataset Kaggle/full release có đủ các thành phần cần dùng.
"""))

cells.append(code(r"""
def count_files(path: Path, pattern: str) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))

def build_file_inventory(iemocap_root: Path) -> pd.DataFrame:
    rows = []
    if iemocap_root is None:
        return pd.DataFrame()
    for session_idx in range(1, 6):
        session = f"Session{session_idx}"
        session_dir = iemocap_root / session
        rows.append({
            "session": session,
            "emo_evaluation_txt": count_files(session_dir / "dialog" / "EmoEvaluation", "*.txt"),
            "dialog_transcription_txt": count_files(session_dir / "dialog" / "transcriptions", "*.txt"),
            "dialog_wav": count_files(session_dir / "dialog" / "wav", "*.wav"),
            "sentence_wav": count_files(session_dir / "sentences" / "wav", "**/*.wav"),
            "sentence_forced_alignment": count_files(session_dir / "sentences" / "ForcedAlignment", "**/*"),
            "dialog_mocap_hand": count_files(session_dir / "dialog" / "MOCAP_hand", "**/*"),
            "dialog_mocap_head": count_files(session_dir / "dialog" / "MOCAP_head", "**/*"),
        })
    return pd.DataFrame(rows)

file_inventory = build_file_inventory(IEMOCAP_ROOT)
if file_inventory.empty:
    display(Markdown("Chưa có file inventory vì chưa tìm thấy IEMOCAP root."))
else:
    file_inventory.to_csv(REPORT_DIR / "iemocap_full_release_file_inventory.csv", index=False, encoding="utf-8-sig")
    display(file_inventory)
    display(file_inventory.drop(columns=["session"]).sum().rename("total").reset_index().rename(columns={"index": "file_group"}))
"""))

cells.append(md(r"""
## Parser Design

Mỗi dòng hợp lệ trong `EmoEvaluation` có dạng:

```text
[start - end] utterance_id emotion [valence, arousal, dominance]
```

Ví dụ:

```text
[6.2901 - 8.2357] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
```

Notebook lưu cả:

- `original_emotion`: nhãn ngắn gốc như `neu`, `ang`, `hap`, `exc`.
- `emotion`: nhãn đầy đủ sau mapping.
- `emotion_4class`: nhãn sau khi lọc/gộp cho benchmark 4-class.
- `valence`, `arousal`, `dominance`: điểm gốc thang 1-5.
- `*_norm`: chuẩn hóa về 0-1 cho regression.
"""))

cells.append(code(r"""
EVAL_PATTERN = re.compile(
    r"^\[(?P<start>[0-9.]+)\s*-\s*(?P<end>[0-9.]+)\]\s+"
    r"(?P<utterance_id>\S+)\s+"
    r"(?P<emotion>\w+)\s+"
    r"\[(?P<valence>[0-9.]+),\s*(?P<arousal>[0-9.]+),\s*(?P<dominance>[0-9.]+)\]"
)

TRANSCRIPT_PATTERN = re.compile(r"^(?P<utterance_id>Ses\d{2}\S+)\s+\[(?P<start>[0-9.]+)-(?P<end>[0-9.]+)\]:\s*(?P<text>.*)$")

EMOTION_NAME_MAP = {
    "neu": "neutral",
    "ang": "angry",
    "sad": "sad",
    "hap": "happy",
    "exc": "happy",
    "sur": "surprise",
    "fea": "fear",
    "dis": "disgust",
    "fru": "frustration",
    "oth": "other",
    "xxx": "unknown",
}

FOUR_CLASS_EMOTIONS = ["neutral", "angry", "sad", "happy"]
EMOTION_ID_MAP = {name: idx for idx, name in enumerate(FOUR_CLASS_EMOTIONS)}

def parse_session_from_utt(utterance_id: str) -> str:
    match = re.match(r"(Ses\d{2})", str(utterance_id))
    return match.group(1) if match else "unknown_session"

def parse_gender_from_utt(utterance_id: str) -> str:
    last = str(utterance_id).split("_")[-1]
    if last.startswith("F"):
        return "female"
    if last.startswith("M"):
        return "male"
    return "unknown"

def parse_speaker_from_utt(utterance_id: str) -> str:
    session = parse_session_from_utt(utterance_id)
    gender = parse_gender_from_utt(utterance_id)
    if gender == "female":
        return f"{session}F"
    if gender == "male":
        return f"{session}M"
    return f"{session}_unknown"

def parse_conversation_id(utterance_id: str) -> str:
    # Ses01F_impro01_F000 -> Ses01F_impro01
    # Ses01F_script01_1_F000 -> Ses01F_script01_1
    parts = str(utterance_id).split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 else str(utterance_id)

def wav_duration_seconds(path: Path):
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            return frames / float(rate), rate, channels
    except Exception:
        return np.nan, np.nan, np.nan

def build_wav_index(iemocap_root: Path) -> dict:
    wav_paths = sorted(iemocap_root.glob("Session*/sentences/wav/**/*.wav"))
    return {path.stem: path.resolve() for path in wav_paths}

def parse_transcripts(iemocap_root: Path) -> pd.DataFrame:
    rows = []
    for transcript_path in sorted(iemocap_root.glob("Session*/dialog/transcriptions/*.txt")):
        with transcript_path.open("r", encoding="utf-8", errors="ignore") as file:
            for raw_line in file:
                line = raw_line.strip()
                match = TRANSCRIPT_PATTERN.match(line)
                if not match:
                    continue
                item = match.groupdict()
                rows.append({
                    "utterance_id": item["utterance_id"],
                    "transcript_start": float(item["start"]),
                    "transcript_end": float(item["end"]),
                    "transcript": item["text"].strip(),
                    "transcript_file": str(transcript_path),
                })
    return pd.DataFrame(rows)

def parse_iemocap_metadata(iemocap_root: Path) -> pd.DataFrame:
    wav_index = build_wav_index(iemocap_root)
    transcript_df = parse_transcripts(iemocap_root)
    transcript_map = {}
    if not transcript_df.empty:
        duplicate_count = int(transcript_df["utterance_id"].duplicated().sum())
        if duplicate_count:
            duplicate_path = REPORT_DIR / "duplicate_transcript_utterance_ids.csv"
            duplicate_rows = transcript_df[transcript_df["utterance_id"].duplicated(keep=False)].sort_values(["utterance_id", "transcript_file"])
            duplicate_rows.to_csv(duplicate_path, index=False, encoding="utf-8-sig")
            print(f"Found {duplicate_count} duplicate transcript rows. Keeping the first transcript per utterance_id.")
            print("Saved duplicate transcript report:", duplicate_path)
        transcript_df = (
            transcript_df
            .sort_values(["utterance_id", "transcript_file"])
            .drop_duplicates(subset=["utterance_id"], keep="first")
            .reset_index(drop=True)
        )
        transcript_map = transcript_df.set_index("utterance_id").to_dict("index")

    rows = []
    eval_files = sorted(iemocap_root.glob("Session*/dialog/EmoEvaluation/*.txt"))
    for eval_path in eval_files:
        with eval_path.open("r", encoding="utf-8", errors="ignore") as file:
            for raw_line in file:
                line = raw_line.strip()
                match = EVAL_PATTERN.match(line)
                if not match:
                    continue
                item = match.groupdict()
                utterance_id = item["utterance_id"]
                wav_path = wav_index.get(utterance_id)
                duration, sample_rate, channels = wav_duration_seconds(wav_path) if wav_path else (np.nan, np.nan, np.nan)
                original_emotion = item["emotion"]
                emotion = EMOTION_NAME_MAP.get(original_emotion, original_emotion)
                transcript_item = transcript_map.get(utterance_id, {})
                rows.append({
                    "utterance_id": utterance_id,
                    "conversation_id": parse_conversation_id(utterance_id),
                    "session": parse_session_from_utt(utterance_id),
                    "speaker_id": parse_speaker_from_utt(utterance_id),
                    "gender": parse_gender_from_utt(utterance_id),
                    "original_emotion": original_emotion,
                    "emotion": emotion,
                    "start_time": float(item["start"]),
                    "end_time": float(item["end"]),
                    "eval_duration": float(item["end"]) - float(item["start"]),
                    "valence": float(item["valence"]),
                    "arousal": float(item["arousal"]),
                    "dominance": float(item["dominance"]),
                    "wav_path": str(wav_path) if wav_path else None,
                    "wav_found": wav_path is not None,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "transcript": transcript_item.get("transcript"),
                    "transcript_found": utterance_id in transcript_map,
                    "transcript_file": transcript_item.get("transcript_file"),
                    "eval_file": str(eval_path),
                })

    metadata = pd.DataFrame(rows)
    if metadata.empty:
        return metadata

    metadata = metadata.sort_values(["session", "conversation_id", "utterance_id"]).reset_index(drop=True)
    metadata.insert(0, "sample_id", [f"iemocap_{i:05d}" for i in range(len(metadata))])
    metadata["is_4class"] = metadata["emotion"].isin(FOUR_CLASS_EMOTIONS)
    metadata["emotion_4class"] = metadata["emotion"].where(metadata["is_4class"], np.nan)
    metadata["emotion_id"] = metadata["emotion_4class"].map(EMOTION_ID_MAP)
    metadata["valence_norm"] = (metadata["valence"] - 1.0) / 4.0
    metadata["arousal_norm"] = (metadata["arousal"] - 1.0) / 4.0
    metadata["dominance_norm"] = (metadata["dominance"] - 1.0) / 4.0
    metadata["source_dataset"] = "IEMOCAP_full_release"
    metadata["word_count"] = metadata["transcript"].fillna("").str.split().str.len()
    return metadata
"""))

cells.append(md(r"""
## Parse Full Metadata

Sau cell này, kiểm tra nhanh:

- `metadata.shape`: tổng số utterances parse được từ annotation gốc.
- `wav_found`: tỷ lệ utterance có audio sentence-level.
- `transcript_found`: tỷ lệ utterance có transcript.
- `original_emotion`: phân phối nhãn gốc.
"""))

cells.append(code(r"""
if IEMOCAP_ROOT is None:
    metadata = pd.DataFrame()
else:
    metadata = parse_iemocap_metadata(IEMOCAP_ROOT)

print("Full metadata shape:", metadata.shape)
if not metadata.empty:
    display(metadata.head(10))
    print("WAV found rate:", round(float(metadata["wav_found"].mean()), 4))
    print("Transcript found rate:", round(float(metadata["transcript_found"].mean()), 4))
    print("Sessions:", sorted(metadata["session"].dropna().unique().tolist()))
    print("Speakers:", sorted(metadata["speaker_id"].dropna().unique().tolist()))
"""))

cells.append(code(r"""
if not metadata.empty:
    metadata_path = METADATA_DIR / "iemocap_metadata_full.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    shutil.copy2(metadata_path, CENTRAL_METADATA_DIR / metadata_path.name)
    print("Saved:", metadata_path)
    print("Copied to:", CENTRAL_METADATA_DIR / metadata_path.name)
"""))

cells.append(md(r"""
## Full Dataset Statistics

Các thống kê này dùng để hiểu toàn bộ IEMOCAP trước khi lọc 4-class.
"""))

cells.append(code(r"""
if metadata.empty:
    display(Markdown("Metadata rỗng nên chưa thể thống kê."))
else:
    overview = pd.DataFrame({
        "metric": [
            "total_utterances",
            "sessions",
            "speakers",
            "conversations",
            "wav_found_rate",
            "transcript_found_rate",
            "mean_duration_sec",
            "median_duration_sec",
            "mean_word_count",
        ],
        "value": [
            len(metadata),
            metadata["session"].nunique(),
            metadata["speaker_id"].nunique(),
            metadata["conversation_id"].nunique(),
            metadata["wav_found"].mean(),
            metadata["transcript_found"].mean(),
            metadata["duration"].mean(),
            metadata["duration"].median(),
            metadata["word_count"].mean(),
        ],
    })
    overview.to_csv(REPORT_DIR / "dataset_overview.csv", index=False, encoding="utf-8-sig")
    display(overview)

    label_counts = metadata["original_emotion"].value_counts(dropna=False).rename_axis("original_emotion").reset_index(name="count")
    label_counts["mapped_emotion"] = label_counts["original_emotion"].map(EMOTION_NAME_MAP).fillna(label_counts["original_emotion"])
    label_counts.to_csv(REPORT_DIR / "full_original_emotion_distribution.csv", index=False, encoding="utf-8-sig")
    display(label_counts)
"""))

cells.append(code(r"""
if not metadata.empty and plt is not None:
    emotion_counts = metadata["emotion"].value_counts().reset_index()
    emotion_counts.columns = ["emotion", "count"]

    plt.figure(figsize=(10, 5))
    sns.barplot(data=emotion_counts, x="emotion", y="count", color="#4C78A8")
    plt.title("Full IEMOCAP Emotion Distribution")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig_path = FIGURE_DIR / "full_emotion_distribution.png"
    plt.savefig(fig_path, dpi=160)
    plt.show()
    print("Saved:", fig_path)
"""))

cells.append(md(r"""
## Build 4-Class Train-Ready Subset

Điều kiện giữ lại:

1. `emotion_4class` thuộc `neutral/angry/sad/happy`.
2. Có file `.wav` sentence-level.
3. Có đầy đủ `valence/arousal/dominance`.

`happy` bao gồm cả `hap` và `exc`.
"""))

cells.append(code(r"""
if metadata.empty:
    train_metadata = pd.DataFrame()
else:
    train_metadata = metadata[
        metadata["is_4class"]
        & metadata["wav_found"]
        & metadata[["valence", "arousal", "dominance"]].notna().all(axis=1)
    ].copy()
    train_metadata = train_metadata.sort_values(["session", "speaker_id", "conversation_id", "utterance_id"]).reset_index(drop=True)
    train_metadata.insert(0, "train_sample_id", [f"iemocap_train_{i:05d}" for i in range(len(train_metadata))])
    train_metadata["emotion_id"] = train_metadata["emotion_id"].astype(int)

print("4-class train-ready shape:", train_metadata.shape)
if not train_metadata.empty:
    display(train_metadata.head(10))
    class_counts = train_metadata["emotion_4class"].value_counts().reindex(FOUR_CLASS_EMOTIONS).fillna(0).astype(int)
    display(class_counts.rename("count").reset_index().rename(columns={"index": "emotion_4class"}))
    print("Expected paper-like size around 5,531. Current:", len(train_metadata))
"""))

cells.append(code(r"""
if not train_metadata.empty:
    train_metadata_path = METADATA_DIR / "iemocap_4class_avd_metadata.csv"
    train_metadata.to_csv(train_metadata_path, index=False, encoding="utf-8-sig")
    shutil.copy2(train_metadata_path, CENTRAL_METADATA_DIR / train_metadata_path.name)
    print("Saved:", train_metadata_path)
    print("Copied to:", CENTRAL_METADATA_DIR / train_metadata_path.name)
"""))

cells.append(md(r"""
## 4-Class Statistical Tables

Các bảng này là phần nên đưa vào báo cáo:

- Emotion count.
- Count theo session/speaker.
- Duration theo class.
- VAD theo class.
- Transcript length theo class.
"""))

cells.append(code(r"""
if train_metadata.empty:
    display(Markdown("Train metadata rỗng nên chưa thể thống kê subset 4-class."))
else:
    emotion_distribution = (
        train_metadata["emotion_4class"]
        .value_counts()
        .reindex(FOUR_CLASS_EMOTIONS)
        .fillna(0)
        .astype(int)
        .rename_axis("emotion_4class")
        .reset_index(name="count")
    )
    emotion_distribution["percent"] = emotion_distribution["count"] / emotion_distribution["count"].sum() * 100
    emotion_distribution.to_csv(REPORT_DIR / "4class_emotion_distribution.csv", index=False, encoding="utf-8-sig")
    display(emotion_distribution)

    session_emotion = pd.crosstab(train_metadata["session"], train_metadata["emotion_4class"]).reindex(columns=FOUR_CLASS_EMOTIONS).fillna(0).astype(int)
    session_emotion.to_csv(REPORT_DIR / "4class_session_emotion_crosstab.csv", encoding="utf-8-sig")
    display(session_emotion)

    speaker_emotion = pd.crosstab(train_metadata["speaker_id"], train_metadata["emotion_4class"]).reindex(columns=FOUR_CLASS_EMOTIONS).fillna(0).astype(int)
    speaker_emotion.to_csv(REPORT_DIR / "4class_speaker_emotion_crosstab.csv", encoding="utf-8-sig")
    display(speaker_emotion)

    duration_stats = (
        train_metadata.groupby("emotion_4class")["duration"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
        .round(3)
        .reindex(FOUR_CLASS_EMOTIONS)
    )
    duration_stats.to_csv(REPORT_DIR / "4class_duration_stats.csv", encoding="utf-8-sig")
    display(duration_stats)

    vad_stats = (
        train_metadata.groupby("emotion_4class")[["valence", "arousal", "dominance"]]
        .agg(["mean", "std", "min", "max"])
        .round(3)
        .reindex(FOUR_CLASS_EMOTIONS)
    )
    vad_stats.to_csv(REPORT_DIR / "4class_vad_stats.csv", encoding="utf-8-sig")
    display(vad_stats)

    transcript_stats = (
        train_metadata.groupby("emotion_4class")["word_count"]
        .describe(percentiles=[0.25, 0.5, 0.75, 0.90])
        .round(3)
        .reindex(FOUR_CLASS_EMOTIONS)
    )
    transcript_stats.to_csv(REPORT_DIR / "4class_transcript_word_count_stats.csv", encoding="utf-8-sig")
    display(transcript_stats)
"""))

cells.append(code(r"""
if not train_metadata.empty and plt is not None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.countplot(data=train_metadata, x="emotion_4class", order=FOUR_CLASS_EMOTIONS, ax=axes[0, 0], color="#4C78A8")
    axes[0, 0].set_title("4-Class Emotion Counts")
    axes[0, 0].set_xlabel("")

    sns.boxplot(data=train_metadata, x="emotion_4class", y="duration", order=FOUR_CLASS_EMOTIONS, ax=axes[0, 1], color="#72B7B2")
    axes[0, 1].set_title("Duration by Emotion")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("seconds")

    vad_long = train_metadata.melt(
        id_vars=["emotion_4class"],
        value_vars=["valence", "arousal", "dominance"],
        var_name="dimension",
        value_name="score",
    )
    sns.boxplot(data=vad_long, x="emotion_4class", y="score", hue="dimension", order=FOUR_CLASS_EMOTIONS, ax=axes[1, 0])
    axes[1, 0].set_title("Valence / Arousal / Dominance by Emotion")
    axes[1, 0].set_xlabel("")
    axes[1, 0].legend(title="")

    corr = train_metadata[["valence", "arousal", "dominance", "duration", "word_count"]].corr()
    sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Matrix")

    plt.tight_layout()
    fig_path = FIGURE_DIR / "4class_dataset_statistics.png"
    plt.savefig(fig_path, dpi=160)
    plt.show()
    print("Saved:", fig_path)
"""))

cells.append(md(r"""
## Split Generation

### 5-fold session split

Với 5 sessions:

```text
fold_1: test Ses01, val Ses02, train Ses03+Ses04+Ses05
fold_2: test Ses02, val Ses03, train Ses01+Ses04+Ses05
...
fold_5: test Ses05, val Ses01, train Ses02+Ses03+Ses04
```

### 10-fold speaker split

Với 10 speakers:

```text
Ses01F, Ses01M, Ses02F, Ses02M, ..., Ses05F, Ses05M
```

Mỗi fold:

```text
1 speaker test
1 speaker validation
8 speakers train
```
"""))

cells.append(code(r"""
def make_long_split(base_df: pd.DataFrame, protocol: str, fold_specs: list[dict]) -> pd.DataFrame:
    rows = []
    for spec in fold_specs:
        fold_name = spec["fold"]
        train_mask = spec["train_mask"](base_df)
        val_mask = spec["val_mask"](base_df)
        test_mask = spec["test_mask"](base_df)
        for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            part = base_df.loc[mask].copy()
            part["protocol"] = protocol
            part["fold"] = fold_name
            part["split"] = split_name
            rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def build_5fold_session(base_df: pd.DataFrame) -> pd.DataFrame:
    sessions = [f"Ses{i:02d}" for i in range(1, 6)]
    specs = []
    for idx, test_session in enumerate(sessions):
        val_session = sessions[(idx + 1) % len(sessions)]
        train_sessions = [s for s in sessions if s not in {test_session, val_session}]
        specs.append({
            "fold": f"fold_{idx + 1}_test_{test_session}_val_{val_session}",
            "train_mask": lambda df, train_sessions=train_sessions: df["session"].isin(train_sessions),
            "val_mask": lambda df, val_session=val_session: df["session"].eq(val_session),
            "test_mask": lambda df, test_session=test_session: df["session"].eq(test_session),
        })
    return make_long_split(base_df, "5fold_session", specs)

def speaker_sort_key(speaker_id: str):
    match = re.match(r"Ses(\d{2})([FM])", speaker_id)
    if not match:
        return (999, speaker_id)
    session_num = int(match.group(1))
    gender_order = 0 if match.group(2) == "F" else 1
    return (session_num, gender_order)

def build_10fold_speaker(base_df: pd.DataFrame) -> pd.DataFrame:
    speakers = sorted(base_df["speaker_id"].dropna().unique().tolist(), key=speaker_sort_key)
    specs = []
    for idx, test_speaker in enumerate(speakers):
        val_speaker = speakers[(idx + 1) % len(speakers)]
        train_speakers = [s for s in speakers if s not in {test_speaker, val_speaker}]
        specs.append({
            "fold": f"fold_{idx + 1}_test_{test_speaker}_val_{val_speaker}",
            "train_mask": lambda df, train_speakers=train_speakers: df["speaker_id"].isin(train_speakers),
            "val_mask": lambda df, val_speaker=val_speaker: df["speaker_id"].eq(val_speaker),
            "test_mask": lambda df, test_speaker=test_speaker: df["speaker_id"].eq(test_speaker),
        })
    return make_long_split(base_df, "10fold_speaker", specs)

if train_metadata.empty:
    split_5fold = pd.DataFrame()
    split_10fold = pd.DataFrame()
else:
    split_5fold = build_5fold_session(train_metadata)
    split_10fold = build_10fold_speaker(train_metadata)

print("5-fold rows:", split_5fold.shape)
print("10-fold rows:", split_10fold.shape)
"""))

cells.append(code(r"""
def split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    if split_df.empty:
        return pd.DataFrame()
    summary = split_df.groupby(["fold", "split"]).size().unstack(fill_value=0)
    summary["total_fold_rows"] = summary.sum(axis=1)
    return summary

summary_5 = split_summary(split_5fold)
summary_10 = split_summary(split_10fold)

if not summary_5.empty:
    summary_5.to_csv(REPORT_DIR / "5fold_session_split_sizes.csv", encoding="utf-8-sig")
    display(summary_5)
if not summary_10.empty:
    summary_10.to_csv(REPORT_DIR / "10fold_speaker_split_sizes.csv", encoding="utf-8-sig")
    display(summary_10)
"""))

cells.append(md(r"""
## Leakage Checks

Leakage check không chứng minh model đúng, nhưng giúp đảm bảo split không vô tình đưa cùng session/speaker vào train và test.

Ta kiểm tra:

- Trùng `utterance_id` giữa train/val/test trong cùng fold.
- 5-fold: session giữa train/val/test không overlap.
- 10-fold: speaker giữa train/val/test không overlap.
"""))

cells.append(code(r"""
def leakage_check(split_df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    rows = []
    if split_df.empty:
        return pd.DataFrame()
    for fold, fold_df in split_df.groupby("fold"):
        split_groups = {name: part for name, part in fold_df.groupby("split")}
        def ids(split):
            return set(split_groups.get(split, pd.DataFrame()).get("utterance_id", pd.Series(dtype=str)).astype(str))
        def sessions(split):
            return set(split_groups.get(split, pd.DataFrame()).get("session", pd.Series(dtype=str)).astype(str))
        def speakers(split):
            return set(split_groups.get(split, pd.DataFrame()).get("speaker_id", pd.Series(dtype=str)).astype(str))
        rows.append({
            "protocol": protocol,
            "fold": fold,
            "train_val_utterance_overlap": sorted(ids("train") & ids("val")),
            "train_test_utterance_overlap": sorted(ids("train") & ids("test")),
            "val_test_utterance_overlap": sorted(ids("val") & ids("test")),
            "train_val_session_overlap": sorted(sessions("train") & sessions("val")),
            "train_test_session_overlap": sorted(sessions("train") & sessions("test")),
            "val_test_session_overlap": sorted(sessions("val") & sessions("test")),
            "train_val_speaker_overlap": sorted(speakers("train") & speakers("val")),
            "train_test_speaker_overlap": sorted(speakers("train") & speakers("test")),
            "val_test_speaker_overlap": sorted(speakers("val") & speakers("test")),
        })
    return pd.DataFrame(rows)

leak_5 = leakage_check(split_5fold, "5fold_session")
leak_10 = leakage_check(split_10fold, "10fold_speaker")

if not leak_5.empty:
    leak_5.to_csv(REPORT_DIR / "5fold_session_leakage_check.csv", index=False, encoding="utf-8-sig")
    display(leak_5)
if not leak_10.empty:
    leak_10.to_csv(REPORT_DIR / "10fold_speaker_leakage_check.csv", index=False, encoding="utf-8-sig")
    display(leak_10)
"""))

cells.append(code(r"""
if not split_5fold.empty and plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plot_5 = split_5fold.groupby(["fold", "split"]).size().reset_index(name="count")
    sns.barplot(data=plot_5, x="fold", y="count", hue="split", ax=axes[0])
    axes[0].set_title("5-fold Session Split Sizes")
    axes[0].tick_params(axis="x", rotation=45)

    plot_10 = split_10fold.groupby(["fold", "split"]).size().reset_index(name="count")
    sns.barplot(data=plot_10, x="fold", y="count", hue="split", ax=axes[1])
    axes[1].set_title("10-fold Speaker Split Sizes")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig_path = FIGURE_DIR / "split_size_overview.png"
    plt.savefig(fig_path, dpi=160)
    plt.show()
    print("Saved:", fig_path)
"""))

cells.append(md(r"""
## Save Split Files

Notebook lưu split ở hai nơi:

1. Folder local của notebook 01:

```text
01_IEMOCAP Dataset Analysis and Speaker-Independent Splits/output/
```

2. Folder dùng chung cho các notebook sau:

```text
06_w2v_based_models/data/
```
"""))

cells.append(code(r"""
def save_split_files(split_df: pd.DataFrame, csv_name: str, json_name: str):
    if split_df.empty:
        print("Skip empty split:", csv_name)
        return
    csv_path = SPLIT_DIR / csv_name
    json_path = SPLIT_DIR / json_name
    split_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    fold_spec = {}
    for fold, fold_df in split_df.groupby("fold"):
        fold_spec[fold] = {}
        for split_name, part in fold_df.groupby("split"):
            fold_spec[fold][split_name] = sorted(part["train_sample_id"].astype(str).tolist())
    json_path.write_text(json.dumps(fold_spec, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.copy2(csv_path, CENTRAL_SPLIT_DIR / csv_path.name)
    shutil.copy2(json_path, CENTRAL_SPLIT_DIR / json_path.name)
    print("Saved:", csv_path)
    print("Saved:", json_path)
    print("Copied to:", CENTRAL_SPLIT_DIR)

save_split_files(split_5fold, "iemocap_5fold_session_long.csv", "iemocap_5fold_session.json")
save_split_files(split_10fold, "iemocap_10fold_speaker_long.csv", "iemocap_10fold_speaker.json")
"""))

cells.append(md(r"""
## Final Research Summary

Cell cuối tạo file markdown tóm tắt để dùng trong báo cáo hoặc kiểm tra nhanh.
"""))

cells.append(code(r"""
def list_status(df: pd.DataFrame, cols: list[str]) -> bool:
    if df.empty:
        return False
    for col in cols:
        if col not in df.columns:
            return False
        if df[col].astype(str).apply(lambda x: x not in {"[]", "set()"}).any():
            return True
    return False

if metadata.empty or train_metadata.empty:
    display(Markdown("Chưa thể tạo final report vì metadata rỗng."))
else:
    emotion_counts = train_metadata["emotion_4class"].value_counts().reindex(FOUR_CLASS_EMOTIONS).fillna(0).astype(int)
    duration_desc = train_metadata["duration"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    vad_mean = train_metadata[["valence", "arousal", "dominance"]].mean()
    vad_by_emotion = train_metadata.groupby("emotion_4class")[["valence", "arousal", "dominance"]].mean().round(3).reindex(FOUR_CLASS_EMOTIONS)
    try:
        vad_table = vad_by_emotion.to_markdown()
    except Exception:
        vad_table = "```text\n" + vad_by_emotion.to_string() + "\n```"
    crop_rates = {sec: float((train_metadata["duration"] > sec).mean()) for sec in [3, 4, 5, 6, 8, 10, 15]}

    report_lines = [
        "# IEMOCAP Full Release Dataset Analysis Summary",
        "",
        "## Source",
        "",
        f"- Root: `{IEMOCAP_ROOT}`",
        "- Annotation source: `Session*/dialog/EmoEvaluation/*.txt`.",
        "- Audio source: `Session*/sentences/wav/**/*.wav`.",
        "- Transcript source: `Session*/dialog/transcriptions/*.txt`.",
        "",
        "## Scope",
        "",
        f"- Full utterances parsed: **{len(metadata):,}**.",
        f"- 4-class train-ready utterances: **{len(train_metadata):,}**.",
        f"- Sessions: **{train_metadata['session'].nunique()}**.",
        f"- Speakers: **{train_metadata['speaker_id'].nunique()}**.",
        f"- Conversations: **{train_metadata['conversation_id'].nunique()}**.",
        f"- WAV found rate: **{metadata['wav_found'].mean():.4f}**.",
        f"- Transcript found rate: **{metadata['transcript_found'].mean():.4f}**.",
        "",
        "## 4-Class Label Rule",
        "",
        "`neu -> neutral`, `ang -> angry`, `sad -> sad`, `hap + exc -> happy`.",
        "",
        "Class counts:",
    ]
    for emotion, count in emotion_counts.items():
        report_lines.append(f"- {emotion}: **{count:,}**")

    report_lines += [
        "",
        "## Duration",
        "",
        f"- Mean: **{duration_desc['mean']:.3f}s**.",
        f"- Median: **{duration_desc['50%']:.3f}s**.",
        f"- 75th percentile: **{duration_desc['75%']:.3f}s**.",
        f"- 95th percentile: **{duration_desc['95%']:.3f}s**.",
        f"- Max: **{duration_desc['max']:.3f}s**.",
        "",
        "Fixed-length truncation risk:",
    ]
    for sec, rate in crop_rates.items():
        report_lines.append(f"- Longer than {sec}s: **{rate * 100:.2f}%**")

    report_lines += [
        "",
        "## Valence / Arousal / Dominance",
        "",
        "Labels are on the original IEMOCAP 1-5 scale. Notebook also stores normalized 0-1 columns.",
        "",
        f"- Mean valence: **{vad_mean['valence']:.3f}**.",
        f"- Mean arousal: **{vad_mean['arousal']:.3f}**.",
        f"- Mean dominance: **{vad_mean['dominance']:.3f}**.",
        "",
        vad_table,
        "",
        "## Split Checks",
        "",
        f"- 5-fold rows: **{len(split_5fold):,}**.",
        f"- 10-fold rows: **{len(split_10fold):,}**.",
        f"- 5-fold leakage detected: **{list_status(leak_5, [c for c in leak_5.columns if 'overlap' in c])}**.",
        f"- 10-fold leakage detected: **{list_status(leak_10, [c for c in leak_10.columns if 'utterance_overlap' in c or 'speaker_overlap' in c])}**.",
        "",
        "## Output Files",
        "",
        f"- Metadata: `{METADATA_DIR}`",
        f"- Splits: `{SPLIT_DIR}`",
        f"- Reports: `{REPORT_DIR}`",
        f"- Figures: `{FIGURE_DIR}`",
        f"- Shared data folder: `{CENTRAL_DATA_DIR}`",
    ]

    report_path = REPORT_DIR / "iemocap_full_release_analysis_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    display(Markdown(report_path.read_text(encoding="utf-8")))
    print("Saved report:", report_path)
"""))

cells.append(code(r"""
zip_path = OUTPUT_DIR.with_suffix(".zip")
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in OUTPUT_DIR.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(OUTPUT_DIR.parent))
print("Saved zip:", zip_path)
"""))


def main():
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print("Rebuilt", NB_PATH)


if __name__ == "__main__":
    main()
