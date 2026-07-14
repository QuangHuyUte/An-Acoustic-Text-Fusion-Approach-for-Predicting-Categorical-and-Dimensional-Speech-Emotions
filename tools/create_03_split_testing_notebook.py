from pathlib import Path

import nbformat as nbf


BASE = Path(r"D:/UTE/Speech Programming/Speech Project")
NB_DIR = BASE / "06_w2v_based_models" / "03IEMOCAP HF Original Split Testing and Manifests"
NB_PATH = NB_DIR / "03_IEMOCAP_HF_Original_Split_Testing_and_Manifests.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def main() -> None:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            """
# 03 - IEMOCAP HF Original Split Testing and Manifests

Notebook này dùng **nguyên setting hiện tại** của project:

- Dataset: `AbstractTTS/IEMOCAP` đã tải về local.
- Subset: `HF-major-emotion 4-class + AVD`.
- Số mẫu: **6,877 utterances**.
- Emotion classes: `neutral`, `angry`, `sad`, `happy`, trong đó `happy = happy + excited`.
- Không undersampling, không cân bằng lại lớp `happy`.

Mục tiêu của notebook:

1. Kiểm tra lại 5-fold session split và 10-fold speaker split đã sinh ở notebook 01.
2. Xem phân bố emotion, speaker, session ở từng fold.
3. Kiểm tra leakage giữa train/validation/test.
4. Tạo manifest CSV cho từng fold để notebook training dùng trực tiếp.
5. Chạy majority-class baseline rất nhẹ để thấy độ lệch lớp ảnh hưởng thế nào tới WA/UA.

Lưu ý: đây **không phải official IEMOCAP 5,531 benchmark setting**. Đây là setting thực nghiệm hiện tại để đi tiếp pipeline multi-task.
"""
        ),
        code(
            """
from pathlib import Path
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display, Markdown
except Exception:
    display = print
    Markdown = lambda x: x

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 160)
pd.set_option("display.max_colwidth", 160)

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name != "Speech Project" and PROJECT_ROOT.parent.name == "Speech Project":
    PROJECT_ROOT = PROJECT_ROOT.parent.resolve()

BASE_DIR = PROJECT_ROOT / "06_w2v_based_models"
FOLDER01 = BASE_DIR / "01IEMOCAP Dataset Analysis and Speaker-Independent Splits"
NOTEBOOK_DIR = BASE_DIR / "03IEMOCAP HF Original Split Testing and Manifests"

METADATA_DIR = FOLDER01 / "metadata"
SPLIT_DIR = FOLDER01 / "splits"
REPORT_DIR = NOTEBOOK_DIR / "reports"
FIGURE_DIR = NOTEBOOK_DIR / "figures"
MANIFEST_DIR = NOTEBOOK_DIR / "manifests"

for folder in [REPORT_DIR, FIGURE_DIR, MANIFEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

FULL_METADATA_PATH = METADATA_DIR / "iemocap_metadata_full.csv"
TRAIN_METADATA_PATH = METADATA_DIR / "iemocap_4class_avd_metadata.csv"
SPLIT_5FOLD_PATH = SPLIT_DIR / "iemocap_5fold_session_long.csv"
SPLIT_10FOLD_PATH = SPLIT_DIR / "iemocap_10fold_speaker_long.csv"

EMOTIONS = ["neutral", "angry", "sad", "happy"]

print("PROJECT_ROOT:", PROJECT_ROOT)
print("NOTEBOOK_DIR:", NOTEBOOK_DIR)
print("TRAIN_METADATA_PATH exists:", TRAIN_METADATA_PATH.exists())
print("SPLIT_5FOLD_PATH exists:", SPLIT_5FOLD_PATH.exists())
print("SPLIT_10FOLD_PATH exists:", SPLIT_10FOLD_PATH.exists())
"""
        ),
        md(
            """
## 1. Load Metadata

Ta load cả hai file:

- `iemocap_metadata_full.csv`: metadata đầy đủ của HuggingFace mirror.
- `iemocap_4class_avd_metadata.csv`: subset 4-class + AVD đã dùng để tạo split.

Notebook 03 sẽ merge thêm transcript và các acoustic metadata nhẹ vào manifest để notebook train sau này có thể dùng cả audio hoặc text nếu cần.
"""
        ),
        code(
            """
full_metadata = pd.read_csv(FULL_METADATA_PATH)
metadata = pd.read_csv(TRAIN_METADATA_PATH)

required_cols = {
    "train_sample_id", "utterance_id", "session", "speaker_id",
    "emotion_4class", "emotion_id", "valence", "arousal", "dominance", "wav_path"
}
missing = required_cols - set(metadata.columns)
if missing:
    raise ValueError(f"Missing required metadata columns: {missing}")

metadata["wav_exists"] = metadata["wav_path"].apply(lambda p: Path(str(p)).exists())
metadata["duration"] = pd.to_numeric(metadata["duration"], errors="coerce")

print("Full metadata:", full_metadata.shape)
print("4-class AVD metadata:", metadata.shape)
print("WAV exists rate:", metadata["wav_exists"].mean())
display(metadata.head())
"""
        ),
        code(
            """
overview = pd.DataFrame({
    "metric": [
        "n_samples", "n_sessions", "n_speakers", "n_conversations",
        "wav_exists_rate", "duration_mean", "duration_median", "duration_max"
    ],
    "value": [
        len(metadata),
        metadata["session"].nunique(),
        metadata["speaker_id"].nunique(),
        metadata["conversation_id"].nunique() if "conversation_id" in metadata.columns else np.nan,
        metadata["wav_exists"].mean(),
        metadata["duration"].mean(),
        metadata["duration"].median(),
        metadata["duration"].max(),
    ]
})
overview.to_csv(REPORT_DIR / "hf_original_dataset_overview.csv", index=False, encoding="utf-8-sig")

emotion_counts = metadata["emotion_4class"].value_counts().reindex(EMOTIONS).fillna(0).astype(int).reset_index()
emotion_counts.columns = ["emotion_4class", "n_samples"]
emotion_counts["rate"] = emotion_counts["n_samples"] / len(metadata)
emotion_counts.to_csv(REPORT_DIR / "hf_original_4class_counts.csv", index=False, encoding="utf-8-sig")

display(overview)
display(emotion_counts)

plt.figure(figsize=(8, 4.5))
ax = sns.barplot(data=emotion_counts, x="emotion_4class", y="n_samples", palette="Set2")
for container in ax.containers:
    ax.bar_label(container, fmt="%d")
ax.set_title("HF-original 4-class emotion distribution (N=6,877)")
ax.set_xlabel("Emotion")
ax.set_ylabel("Samples")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "hf_original_emotion_distribution.png", dpi=180)
plt.show()
"""
        ),
        md(
            """
## 2. Load Existing Split Files

Notebook 01 đã sinh sẵn:

- `iemocap_5fold_session_long.csv`: mỗi fold test 1 session, validation là session kế tiếp.
- `iemocap_10fold_speaker_long.csv`: mỗi fold test 1 speaker, validation 1 speaker, train 8 speakers.

Notebook 03 không tạo lại split mới; nó kiểm tra và đóng gói split hiện tại thành manifest.
"""
        ),
        code(
            """
split_5 = pd.read_csv(SPLIT_5FOLD_PATH)
split_10 = pd.read_csv(SPLIT_10FOLD_PATH)

print("5-fold long split:", split_5.shape)
print("10-fold long split:", split_10.shape)
display(split_5.head())
display(split_10.head())
"""
        ),
        code(
            """
extra_cols = [
    "train_sample_id", "transcription", "speaking_rate", "pitch_mean", "pitch_std",
    "rms", "relative_db", "original_emotion", "source_dataset",
    "prob_frustrated", "prob_angry", "prob_sad", "prob_disgust", "prob_excited",
    "prob_fear", "prob_neutral", "prob_surprise", "prob_happy",
]
extra_cols = [c for c in extra_cols if c in full_metadata.columns]
extra = full_metadata[extra_cols].drop_duplicates("train_sample_id")

def enrich_split(df):
    keep_extra = [c for c in extra.columns if c == "train_sample_id" or c not in df.columns]
    out = df.merge(extra[keep_extra], on="train_sample_id", how="left")
    out["wav_exists"] = out["wav_path"].apply(lambda p: Path(str(p)).exists())
    return out

split_5_enriched = enrich_split(split_5)
split_10_enriched = enrich_split(split_10)

print("5-fold enriched:", split_5_enriched.shape)
print("10-fold enriched:", split_10_enriched.shape)
display(split_5_enriched.head())
"""
        ),
        md(
            """
## 3. Integrity Checks

Các kiểm tra chính:

- Mỗi fold phải có đủ `train`, `validation`, `test`.
- Trong cùng một fold, sample ID không được overlap giữa các split.
- 5-fold session split: không nên có overlap session/speaker giữa train/validation/test.
- 10-fold speaker split: không được overlap speaker; session overlap có thể xảy ra vì hai speaker cùng session có thể nằm ở split khác nhau.
"""
        ),
        code(
            """
def split_set(df, fold, split, col):
    return set(df[(df["fold"].eq(fold)) & (df["split"].eq(split))][col].dropna().astype(str))

def list_overlap(a, b):
    return sorted(a & b)

def check_protocol(df, protocol_name):
    rows = []
    for fold in sorted(df["fold"].unique()):
        fold_df = df[df["fold"].eq(fold)]
        ids_train = split_set(df, fold, "train", "train_sample_id")
        ids_val = split_set(df, fold, "validation", "train_sample_id")
        ids_test = split_set(df, fold, "test", "train_sample_id")
        sp_train = split_set(df, fold, "train", "speaker_id")
        sp_val = split_set(df, fold, "validation", "speaker_id")
        sp_test = split_set(df, fold, "test", "speaker_id")
        se_train = split_set(df, fold, "train", "session")
        se_val = split_set(df, fold, "validation", "session")
        se_test = split_set(df, fold, "test", "session")

        all_ids = ids_train | ids_val | ids_test
        rows.append({
            "protocol": protocol_name,
            "fold": fold,
            "n_train": len(ids_train),
            "n_validation": len(ids_val),
            "n_test": len(ids_test),
            "n_unique_ids_in_fold": len(all_ids),
            "expected_unique_ids": len(metadata),
            "covers_all_metadata": len(all_ids) == len(metadata),
            "duplicate_rows_in_fold": int(fold_df.duplicated(["train_sample_id", "split"]).sum()),
            "wav_exists_rate": fold_df["wav_exists"].mean(),
            "sample_train_val_overlap": list_overlap(ids_train, ids_val),
            "sample_train_test_overlap": list_overlap(ids_train, ids_test),
            "sample_val_test_overlap": list_overlap(ids_val, ids_test),
            "speaker_train_val_overlap": list_overlap(sp_train, sp_val),
            "speaker_train_test_overlap": list_overlap(sp_train, sp_test),
            "speaker_val_test_overlap": list_overlap(sp_val, sp_test),
            "session_train_val_overlap": list_overlap(se_train, se_val),
            "session_train_test_overlap": list_overlap(se_train, se_test),
            "session_val_test_overlap": list_overlap(se_val, se_test),
        })
    return pd.DataFrame(rows)

checks = pd.concat([
    check_protocol(split_5_enriched, "5fold_session"),
    check_protocol(split_10_enriched, "10fold_speaker"),
], ignore_index=True)

checks_for_csv = checks.copy()
for col in checks_for_csv.columns:
    if checks_for_csv[col].apply(lambda x: isinstance(x, list)).any():
        checks_for_csv[col] = checks_for_csv[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else x)

checks_for_csv.to_csv(REPORT_DIR / "fold_integrity_checks.csv", index=False, encoding="utf-8-sig")
display(checks_for_csv)

hard_error_cols = ["sample_train_val_overlap", "sample_train_test_overlap", "sample_val_test_overlap"]
has_sample_leak = any(len(v) > 0 for col in hard_error_cols for v in checks[col])
has_missing_wav = checks["wav_exists_rate"].lt(1.0).any()
print("Sample leakage detected:", has_sample_leak)
print("Any missing WAV:", has_missing_wav)
"""
        ),
        md(
            """
## 4. Fold Distributions

Phần này visualize số mẫu và class distribution theo từng fold. Với setting hiện tại, `happy` sẽ cao hơn các lớp khác vì `happy + excited` trong HuggingFace mirror đang lớn.
"""
        ),
        code(
            """
def summarize_by_fold_split(df, protocol_name):
    rows = []
    for (fold, split), part in df.groupby(["fold", "split"]):
        row = {
            "protocol": protocol_name,
            "fold": fold,
            "split": split,
            "n_samples": len(part),
            "n_speakers": part["speaker_id"].nunique(),
            "n_sessions": part["session"].nunique(),
        }
        counts = part["emotion_4class"].value_counts().reindex(EMOTIONS).fillna(0).astype(int)
        for emotion in EMOTIONS:
            row[f"n_{emotion}"] = int(counts[emotion])
            row[f"rate_{emotion}"] = float(counts[emotion] / len(part)) if len(part) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

fold_summary = pd.concat([
    summarize_by_fold_split(split_5_enriched, "5fold_session"),
    summarize_by_fold_split(split_10_enriched, "10fold_speaker"),
], ignore_index=True)
fold_summary.to_csv(REPORT_DIR / "fold_split_distribution_summary.csv", index=False, encoding="utf-8-sig")
display(fold_summary.head(12))
"""
        ),
        code(
            """
for protocol, df in fold_summary.groupby("protocol"):
    plt.figure(figsize=(12, 4.5))
    sns.barplot(data=df, x="fold", y="n_samples", hue="split")
    plt.xticks(rotation=60, ha="right")
    plt.title(f"{protocol}: sample counts by fold/split")
    plt.xlabel("Fold")
    plt.ylabel("Samples")
    plt.tight_layout()
    out_path = FIGURE_DIR / f"{protocol}_sample_counts_by_split.png"
    plt.savefig(out_path, dpi=180)
    plt.show()
"""
        ),
        code(
            """
def plot_class_heatmap(df, protocol):
    for split in ["train", "validation", "test"]:
        part = df[(df["protocol"].eq(protocol)) & (df["split"].eq(split))]
        if part.empty:
            continue
        matrix = part.set_index("fold")[[f"rate_{e}" for e in EMOTIONS]]
        matrix.columns = EMOTIONS
        plt.figure(figsize=(8, max(3, 0.45 * len(matrix))))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=0.60)
        plt.title(f"{protocol}: emotion rate heatmap ({split})")
        plt.xlabel("Emotion")
        plt.ylabel("Fold")
        plt.tight_layout()
        out_path = FIGURE_DIR / f"{protocol}_{split}_emotion_rate_heatmap.png"
        plt.savefig(out_path, dpi=180)
        plt.show()

plot_class_heatmap(fold_summary, "5fold_session")
plot_class_heatmap(fold_summary, "10fold_speaker")
"""
        ),
        md(
            """
## 5. Majority-Class Baseline

Đây không phải model thật. Baseline này chỉ lấy class xuất hiện nhiều nhất ở train rồi predict toàn bộ test là class đó.

Nó giúp kiểm tra nhanh:

- Nếu WA/accuracy nhìn không quá thấp nhưng UA thấp, dataset đang lệch lớp.
- Với 4 class, majority baseline thường có UA khoảng 0.25 nếu test có đủ cả 4 class.
"""
        ),
        code(
            """
def majority_baseline(df, protocol_name):
    rows = []
    for fold in sorted(df["fold"].unique()):
        train = df[(df["fold"].eq(fold)) & (df["split"].eq("train"))]
        test = df[(df["fold"].eq(fold)) & (df["split"].eq("test"))]
        majority = train["emotion_4class"].value_counts().idxmax()
        y_true = test["emotion_4class"].astype(str).tolist()
        y_pred = [majority] * len(y_true)
        wa = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else np.nan
        recalls = []
        for emotion in EMOTIONS:
            denom = sum(1 for y in y_true if y == emotion)
            correct = sum(1 for y, pred in zip(y_true, y_pred) if y == emotion and pred == emotion)
            recalls.append(correct / denom if denom else np.nan)
        ua = float(np.nanmean(recalls))
        rows.append({
            "protocol": protocol_name,
            "fold": fold,
            "majority_train_label": majority,
            "test_n": len(test),
            "WA_majority": wa,
            "UA_majority": ua,
            **{f"recall_{emotion}": recalls[i] for i, emotion in enumerate(EMOTIONS)},
        })
    return pd.DataFrame(rows)

majority_results = pd.concat([
    majority_baseline(split_5_enriched, "5fold_session"),
    majority_baseline(split_10_enriched, "10fold_speaker"),
], ignore_index=True)

majority_results.to_csv(REPORT_DIR / "majority_class_baseline_by_fold.csv", index=False, encoding="utf-8-sig")
display(majority_results)

majority_summary = majority_results.groupby("protocol")[["WA_majority", "UA_majority"]].agg(["mean", "std"]).round(4)
display(majority_summary)
"""
        ),
        md(
            """
## 6. Export Fold Manifests

Mỗi manifest CSV có một dòng cho một utterance, gồm:

- audio path;
- emotion label/id;
- valence/arousal/dominance;
- speaker/session;
- transcript nếu có;
- một số metadata nhẹ như pitch/rms/speaking rate.

Folder output:

```text
03.../manifests/
  5fold_session/
    fold_xxx/
      train.csv
      validation.csv
      test.csv
  10fold_speaker/
    fold_xxx/
      train.csv
      validation.csv
      test.csv
```
"""
        ),
        code(
            """
def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")

manifest_columns_preferred = [
    "train_sample_id", "utterance_id", "conversation_id", "session", "speaker_id",
    "emotion_4class", "emotion_id", "original_emotion",
    "valence", "arousal", "dominance", "valence_norm", "arousal_norm", "dominance_norm",
    "duration", "sample_rate", "channels", "wav_path", "wav_exists",
    "transcription", "speaking_rate", "pitch_mean", "pitch_std", "rms", "relative_db",
    "fold", "split",
]

def export_manifests(df, protocol_name):
    inventory = []
    protocol_dir = MANIFEST_DIR / protocol_name
    protocol_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in manifest_columns_preferred if c in df.columns]
    for fold in sorted(df["fold"].unique()):
        fold_dir = protocol_dir / safe_name(fold)
        fold_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "validation", "test"]:
            part = df[(df["fold"].eq(fold)) & (df["split"].eq(split))].copy()
            out_path = fold_dir / f"{split}.csv"
            part[cols].to_csv(out_path, index=False, encoding="utf-8-sig")
            inventory.append({
                "protocol": protocol_name,
                "fold": fold,
                "split": split,
                "path": str(out_path),
                "n_samples": len(part),
                "n_speakers": part["speaker_id"].nunique(),
                "n_sessions": part["session"].nunique(),
                "n_neutral": int(part["emotion_4class"].eq("neutral").sum()),
                "n_angry": int(part["emotion_4class"].eq("angry").sum()),
                "n_sad": int(part["emotion_4class"].eq("sad").sum()),
                "n_happy": int(part["emotion_4class"].eq("happy").sum()),
            })
    return pd.DataFrame(inventory)

manifest_inventory = pd.concat([
    export_manifests(split_5_enriched, "5fold_session"),
    export_manifests(split_10_enriched, "10fold_speaker"),
], ignore_index=True)

manifest_inventory.to_csv(REPORT_DIR / "manifest_inventory.csv", index=False, encoding="utf-8-sig")
display(manifest_inventory.head(20))
print("Saved manifest inventory:", REPORT_DIR / "manifest_inventory.csv")
"""
        ),
        md(
            """
## 7. Final Summary Report

Cell cuối ghi lại kết quả kiểm tra chính vào Markdown để dễ đọc ngoài notebook.
"""
        ),
        code(
            """
sample_leak = any(len(v) > 0 for col in ["sample_train_val_overlap", "sample_train_test_overlap", "sample_val_test_overlap"] for v in checks[col])
speaker_leak_5 = any(len(v) > 0 for col in ["speaker_train_val_overlap", "speaker_train_test_overlap", "speaker_val_test_overlap"] for v in checks[checks["protocol"].eq("5fold_session")][col])
speaker_leak_10 = any(len(v) > 0 for col in ["speaker_train_val_overlap", "speaker_train_test_overlap", "speaker_val_test_overlap"] for v in checks[checks["protocol"].eq("10fold_speaker")][col])
session_overlap_10 = any(len(v) > 0 for col in ["session_train_val_overlap", "session_train_test_overlap", "session_val_test_overlap"] for v in checks[checks["protocol"].eq("10fold_speaker")][col])

counts_line = ", ".join(f"{row.emotion_4class}: {row.n_samples}" for row in emotion_counts.itertuples())
maj_avg = majority_results.groupby("protocol")[["WA_majority", "UA_majority"]].mean().round(4)

report_lines = [
    "# 03 Split Testing Summary",
    "",
    "## Dataset setting",
    "",
    "- Setting: **HF-major-emotion 4-class + AVD**.",
    f"- Samples: **{len(metadata):,}**.",
    f"- Emotion counts: **{counts_line}**.",
    "- This is not the official IEMOCAP 5,531 benchmark subset.",
    "- No undersampling/balancing was applied in this notebook.",
    "",
    "## Split integrity",
    "",
    f"- Sample leakage detected: **{sample_leak}**.",
    f"- 5-fold speaker leakage detected: **{speaker_leak_5}**.",
    f"- 10-fold speaker leakage detected: **{speaker_leak_10}**.",
    f"- 10-fold session overlap exists: **{session_overlap_10}**. This is expected for speaker-independent split, not session-independent split.",
    f"- Manifest files exported: **{len(manifest_inventory)}** CSV files.",
    "",
    "## Majority-class baseline",
    "",
    maj_avg.to_markdown(),
    "",
    "Interpretation: majority baseline mostly predicts `happy`, so WA reflects class imbalance while UA stays low. During real training, report both WA and UA/Macro-F1.",
    "",
    "## Outputs",
    "",
    f"- Reports: `{REPORT_DIR}`",
    f"- Figures: `{FIGURE_DIR}`",
    f"- Manifests: `{MANIFEST_DIR}`",
    "",
    "## Next notebook suggestion",
    "",
    "Notebook 04 can read these manifests and start with a simple multi-task baseline: emotion classification head + valence/arousal/dominance regression head.",
]

report_path = REPORT_DIR / "03_split_testing_summary.md"
report_path.write_text("\\n".join(report_lines), encoding="utf-8")
display(Markdown("\\n".join(report_lines)))
print("Saved:", report_path)
"""
        ),
    ]

    nbf.write(nb, NB_PATH)
    print(f"Created {NB_PATH}")


if __name__ == "__main__":
    main()
