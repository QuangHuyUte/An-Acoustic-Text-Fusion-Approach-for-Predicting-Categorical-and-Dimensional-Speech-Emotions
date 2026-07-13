from pathlib import Path
import json
import re
import io

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "06_w2v_based_models" / "datasets" / "AbstractTTS_IEMOCAP"
AUDIO_DIR = OUT_ROOT / "audio_wav"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "AbstractTTS/IEMOCAP"


def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^\w.\-]+", "_", name)
    return name or "unknown.wav"


def gender_to_std(g):
    g = str(g).lower()
    if g.startswith("f"):
        return "female"
    if g.startswith("m"):
        return "male"
    return "unknown"


def parse_session(file_name: str) -> str:
    match = re.match(r"(Ses\d{2})", str(file_name))
    return match.group(1) if match else "unknown_session"


def parse_speaker(file_name: str, gender: str) -> str:
    session = parse_session(file_name)
    g = gender_to_std(gender)
    if g == "female":
        return f"{session}F"
    if g == "male":
        return f"{session}M"
    # Fallback from utterance id suffix.
    stem = Path(str(file_name)).stem
    suffix = stem.split("_")[-1]
    if suffix.startswith("F"):
        return f"{session}F"
    if suffix.startswith("M"):
        return f"{session}M"
    return f"{session}_unknown"


def parse_conversation(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def choose_major(row):
    if row.get("major_emotion") and str(row["major_emotion"]).lower() != "nan":
        return str(row["major_emotion"]).lower()
    probs = {
        k: row.get(k, np.nan)
        for k in ["frustrated", "angry", "sad", "disgust", "excited", "fear", "neutral", "surprise", "happy"]
    }
    valid = {k: float(v) for k, v in probs.items() if pd.notna(v)}
    return max(valid, key=valid.get) if valid else "unknown"


def normalize_4class(emotion: str):
    e = str(emotion).lower()
    if e == "neutral":
        return "neutral"
    if e == "angry":
        return "angry"
    if e == "sad":
        return "sad"
    if e in {"happy", "excited"}:
        return "happy"
    return None


def extract_audio_array(audio_obj):
    if isinstance(audio_obj, dict):
        if audio_obj.get("bytes") is not None:
            data, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32")
            return np.asarray(data, dtype=np.float32), int(sr)
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        path = audio_obj.get("path")
        if arr is None and path:
            data, sr2 = sf.read(path, dtype="float32")
            arr = data
            sr = sr or sr2
        return np.asarray(arr, dtype=np.float32), int(sr)
    raise ValueError(f"Unsupported audio object type: {type(audio_obj)}")


def main():
    print(f"Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    print(ds)
    rows = []
    for idx, item in enumerate(ds):
        file_name = sanitize_filename(item.get("file", f"iemocap_hf_{idx:05d}.wav"))
        if not file_name.lower().endswith(".wav"):
            file_name = f"{file_name}.wav"
        wav_path = AUDIO_DIR / file_name
        audio_arr, sr = extract_audio_array(item["audio"])
        if audio_arr.ndim == 2:
            audio_arr = audio_arr.mean(axis=1)
        if not wav_path.exists():
            sf.write(str(wav_path), audio_arr, sr)
        major = choose_major(item)
        emotion_4 = normalize_4class(major)
        sample_id = f"iemocap_hf_{idx:05d}"
        row = {
            "sample_id": sample_id,
            "train_sample_id": sample_id,
            "utterance_id": Path(file_name).stem,
            "file": file_name,
            "conversation_id": parse_conversation(file_name),
            "session": parse_session(file_name),
            "speaker_id": parse_speaker(file_name, item.get("gender", "")),
            "gender": gender_to_std(item.get("gender", "")),
            "original_emotion": major,
            "emotion": major,
            "emotion_4class": emotion_4,
            "is_4class": emotion_4 is not None,
            "emotion_id": {"neutral": 0, "angry": 1, "sad": 2, "happy": 3}.get(emotion_4, -1),
            "transcription": item.get("transcription", ""),
            "valence": float(item.get("EmoVal")) if item.get("EmoVal") is not None else np.nan,
            "arousal": float(item.get("EmoAct")) if item.get("EmoAct") is not None else np.nan,
            "dominance": float(item.get("EmoDom")) if item.get("EmoDom") is not None else np.nan,
            "EmoAct": float(item.get("EmoAct")) if item.get("EmoAct") is not None else np.nan,
            "EmoVal": float(item.get("EmoVal")) if item.get("EmoVal") is not None else np.nan,
            "EmoDom": float(item.get("EmoDom")) if item.get("EmoDom") is not None else np.nan,
            "valence_norm": (float(item.get("EmoVal")) - 1.0) / 4.0 if item.get("EmoVal") is not None else np.nan,
            "arousal_norm": (float(item.get("EmoAct")) - 1.0) / 4.0 if item.get("EmoAct") is not None else np.nan,
            "dominance_norm": (float(item.get("EmoDom")) - 1.0) / 4.0 if item.get("EmoDom") is not None else np.nan,
            "duration": float(item.get("audio duration (s)", len(audio_arr) / sr)),
            "start_time": 0.0,
            "end_time": float(item.get("audio duration (s)", len(audio_arr) / sr)),
            "eval_duration": float(item.get("audio duration (s)", len(audio_arr) / sr)),
            "sample_rate": sr,
            "channels": 1,
            "wav_path": str(wav_path),
            "wav_found": True,
            "source_dataset": DATASET_ID,
            "speaking_rate": item.get("speaking_rate", np.nan),
            "pitch_mean": item.get("pitch_mean", np.nan),
            "pitch_std": item.get("pitch_std", np.nan),
            "rms": item.get("rms", np.nan),
            "relative_db": item.get("relative_db", np.nan),
        }
        for prob_col in ["frustrated", "angry", "sad", "disgust", "excited", "fear", "neutral", "surprise", "happy"]:
            row[f"prob_{prob_col}"] = item.get(prob_col, np.nan)
        rows.append(row)
        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1} rows...")

    df = pd.DataFrame(rows)
    full_csv = OUT_ROOT / "abstracttts_iemocap_metadata_full.csv"
    train_csv = OUT_ROOT / "abstracttts_iemocap_4class_avd_metadata.csv"
    df.to_csv(full_csv, index=False, encoding="utf-8-sig")
    df[df["is_4class"]].copy().reset_index(drop=True).to_csv(train_csv, index=False, encoding="utf-8-sig")
    summary = {
        "dataset_id": DATASET_ID,
        "n_rows": int(len(df)),
        "n_4class": int(df["is_4class"].sum()),
        "audio_dir": str(AUDIO_DIR),
        "full_csv": str(full_csv),
        "train_csv": str(train_csv),
        "major_emotion_counts": df["emotion"].value_counts().to_dict(),
        "emotion_4class_counts": df.loc[df["is_4class"], "emotion_4class"].value_counts().to_dict(),
    }
    (OUT_ROOT / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
