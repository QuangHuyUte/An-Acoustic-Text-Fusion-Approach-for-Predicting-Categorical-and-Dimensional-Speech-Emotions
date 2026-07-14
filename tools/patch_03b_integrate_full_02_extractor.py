import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"


FULL02_MARKDOWN = """## Tích hợp core feature extractor từ notebook 02

03B ưu tiên đọc cache đã xuất từ notebook 02. Nếu cache full 06D chưa có, notebook này có thể tự dựng lại phần feature cốt lõi từ raw audio:

```text
raw wav -> load/resample/mono/pad-truncate
        -> X_temporal: MFCC, delta, delta-delta, RMS, ZCR, centroid, bandwidth, rolloff, flatness, contrast, F0, voiced flag
        -> X_spectral: log-Mel, delta log-Mel, delta-delta log-Mel
        -> X_stats: mean/std/min/max/median/p10/p90/IQR trên temporal + chroma + tonnetz
        -> X_e2v: lấy từ cache Emotion2Vec hoặc tự trích bằng FunASR nếu bật EXTRACT_EMOTION2VEC=1
```

Phần này chỉ tái tạo **core feature cache** để train 03B. Các bảng phân tích, biểu đồ minh họa đặc trưng, report chất lượng feature và file zip nghiên cứu vẫn thuộc notebook 02, vì notebook 02 là bước chuyên về phân tích/trích đặc trưng.
"""


FULL02_EXTRACTOR = r'''def safe_delta_feature(x, order=1):
    if x.shape[-1] < 3:
        return np.zeros_like(x, dtype=np.float32)
    width = min(9, x.shape[-1])
    if width % 2 == 0:
        width -= 1
    width = max(3, width)
    return librosa.feature.delta(x, order=order, width=width).astype(np.float32)


def stat_functionals(feature_dict):
    vectors = []
    names = []
    for feature_name, values in feature_dict.items():
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        stats = {
            "mean": np.mean(arr, axis=1),
            "std": np.std(arr, axis=1),
            "min": np.min(arr, axis=1),
            "max": np.max(arr, axis=1),
            "median": np.median(arr, axis=1),
            "p10": np.percentile(arr, 10, axis=1),
            "p90": np.percentile(arr, 90, axis=1),
            "iqr": np.percentile(arr, 75, axis=1) - np.percentile(arr, 25, axis=1),
        }
        for stat_name, stat_values in stats.items():
            vectors.append(stat_values.astype(np.float32))
            names.extend([f"{feature_name}_{idx}_{stat_name}" for idx in range(arr.shape[0])])
    return np.concatenate(vectors, axis=0).astype(np.float32), names


def extract_pitch_yin(y, target_frames):
    try:
        f0 = librosa.yin(
            y,
            fmin=50,
            fmax=500,
            sr=SAMPLE_RATE,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        voiced = (f0 > 0).astype(np.float32)
        log_f0 = np.zeros_like(f0, dtype=np.float32)
        positive = f0 > 0
        log_f0[positive] = np.log(f0[positive])
        return (
            fix_feature_time(log_f0.reshape(1, -1), target_frames),
            fix_feature_time(voiced.reshape(1, -1), target_frames),
        )
    except Exception:
        return (
            np.zeros((1, target_frames), dtype=np.float32),
            np.zeros((1, target_frames), dtype=np.float32),
        )


def extract_06d_features_from_waveform(y):
    target_frames = 1 + MAX_AUDIO_SAMPLES // HOP_LENGTH

    mfcc = fix_feature_time(
        librosa.feature.mfcc(
            y=y,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
        ),
        target_frames,
    )
    delta = safe_delta_feature(mfcc, order=1)
    delta2 = safe_delta_feature(mfcc, order=2)

    rms = fix_feature_time(librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH), target_frames)
    zcr = fix_feature_time(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH), target_frames)
    centroid = fix_feature_time(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH), target_frames)
    bandwidth = fix_feature_time(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH), target_frames)
    rolloff = fix_feature_time(librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH), target_frames)
    flatness = fix_feature_time(librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH), target_frames)

    try:
        contrast = fix_feature_time(librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH), target_frames)
    except Exception:
        contrast = np.zeros((7, target_frames), dtype=np.float32)

    f0, voiced_flag = extract_pitch_yin(y, target_frames)

    temporal_parts = {
        "mfcc": mfcc,
        "delta_mfcc": delta,
        "delta2_mfcc": delta2,
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
        "spectral_contrast": contrast,
        "f0": f0,
        "voiced_flag": voiced_flag,
    }
    temporal = np.concatenate(list(temporal_parts.values()), axis=0).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = fix_feature_time(librosa.power_to_db(mel, ref=np.max), target_frames)
    delta_logmel = safe_delta_feature(logmel, order=1)
    delta2_logmel = safe_delta_feature(logmel, order=2)
    spectral = np.stack([logmel, delta_logmel, delta2_logmel], axis=0).astype(np.float32)

    try:
        chroma = fix_feature_time(librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH), target_frames)
    except Exception:
        chroma = np.zeros((12, target_frames), dtype=np.float32)

    try:
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = fix_feature_time(librosa.feature.tonnetz(y=y_harmonic, sr=SAMPLE_RATE), target_frames)
    except Exception:
        tonnetz = np.zeros((6, target_frames), dtype=np.float32)

    stats_parts = dict(temporal_parts)
    stats_parts["chroma"] = chroma
    stats_parts["tonnetz"] = tonnetz
    stats, _ = stat_functionals(stats_parts)

    return temporal, spectral, stats


'''


def replace_extractor_source(source: str) -> str:
    anchor = source.index("def load_audio_for_06d(path):")
    candidates = [
        source.find("def safe_delta_feature(x, order=1):", anchor),
        source.find("def stat_functionals(feature_dict):", anchor),
        source.find("def extract_pitch_yin(y, target_frames):", anchor),
        source.find("def extract_06d_features_from_waveform(y):", anchor),
    ]
    candidates = [idx for idx in candidates if idx != -1]
    start = min(candidates)
    end = source.index("def load_light_emotion2vec_cache():", start)
    return source[:start] + FULL02_EXTRACTOR + source[end:]


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    has_markdown = any(
        cell.get("cell_type") == "markdown" and "Tích hợp core feature extractor từ notebook 02" in "".join(cell.get("source", ""))
        for cell in nb["cells"]
    )
    if not has_markdown:
        insert_at = 0
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") == "markdown" and "Data input cần có" in "".join(cell.get("source", "")):
                insert_at = idx + 1
                break
        nb["cells"].insert(
            insert_at,
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": FULL02_MARKDOWN,
            },
        )

    replaced = False
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", ""))
            source = source.replace(
                'MAX_SECONDS = float(os.getenv("FULL06D_MAX_SECONDS", "6.0"))',
                'MAX_SECONDS = float(os.getenv("FULL06D_MAX_SECONDS", "12.0"))',
            )
            if "def extract_06d_features_from_waveform(y):" in source and "def load_light_emotion2vec_cache():" in source:
                source = replace_extractor_source(source)
                replaced = True
            cell["source"] = source

    if not replaced:
        raise RuntimeError("Không tìm thấy cell chứa extract_06d_features_from_waveform và load_light_emotion2vec_cache.")

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    main()
