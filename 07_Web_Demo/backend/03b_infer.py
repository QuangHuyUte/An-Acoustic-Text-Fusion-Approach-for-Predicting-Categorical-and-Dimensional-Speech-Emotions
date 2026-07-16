import argparse
import json
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODELS_DIR = WEB_ROOT / "models"
DEFAULT_ARTIFACT_DIR = DEFAULT_MODELS_DIR / "checkpoints" / "03b"
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT_DIR / "5fold_session" / "fold_1_test_Ses01_val_Ses02_acoustic_text_bridge_rmm_best.pt"
DEFAULT_SCALER = DEFAULT_MODELS_DIR / "scalers" / "03b_frozen_text_5fold_scalers.npz"
DEFAULT_CACHE = ROOT / "06_w2v_based_models" / "02_IEMOCAP Feature Extraction Emotion2Vec Acoustic" / "output" / "features" / "iemocap_full_06d_multibranch_cache.npz"
DEFAULT_SPLIT = ROOT / "06_w2v_based_models" / "02_IEMOCAP Feature Extraction Emotion2Vec Acoustic" / "output" / "splits" / "iemocap_5fold_session_long.csv"
DEFAULT_TEXT_MODEL = DEFAULT_MODELS_DIR / "base" / "roberta-base-kaggle"
DEFAULT_WAV2VEC2_MODEL = DEFAULT_MODELS_DIR / "base" / "wav2vec2-base-kaggle"
DEFAULT_EMOTION2VEC_MODEL = DEFAULT_MODELS_DIR / "base" / "funasr" / "models" / "iic--emotion2vec_base" / "snapshots" / "master"
DEFAULT_ASR_MODEL = DEFAULT_MODELS_DIR / "base" / "whisper-tiny-en"
REGISTRY_PATH = DEFAULT_MODELS_DIR / "live_model_registry.json"
DEFAULT_PROFILE_ID = os.getenv("MODEL_PROFILE_ID", "03b_frozen_text_5fold")

EMOTIONS = ["neutral", "angry", "sad", "happy"]
WEB_EMOTION_ORDER = ["neutral", "happy", "sad", "angry"]

SAMPLE_RATE = 16000
MAX_SECONDS = 12.0
SEGMENT_SECONDS = float(os.getenv("LIVE_SEGMENT_SECONDS", MAX_SECONDS))
SEGMENT_OVERLAP_SECONDS = float(os.getenv("LIVE_SEGMENT_OVERLAP_SECONDS", "2.0"))
ASR_SEGMENT_SECONDS = float(os.getenv("LIVE_ASR_SEGMENT_SECONDS", "30.0"))
TARGET_FRAMES = 601
N_MFCC = 40
N_MELS = 96
N_FFT = 640
HOP_LENGTH = 320
WIN_LENGTH = 640

HIDDEN_DIM = 256
NUM_HEADS = 8
BRIDGE_TOKENS = 16
DROPOUT = 0.30
FUSION_MODE = "acoustic_text_bridge_rmm"
VALID_FUSION_MODES = {
    "acoustic_only",
    "text_only",
    "acoustic_text_concat",
    "acoustic_text_bridge",
    "acoustic_text_bridge_rmm",
}
VAD_REPRESENTATION = "acoustic_heavy"

DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.getenv("FORCE_CPU", "0") != "1" else "cpu")


def log(message):
    if os.getenv("MODEL_03B_VERBOSE", "1") == "1":
        print(f"[03B] {message}", file=sys.stderr, flush=True)


def fail(message, **extra):
    print(json.dumps({"ok": False, "error": message, **extra}, ensure_ascii=False))
    raise SystemExit(2)


def resolve_project_path(value):
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


@lru_cache(maxsize=1)
def load_live_model_registry():
    if not REGISTRY_PATH.exists():
        return {
            "defaultProfileId": DEFAULT_PROFILE_ID,
            "profiles": []
        }
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def list_model_profiles():
    registry = load_live_model_registry()
    return registry.get("profiles", [])


def get_model_profile(profile_id=None):
    registry = load_live_model_registry()
    selected_id = profile_id or os.getenv("MODEL_PROFILE_ID") or registry.get("defaultProfileId") or DEFAULT_PROFILE_ID
    profiles = {profile.get("id"): profile for profile in registry.get("profiles", [])}
    profile = profiles.get(selected_id)
    if profile is None:
        available = ", ".join(sorted(k for k in profiles if k))
        fail(f"Unknown model profile: {selected_id}", availableProfiles=available)
    return profile


def profile_paths(profile_id=None):
    profile = get_model_profile(profile_id)
    return {
        "profile": profile,
        "checkpoint": resolve_project_path(profile.get("checkpoint")) or DEFAULT_CHECKPOINT,
        "scaler": resolve_project_path(profile.get("scaler")) or DEFAULT_SCALER,
        "split": resolve_project_path(profile.get("split")) or DEFAULT_SPLIT,
        "text_model": resolve_project_path(profile.get("textModel")) or DEFAULT_TEXT_MODEL,
    }


def profile_uses_text(profile):
    runner = profile.get("runner")
    if runner in {"03c_text_tuned", "03d_live_weighted_03b_03c"}:
        return True
    if runner == "03b_bridge":
        return str(profile.get("fusionMode") or FUSION_MODE) != "acoustic_only"
    return True


def profile_uses_acoustic(profile):
    return profile.get("runner") != "03c_text_tuned"


def load_audio_16k(path, max_seconds=MAX_SECONDS):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if sr != SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = np.gcd(int(sr), int(SAMPLE_RATE))
        y = resample_poly(y, SAMPLE_RATE // gcd, sr // gcd).astype(np.float32)
    if max_seconds is not None and max_seconds > 0:
        max_len = int(SAMPLE_RATE * max_seconds)
        y = y[:max_len]
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:
        y = y / peak
    return y.astype(np.float32)


def make_live_segments(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return [{"index": 0, "start": 0.0, "end": 0.0, "audio": audio}]
    seg_len = max(1, int(SAMPLE_RATE * SEGMENT_SECONDS))
    overlap = max(0, int(SAMPLE_RATE * SEGMENT_OVERLAP_SECONDS))
    hop = max(1, seg_len - min(overlap, seg_len - 1))
    total = audio.size
    segments = []
    start = 0
    while start < total:
        end = min(start + seg_len, total)
        chunk = audio[start:end]
        if chunk.size < int(SAMPLE_RATE * 0.25) and segments:
            break
        segments.append({
            "index": len(segments),
            "start": start / SAMPLE_RATE,
            "end": end / SAMPLE_RATE,
            "audio": chunk,
        })
        if end >= total:
            break
        start += hop
    return segments


def write_segment_wav(segment_audio):
    handle = tempfile.NamedTemporaryFile(prefix="speech-demo-segment-", suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, segment_audio, SAMPLE_RATE)
    return Path(handle.name)


def make_asr_segments(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return []
    seg_len = max(1, int(SAMPLE_RATE * ASR_SEGMENT_SECONDS))
    segments = []
    for start in range(0, audio.size, seg_len):
        end = min(start + seg_len, audio.size)
        chunk = audio[start:end]
        if chunk.size < int(SAMPLE_RATE * 0.20):
            continue
        segments.append({
            "index": len(segments),
            "start": start / SAMPLE_RATE,
            "end": end / SAMPLE_RATE,
            "audio": chunk,
        })
    return segments


@lru_cache(maxsize=1)
def load_asr_model():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_path = Path(os.getenv("LIVE_ASR_MODEL", DEFAULT_ASR_MODEL))
    if not model_path.exists():
        fail(f"Missing ASR model: {model_path}")
    processor = WhisperProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(model_path), local_files_only=True).to(DEVICE)
    model.eval()
    return processor, model, model_path


def transcribe_audio_array(audio):
    if os.getenv("ENABLE_LOCAL_ASR", "1") != "1":
        return {"text": "", "segments": [], "source": "disabled"}
    processor, model, model_path = load_asr_model()
    rows = []
    texts = []
    for segment in make_asr_segments(audio):
        inputs = processor(segment["audio"], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_features = inputs.input_features.to(DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(input_features, max_new_tokens=int(os.getenv("LIVE_ASR_MAX_NEW_TOKENS", "96")))
        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        if text:
            texts.append(text)
        rows.append({
            "index": int(segment["index"]),
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": text,
        })
    return {
        "text": " ".join(texts).strip(),
        "segments": rows,
        "source": f"local Whisper ASR: {model_path.name}",
    }


def transcribe_audio_file(audio_path):
    audio = load_audio_16k(audio_path, max_seconds=None)
    return transcribe_audio_array(audio)


def fix_frames(x, target=TARGET_FRAMES):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] == target:
        return x
    if x.shape[-1] > target:
        return x[..., :target]
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target - x.shape[-1])
    return np.pad(x, pad_width, mode="constant").astype(np.float32)


def functionals(mat):
    mat = np.asarray(mat, dtype=np.float32)
    stats = []
    for row in mat:
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            finite = np.asarray([0.0], dtype=np.float32)
        stats.extend([
            np.mean(finite),
            np.std(finite),
            np.min(finite),
            np.max(finite),
            np.median(finite),
            np.percentile(finite, 10),
            np.percentile(finite, 90),
            np.percentile(finite, 75) - np.percentile(finite, 25),
        ])
    return np.asarray(stats, dtype=np.float32)


@lru_cache(maxsize=1)
def load_wav2vec2_fallback():
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    model_path = Path(os.getenv("WAV2VEC2_FALLBACK_MODEL", DEFAULT_WAV2VEC2_MODEL))
    if not model_path.exists():
        return None, None, model_path
    extractor = AutoFeatureExtractor.from_pretrained(str(model_path), local_files_only=True)
    model = Wav2Vec2Model.from_pretrained(str(model_path), local_files_only=True).to(DEVICE)
    model.eval()
    return extractor, model, model_path


def extract_wav2vec2_fallback(wav_path, warnings):
    if os.getenv("USE_WAV2VEC2_FALLBACK", "1") != "1":
        warnings.append("X_e2v uses zero vector because USE_WAV2VEC2_FALLBACK=0 and emotion2vec is disabled/unavailable.")
        return np.zeros(768, dtype=np.float32)
    try:
        extractor, model, model_path = load_wav2vec2_fallback()
        if extractor is None or model is None:
            warnings.append(f"X_e2v uses zero vector because local wav2vec2 fallback was not found: {model_path}")
            return np.zeros(768, dtype=np.float32)
        y = load_audio_16k(wav_path)
        inputs = extractor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(DEVICE)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)
        with torch.no_grad():
            out = model(input_values, attention_mask=attention_mask)
            emb = out.last_hidden_state.mean(dim=1).detach().cpu().numpy()[0]
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size != 768:
            fixed = np.zeros(768, dtype=np.float32)
            fixed[: min(768, emb.size)] = emb[:768]
            emb = fixed
        warnings.append(f"X_e2v uses local wav2vec2 fallback instead of FunASR emotion2vec: {model_path.name}")
        return emb
    except Exception as exc:
        warnings.append(f"X_e2v wav2vec2 fallback failed; zero vector used: {type(exc).__name__}: {exc}")
        return np.zeros(768, dtype=np.float32)


@lru_cache(maxsize=1)
def load_funasr_emotion2vec(model_name):
    from funasr import AutoModel

    try:
        return AutoModel(model=model_name, disable_update=True)
    except TypeError:
        return AutoModel(model=model_name)


def extract_emotion2vec_optional(wav_path, warnings):
    local_emotion2vec_ready = (DEFAULT_EMOTION2VEC_MODEL / "emotion2vec_base.pt").exists()
    extract_flag = os.getenv("EXTRACT_EMOTION2VEC")
    should_extract = extract_flag == "1" or (extract_flag is None and local_emotion2vec_ready)
    if not should_extract:
        return extract_wav2vec2_fallback(wav_path, warnings)
    try:
        model_name = os.getenv("EMOTION2VEC_MODEL_NAME")
        if not model_name:
            model_name = str(DEFAULT_EMOTION2VEC_MODEL) if local_emotion2vec_ready else "iic/emotion2vec_base"
        model = load_funasr_emotion2vec(model_name)
        result = model.generate(str(wav_path), output_dir=None, granularity="utterance", extract_embedding=True)
        emb = None
        if isinstance(result, list) and result:
            item = result[0]
            if isinstance(item, dict):
                if "feats" in item and item["feats"] is not None:
                    emb = item["feats"]
                elif "embedding" in item and item["embedding"] is not None:
                    emb = item["embedding"]
        if emb is None:
            raise RuntimeError("FunASR did not return an embedding.")
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        if emb.size != 768:
            fixed = np.zeros(768, dtype=np.float32)
            fixed[: min(768, emb.size)] = emb[:768]
            emb = fixed
        warnings.append(f"X_e2v uses FunASR emotion2vec: {model_name}")
        return emb
    except Exception as exc:
        warnings.append(f"FunASR emotion2vec failed, trying wav2vec2 fallback: {type(exc).__name__}: {exc}")
        return extract_wav2vec2_fallback(wav_path, warnings)


def extract_06d_features(wav_path):
    warnings = []
    log("Load and normalize audio")
    y = load_audio_16k(wav_path)
    import torchaudio

    waveform = torch.from_numpy(y).float().unsqueeze(0)
    window = torch.hann_window(WIN_LENGTH)

    log("Extract MFCC and deltas with torchaudio")
    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "n_mels": N_MELS,
            "center": True,
            "power": 2.0,
        },
    )
    mfcc_t = mfcc_transform(waveform)[0]
    delta_t = torchaudio.functional.compute_deltas(mfcc_t.unsqueeze(0))[0]
    delta2_t = torchaudio.functional.compute_deltas(delta_t.unsqueeze(0))[0]
    mfcc = mfcc_t.numpy()
    delta = delta_t.numpy()
    delta2 = delta2_t.numpy()

    log("Extract STFT descriptors")
    spec_complex = torch.stft(
        waveform[0],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        center=True,
        return_complex=True,
    )
    power = (spec_complex.abs().numpy() ** 2).astype(np.float32)
    freqs = np.linspace(0.0, SAMPLE_RATE / 2.0, power.shape[0], dtype=np.float32).reshape(-1, 1)
    eps = 1e-8
    power_sum = power.sum(axis=0, keepdims=True) + eps
    centroid = (freqs * power).sum(axis=0, keepdims=True) / power_sum
    bandwidth = np.sqrt((((freqs - centroid) ** 2) * power).sum(axis=0, keepdims=True) / power_sum)
    cumulative = np.cumsum(power, axis=0)
    rolloff_idx = (cumulative >= 0.85 * power_sum).argmax(axis=0)
    rolloff = freqs.reshape(-1)[rolloff_idx].reshape(1, -1)
    flatness = (np.exp(np.mean(np.log(power + eps), axis=0, keepdims=True)) / (np.mean(power, axis=0, keepdims=True) + eps)).astype(np.float32)
    rms = np.sqrt(power_sum / N_FFT).astype(np.float32)

    padded = np.pad(y, (N_FFT // 2, N_FFT // 2), mode="constant")
    if padded.size < N_FFT:
        padded = np.pad(padded, (0, N_FFT - padded.size), mode="constant")
    frame_count = 1 + max(0, (padded.size - N_FFT) // HOP_LENGTH)
    frames = np.stack([padded[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] for i in range(frame_count)], axis=0)
    zcr = (np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1, keepdims=True).T).astype(np.float32)

    log_power_db = 10.0 * np.log10(power + eps)
    bands = np.array_split(np.arange(power.shape[0]), 7)
    contrast_rows = []
    for band in bands:
        band_values = log_power_db[band]
        contrast_rows.append(np.percentile(band_values, 90, axis=0) - np.percentile(band_values, 10, axis=0))
    contrast = np.vstack(contrast_rows).astype(np.float32)

    log("Estimate F0 with lightweight autocorrelation")
    f0_values = []
    voiced_values = []
    min_lag = int(SAMPLE_RATE / 500)
    max_lag = int(SAMPLE_RATE / 50)
    energy_threshold = max(float(np.mean(frames ** 2)) * 0.05, 1e-8)
    for frame in frames:
        energy = float(np.mean(frame ** 2))
        if energy <= energy_threshold:
            f0_values.append(0.0)
            voiced_values.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")[N_FFT - 1 :]
        corr[:min_lag] = 0.0
        search = corr[min_lag:max_lag]
        if search.size == 0 or np.max(search) <= 0:
            f0_values.append(0.0)
            voiced_values.append(0.0)
            continue
        lag = int(np.argmax(search) + min_lag)
        f0_values.append(float(SAMPLE_RATE / max(lag, 1)))
        voiced_values.append(1.0)
    f0 = np.asarray(f0_values, dtype=np.float32).reshape(1, -1)
    voiced_flag = np.asarray(voiced_values, dtype=np.float32).reshape(1, -1)
    warnings.append("F0 uses lightweight autocorrelation for web inference.")

    temporal = np.vstack([
        fix_frames(mfcc),
        fix_frames(delta),
        fix_frames(delta2),
        fix_frames(rms),
        fix_frames(zcr),
        fix_frames(centroid),
        fix_frames(bandwidth),
        fix_frames(rolloff),
        fix_frames(flatness),
        fix_frames(contrast),
        fix_frames(f0),
        fix_frames(voiced_flag),
    ])
    temporal = fix_frames(temporal)

    log("Extract log-Mel spectral stack with torchaudio")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        center=True,
    )
    mel = mel_transform(waveform)[0]
    logmel_t = torchaudio.transforms.AmplitudeToDB(stype="power")(mel)
    d_logmel_t = torchaudio.functional.compute_deltas(logmel_t.unsqueeze(0))[0]
    d2_logmel_t = torchaudio.functional.compute_deltas(d_logmel_t.unsqueeze(0))[0]
    logmel = logmel_t.numpy()
    d_logmel = d_logmel_t.numpy()
    d2_logmel = d2_logmel_t.numpy()
    spectral = np.stack([fix_frames(logmel), fix_frames(d_logmel), fix_frames(d2_logmel)], axis=0).astype(np.float32)

    log("Extract lightweight chroma statistics")
    chroma = np.zeros((12, power.shape[1]), dtype=np.float32)
    valid_freqs = freqs.reshape(-1)
    nonzero = valid_freqs > 0
    midi = 69.0 + 12.0 * np.log2(np.maximum(valid_freqs[nonzero], 1e-6) / 440.0)
    pitch_classes = np.mod(np.round(midi).astype(int), 12)
    for bin_idx, pc in zip(np.where(nonzero)[0], pitch_classes):
        chroma[pc] += power[bin_idx]
    chroma = chroma / (chroma.sum(axis=0, keepdims=True) + eps)
    tonnetz = np.zeros((6, chroma.shape[1]), dtype=np.float32)
    warnings.append("Tonnetz uses zero fallback in web inference for speed.")
    stats_source = np.vstack([temporal, fix_frames(chroma), fix_frames(tonnetz)])
    stats = functionals(stats_source)
    if stats.size != 1224:
        fixed = np.zeros(1224, dtype=np.float32)
        fixed[: min(1224, stats.size)] = stats[:1224]
        stats = fixed

    log("Extract or fallback emotion2vec vector")
    e2v = extract_emotion2vec_optional(wav_path, warnings)
    log("Feature extraction done")
    return {
        "temporal": temporal.astype(np.float32),
        "spectral": spectral.astype(np.float32),
        "stats": stats.astype(np.float32),
        "e2v": e2v.astype(np.float32),
        "warnings": warnings,
    }


class TemporalBranch(nn.Module):
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.net(x).transpose(1, 2)


class SpectralBranch(nn.Module):
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout * 0.3),
            nn.Conv2d(32, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout * 0.3),
        )

    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=2)
        return h.transpose(1, 2)


class VectorToken(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)


class FrozenTextBranch(nn.Module):
    def __init__(self, model_name, hidden, dropout, load_backbone=True):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(str(model_name), local_files_only=True) if load_backbone else None
        if self.backbone is not None:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            text_hidden = int(self.backbone.config.hidden_size)
        else:
            text_hidden = 768
        self.proj = nn.Sequential(nn.Linear(text_hidden, hidden), nn.GELU(), nn.Dropout(dropout))

    def forward(self, input_ids, attention_mask):
        if self.backbone is None:
            batch = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            empty = torch.zeros(batch, seq_len, self.proj[0].in_features, device=input_ids.device, dtype=torch.float32)
            return self.proj(empty), attention_mask.bool()
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state), attention_mask.bool()


class AcousticTextBridgeFusionSER(nn.Module):
    def __init__(
        self,
        temporal_channels,
        spectral_channels,
        stats_dim,
        e2v_dim,
        hidden=256,
        heads=8,
        bridge_tokens=16,
        dropout=0.3,
        text_model=DEFAULT_TEXT_MODEL,
        fusion_mode=FUSION_MODE,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode if fusion_mode in VALID_FUSION_MODES else FUSION_MODE
        self.uses_text = self.fusion_mode != "acoustic_only"
        self.temporal = TemporalBranch(temporal_channels, hidden, dropout)
        self.spectral = SpectralBranch(spectral_channels, hidden, dropout)
        self.stats = VectorToken(stats_dim, hidden, dropout)
        self.e2v = VectorToken(e2v_dim, hidden, dropout)
        self.text = FrozenTextBranch(text_model, hidden, dropout, load_backbone=self.uses_text)

        audio_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        text_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.audio_self_attention = nn.TransformerEncoder(audio_layer, num_layers=1)
        self.text_self_attention = nn.TransformerEncoder(text_layer, num_layers=1)

        self.bridge_audio_query = nn.Parameter(torch.randn(bridge_tokens, hidden) * 0.02)
        self.bridge_text_query = nn.Parameter(torch.randn(bridge_tokens, hidden) * 0.02)
        self.audio_key_text_value = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.text_key_audio_value = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)

        self.concat_fusion = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.bridge_fusion = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.vad_acoustic_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.vad_fused_fusion = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.emotion_head = nn.Linear(hidden, 4)
        self.vad_head = nn.Sequential(nn.Linear(hidden, 3), nn.Sigmoid())

    @staticmethod
    def _pad_tokens_to_same_length(audio, text, text_mask):
        audio_mask = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        target = max(audio.shape[1], text.shape[1])

        def pad_token(x):
            return x if x.shape[1] == target else F.pad(x, (0, 0, 0, target - x.shape[1]), value=0.0)

        def pad_mask(mask):
            return mask if mask.shape[1] == target else F.pad(mask, (0, target - mask.shape[1]), value=False)

        return pad_token(audio), pad_mask(audio_mask), pad_token(text), pad_mask(text_mask)

    def acoustic_tokens(self, batch):
        return torch.cat([self.temporal(batch["temporal"]), self.spectral(batch["spectral"]), self.stats(batch["stats"]), self.e2v(batch["e2v"])], dim=1)

    def forward(self, batch, mask_modality=None, return_embedding=False):
        audio = self.acoustic_tokens(batch)
        if mask_modality == "acoustic":
            audio = torch.zeros_like(audio)
        audio = self.audio_self_attention(audio)
        audio_pool = audio.mean(dim=1)

        if self.fusion_mode == "acoustic_only":
            z = audio_pool
            attn_a, attn_t = None, None
        else:
            text, text_mask = self.text(batch["input_ids"], batch["attention_mask"])
            if mask_modality == "text":
                text = torch.zeros_like(text)
            text = self.text_self_attention(text, src_key_padding_mask=~text_mask)
            text_pool = (text * text_mask.unsqueeze(-1)).sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)

            if self.fusion_mode == "text_only":
                z = text_pool
                attn_a, attn_t = None, None
            elif self.fusion_mode == "acoustic_text_concat":
                z = self.concat_fusion(torch.cat([audio_pool, text_pool], dim=-1))
                attn_a, attn_t = None, None
            else:
                audio_pad, audio_mask, text_pad, text_mask_pad = self._pad_tokens_to_same_length(audio, text, text_mask)
                qa = self.bridge_audio_query.unsqueeze(0).expand(audio.size(0), -1, -1)
                qt = self.bridge_text_query.unsqueeze(0).expand(audio.size(0), -1, -1)
                e_audio, attn_a = self.audio_key_text_value(query=qa, key=audio_pad, value=text_pad, key_padding_mask=~audio_mask, need_weights=return_embedding)
                e_text, attn_t = self.text_key_audio_value(query=qt, key=text_pad, value=audio_pad, key_padding_mask=~text_mask_pad, need_weights=return_embedding)
                bridge = 0.5 * (e_audio.mean(dim=1) + e_text.mean(dim=1))
                gate = self.gate(torch.cat([audio_pool, text_pool], dim=-1))
                z = self.bridge_fusion(bridge + gate * audio_pool + (1.0 - gate) * text_pool)

        z_vad = self.vad_acoustic_fusion(torch.cat([audio_pool, z], dim=-1))
        out = {"emotion_logits": self.emotion_head(z), "vad_pred": self.vad_head(z_vad)}
        if return_embedding:
            out["embedding"] = z
            out["attn_audio_query"] = attn_a
            out["attn_text_query"] = attn_t
        return out


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
        return torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)


class TranscriptMultiTaskSER(nn.Module):
    def __init__(self, model_name, num_classes=4, dropout=0.25):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(str(model_name), local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
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
        output = {
            "logits": self.emotion_head(emb),
            "vad": self.vad_head(emb),
        }
        if return_embedding:
            output["embedding"] = emb
        return output


@lru_cache(maxsize=4)
def load_cache_and_scalers(profile_id=DEFAULT_PROFILE_ID):
    paths = profile_paths(profile_id)
    profile = paths["profile"]
    scaler_path = Path(os.getenv("MODEL_03B_SCALER", paths["scaler"]))
    if scaler_path.exists():
        log(f"Load cached fold scalers: {scaler_path.name}")
        scalers_np = np.load(scaler_path, allow_pickle=False)
        return {
            "temporal_mean": scalers_np["temporal_mean"].astype(np.float32),
            "temporal_std": scalers_np["temporal_std"].astype(np.float32),
            "spectral_mean": scalers_np["spectral_mean"].astype(np.float32),
            "spectral_std": scalers_np["spectral_std"].astype(np.float32),
            "stats_mean": scalers_np["stats_mean"].astype(np.float32),
            "stats_scale": scalers_np["stats_scale"].astype(np.float32),
            "e2v_mean": scalers_np["e2v_mean"].astype(np.float32),
            "e2v_scale": scalers_np["e2v_scale"].astype(np.float32),
        }

    log("Load feature cache and fit fold scalers")
    cache_path = Path(os.getenv("MODEL_03B_CACHE", DEFAULT_CACHE))
    split_path = Path(os.getenv("MODEL_03B_SPLIT", paths["split"]))
    if not cache_path.exists():
        fail(f"Missing 03B feature cache: {cache_path}")
    if not split_path.exists():
        fail(f"Missing 03B split file: {split_path}")
    cache = np.load(cache_path, allow_pickle=True)
    sample_ids = cache["train_sample_id"].astype(str)
    feature_index = {sid: i for i, sid in enumerate(sample_ids)}
    split_df = pd.read_csv(split_path)
    fold_name = os.getenv("MODEL_03B_FOLD", profile.get("fold") or "fold_1_test_Ses01_val_Ses02")
    train_ids = split_df[(split_df["fold"] == fold_name) & (split_df["split"] == "train")]["train_sample_id"].astype(str)
    train_idx = np.asarray([feature_index[sid] for sid in train_ids if sid in feature_index], dtype=np.int64)
    if train_idx.size == 0:
        fail(f"Cannot fit scalers: no training rows found for {fold_name}")
    x_temporal = cache["X_temporal"].astype(np.float32)
    x_spectral = cache["X_spectral"].astype(np.float32)
    x_stats = cache["X_stats"].astype(np.float32)
    x_e2v = cache["X_e2v"].astype(np.float32)
    stats_scaler = StandardScaler().fit(x_stats[train_idx])
    e2v_scaler = StandardScaler().fit(x_e2v[train_idx])
    scalers = {
        "temporal_mean": x_temporal[train_idx].mean(axis=(0, 2), keepdims=True).astype(np.float32),
        "temporal_std": (x_temporal[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32),
        "spectral_mean": x_spectral[train_idx].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32),
        "spectral_std": (x_spectral[train_idx].std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32),
        "stats_mean": stats_scaler.mean_.astype(np.float32),
        "stats_scale": stats_scaler.scale_.astype(np.float32),
        "e2v_mean": e2v_scaler.mean_.astype(np.float32),
        "e2v_scale": e2v_scaler.scale_.astype(np.float32),
    }
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scaler_path, **scalers)
    log(f"Saved cached fold scalers: {scaler_path}")
    log("Scalers ready")
    return scalers


@lru_cache(maxsize=3)
def load_model_and_tokenizer(profile_id=DEFAULT_PROFILE_ID):
    from transformers import AutoTokenizer

    paths = profile_paths(profile_id)
    profile = paths["profile"]
    checkpoint_path = Path(os.getenv("MODEL_03B_CHECKPOINT", paths["checkpoint"]))
    text_model_path = Path(os.getenv("MODEL_03B_TEXT_MODEL", paths["text_model"]))
    fusion_mode = str(profile.get("fusionMode") or FUSION_MODE)
    if fusion_mode not in VALID_FUSION_MODES:
        fail(f"Unsupported 03B fusion mode: {fusion_mode}", modelProfile=profile)
    if not checkpoint_path.exists():
        fail(f"Missing 03B checkpoint: {checkpoint_path}")
    uses_text = profile_uses_text(profile)
    if uses_text and not text_model_path.exists():
        fail(f"Missing local text model: {text_model_path}")

    log(f"Build 03B model, fusion_mode={fusion_mode}")
    model = AcousticTextBridgeFusionSER(
        temporal_channels=135,
        spectral_channels=3,
        stats_dim=1224,
        e2v_dim=768,
        hidden=HIDDEN_DIM,
        heads=NUM_HEADS,
        bridge_tokens=BRIDGE_TOKENS,
        dropout=DROPOUT,
        text_model=text_model_path,
        fusion_mode=fusion_mode,
    ).to(DEVICE)
    log(f"Load checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    omit_text_backbone = bool(isinstance(checkpoint, dict) and checkpoint.get("frozen_text_backbone_omitted"))
    load_result = model.load_state_dict(state_dict, strict=not omit_text_backbone)
    if omit_text_backbone:
        missing = [key for key in load_result.missing_keys if not key.startswith("text.backbone.")]
        if missing:
            fail("Checkpoint is missing non-backbone parameters.", missingKeys=missing[:20], checkpoint=str(checkpoint_path))
        log("Loaded lightweight 03B checkpoint; frozen RoBERTa backbone was loaded from local base model.")
    model.eval()
    tokenizer = None
    if uses_text:
        log(f"Load tokenizer: {text_model_path}")
        tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), local_files_only=True)
    log("Model and tokenizer ready")
    return model, tokenizer


def prewarm_profile(profile_id=None):
    profile = get_model_profile(profile_id)
    runner = profile.get("runner")
    selected_id = profile.get("id") or profile_id or DEFAULT_PROFILE_ID
    if runner == "03b_bridge":
        load_cache_and_scalers(selected_id)
        load_model_and_tokenizer(selected_id)
        return {"profileId": selected_id, "runner": runner, "loaded": ["03b_bridge"]}
    if runner == "03c_text_tuned":
        load_text_model_and_tokenizer(selected_id)
        return {"profileId": selected_id, "runner": runner, "loaded": ["03c_text_tuned"]}
    if runner == "03d_live_weighted_03b_03c":
        acoustic_profile_id = profile.get("acousticProfileId")
        text_profile_id = profile.get("textProfileId")
        if acoustic_profile_id:
            prewarm_profile(acoustic_profile_id)
        if text_profile_id:
            prewarm_profile(text_profile_id)
        return {
            "profileId": selected_id,
            "runner": runner,
            "loaded": [value for value in [acoustic_profile_id, text_profile_id] if value],
        }
    fail(f"Cannot prewarm unsupported live runner: {runner}", modelProfile=profile)


@lru_cache(maxsize=4)
def load_text_model_and_tokenizer(profile_id=DEFAULT_PROFILE_ID):
    from transformers import AutoTokenizer

    paths = profile_paths(profile_id)
    checkpoint_path = Path(os.getenv("MODEL_03C_CHECKPOINT", paths["checkpoint"]))
    text_model_path = Path(os.getenv("MODEL_03C_TEXT_MODEL", paths["text_model"]))
    if not checkpoint_path.exists():
        fail(f"Missing 03C text checkpoint: {checkpoint_path}")
    if not text_model_path.exists():
        fail(f"Missing local text model: {text_model_path}")

    log("Build 03C tuned transcript model")
    model = TranscriptMultiTaskSER(
        text_model_path,
        num_classes=4,
        dropout=float(os.getenv("MODEL_03C_DROPOUT", "0.25")),
    ).to(DEVICE)
    log(f"Load 03C checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(text_model_path), local_files_only=True)
    log("03C model and tokenizer ready")
    return model, tokenizer


def build_batch(features, transcript, profile_id=DEFAULT_PROFILE_ID):
    scalers = load_cache_and_scalers(profile_id)
    model, tokenizer = load_model_and_tokenizer(profile_id)
    if tokenizer is None:
        max_length = int(os.getenv("MODEL_03B_MAX_TEXT_LENGTH", "96"))
        encoded = {
            "input_ids": torch.zeros((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long),
        }
    else:
        encoded = tokenizer(
            transcript or "",
            max_length=int(os.getenv("MODEL_03B_MAX_TEXT_LENGTH", "96")),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
    temporal = (features["temporal"] - scalers["temporal_mean"][0]) / scalers["temporal_std"][0]
    spectral = (features["spectral"] - scalers["spectral_mean"][0]) / scalers["spectral_std"][0]
    stats = ((features["stats"] - scalers["stats_mean"]) / np.maximum(scalers["stats_scale"], 1e-6)).astype(np.float32)
    e2v = ((features["e2v"] - scalers["e2v_mean"]) / np.maximum(scalers["e2v_scale"], 1e-6)).astype(np.float32)
    return {
        "temporal": torch.tensor(temporal[None, :, :], dtype=torch.float32, device=DEVICE),
        "spectral": torch.tensor(spectral[None, :, :, :], dtype=torch.float32, device=DEVICE),
        "stats": torch.tensor(stats[None, :], dtype=torch.float32, device=DEVICE),
        "e2v": torch.tensor(e2v[None, :], dtype=torch.float32, device=DEVICE),
        "input_ids": encoded["input_ids"].to(DEVICE),
        "attention_mask": encoded["attention_mask"].to(DEVICE),
    }


def predict_from_text(transcript, profile_id=DEFAULT_PROFILE_ID):
    profile = get_model_profile(profile_id)
    model, tokenizer = load_text_model_and_tokenizer(profile_id)
    text = (transcript or "").strip() or "[EMPTY]"
    encoded = tokenizer(
        text,
        max_length=int(profile.get("maxLength") or os.getenv("MODEL_03C_MAX_TEXT_LENGTH", "96")),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    with torch.no_grad():
        out = model(input_ids, attention_mask, return_embedding=False)
        probs = torch.softmax(out["logits"], dim=1).cpu().numpy()[0]
        vad = out["vad"].cpu().numpy()[0]
    probs_by_label = {label: float(probs[i]) for i, label in enumerate(EMOTIONS)}
    web_probs = {label: float(probs_by_label.get(label, 0.0)) for label in WEB_EMOTION_ORDER}
    emotion = max(web_probs, key=web_probs.get)
    return {
        "ok": True,
        "source": f"{profile.get('shortLabel') or profile.get('label') or profile_id} + local transcript tuned checkpoint",
        "modelProfile": profile,
        "prediction": {
            "emotion": emotion,
            "probabilities": web_probs,
            "valence": float(vad[0]),
            "arousal": float(vad[1]),
            "dominance": float(vad[2]),
            "confidence": float(web_probs[emotion]),
        },
        "debug": {
            "device": str(DEVICE),
            "feature_shapes": {
                "Text tokens": list(input_ids.shape),
            },
            "warnings": [],
        },
    }


def predict_from_features(features, transcript, profile_id=DEFAULT_PROFILE_ID):
    profile = get_model_profile(profile_id)
    model, _ = load_model_and_tokenizer(profile_id)
    batch = build_batch(features, transcript, profile_id)
    log("Run forward pass")
    with torch.no_grad():
        out = model(batch, return_embedding=False)
        probs = torch.softmax(out["emotion_logits"], dim=1).cpu().numpy()[0]
        vad = out["vad_pred"].cpu().numpy()[0]
    probs_by_label = {label: float(probs[i]) for i, label in enumerate(EMOTIONS)}
    web_probs = {label: float(probs_by_label.get(label, 0.0)) for label in WEB_EMOTION_ORDER}
    emotion = max(web_probs, key=web_probs.get)
    return {
        "ok": True,
        "source": f"{profile.get('shortLabel') or profile.get('label') or profile_id} + local 06D feature extractor",
        "modelProfile": profile,
        "prediction": {
            "emotion": emotion,
            "probabilities": web_probs,
            "valence": float(vad[0]),
            "arousal": float(vad[1]),
            "dominance": float(vad[2]),
            "confidence": float(web_probs[emotion]),
        },
        "debug": {
            "device": str(DEVICE),
            "feature_shapes": {
                "X_temporal": list(features["temporal"].shape),
                "X_spectral": list(features["spectral"].shape),
                "X_stats": list(features["stats"].shape),
                "X_e2v": list(features["e2v"].shape),
            },
            "warnings": features["warnings"],
        },
    }


def fuse_predictions(acoustic_prediction, text_prediction, acoustic_weight=0.55):
    acoustic_weight = float(np.clip(acoustic_weight, 0.0, 1.0))
    text_weight = 1.0 - acoustic_weight
    probabilities = {}
    for label in WEB_EMOTION_ORDER:
        probabilities[label] = float(
            acoustic_weight * acoustic_prediction["probabilities"].get(label, 0.0)
            + text_weight * text_prediction["probabilities"].get(label, 0.0)
        )
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {label: float(value / total) for label, value in probabilities.items()}
    emotion = max(probabilities, key=probabilities.get)
    return {
        "emotion": emotion,
        "probabilities": probabilities,
        "valence": float(acoustic_weight * acoustic_prediction["valence"] + text_weight * text_prediction["valence"]),
        "arousal": float(acoustic_weight * acoustic_prediction["arousal"] + text_weight * text_prediction["arousal"]),
        "dominance": float(acoustic_weight * acoustic_prediction["dominance"] + text_weight * text_prediction["dominance"]),
        "confidence": float(probabilities[emotion]),
    }


def diagnose_segment(prediction):
    notes = []
    if prediction["arousal"] < 0.35:
        notes.append("low arousal")
    if prediction["valence"] < 0.35:
        notes.append("low valence")
    if prediction["dominance"] < 0.35:
        notes.append("low dominance")
    if prediction["confidence"] < 0.45:
        notes.append("uncertain emotion")
    return notes


def aggregate_segment_predictions(segment_results):
    weights = np.asarray([max(0.05, item["duration"]) for item in segment_results], dtype=np.float32)
    weights = weights / np.maximum(weights.sum(), 1e-8)
    probabilities = {}
    for label in WEB_EMOTION_ORDER:
        probabilities[label] = float(sum(weights[i] * segment_results[i]["prediction"]["probabilities"].get(label, 0.0) for i in range(len(segment_results))))
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {label: float(value / total) for label, value in probabilities.items()}
    emotion = max(probabilities, key=probabilities.get)
    valence = float(sum(weights[i] * segment_results[i]["prediction"]["valence"] for i in range(len(segment_results))))
    arousal = float(sum(weights[i] * segment_results[i]["prediction"]["arousal"] for i in range(len(segment_results))))
    dominance = float(sum(weights[i] * segment_results[i]["prediction"]["dominance"] for i in range(len(segment_results))))
    return {
        "emotion": emotion,
        "probabilities": probabilities,
        "valence": valence,
        "arousal": arousal,
        "dominance": dominance,
        "confidence": float(probabilities[emotion]),
    }


def infer_segment(segment_audio, transcript, segment_meta, profile_id=DEFAULT_PROFILE_ID):
    segment_path = write_segment_wav(segment_audio)
    try:
        features = extract_06d_features(segment_path)
        result = predict_from_features(features, transcript, profile_id)
    finally:
        try:
            segment_path.unlink(missing_ok=True)
        except Exception:
            pass
    prediction = result["prediction"]
    return {
        "index": int(segment_meta["index"]),
        "start": float(segment_meta["start"]),
        "end": float(segment_meta["end"]),
        "duration": float(max(0.0, segment_meta["end"] - segment_meta["start"])),
        "prediction": prediction,
        "diagnostics": diagnose_segment(prediction),
        "debug": result["debug"],
    }


def infer(audio_path, transcript, auto_transcribe=True, model_profile_id=DEFAULT_PROFILE_ID):
    log("Start live inference")
    profile = get_model_profile(model_profile_id)
    runner = profile.get("runner")
    if runner == "03c_text_tuned":
        return infer_text_tuned(audio_path, transcript, auto_transcribe, model_profile_id)
    if runner == "03d_live_weighted_03b_03c":
        return infer_live_weighted_expert_fusion(audio_path, transcript, auto_transcribe, model_profile_id)
    if runner != "03b_bridge":
        fail(
            f"Model profile '{model_profile_id}' does not have a supported live runner.",
            modelProfile=profile,
        )
    full_audio = load_audio_16k(audio_path, max_seconds=None)
    uses_text = profile_uses_text(profile)
    transcript_info = {
        "text": (transcript or "").strip(),
        "segments": [],
        "source": "provided transcript" if (transcript or "").strip() else "empty",
    }
    if not uses_text and not transcript_info["text"]:
        transcript_info["source"] = "not required for acoustic-only profile"
    elif auto_transcribe and not transcript_info["text"]:
        log("Transcript is empty; run local ASR")
        transcript_info = transcribe_audio_array(full_audio)
    effective_transcript = transcript_info["text"]
    segments = make_live_segments(full_audio)
    log(f"Live segmentation: {len(segments)} segment(s), {SEGMENT_SECONDS:.1f}s window, {SEGMENT_OVERLAP_SECONDS:.1f}s overlap")
    segment_results = []
    for segment in segments:
        log(f"Run segment {segment['index'] + 1}/{len(segments)}: {segment['start']:.2f}s-{segment['end']:.2f}s")
        segment_results.append(infer_segment(segment["audio"], effective_transcript, segment, model_profile_id))
    prediction = aggregate_segment_predictions(segment_results)
    warnings = []
    for item in segment_results:
        warnings.extend(item["debug"].get("warnings", []))
    unique_warnings = list(dict.fromkeys(warnings))
    first_debug = segment_results[0]["debug"] if segment_results else {}
    return {
        "ok": True,
        "source": f"{profile.get('shortLabel') or profile.get('label') or model_profile_id} + local 06D live segmented inference",
        "modelProfile": profile,
        "transcript": transcript_info,
        "prediction": prediction,
        "segments": [
            {
                "index": item["index"],
                "start": item["start"],
                "end": item["end"],
                "duration": item["duration"],
                "prediction": item["prediction"],
                "diagnostics": item["diagnostics"],
            }
            for item in segment_results
        ],
        "aggregation": {
            "method": "duration_weighted_mean_probabilities_and_vad",
            "segmentSeconds": SEGMENT_SECONDS,
            "overlapSeconds": SEGMENT_OVERLAP_SECONDS,
            "segmentCount": len(segment_results),
        },
        "debug": {
            "device": str(DEVICE),
            "duration": float(len(full_audio) / SAMPLE_RATE),
            "feature_shapes": first_debug.get("feature_shapes", {}),
            "warnings": unique_warnings,
        },
    }


def infer_text_tuned(audio_path, transcript, auto_transcribe=True, model_profile_id=DEFAULT_PROFILE_ID):
    profile = get_model_profile(model_profile_id)
    full_audio = load_audio_16k(audio_path, max_seconds=None)
    transcript_info = {
        "text": (transcript or "").strip(),
        "segments": [],
        "source": "provided transcript" if (transcript or "").strip() else "empty",
    }
    if auto_transcribe and not transcript_info["text"]:
        log("Transcript is empty; run local ASR for 03C text model")
        transcript_info = transcribe_audio_array(full_audio)
    effective_transcript = transcript_info["text"]
    if transcript_info.get("segments"):
        segment_results = []
        for segment in transcript_info["segments"]:
            segment_pred = predict_from_text(segment.get("text", ""), model_profile_id)["prediction"]
            segment_results.append({
                "index": int(segment["index"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "duration": float(max(0.0, segment["end"] - segment["start"])),
                "prediction": segment_pred,
                "diagnostics": diagnose_segment(segment_pred),
            })
        prediction = aggregate_segment_predictions(segment_results) if segment_results else predict_from_text(effective_transcript, model_profile_id)["prediction"]
    else:
        text_result = predict_from_text(effective_transcript, model_profile_id)
        prediction = text_result["prediction"]
        segment_results = []
    return {
        "ok": True,
        "source": f"{profile.get('shortLabel') or profile.get('label') or model_profile_id} + local transcript tuned inference",
        "modelProfile": profile,
        "transcript": transcript_info,
        "prediction": prediction,
        "segments": segment_results,
        "aggregation": {
            "method": "text_segments_duration_weighted_mean" if segment_results else "single_transcript_prediction",
            "segmentCount": len(segment_results),
        },
        "debug": {
            "device": str(DEVICE),
            "duration": float(len(full_audio) / SAMPLE_RATE),
            "feature_shapes": {"Text tokens": [1, int(profile.get("maxLength") or 96)]},
            "warnings": [],
        },
    }


def infer_live_weighted_expert_fusion(audio_path, transcript, auto_transcribe=True, model_profile_id=DEFAULT_PROFILE_ID):
    profile = get_model_profile(model_profile_id)
    acoustic_profile_id = profile.get("acousticProfileId")
    text_profile_id = profile.get("textProfileId")
    if not acoustic_profile_id or not text_profile_id:
        fail("03D live fusion profile must define acousticProfileId and textProfileId.", modelProfile=profile)
    full_audio = load_audio_16k(audio_path, max_seconds=None)
    transcript_info = {
        "text": (transcript or "").strip(),
        "segments": [],
        "source": "provided transcript" if (transcript or "").strip() else "empty",
    }
    if auto_transcribe and not transcript_info["text"]:
        log("Transcript is empty; run local ASR before 03D live fusion")
        transcript_info = transcribe_audio_array(full_audio)
    effective_transcript = transcript_info["text"]
    acoustic_result = infer(audio_path, effective_transcript, auto_transcribe=False, model_profile_id=acoustic_profile_id)
    text_result = infer_text_tuned(audio_path, effective_transcript, auto_transcribe=False, model_profile_id=text_profile_id)
    acoustic_weight = float(profile.get("acousticWeight", os.getenv("MODEL_03D_ACOUSTIC_WEIGHT", "0.55")))
    prediction = fuse_predictions(acoustic_result["prediction"], text_result["prediction"], acoustic_weight)
    return {
        "ok": True,
        "source": f"{profile.get('shortLabel') or profile.get('label') or model_profile_id} + live weighted 03B/03C checkpoint fusion",
        "modelProfile": profile,
        "transcript": transcript_info,
        "prediction": prediction,
        "segments": acoustic_result.get("segments", []),
        "aggregation": {
            "method": "live_weighted_prediction_fusion",
            "acousticWeight": acoustic_weight,
            "textWeight": 1.0 - acoustic_weight,
            "acousticProfileId": acoustic_profile_id,
            "textProfileId": text_profile_id,
            "note": "Live demo fusion uses saved 03B and 03C checkpoints. It is not a standalone saved 03D fusion-head checkpoint.",
        },
        "debug": {
            "device": str(DEVICE),
            "duration": float(len(full_audio) / SAMPLE_RATE),
            "feature_shapes": {
                "Acoustic": acoustic_result.get("debug", {}).get("feature_shapes", {}),
                "Text": text_result.get("debug", {}).get("feature_shapes", {}),
            },
            "warnings": list(dict.fromkeys(
                (acoustic_result.get("debug", {}).get("warnings", []) or [])
                + (text_result.get("debug", {}).get("warnings", []) or [])
            )),
        },
        "experts": {
            "acoustic": {
                "profile": acoustic_result.get("modelProfile"),
                "prediction": acoustic_result.get("prediction"),
            },
            "text": {
                "profile": text_result.get("modelProfile"),
                "prediction": text_result.get("prediction"),
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--features-only", action="store_true")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--no-auto-transcribe", action="store_true")
    parser.add_argument("--model-profile", default=DEFAULT_PROFILE_ID)
    args = parser.parse_args()
    try:
        if args.asr_only:
            result = {
                "ok": True,
                "source": "local ASR only",
                "transcript": transcribe_audio_file(Path(args.audio)),
            }
        elif args.features_only:
            features = extract_06d_features(Path(args.audio))
            result = {
                "ok": True,
                "source": "03B feature extractor only",
                "feature_shapes": {
                    "X_temporal": list(features["temporal"].shape),
                    "X_spectral": list(features["spectral"].shape),
                    "X_stats": list(features["stats"].shape),
                    "X_e2v": list(features["e2v"].shape),
                },
                "warnings": features["warnings"],
            }
        else:
            result = infer(
                Path(args.audio),
                args.transcript,
                auto_transcribe=not args.no_auto_transcribe,
                model_profile_id=args.model_profile,
            )
        print(json.dumps(result, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
