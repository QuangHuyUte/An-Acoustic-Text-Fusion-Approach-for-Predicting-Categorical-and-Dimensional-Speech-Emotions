from pathlib import Path

import nbformat


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")

NB01 = ROOT / "06_w2v_based_models" / "01_IEMOCAP Dataset Analysis and Speaker-Independent Splits" / "01_IEMOCAP_Dataset_Analysis_and_Splits.ipynb"
NB03A = ROOT / "06_w2v_based_models" / "03_Emotion2Vec RawAudio Backbone Finetune 5Fold 10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
NB03B = ROOT / "06_w2v_based_models" / "03_Emotion2Vec Downstream Finetune 5Fold 10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"


ROOT_DISCOVERY_01 = r'''IEMOCAP_ROOT = os.environ.get("IEMOCAP_ROOT", "").strip()

def looks_like_iemocap_root(path: Path) -> bool:
    """Nhận diện IEMOCAP full release: Session*/dialog/EmoEvaluation và Session*/sentences/wav."""
    if not path.exists() or not path.is_dir():
        return False
    sessions = [path / f"Session{i}" for i in range(1, 6)]
    has_sessions = sum(s.exists() for s in sessions) >= 3
    has_eval = any((s / "dialog" / "EmoEvaluation").exists() for s in sessions)
    has_wav = any((s / "sentences" / "wav").exists() for s in sessions)
    return has_sessions and has_eval and has_wav

def candidate_iemocap_roots():
    seeds = [
        Path(IEMOCAP_ROOT) if IEMOCAP_ROOT else None,
        PROJECT_ROOT / "IEMOCAP",
        PROJECT_ROOT / "IEMOCAP_full_release",
        PROJECT_ROOT / "data" / "IEMOCAP",
        PROJECT_ROOT / "data" / "IEMOCAP_full_release",
        PROJECT_ROOT / "datasets" / "IEMOCAP",
        PROJECT_ROOT / "datasets" / "IEMOCAP_full_release",
        PROJECT_ROOT / "Datasets" / "IEMOCAP",
        Path(r"D:\UTE\Speech Programming\Speech Project\IEMOCAP"),
        Path(r"D:\UTE\Speech Programming\Speech Project\IEMOCAP_full_release"),
        Path(r"D:\UTE\Speech Programming\Speech Project\data\IEMOCAP"),
        Path(r"D:\UTE\Speech Programming\Speech Project\data\IEMOCAP_full_release"),
        Path(r"D:\UTE\Speech Programming\Speech Project\datasets\IEMOCAP"),
        Path(r"/kaggle/input"),
        Path(r"/kaggle/working"),
    ]
    seen = set()
    for seed in seeds:
        if seed is None or not seed.exists():
            continue
        for candidate in [
            seed,
            seed / "IEMOCAP",
            seed / "IEMOCAP_full_release",
            seed / "iemocapfullrelease" / "IEMOCAP_full_release",
            seed / "data" / "IEMOCAP_full_release",
        ]:
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
                parent = session1.parent
                key = str(parent).lower()
                if key not in seen:
                    seen.add(key)
                    yield parent
        except Exception:
            pass

resolved_root = None
for candidate in candidate_iemocap_roots():
    if looks_like_iemocap_root(candidate):
        resolved_root = candidate.resolve()
        break

if resolved_root is None:
    display(Markdown("""
### Chưa tìm thấy IEMOCAP full release root

Notebook chưa thấy cấu trúc IEMOCAP gốc trong các đường dẫn gợi ý.

Với Kaggle dataset `dejolilandry/iemocapfullrelease`, root thường là:

```python
IEMOCAP_ROOT = "/kaggle/input/iemocapfullrelease/IEMOCAP_full_release"
```

Root hợp lệ cần có:

```text
IEMOCAP_full_release/
  Session1/dialog/EmoEvaluation/
  Session1/sentences/wav/
  ...
  Session5/
```
"""))
else:
    IEMOCAP_ROOT = resolved_root
    print("Found IEMOCAP_ROOT:", IEMOCAP_ROOT)
'''


RAW_AUDIO_DISCOVERY = r'''def looks_like_iemocap_release(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    sessions = [path / f"Session{i}" for i in range(1, 6)]
    has_sessions = sum(s.exists() for s in sessions) >= 3
    has_wav = any((s / "sentences" / "wav").exists() for s in sessions)
    return has_sessions and has_wav

def has_any_wav(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        return any(path.rglob("*.wav"))
    except Exception:
        return False

def find_audio_dir():
    """Trả về thư mục có thể quét được wav.

    Hỗ trợ cả:
    - data/audio_wav/*.wav hoặc output/audio_wav/*.wav
    - IEMOCAP_full_release/Session*/sentences/wav/**/*.wav
    """
    if os.getenv("IEMOCAP_AUDIO_DIR"):
        candidate = Path(os.getenv("IEMOCAP_AUDIO_DIR"))
        if candidate.exists() and (has_any_wav(candidate) or looks_like_iemocap_release(candidate)):
            return candidate.resolve()
    if os.getenv("IEMOCAP_ROOT"):
        candidate = Path(os.getenv("IEMOCAP_ROOT"))
        if candidate.exists() and (has_any_wav(candidate) or looks_like_iemocap_release(candidate)):
            return candidate.resolve()
    for root in search_roots():
        for candidate in [
            root / "audio_wav",
            root / "data" / "audio_wav",
            root / "output" / "audio_wav",
            root / "datasets" / "AbstractTTS_IEMOCAP" / "audio_wav",
            root / "IEMOCAP_full_release",
            root / "iemocapfullrelease" / "IEMOCAP_full_release",
            root / "data" / "IEMOCAP_full_release",
        ]:
            if candidate.exists() and (has_any_wav(candidate) or looks_like_iemocap_release(candidate)):
                return candidate.resolve()
        try:
            for candidate in root.rglob("audio_wav"):
                if candidate.is_dir() and has_any_wav(candidate):
                    return candidate.resolve()
            for candidate in root.rglob("IEMOCAP_full_release"):
                if looks_like_iemocap_release(candidate):
                    return candidate.resolve()
            for session1 in root.rglob("Session1"):
                candidate = session1.parent
                if looks_like_iemocap_release(candidate):
                    return candidate.resolve()
        except Exception:
            pass
    roots_text = "\n".join(f"- {root}" for root in search_roots())
    raise FileNotFoundError(
        "Không tìm thấy raw audio IEMOCAP. Notebook đã quét cả dạng audio_wav và "
        f"IEMOCAP_full_release/Session*/sentences/wav:\n{roots_text}"
    )
'''


RAW_AUDIO_RESOLVER_03A = r'''_AUDIO_INDEX = None

def audio_index():
    global _AUDIO_INDEX
    if _AUDIO_INDEX is None:
        _AUDIO_INDEX = {}
        for p in AUDIO_DIR.rglob("*.wav"):
            resolved = p.resolve()
            _AUDIO_INDEX[p.name] = resolved
            _AUDIO_INDEX[p.stem] = resolved
    return _AUDIO_INDEX

def resolve_wav_path(row):
    name = Path(str(row.get("wav_path", "")).replace("\\", "/")).name
    direct = AUDIO_DIR / name
    if direct.exists():
        return direct.resolve()
    idx = audio_index()
    if name in idx:
        return idx[name]
    stem = Path(name).stem
    if stem in idx:
        return idx[stem]
    utterance = str(row.get("utterance_id", ""))
    direct = AUDIO_DIR / f"{utterance}.wav"
    if direct.exists():
        return direct.resolve()
    if utterance in idx:
        return idx[utterance]
    if f"{utterance}.wav" in idx:
        return idx[f"{utterance}.wav"]
    raise FileNotFoundError(f"Không tìm thấy wav cho {utterance} ({name})")

'''


RAW_AUDIO_RESOLVER_03B = r'''_AUDIO_INDEX = None

def audio_index(audio_dir):
    global _AUDIO_INDEX
    if _AUDIO_INDEX is None:
        _AUDIO_INDEX = {}
        for p in audio_dir.rglob("*.wav"):
            resolved = p.resolve()
            _AUDIO_INDEX[p.name] = resolved
            _AUDIO_INDEX[p.stem] = resolved
    return _AUDIO_INDEX

def resolve_wav_path_from_row(row, audio_dir):
    name = Path(str(row.get("wav_path", "")).replace("\\\\", "/")).name
    direct = audio_dir / name
    if direct.exists():
        return direct.resolve()
    idx = audio_index(audio_dir)
    if name in idx:
        return idx[name]
    stem = Path(name).stem
    if stem in idx:
        return idx[stem]
    utterance = str(row.get("utterance_id", ""))
    direct = audio_dir / f"{utterance}.wav"
    if direct.exists():
        return direct.resolve()
    if utterance in idx:
        return idx[utterance]
    if f"{utterance}.wav" in idx:
        return idx[f"{utterance}.wav"]
    raise FileNotFoundError(f"Không tìm thấy wav cho {utterance} ({name}) trong {audio_dir}")

'''


def replace_func_block(src: str, start_marker: str, end_marker: str, new_block: str) -> str:
    start = src.index(start_marker)
    end = src.index(end_marker, start)
    return src[:start] + new_block.rstrip() + "\n\n" + src[end:]


def update_01():
    nb = nbformat.read(NB01, as_version=4)
    nb.cells[1].source = """## 1. Dataset Expected Structure

Notebook này ưu tiên đọc **IEMOCAP full release** theo cấu trúc gốc. Với Kaggle dataset `dejolilandry/iemocapfullrelease`, root thường là:

```text
/kaggle/input/iemocapfullrelease/IEMOCAP_full_release/
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

Trong notebook này:

- Label emotion và VAD được parse từ `Session*/dialog/EmoEvaluation/*.txt`.
- Audio utterance-level được tìm trong `Session*/sentences/wav/**/*.wav`.
- Transcript nếu cần mở rộng về sau nằm trong `Session*/dialog/transcriptions/*.txt`.

Setting mặc định dùng 4-class phổ biến:

```text
neutral, angry, sad, happy
```

Trong đó `hap` và `exc` được gộp thành `happy`. Đây là cách thường tạo subset khoảng **5,531 utterances** khi parse từ annotation gốc.
"""
    nb.cells[9].source = ROOT_DISCOVERY_01
    nb.cells[11].source = nb.cells[11].source.replace(
        '''def parse_conversation_id(utterance_id: str) -> str:
    parts = utterance_id.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else utterance_id''',
        '''def parse_conversation_id(utterance_id: str) -> str:
    # Bỏ hậu tố lượt nói F000/M000 để giữ đúng dialog id:
    # Ses01F_impro01_F000 -> Ses01F_impro01
    # Ses01F_script01_1_F000 -> Ses01F_script01_1
    parts = utterance_id.split("_")
    return "_".join(parts[:-1]) if len(parts) >= 2 else utterance_id''',
    )
    nb.cells[14].source = '''HF_METADATA_FULL_PATH = PROJECT_ROOT / "06_w2v_based_models" / "datasets" / "AbstractTTS_IEMOCAP" / "abstracttts_iemocap_metadata_full.csv"

DATA_SOURCE_KIND = "unknown"
DATA_SOURCE_NAME = "unknown"

if resolved_root is None and HF_METADATA_FULL_PATH.exists():
    DATA_SOURCE_KIND = "hf_abstracttts"
    DATA_SOURCE_NAME = "AbstractTTS/IEMOCAP Hugging Face fallback"
    metadata = pd.read_csv(HF_METADATA_FULL_PATH)
    metadata["is_4class"] = metadata["emotion_4class"].notna()
    metadata["source_dataset"] = DATA_SOURCE_KIND
    print("Loaded Hugging Face AbstractTTS/IEMOCAP metadata:", metadata.shape)
    display(Markdown("""
**Đang dùng fallback Hugging Face AbstractTTS/IEMOCAP.**

Notebook không thấy IEMOCAP full release root, nhưng đã thấy metadata HF đã tải trong workspace.
"""))
    display(metadata.head())

    metadata_path = METADATA_DIR / "iemocap_metadata_full.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print("Saved full metadata:", metadata_path)
elif resolved_root is None:
    metadata = pd.DataFrame()
    display(Markdown("**Dataset chưa sẵn sàng.** Hãy cấu hình `IEMOCAP_ROOT` hoặc tải Hugging Face AbstractTTS/IEMOCAP, sau đó chạy lại notebook."))
else:
    DATA_SOURCE_KIND = "official_full_release"
    DATA_SOURCE_NAME = f"IEMOCAP full release ({IEMOCAP_ROOT})"
    metadata = parse_iemocap_metadata(IEMOCAP_ROOT)
    metadata["source_dataset"] = DATA_SOURCE_KIND
    print("Metadata shape:", metadata.shape)
    display(metadata.head())

    metadata_path = METADATA_DIR / "iemocap_metadata_full.csv"
    metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    print("Saved full metadata:", metadata_path)
'''
    nb.cells[39].source = nb.cells[39].source.replace(
        '        "- Source used in this notebook: AbstractTTS/IEMOCAP on HuggingFace.",',
        '        f"- Source used in this notebook: **{DATA_SOURCE_NAME}**.",',
    )
    old_note = '''        "Important: this 4-class subset is built from the HuggingFace `major_emotion` style labels. It has "
        f"**{len(train_metadata):,}** samples, so it is not automatically the same as the canonical IEMOCAP 4-class "
        "benchmark with about 5,531 utterances used by many papers. Results should be reported as "
        "**HF-major-emotion 4-class setting** unless we reproduce the exact paper filtering rule.",'''
    new_note = '''        (
            "Important: this 4-class subset is built from the original `EmoEvaluation` labels "
            "with `hap` and `exc` merged into `happy`. This is the standard setting used by many IEMOCAP SER papers; "
            "the expected size is around 5,531 utterances when all canonical labels are present."
            if DATA_SOURCE_KIND == "official_full_release"
            else
            "Important: this 4-class subset is built from the HuggingFace `major_emotion` style labels. "
            f"It has **{len(train_metadata):,}** samples, so it is not automatically the same as the canonical "
            "IEMOCAP 4-class benchmark with about 5,531 utterances used by many papers."
        ),'''
    nb.cells[39].source = nb.cells[39].source.replace(old_note, new_note)
    nbformat.write(nb, NB01)


def update_raw_audio_notebook(path: Path):
    nb = nbformat.read(path, as_version=4)
    cell6 = nb.cells[6].source
    helper_start = cell6.find("def looks_like_iemocap_release")
    if helper_start != -1:
        cell6 = cell6[:helper_start] + cell6[cell6.index("SPLIT_5FOLD_PATH", helper_start):]
        nb.cells[6].source = cell6
    if "def find_audio_dir():" in nb.cells[6].source:
        nb.cells[6].source = replace_func_block(
            nb.cells[6].source,
            "def find_audio_dir():",
            "SPLIT_5FOLD_PATH",
            RAW_AUDIO_DISCOVERY,
        )
    else:
        split_marker = "SPLIT_5FOLD_PATH"
        split_start = nb.cells[6].source.index(split_marker)
        nb.cells[6].source = (
            nb.cells[6].source[:split_start]
            + RAW_AUDIO_DISCOVERY.rstrip()
            + "\n\n"
            + nb.cells[6].source[split_start:]
        )
    for idx in range(len(nb.cells)):
        src = nb.cells[idx].source
        if "def audio_index" in src and "def resolve_wav_path" in src:
            resolver = RAW_AUDIO_RESOLVER_03B if "resolve_wav_path_from_row" in src else RAW_AUDIO_RESOLVER_03A
            src = replace_func_block(src, "_AUDIO_INDEX = None", "def load_audio", resolver)
            nb.cells[idx].source = src
    nbformat.write(nb, path)


def main():
    update_01()
    update_raw_audio_notebook(NB03A)
    update_raw_audio_notebook(NB03B)
    print("Updated IEMOCAP full release support in:")
    print("-", NB01)
    print("-", NB03A)
    print("-", NB03B)


if __name__ == "__main__":
    main()
