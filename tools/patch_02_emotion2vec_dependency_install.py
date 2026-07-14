import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "02_IEMOCAP Feature Extraction Emotion2Vec Acoustic" / "02-iemocap-feature-extraction-emotion2vec-acoustic.ipynb"


OLD_BLOCK = '''def maybe_install_emotion2vec_deps():
    if not INSTALL_EMOTION2VEC_DEPS:
        return
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "funasr", "modelscope"])

def load_emotion2vec_model():
    if not EXTRACT_EMOTION2VEC:
        if REQUIRE_EMOTION2VEC:
            raise ValueError("REQUIRE_EMOTION2VEC=1 nhưng EXTRACT_EMOTION2VEC=0. Hãy bật EXTRACT_EMOTION2VEC=1 để chạy bản full.")
        print("EXTRACT_EMOTION2VEC=0, using zero X_e2v vectors.")
        return None
    maybe_install_emotion2vec_deps()
    try:
        from funasr import AutoModel
        model = AutoModel(model=EMOTION2VEC_MODEL_NAME)
        print("Loaded emotion2vec model:", EMOTION2VEC_MODEL_NAME)
        return model
    except Exception as exc:
        if REQUIRE_EMOTION2VEC:
            raise
        print("Cannot load emotion2vec; using zero vectors. Error:", repr(exc))
        return None
'''

NEW_BLOCK = '''def maybe_install_emotion2vec_deps():
    if not INSTALL_EMOTION2VEC_DEPS:
        return
    import importlib
    import subprocess
    print("Installing Emotion2Vec dependencies: funasr, modelscope ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "funasr", "modelscope"])
    importlib.invalidate_caches()

def load_emotion2vec_model():
    if not EXTRACT_EMOTION2VEC:
        if REQUIRE_EMOTION2VEC:
            raise ValueError("REQUIRE_EMOTION2VEC=1 nhưng EXTRACT_EMOTION2VEC=0. Hãy bật EXTRACT_EMOTION2VEC=1 để chạy bản full.")
        print("EXTRACT_EMOTION2VEC=0, using zero X_e2v vectors.")
        return None
    maybe_install_emotion2vec_deps()
    try:
        from funasr import AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Không import được `funasr`. Trên Kaggle hãy bật Internet và chạy lại cell này, "
            "hoặc set INSTALL_EMOTION2VEC_DEPS=1 trước khi chạy notebook. "
            "Nếu vẫn lỗi, chạy thủ công: `pip install -U funasr modelscope` rồi Restart session."
        ) from exc
    try:
        try:
            model = AutoModel(model=EMOTION2VEC_MODEL_NAME, disable_update=True)
        except TypeError:
            model = AutoModel(model=EMOTION2VEC_MODEL_NAME)
        print("Loaded emotion2vec model:", EMOTION2VEC_MODEL_NAME)
        return model
    except Exception as exc:
        if REQUIRE_EMOTION2VEC:
            raise RuntimeError(
                "Đã import được funasr nhưng chưa load được Emotion2Vec model. "
                "Hãy kiểm tra Internet/Kaggle accelerator và model name."
            ) from exc
        print("Cannot load emotion2vec; using zero vectors. Error:", repr(exc))
        return None
'''


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        source = "".join(cell.get("source", ""))
        updated = source
        updated = updated.replace(
            'INSTALL_EMOTION2VEC_DEPS = os.getenv("INSTALL_EMOTION2VEC_DEPS", "0") == "1"',
            'INSTALL_EMOTION2VEC_DEPS = os.getenv("INSTALL_EMOTION2VEC_DEPS", "1") == "1"',
        )
        updated = updated.replace(
            'INSTALL_EMOTION2VEC_DEPS=1\\n',
            'INSTALL_EMOTION2VEC_DEPS=1\\n',
        )
        if OLD_BLOCK in updated:
            updated = updated.replace(OLD_BLOCK, NEW_BLOCK)
        if updated != source:
            cell["source"] = updated
            changed = True

    if not changed:
        raise RuntimeError("Không tìm thấy đoạn cần vá trong notebook 02.")
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    main()
