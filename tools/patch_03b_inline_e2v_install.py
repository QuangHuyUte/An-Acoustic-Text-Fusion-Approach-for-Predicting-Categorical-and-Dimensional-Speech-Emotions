import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"


OLD = '''def build_emotion2vec_extractor_if_needed():
    if os.getenv("EXTRACT_EMOTION2VEC", "1") != "1":
        return None
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise ImportError(
            "03B đang cần X_e2v nhưng chưa tìm thấy cache Emotion2Vec từ notebook 02. "
            "Bạn có 2 cách: (1) upload output của notebook 02 có file "
            "`features/iemocap_full_emotion2vec_acoustic_cache.npz` hoặc "
            "`features/iemocap_full_06d_multibranch_cache.npz`; "
            "(2) bật Internet trên Kaggle và set INSTALL_EMOTION2VEC_DEPS=1 để notebook tự cài funasr và tự trích Emotion2Vec."
        ) from exc
    model_name = os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base")
    print("Loading emotion2vec extractor:", model_name)
    return AutoModel(model=model_name, disable_update=True)
'''

NEW = '''def build_emotion2vec_extractor_if_needed():
    if os.getenv("EXTRACT_EMOTION2VEC", "1") != "1":
        return None
    if os.getenv("INSTALL_EMOTION2VEC_DEPS", "0") == "1":
        import subprocess
        print("Installing FunASR/ModelScope for Emotion2Vec...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "funasr", "modelscope"])
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise ImportError(
            "03B đang cần X_e2v nhưng chưa tìm thấy cache Emotion2Vec từ notebook 02. "
            "Bạn có 2 cách: (1) upload output của notebook 02 có file "
            "`features/iemocap_full_emotion2vec_acoustic_cache.npz` hoặc "
            "`features/iemocap_full_06d_multibranch_cache.npz`; "
            "(2) bật Internet trên Kaggle và set INSTALL_EMOTION2VEC_DEPS=1 để notebook tự cài funasr và tự trích Emotion2Vec."
        ) from exc
    model_name = os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base")
    print("Loading emotion2vec extractor:", model_name)
    try:
        return AutoModel(model=model_name, disable_update=True)
    except TypeError:
        return AutoModel(model=model_name)
'''


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        if OLD in source:
            cell["source"] = source.replace(OLD, NEW)
            changed = True
            break
    if not changed:
        raise RuntimeError("Không tìm thấy build_emotion2vec_extractor_if_needed để vá.")
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    main()
