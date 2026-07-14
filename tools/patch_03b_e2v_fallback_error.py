import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    changed = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        old = 'if os.getenv("EXTRACT_EMOTION2VEC", "0") != "1":\n        return None'
        new = 'if os.getenv("EXTRACT_EMOTION2VEC", "1") != "1":\n        return None'
        if old in source:
            source = source.replace(old, new)
            changed = True

        old_import_error = 'raise ImportError("Muốn tự trích emotion2vec thì cần funasr. Bật Internet và cài funasr/modelscope, hoặc dùng light cache có sẵn.") from exc'
        new_import_error = '''raise ImportError(
            "03B đang cần X_e2v nhưng chưa tìm thấy cache Emotion2Vec từ notebook 02. "
            "Bạn có 2 cách: (1) upload output của notebook 02 có file "
            "`features/iemocap_full_emotion2vec_acoustic_cache.npz` hoặc "
            "`features/iemocap_full_06d_multibranch_cache.npz`; "
            "(2) bật Internet trên Kaggle và set INSTALL_EMOTION2VEC_DEPS=1 để notebook tự cài funasr và tự trích Emotion2Vec."
        ) from exc'''
        if old_import_error in source:
            source = source.replace(old_import_error, new_import_error)
            changed = True

        old_value_error = '''raise ValueError(
        "Không có X_e2v cho sample và chưa bật EXTRACT_EMOTION2VEC=1. "
        "Cách nhanh nhất: upload thêm iemocap_full_emotion2vec_acoustic_cache.npz để lấy sẵn emotion2vec."
    )'''
        new_value_error = '''raise ValueError(
        "Không có X_e2v cho sample này. 03B đã tìm raw audio để dựng X_temporal/X_spectral/X_stats, "
        "nhưng vẫn cần Emotion2Vec thật cho X_e2v. Hãy upload output của notebook 02 có "
        "`features/iemocap_full_emotion2vec_acoustic_cache.npz` hoặc bật EXTRACT_EMOTION2VEC=1 "
        "kèm INSTALL_EMOTION2VEC_DEPS=1 trên Kaggle. Nếu chỉ debug pipeline, có thể đặt ALLOW_ZERO_E2V=1."
    )'''
        if old_value_error in source:
            source = source.replace(old_value_error, new_value_error)
            changed = True

        cell["source"] = source

    if not changed:
        raise RuntimeError("Không tìm thấy đoạn cần vá trong notebook 03B.")
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    main()
