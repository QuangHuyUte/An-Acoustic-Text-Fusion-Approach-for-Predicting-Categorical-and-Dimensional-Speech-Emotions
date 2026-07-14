from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = (
    ROOT
    / "06_w2v_based_models"
    / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold"
    / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
)


E2V_MARKDOWN = """<!-- AUTO-03B-E2V-PATCH -->
## Emotion2Vec thật cho 03B

Output 02 có thể chứa `X_e2v` toàn zero nếu lúc trích feature chưa cài được FunASR/ModelScope. Khi đó 03B vẫn chạy được acoustic 06D, nhưng chưa phải Emotion2Vec-guided co-attention đúng nghĩa.

Notebook 03B hiện đã có cơ chế vá tự động:

```text
1. Đọc cache 06D từ notebook 02.
2. Kiểm tra norm của X_e2v.
3. Nếu X_e2v toàn zero:
   - tìm file patch emotion2vec đã có, hoặc
   - dùng FunASR emotion2vec để trích lại embedding từ raw audio.
4. Lưu patch nhỏ: features/iemocap_03b_emotion2vec_patch.npz.
5. Thay X_e2v trong RAM trước khi train.
```

Khi chạy Kaggle để có Emotion2Vec thật, nên bật:

```text
INSTALL_EMOTION2VEC_DEPS=1
EXTRACT_EMOTION2VEC=1
REQUIRE_REAL_E2V=1
```

Và cần upload raw audio IEMOCAP full release, vì Emotion2Vec nhận đầu vào là waveform/WAV, không thể sinh embedding thật từ `X_temporal/X_spectral/X_stats` đã cache.
"""


PATCH_CODE = r'''

# AUTO-03B-E2V-PATCH
def install_emotion2vec_deps_if_needed():
    install_e2v = os.getenv("INSTALL_EMOTION2VEC_DEPS", "0") == "1"
    if not install_e2v:
        return
    import subprocess
    print("Installing FunASR/ModelScope for Emotion2Vec...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "funasr", "modelscope"])

def load_emotion2vec_extractor_for_patch():
    if os.getenv("EXTRACT_EMOTION2VEC", "1") != "1":
        return None
    install_emotion2vec_deps_if_needed()
    try:
        from funasr import AutoModel
    except Exception as exc:
        raise ImportError(
            "Không import được funasr. Trên Kaggle hãy bật Internet và set INSTALL_EMOTION2VEC_DEPS=1, "
            "hoặc upload sẵn features/iemocap_03b_emotion2vec_patch.npz."
        ) from exc
    model_name = os.getenv("EMOTION2VEC_MODEL_NAME", os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base"))
    print("Loading Emotion2Vec extractor:", model_name)
    try:
        return AutoModel(model=model_name, disable_update=True)
    except TypeError:
        return AutoModel(model=model_name)

def find_first_numeric_embedding(obj):
    candidates = []
    def visit(x):
        if isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number):
            arr = np.asarray(x, dtype=np.float32).squeeze()
            if arr.size >= 32:
                candidates.append(arr)
        elif torch.is_tensor(x):
            arr = x.detach().cpu().numpy().astype(np.float32).squeeze()
            if arr.size >= 32:
                candidates.append(arr)
        elif isinstance(x, dict):
            preferred = ["feats", "embedding", "embeddings", "x", "hidden_states", "value"]
            for key in preferred:
                if key in x:
                    visit(x[key])
            for value in x.values():
                visit(value)
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
    visit(obj)
    if not candidates:
        raise ValueError(f"Không tìm thấy numeric embedding trong output Emotion2Vec type={type(obj)}")
    arr = sorted(candidates, key=lambda a: a.size, reverse=True)[0]
    if arr.ndim > 1:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    return arr.reshape(-1).astype(np.float32)

def find_existing_e2v_patch():
    candidates = [
        "iemocap_03b_emotion2vec_patch.npz",
        "iemocap_emotion2vec_patch.npz",
        "iemocap_full_emotion2vec_patch.npz",
    ]
    for name in candidates:
        found = find_optional_file(name)
        if found is not None:
            return found
    return None

def load_e2v_patch_for_cache(cache_ids):
    patch_path = find_existing_e2v_patch()
    if patch_path is None:
        return None, None
    patch = np.load(patch_path, allow_pickle=True)
    if "X_e2v" not in patch.files:
        return patch_path, None
    patch_id_key = None
    for key in ["train_sample_id", "sample_ids", "utterance_id"]:
        if key in patch.files:
            patch_id_key = key
            break
    if patch_id_key is None:
        return patch_path, None
    patch_ids = patch[patch_id_key].astype(str)
    patch_map = {sid: i for i, sid in enumerate(patch_ids)}
    missing = [sid for sid in cache_ids.astype(str) if sid not in patch_map]
    if missing:
        print(f"Emotion2Vec patch tồn tại nhưng thiếu {len(missing)} ids. Ví dụ:", missing[:5])
        return patch_path, None
    X_patch = np.stack([patch["X_e2v"][patch_map[sid]] for sid in cache_ids.astype(str)]).astype(np.float32)
    print("Loaded Emotion2Vec patch:", patch_path, X_patch.shape)
    return patch_path, X_patch

def extract_e2v_vector_for_row(row, wav_path, light_map, extractor):
    for key in ["train_sample_id", "utterance_id"]:
        value = str(row.get(key, ""))
        if value in light_map:
            return light_map[value].astype(np.float32), "ok_light_cache"
    if extractor is None:
        raise RuntimeError("Extractor Emotion2Vec chưa được load vì EXTRACT_EMOTION2VEC chưa bật.")
    result = extractor.generate(input=str(wav_path), granularity="utterance", extract_embedding=True)
    return find_first_numeric_embedding(result), "ok_extracted"

def build_e2v_patch_for_cache(cache_ids, id_key_for_cache):
    audio_dir = find_audio_dir()
    print("AUDIO_DIR for Emotion2Vec patch:", audio_dir)
    _, light_map = load_light_emotion2vec_cache()
    extractor = load_emotion2vec_extractor_for_patch()
    if extractor is None and not light_map:
        raise RuntimeError(
            "X_e2v trong cache đang zero nhưng chưa có nguồn Emotion2Vec thật. "
            "Hãy set EXTRACT_EMOTION2VEC=1 và INSTALL_EMOTION2VEC_DEPS=1, hoặc upload patch/cache emotion2vec đã trích sẵn."
        )

    all_rows = pd.concat([df for df in SPLIT_TABLES.values()], ignore_index=True)
    all_rows = all_rows.drop_duplicates("train_sample_id").copy()
    by_train = {str(r["train_sample_id"]): r for _, r in all_rows.iterrows()}
    by_utt = {str(r["utterance_id"]): r for _, r in all_rows.iterrows()}

    vectors = []
    statuses = []
    failures = []
    target_dim = None
    for idx, cache_id in enumerate(cache_ids.astype(str)):
        try:
            row = by_train.get(cache_id)
            if row is None:
                row = by_utt.get(cache_id)
            if row is None:
                raise KeyError(f"Không tìm thấy row trong split cho cache id {cache_id}")
            wav_path = resolve_wav_path_from_row(row, audio_dir)
            vec, status = extract_e2v_vector_for_row(row, wav_path, light_map, extractor)
            if target_dim is None:
                target_dim = int(vec.shape[0])
            if vec.shape[0] != target_dim:
                fixed = np.zeros(target_dim, dtype=np.float32)
                n = min(target_dim, vec.shape[0])
                fixed[:n] = vec[:n]
                vec = fixed
                status = status + "_dim_fixed"
            vectors.append(vec.astype(np.float32))
            statuses.append(status)
        except Exception as exc:
            failures.append({"cache_id": cache_id, "error": repr(exc)})
            if os.getenv("REQUIRE_REAL_E2V", "1") == "1":
                break
            if target_dim is None:
                target_dim = int(os.getenv("E2V_DIM", "768"))
            vectors.append(np.zeros(target_dim, dtype=np.float32))
            statuses.append(f"error:{type(exc).__name__}")
        if (idx + 1) % 100 == 0 or idx == len(cache_ids) - 1:
            print(f"Emotion2Vec patch: {idx + 1}/{len(cache_ids)} | failures={len(failures)}")

    if failures and os.getenv("REQUIRE_REAL_E2V", "1") == "1":
        failure_dir = Path(os.getenv("OUTPUT_DIR", "output_03b_timnet_guided_06d_coattention")).resolve() / "features"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / "iemocap_03b_emotion2vec_patch_failures.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
        raise RuntimeError(f"Emotion2Vec patch lỗi ở {len(failures)} mẫu. Xem {failure_path}")

    X_patch = np.stack(vectors).astype(np.float32)
    patch_dir = Path(os.getenv("OUTPUT_DIR", "output_03b_timnet_guided_06d_coattention")).resolve() / "features"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / "iemocap_03b_emotion2vec_patch.npz"
    np.savez_compressed(
        patch_path,
        X_e2v=X_patch,
        train_sample_id=cache_ids.astype(str) if id_key_for_cache != "utterance_id" else np.asarray([], dtype=str),
        sample_ids=cache_ids.astype(str),
        utterance_id=cache_ids.astype(str) if id_key_for_cache == "utterance_id" else np.asarray([], dtype=str),
        emotion2vec_status=np.asarray(statuses),
        model_name=np.asarray([os.getenv("EMOTION2VEC_MODEL_NAME", os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_base"))]),
    )
    status_path = patch_dir / "iemocap_03b_emotion2vec_patch_status.csv"
    pd.DataFrame({"cache_id": cache_ids.astype(str), "status": statuses}).to_csv(status_path, index=False, encoding="utf-8-sig")
    print("Saved Emotion2Vec patch:", patch_path, X_patch.shape)
    return X_patch

def ensure_real_x_e2v(X_e2v_current, cache_ids, id_key_for_cache):
    norms = np.linalg.norm(X_e2v_current, axis=1)
    nonzero = int((norms > 1e-8).sum())
    print(f"Emotion2Vec nonzero rows before patch: {nonzero}/{len(norms)}")
    if nonzero == len(norms):
        print("X_e2v đã có embedding thật, không cần patch.")
        return X_e2v_current.astype(np.float32)
    if os.getenv("PATCH_ZERO_E2V", "1") != "1":
        if os.getenv("REQUIRE_REAL_E2V", "1") == "1":
            raise RuntimeError("X_e2v chưa đủ embedding thật và PATCH_ZERO_E2V=0.")
        print("PATCH_ZERO_E2V=0, tiếp tục dùng X_e2v hiện tại.")
        return X_e2v_current.astype(np.float32)
    patch_path, X_patch = load_e2v_patch_for_cache(cache_ids)
    if X_patch is not None:
        return X_patch.astype(np.float32)
    print("Không có patch Emotion2Vec phù hợp. Bắt đầu trích lại X_e2v từ raw audio...")
    return build_e2v_patch_for_cache(cache_ids, id_key_for_cache)
'''


def main():
    nb = nbf.read(NB_PATH, as_version=4)

    # Remove old auto markdown if the patch is re-run. Do not remove code cells:
    # the code cell also contains the marker after patching.
    nb.cells = [
        cell for cell in nb.cells
        if not (cell.cell_type == "markdown" and "AUTO-03B-E2V-PATCH" in "".join(cell.get("source", "")))
    ]

    # Insert markdown after the data input cell.
    insert_md_at = 6
    nb.cells.insert(insert_md_at, nbf.v4.new_markdown_cell(E2V_MARKDOWN))

    # Patch the main cache loading cell.
    patched = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if "# AUTO-03B-E2V-PATCH" in src:
            start = src.index("# AUTO-03B-E2V-PATCH")
            end_marker = "\n\nid_key = None"
            end = src.index(end_marker, start)
            src = src[:start] + PATCH_CODE.lstrip("\n") + src[end:]
            cell.source = src
            patched = True
            break
        target = 'X_e2v = cache["X_e2v"].astype(np.float32)\n\nid_key = None'
        if target not in src:
            continue
        replacement = 'X_e2v = cache["X_e2v"].astype(np.float32)' + PATCH_CODE + "\n\nid_key = None"
        src = src.replace(target, replacement)
        post_target = '''cache_ids = cache[id_key].astype(str)
feature_index = {sid: i for i, sid in enumerate(cache_ids)}

print("id_key:", id_key)'''
        post_replacement = '''cache_ids = cache[id_key].astype(str)
X_e2v = ensure_real_x_e2v(X_e2v, cache_ids, id_key)
feature_index = {sid: i for i, sid in enumerate(cache_ids)}

print("id_key:", id_key)'''
        if post_target in src:
            src = src.replace(post_target, post_replacement)
        else:
            raise RuntimeError("Không tìm thấy đoạn cache_ids để chèn ensure_real_x_e2v.")
        cell.source = src
        patched = True
        break

    if not patched:
        raise RuntimeError("Không tìm thấy cell load cache 03B để patch.")

    nbf.write(nb, NB_PATH)
    print("Patched", NB_PATH)


if __name__ == "__main__":
    main()
