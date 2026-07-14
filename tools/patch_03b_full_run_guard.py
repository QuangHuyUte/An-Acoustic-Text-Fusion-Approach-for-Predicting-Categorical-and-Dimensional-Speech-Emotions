from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_03B = ROOT / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"


FULL_MARKDOWN_03B = """<!-- AUTO-03B-FULL-RUN-GUARD -->
## Chế độ chạy full cho 03B

Notebook 03B hiện mặc định chạy full:

```text
RUN_PROTOCOLS=5fold_session,10fold_speaker
MAX_FOLDS=0
PATCH_ZERO_E2V=1
EXTRACT_EMOTION2VEC=1
REQUIRE_REAL_E2V=1
```

Nếu muốn notebook dừng ngay khi cấu hình chưa phải full, set:

```text
REQUIRE_FULL_RUN=1
```

Với output 02 mới, `X_e2v` đã phải là Emotion2Vec thật. Cơ chế patch trong 03B chỉ là lớp bảo vệ cuối nếu cache đầu vào vẫn bị zero.
"""


FULL_GUARD = """

REQUIRE_FULL_RUN = os.getenv("REQUIRE_FULL_RUN", "0") == "1"
if REQUIRE_FULL_RUN:
    expected_protocols = {"5fold_session", "10fold_speaker"}
    if MAX_FOLDS != 0 or set(RUN_PROTOCOLS) != expected_protocols:
        raise ValueError(
            "REQUIRE_FULL_RUN=1 nhưng 03B chưa cấu hình full. "
            f"MAX_FOLDS={MAX_FOLDS}, RUN_PROTOCOLS={RUN_PROTOCOLS}. "
            "Cần MAX_FOLDS=0 và RUN_PROTOCOLS=5fold_session,10fold_speaker."
        )
"""


def main():
    nb = nbf.read(NB_03B, as_version=4)
    nb.cells = [
        cell for cell in nb.cells
        if not (cell.cell_type == "markdown" and "AUTO-03B-FULL-RUN-GUARD" in "".join(cell.get("source", "")))
    ]
    insert_at = 0
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "## Data input cần có" in "".join(cell.get("source", "")):
            insert_at = idx + 1
            break
    nb.cells.insert(insert_at, nbf.v4.new_markdown_cell(FULL_MARKDOWN_03B))

    patched = False
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        marker = 'RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]\n'
        if marker in src:
            if "REQUIRE_FULL_RUN = os.getenv" not in src:
                src = src.replace(marker, marker + FULL_GUARD)
            cell.source = src
            patched = True
            break
    if not patched:
        raise RuntimeError("Không tìm thấy cell config của 03B.")
    nbf.write(nb, NB_03B)
    print("Patched 03B full-run guard:", NB_03B)


if __name__ == "__main__":
    main()
