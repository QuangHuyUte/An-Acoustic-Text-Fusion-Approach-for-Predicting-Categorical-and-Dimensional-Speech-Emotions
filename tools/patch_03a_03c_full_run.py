from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_03A = ROOT / "06_w2v_based_models" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
NB_03C = ROOT / "06_w2v_based_models" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"


FULL_MARKDOWN_03A = """<!-- AUTO-FULL-RUN-GUARD -->
## Chế độ chạy full

Notebook này hiện mặc định chạy full:

```text
RUN_MODE=full
RUN_PROTOCOLS=5fold_session,10fold_speaker
MAX_FOLDS=0
EVAL_TRAIN_SPLIT=1
```

Nếu muốn notebook dừng ngay khi cấu hình chưa phải full, set:

```text
REQUIRE_FULL_RUN=1
```

03A cần raw audio IEMOCAP full release vì pretrained speech backbone nhận waveform trực tiếp.
"""


FULL_MARKDOWN_03C = """<!-- AUTO-FULL-RUN-GUARD -->
## Chế độ chạy full

Notebook này mặc định chạy full 5-fold + 10-fold cho nhánh transcript:

```text
RUN_MODE=full
RUN_PROTOCOLS=5fold_session,10fold_speaker
MAX_FOLDS=0
EVAL_TRAIN_SPLIT=1
```

Nếu muốn notebook dừng ngay khi cấu hình chưa phải full, set:

```text
REQUIRE_FULL_RUN=1
```

03C không cần raw audio. Nó chỉ cần metadata/splits/text-ready từ output 02.
"""


FULL_GUARD = """

REQUIRE_FULL_RUN = os.getenv("REQUIRE_FULL_RUN", "0") == "1"
if REQUIRE_FULL_RUN:
    expected_protocols = {"5fold_session", "10fold_speaker"}
    if RUN_MODE != "full" or MAX_FOLDS != 0 or set(RUN_PROTOCOLS) != expected_protocols or not EVAL_TRAIN_SPLIT:
        raise ValueError(
            "REQUIRE_FULL_RUN=1 nhưng cấu hình chưa phải full. "
            f"RUN_MODE={RUN_MODE}, MAX_FOLDS={MAX_FOLDS}, RUN_PROTOCOLS={RUN_PROTOCOLS}, "
            f"EVAL_TRAIN_SPLIT={EVAL_TRAIN_SPLIT}. "
            "Cần RUN_MODE=full, MAX_FOLDS=0, RUN_PROTOCOLS=5fold_session,10fold_speaker, EVAL_TRAIN_SPLIT=1."
        )
"""


def remove_marker(nb):
    nb.cells = [
        cell for cell in nb.cells
        if not (cell.cell_type == "markdown" and "AUTO-FULL-RUN-GUARD" in "".join(cell.get("source", "")))
    ]


def patch_03a():
    nb = nbf.read(NB_03A, as_version=4)
    remove_marker(nb)
    nb.cells.insert(1, nbf.v4.new_markdown_cell(FULL_MARKDOWN_03A))
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if 'RUN_MODE = os.getenv("RUN_MODE", "tune").strip().lower()' in src:
            src = src.replace('RUN_MODE = os.getenv("RUN_MODE", "tune").strip().lower()', 'RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()')
            src = src.replace('OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03a_pretrained_backbone_tuned")).resolve()', 'OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03a_pretrained_backbone")).resolve()')
            marker = 'EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "0" if IS_TUNE_MODE else "1") == "1"\n'
            if "REQUIRE_FULL_RUN = os.getenv" not in src:
                src = src.replace(marker, marker + FULL_GUARD)
            cell.source = src
            break
    else:
        raise RuntimeError("Không tìm thấy cell config của 03A.")
    nbf.write(nb, NB_03A)


def patch_03c():
    nb = nbf.read(NB_03C, as_version=4)
    remove_marker(nb)
    nb.cells.insert(1, nbf.v4.new_markdown_cell(FULL_MARKDOWN_03C))
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source)
        if 'RUN_MODE = os.getenv("RUN_MODE", "full").strip().lower()' in src and 'TEXT_MODEL_NAME' in src:
            marker = 'EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "1" if not IS_TUNE_MODE else "0") == "1"\n'
            if "REQUIRE_FULL_RUN = os.getenv" not in src:
                src = src.replace(marker, marker + FULL_GUARD)
            cell.source = src
            break
    else:
        raise RuntimeError("Không tìm thấy cell config của 03C.")
    nbf.write(nb, NB_03C)


def main():
    patch_03a()
    patch_03c()
    print("Patched full-run defaults:", NB_03A)
    print("Patched full-run guard:", NB_03C)


if __name__ == "__main__":
    main()
