import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

replacements = {
    'TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "microsoft/deberta-v3-base")':
        'TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "roberta-base")',
}

insert_note = (
    "\n\n> Kaggle note: notebook mặc định dùng `roberta-base` để tránh lỗi/kẹt tải "
    "`spm.model` thường gặp với DeBERTaV3. Nếu môi trường ổn định, có thể set "
    "`TEXT_MODEL_NAME=microsoft/deberta-v3-base` để thử lại DeBERTaV3.\n"
)

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    original = src
    for old, new in replacements.items():
        src = src.replace(old, new)
    if cell.get("cell_type") == "markdown" and "# 03C" in src and "spm.model" not in src:
        src += insert_note
    if src != original:
        cell["source"] = src.splitlines(True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NB_PATH)
