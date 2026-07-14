import json
from pathlib import Path


NOTEBOOK = Path(
    r"06_w2v_based_models\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold"
    r"\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"
)


def replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise SystemExit(f"Pattern not found:\n{old[:400]}")
    return src.replace(old, new, 1)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))

    if "def to_device(batch):" in src and "def multitask_loss" in src:
        src = replace_once(
            src,
            "def multitask_loss(outputs, y, vad_true, class_weight=None):\n",
            "def forward_text_model(model, input_ids, attention_mask, return_embedding=False):\n"
            "    # DataParallel can fail on the final small batch when one GPU receives no samples.\n"
            "    # Fall back to the underlying module for that batch.\n"
            "    if isinstance(model, nn.DataParallel) and input_ids.size(0) < len(model.device_ids):\n"
            "        return model.module(input_ids, attention_mask, return_embedding=return_embedding)\n"
            "    return model(input_ids, attention_mask, return_embedding=return_embedding)\n\n"
            "def multitask_loss(outputs, y, vad_true, class_weight=None):\n",
        )
        src = src.replace(
            'outputs = model(batch["input_ids"], batch["attention_mask"], return_embedding=return_features)',
            'outputs = forward_text_model(model, batch["input_ids"], batch["attention_mask"], return_embedding=return_features)',
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "def train_fold(protocol, fold_name, fold_df, seed):" in src:
        src = src.replace(
            'outputs = model(batch["input_ids"], batch["attention_mask"])',
            'outputs = forward_text_model(model, batch["input_ids"], batch["attention_mask"])',
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK)
