import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

note = """## Kaggle-safe model default

Notebook 03A trước đó mặc định dùng `microsoft/wavlm-base-plus`, nhưng Kaggle có thể bị kẹt ở bước tải `pytorch_model.bin`.

Bản này đổi mặc định sang:

```text
PRETRAINED_MODEL_NAME=facebook/wav2vec2-base
```

`wav2vec2-base` vẫn là pretrained raw-audio backbone, nhẹ và thường tải ổn hơn trên Kaggle. Nếu muốn thử lại WavLM, set:

```text
PRETRAINED_MODEL_NAME=microsoft/wavlm-base-plus
```
"""

if not any("Kaggle-safe model default" in "".join(c.get("source", [])) for c in nb["cells"]):
    insert_at = 2 if len(nb["cells"]) > 2 else 1
    nb["cells"].insert(insert_at, {"cell_type": "markdown", "metadata": {}, "source": note.splitlines(True)})

replacements = {
    'PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "microsoft/wavlm-base-plus")':
        'PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "facebook/wav2vec2-base")',
    "Mặc định notebook dùng `microsoft/wavlm-base-plus` vì HuggingFace hỗ trợ fine-tune raw waveform ổn định.":
        "Mặc định notebook dùng `facebook/wav2vec2-base` để tải ổn hơn trên Kaggle; có thể đổi sang `microsoft/wavlm-base-plus` nếu cache/download ổn định.",
}

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    original = src
    for old, new in replacements.items():
        src = src.replace(old, new)
    if src != original:
        cell["source"] = src.splitlines(True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NB_PATH)
