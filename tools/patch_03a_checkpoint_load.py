import json
from pathlib import Path


NOTEBOOK = Path(
    r"06_w2v_based_models\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold"
    r"\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
)

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "torch.save({\"model_state_dict\": module_state_dict(model)" in src:
        src = src.replace(
            "torch.save({\"model_state_dict\": module_state_dict(model), \"best_epoch\": best_epoch, \"best_val_score\": best_score}, best_path)",
            "torch.save({\"model_state_dict\": module_state_dict(model), \"best_epoch\": int(best_epoch), \"best_val_score\": float(best_score)}, best_path)",
        )
        src = src.replace(
            "checkpoint = torch.load(best_path, map_location=DEVICE)",
            "checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=False)",
        )
        cell["source"] = src.splitlines(keepends=True)

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK)
