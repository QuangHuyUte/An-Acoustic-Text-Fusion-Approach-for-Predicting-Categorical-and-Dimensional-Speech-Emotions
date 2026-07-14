import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

light_note = """## 03A update: lightweight pretrained speech branch

Bản update này biến 03A thành nhánh pretrained speech **nhẹ hơn** để có thể chạy trước và tái sử dụng trong fusion sau này.

Thiết lập mặc định:

- `RUN_PROTOCOLS=5fold_session`: chạy 5-fold trước, chưa chạy 10-fold.
- `UNFREEZE_LAST_N=0`: freeze toàn bộ pretrained backbone, chỉ train pooling/adapter/head.
- `EVAL_TRAIN_SPLIT=1`: xuất đủ train/val/test features theo từng fold.
- Output fusion vẫn là `*_pretrained_features.npz`, có thể ghép với 03B/04 sau này.

Ý nghĩa: 03A không còn là full fine-tune nặng. Nó là một pretrained speech feature branch fold-safe, giúp kiểm tra raw-audio pretrained signal và làm nguồn embedding cho fusion.
"""

if not any("03A update: lightweight pretrained speech branch" in "".join(c.get("source", [])) for c in nb["cells"]):
    nb["cells"].insert(1, {"cell_type": "markdown", "metadata": {}, "source": light_note.splitlines(True)})

replacements = {
    'INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"':
        'INSTALL_DEPS = os.getenv("INSTALL_DEPS", "1") == "1"',
    'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile", "librosa", "transformers", "accelerate"])':
        'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-U", "soundfile", "librosa", "transformers==4.51.3", "tokenizers==0.21.1", "huggingface_hub==0.30.2", "accelerate", "safetensors"])',
    'EPOCHS = int(os.getenv("EPOCHS", "20" if IS_TUNE_MODE else "12"))':
        'EPOCHS = int(os.getenv("EPOCHS", "20" if IS_TUNE_MODE else "20"))',
    'PATIENCE = int(os.getenv("PATIENCE", "6" if IS_TUNE_MODE else "4"))':
        'PATIENCE = int(os.getenv("PATIENCE", "6" if IS_TUNE_MODE else "6"))',
    'BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))':
        'BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))',
    'GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))':
        'GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))',
    'LR_BACKBONE = float(os.getenv("LR_BACKBONE", "8e-6"))':
        'LR_BACKBONE = float(os.getenv("LR_BACKBONE", "0.0"))',
    'UNFREEZE_LAST_N = int(os.getenv("UNFREEZE_LAST_N", "4"))':
        'UNFREEZE_LAST_N = int(os.getenv("UNFREEZE_LAST_N", "0"))',
    'RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session" if IS_TUNE_MODE else "5fold_session,10fold_speaker").split(",") if x.strip()]':
        'RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session").split(",") if x.strip()]',
    'EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "0" if IS_TUNE_MODE else "1") == "1"':
        'EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "1") == "1"',
}

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    original = src
    for old, new in replacements.items():
        src = src.replace(old, new)
    if src != original:
        cell["source"] = src.splitlines(True)

save_old = 'torch.save({"model_state_dict": module_state_dict(model), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)'
save_new = 'if best_path.exists():\n                best_path.unlink()\n            torch.save({"model_state_dict": module_state_dict(model), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)'

for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if save_old in src and "best_path.unlink()" not in src:
        src = src.replace(save_old, save_new)
        cell["source"] = src.splitlines(True)

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NB_PATH)
