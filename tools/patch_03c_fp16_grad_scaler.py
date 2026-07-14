import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"


HELPERS = r'''

def force_trainable_parameters_fp32(model):
    """GradScaler needs trainable parameter gradients in FP32; autocast handles FP16 activations."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.float()
    bad = [(name, str(param.dtype)) for name, param in target.named_parameters() if param.requires_grad and param.dtype != torch.float32]
    if bad:
        preview = bad[:10]
        raise ValueError(f"Trainable parameters must stay FP32 when using GradScaler. Bad dtype examples: {preview}")
    return model


def amp_enabled():
    return USE_AMP and DEVICE.type == "cuda"

'''


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    changed = False

    # Insert helper before train_fold cell if absent.
    if not any("def force_trainable_parameters_fp32" in "".join(c.get("source", "")) for c in nb["cells"]):
        for idx, cell in enumerate(nb["cells"]):
            source = "".join(cell.get("source", ""))
            if cell.get("cell_type") == "code" and "PROTOCOLS = []" in source and "def train_fold" in source:
                nb["cells"].insert(idx, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": HELPERS})
                changed = True
                break

    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", ""))
        updated = source

        updated = updated.replace(
            'model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)\n    if USE_DATA_PARALLEL:',
            'model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)\n    model = force_trainable_parameters_fp32(model)\n    if USE_DATA_PARALLEL:',
        )
        updated = updated.replace(
            '    if USE_DATA_PARALLEL:\n        model = nn.DataParallel(model)\n\n    no_decay =',
            '    if USE_DATA_PARALLEL:\n        model = nn.DataParallel(model)\n    model = force_trainable_parameters_fp32(model)\n\n    no_decay =',
        )
        updated = updated.replace(
            'scaler = GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")',
            'scaler = GradScaler("cuda", enabled=amp_enabled())',
        )
        updated = updated.replace(
            'with autocast(device_type="cuda", enabled=USE_AMP and DEVICE.type == "cuda"):',
            'with autocast(device_type="cuda", enabled=amp_enabled()):',
        )

        if updated != source:
            cell["source"] = updated
            changed = True

    if not changed:
        raise RuntimeError("Không tìm thấy đoạn cần vá trong notebook 03C.")

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Patched {NB_PATH}")


if __name__ == "__main__":
    main()
