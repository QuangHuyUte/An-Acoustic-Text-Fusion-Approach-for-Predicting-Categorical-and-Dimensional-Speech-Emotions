from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
OUT_DIR = ROOT / "pretrained_models" / "roberta-base"

OUT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_path = snapshot_download(
    repo_id="roberta-base",
    local_dir=str(OUT_DIR),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ],
)

print(snapshot_path)
