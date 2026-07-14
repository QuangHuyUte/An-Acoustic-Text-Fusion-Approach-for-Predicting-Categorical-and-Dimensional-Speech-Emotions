from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
OUT_DIR = ROOT / "pretrained_models" / "wav2vec2-base"
OUT_DIR.mkdir(parents=True, exist_ok=True)

snapshot_path = snapshot_download(
    repo_id="facebook/wav2vec2-base",
    local_dir=str(OUT_DIR),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "preprocessor_config.json",
        "feature_extractor_config.json",
    ],
)

print(snapshot_path)
