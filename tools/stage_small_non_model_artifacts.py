import subprocess
from pathlib import Path


ROOT = Path.cwd()
ALLOWED_SUFFIXES = {
    ".ipynb",
    ".py",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".jsonl",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".html",
    ".css",
    ".js",
    ".cmd",
    ".xlsx",
}
BLOCKED_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".bin",
    ".onnx",
    ".pkl",
    ".joblib",
    ".npz",
    ".zip",
    ".rar",
    ".wav",
    ".mp3",
    ".mp4",
}
BLOCKED_PARTS = {
    ".git",
    "models",
    "cache",
    "__pycache__",
    ".ipynb_checkpoints",
    "node_modules",
    "pretrained_models",
}
MAX_BYTES = 95_000_000


def should_stage(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(ROOT)
    parts_lower = {part.lower() for part in rel.parts}
    if parts_lower & BLOCKED_PARTS:
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    if path.stat().st_size > MAX_BYTES:
        return False
    return True


def main() -> None:
    candidates = [str(path.relative_to(ROOT)) for path in ROOT.rglob("*") if should_stage(path)]
    print("candidate files:", len(candidates))
    for idx in range(0, len(candidates), 100):
        subprocess.run(["git", "add", "-f", "--", *candidates[idx : idx + 100]], check=True)
    print("force add done")


if __name__ == "__main__":
    main()
