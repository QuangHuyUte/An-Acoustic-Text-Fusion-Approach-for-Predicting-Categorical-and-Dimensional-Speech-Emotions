from pathlib import Path
from zipfile import ZipFile, ZIP_STORED


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
SRC = ROOT / "pretrained_models" / "wav2vec2-base-kaggle"
RAW = ROOT / "pretrained_models" / "wav2vec2-base"
ZIP_PATH = ROOT / "pretrained_models" / "wav2vec2-base-kaggle.zip"

SRC.mkdir(parents=True, exist_ok=True)
for name in ["config.json", "pytorch_model.bin", "preprocessor_config.json"]:
    (SRC / name).write_bytes((RAW / name).read_bytes())

if ZIP_PATH.exists():
    ZIP_PATH.unlink()

with ZipFile(ZIP_PATH, "w", compression=ZIP_STORED, allowZip64=True) as zf:
    for path in sorted(SRC.rglob("*")):
        if path.is_file():
            arcname = path.relative_to(SRC.parent)
            print("Adding", arcname)
            zf.write(path, arcname.as_posix())

print(ZIP_PATH)
print(ZIP_PATH.stat().st_size)
