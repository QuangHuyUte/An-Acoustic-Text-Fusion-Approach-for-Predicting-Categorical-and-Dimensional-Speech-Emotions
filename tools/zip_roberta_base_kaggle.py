from pathlib import Path
from zipfile import ZipFile, ZIP_STORED


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
SRC = ROOT / "pretrained_models" / "roberta-base-kaggle"
ZIP_PATH = ROOT / "pretrained_models" / "roberta-base-kaggle.zip"

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
