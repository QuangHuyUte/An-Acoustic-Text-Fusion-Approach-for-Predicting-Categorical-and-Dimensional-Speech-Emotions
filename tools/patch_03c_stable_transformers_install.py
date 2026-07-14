import json
from pathlib import Path


PROJECT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = PROJECT / "06_w2v_based_models" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold" / "03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

old = '''INSTALL_DEPS = os.getenv("INSTALL_DEPS", "0") == "1"
if INSTALL_DEPS:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "sentencepiece", "accelerate"])
'''

new = '''INSTALL_DEPS = os.getenv("INSTALL_DEPS", "1") == "1"
if INSTALL_DEPS:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "-U",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface_hub==0.30.2",
        "sentencepiece",
        "accelerate",
        "safetensors",
    ])
'''

patched = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if old in src:
        src = src.replace(old, new)
        cell["source"] = src.splitlines(True)
        patched = True
        break

if not patched:
    raise RuntimeError("Không tìm thấy cell INSTALL_DEPS trong 03C để vá.")

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NB_PATH)
