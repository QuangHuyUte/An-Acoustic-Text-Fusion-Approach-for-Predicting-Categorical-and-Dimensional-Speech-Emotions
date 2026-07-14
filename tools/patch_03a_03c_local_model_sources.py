import json
from pathlib import Path


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_03C = ROOT / (
    r"06_w2v_based_models\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold"
    r"\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"
)
NB_03A = ROOT / (
    r"06_w2v_based_models\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold"
    r"\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
)


def replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise SystemExit(f"Pattern not found:\n{old[:500]}")
    return src.replace(old, new, 1)


def patch_nb(path: Path, patcher):
    nb = json.loads(path.read_text(encoding="utf-8"))
    patcher(nb)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path)


def patch_03c(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "TEXT_MODEL_NAME = os.getenv" in src and "03C configuration is ready" in src:
            src = replace_once(
                src,
                "TEXT_MODEL_NAME = os.getenv(\"TEXT_MODEL_NAME\", \"roberta-base\")\n",
                "TEXT_MODEL_NAME = os.getenv(\"TEXT_MODEL_NAME\", \"roberta-base\")\n"
                "TEXT_MODEL_PATH = os.getenv(\"TEXT_MODEL_PATH\", \"\").strip()\n"
                "ALLOW_HF_DOWNLOAD = os.getenv(\"ALLOW_HF_DOWNLOAD\", \"0\") == \"1\"\n"
                "TEXT_TUNE_LAST_N = int(os.getenv(\"TEXT_TUNE_LAST_N\", \"2\"))\n",
            )
            src = replace_once(
                src,
                "    \"TEXT_MODEL_NAME\": TEXT_MODEL_NAME,\n",
                "    \"TEXT_MODEL_NAME\": TEXT_MODEL_NAME,\n"
                "    \"TEXT_MODEL_PATH\": TEXT_MODEL_PATH,\n"
                "    \"ALLOW_HF_DOWNLOAD\": ALLOW_HF_DOWNLOAD,\n"
                "    \"TEXT_TUNE_LAST_N\": TEXT_TUNE_LAST_N,\n",
            )
            src = replace_once(
                src,
                "live_log(\"03C configuration is ready.\")",
                r'''
def maybe_extract_model_zips_03c():
    extract_dir = Path("extracted_03c_models")
    extract_dir.mkdir(parents=True, exist_ok=True)
    for root in search_roots():
        try:
            zips = list(root.rglob("*.zip"))
        except Exception:
            continue
        for z in zips:
            lname = z.name.lower()
            if not any(k in lname for k in ["roberta", "deberta", "text", "pretrained", "huggingface", "hf-model"]):
                continue
            target = extract_dir / z.stem
            marker = target / ".extracted"
            if marker.exists():
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zipf:
                    zipf.extractall(target)
                marker.write_text("ok", encoding="utf-8")
                live_log(f"Extracted local text model zip: {z}")
            except Exception as exc:
                live_log(f"Không giải nén được text model zip {z}: {exc}")

def _norm_model_name(value):
    return str(value).lower().replace("\\", "/").replace("_", "-").replace(".", "-")

def _looks_like_hf_text_model(path):
    path = Path(path)
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weight = any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
    has_tokenizer = any((path / name).exists() for name in [
        "tokenizer.json", "vocab.json", "vocab.txt", "merges.txt", "spm.model", "sentencepiece.bpe.model"
    ])
    return has_config and has_weight and has_tokenizer

def find_local_text_model(model_name):
    maybe_extract_model_zips_03c()
    if TEXT_MODEL_PATH:
        p = Path(TEXT_MODEL_PATH)
        if _looks_like_hf_text_model(p):
            return p.resolve()
        raise FileNotFoundError(
            f"TEXT_MODEL_PATH không hợp lệ: {p}. Cần folder Hugging Face có config.json, weight file và tokenizer."
        )
    wanted = _norm_model_name(model_name).split("/")[-1]
    candidates = []
    for root in search_roots() + [Path("extracted_03c_models")]:
        try:
            for cfg in root.rglob("config.json"):
                folder = cfg.parent
                if _looks_like_hf_text_model(folder):
                    score = 0 if wanted in _norm_model_name(folder) else 1
                    candidates.append((score, len(folder.parts), str(folder).lower(), folder.resolve()))
        except Exception:
            continue
    if candidates:
        return sorted(candidates)[0][-1]
    return None

TEXT_MODEL_SOURCE = find_local_text_model(TEXT_MODEL_NAME)
if TEXT_MODEL_SOURCE is None:
    if not ALLOW_HF_DOWNLOAD:
        roots = "\n".join(f"- {r}" for r in search_roots())
        raise FileNotFoundError(
            "03C không tìm thấy local text model. Notebook chặn tải online mặc định để tránh kẹt 0.00/499M.\n"
            "Hãy upload roberta-base-kaggle.zip vào Kaggle Input, hoặc set TEXT_MODEL_PATH tới folder model.\n"
            "Nếu vẫn muốn tải online, set ALLOW_HF_DOWNLOAD=1.\n\n"
            f"Đã quét:\n{roots}"
        )
    TEXT_MODEL_SOURCE = TEXT_MODEL_NAME
else:
    TEXT_MODEL_SOURCE = str(TEXT_MODEL_SOURCE)

live_log("03C configuration is ready.")
live_log(f"TEXT_MODEL_SOURCE={TEXT_MODEL_SOURCE}")
live_log(f"ALLOW_HF_DOWNLOAD={ALLOW_HF_DOWNLOAD}")
live_log(f"TEXT_TUNE_LAST_N={TEXT_TUNE_LAST_N}")
'''.strip(),
            )
            cell["source"] = src.splitlines(keepends=True)
            continue

        if "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)" in src:
            src = src.replace(
                "live_log(f\"Loading tokenizer: {TEXT_MODEL_NAME}\")\n"
                "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)\n"
                "live_log(\"Tokenizer loaded.\")",
                "live_log(f\"Loading tokenizer from: {TEXT_MODEL_SOURCE}\")\n"
                "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_SOURCE, local_files_only=not ALLOW_HF_DOWNLOAD)\n"
                "live_log(\"Tokenizer loaded.\")",
            )
            cell["source"] = src.splitlines(keepends=True)
            continue

        if "class TranscriptMultiTaskSER" in src and "AutoModel.from_pretrained(model_name)" in src:
            src = src.replace(
                "        self.backbone = AutoModel.from_pretrained(model_name)\n",
                "        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=not ALLOW_HF_DOWNLOAD)\n",
            )
            src = replace_once(
                src,
                "        hidden = self.backbone.config.hidden_size\n",
                "        self.configure_light_tuning()\n"
                "        hidden = self.backbone.config.hidden_size\n",
            )
            src = replace_once(
                src,
                "    def forward(self, input_ids, attention_mask, return_embedding=False):\n",
                "    def configure_light_tuning(self):\n"
                "        # 03C là text branch tune nhẹ: mặc định chỉ mở 2 layer cuối của RoBERTa.\n"
                "        # Set TEXT_TUNE_LAST_N=-1 nếu muốn full fine-tune; set 0 nếu muốn frozen backbone.\n"
                "        if TEXT_TUNE_LAST_N >= 0:\n"
                "            for p in self.backbone.parameters():\n"
                "                p.requires_grad = False\n"
                "            layers = getattr(getattr(self.backbone, \"encoder\", None), \"layer\", None)\n"
                "            if layers is not None and TEXT_TUNE_LAST_N > 0:\n"
                "                for layer in layers[-TEXT_TUNE_LAST_N:]:\n"
                "                    for p in layer.parameters():\n"
                "                        p.requires_grad = True\n"
                "            live_log(f\"Text backbone light tuning: last {TEXT_TUNE_LAST_N} layers trainable.\")\n"
                "        else:\n"
                "            for p in self.backbone.parameters():\n"
                "                p.requires_grad = True\n"
                "            live_log(\"Text backbone full fine-tuning enabled.\")\n\n"
                "    def forward(self, input_ids, attention_mask, return_embedding=False):\n",
            )
            src = replace_once(
                src,
                "preview_model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT)\n"
                "print(\"Trainable parameters:\", sum(p.numel() for p in preview_model.parameters() if p.requires_grad))\n"
                "del preview_model\n"
                "torch.cuda.empty_cache()",
                "live_log(\"Skip heavy preview_model construction; model will be built inside each fold.\")",
            )
            cell["source"] = src.splitlines(keepends=True)
            continue

        if "Build TranscriptMultiTaskSER with backbone={TEXT_MODEL_NAME}" in src:
            src = src.replace(
                "    live_log(f\"Build TranscriptMultiTaskSER with backbone={TEXT_MODEL_NAME}\")\n"
                "    model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)\n",
                "    live_log(f\"Build TranscriptMultiTaskSER with backbone={TEXT_MODEL_SOURCE}\")\n"
                "    model = TranscriptMultiTaskSER(TEXT_MODEL_SOURCE, num_classes=4, dropout=DROPOUT).to(DEVICE)\n",
            )
            src = src.replace(
                "\"params\": [p for n, p in target.backbone.named_parameters() if not any(nd in n for nd in no_decay)],",
                "\"params\": [p for n, p in target.backbone.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],",
            )
            src = src.replace(
                "\"params\": [p for n, p in target.backbone.named_parameters() if any(nd in n for nd in no_decay)],",
                "\"params\": [p for n, p in target.backbone.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],",
            )
            cell["source"] = src.splitlines(keepends=True)
            continue


def patch_03a(nb):
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))

        if "PRETRAINED_MODEL_NAME = os.getenv" in src and "03A configuration is ready" in src:
            src = replace_once(
                src,
                "PRETRAINED_MODEL_NAME = os.getenv(\"PRETRAINED_MODEL_NAME\", \"facebook/wav2vec2-base\")\n",
                "PRETRAINED_MODEL_NAME = os.getenv(\"PRETRAINED_MODEL_NAME\", \"facebook/wav2vec2-base\")\n"
                "PRETRAINED_MODEL_PATH = os.getenv(\"PRETRAINED_MODEL_PATH\", \"\").strip()\n"
                "ALLOW_HF_DOWNLOAD = os.getenv(\"ALLOW_HF_DOWNLOAD\", \"0\") == \"1\"\n",
            )
            src = replace_once(
                src,
                "print(\"Pretrained model:\", PRETRAINED_MODEL_NAME)\n",
                "print(\"Pretrained model:\", PRETRAINED_MODEL_NAME)\n"
                "print(\"Pretrained model path:\", PRETRAINED_MODEL_PATH)\n"
                "print(\"Allow HF download:\", ALLOW_HF_DOWNLOAD)\n",
            )
            src = replace_once(
                src,
                "    \"LOG_EVERY_STEPS\": LOG_EVERY_STEPS,\n",
                "    \"LOG_EVERY_STEPS\": LOG_EVERY_STEPS,\n"
                "    \"PRETRAINED_MODEL_PATH\": PRETRAINED_MODEL_PATH,\n"
                "    \"ALLOW_HF_DOWNLOAD\": ALLOW_HF_DOWNLOAD,\n",
            )
            src = replace_once(
                src,
                "live_log(\"03A configuration is ready.\")",
                r'''
def maybe_extract_model_zips_03a():
    extract_dir = Path("extracted_03a_models")
    extract_dir.mkdir(parents=True, exist_ok=True)
    for root in search_roots():
        try:
            zips = list(root.rglob("*.zip"))
        except Exception:
            continue
        for z in zips:
            lname = z.name.lower()
            if not any(k in lname for k in ["wav2vec", "wavlm", "hubert", "speech", "pretrained", "huggingface", "hf-model"]):
                continue
            target = extract_dir / z.stem
            marker = target / ".extracted"
            if marker.exists():
                continue
            try:
                target.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zipf:
                    zipf.extractall(target)
                marker.write_text("ok", encoding="utf-8")
                live_log(f"Extracted local speech model zip: {z}")
            except Exception as exc:
                live_log(f"Không giải nén được speech model zip {z}: {exc}")

def _norm_model_name(value):
    return str(value).lower().replace("\\", "/").replace("_", "-").replace(".", "-")

def _looks_like_hf_speech_model(path):
    path = Path(path)
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weight = any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
    has_feature = any((path / name).exists() for name in ["preprocessor_config.json", "feature_extractor_config.json"])
    return has_config and has_weight and has_feature

def find_local_speech_model(model_name):
    maybe_extract_model_zips_03a()
    if PRETRAINED_MODEL_PATH:
        p = Path(PRETRAINED_MODEL_PATH)
        if _looks_like_hf_speech_model(p):
            return p.resolve()
        raise FileNotFoundError(
            f"PRETRAINED_MODEL_PATH không hợp lệ: {p}. Cần folder có config.json, weight file và preprocessor_config.json."
        )
    wanted = _norm_model_name(model_name).split("/")[-1]
    candidates = []
    for root in search_roots() + [Path("extracted_03a_models")]:
        try:
            for cfg in root.rglob("config.json"):
                folder = cfg.parent
                if _looks_like_hf_speech_model(folder):
                    score = 0 if wanted in _norm_model_name(folder) else 1
                    candidates.append((score, len(folder.parts), str(folder).lower(), folder.resolve()))
        except Exception:
            continue
    if candidates:
        return sorted(candidates)[0][-1]
    return None

PRETRAINED_MODEL_SOURCE = find_local_speech_model(PRETRAINED_MODEL_NAME)
if PRETRAINED_MODEL_SOURCE is None:
    if not ALLOW_HF_DOWNLOAD:
        roots = "\n".join(f"- {r}" for r in search_roots())
        raise FileNotFoundError(
            "03A không tìm thấy local pretrained speech model. Notebook chặn tải online mặc định để tránh treo.\n"
            "Hãy upload wav2vec2-base-kaggle.zip vào Kaggle Input, hoặc set PRETRAINED_MODEL_PATH tới folder model.\n"
            "Nếu vẫn muốn tải online, set ALLOW_HF_DOWNLOAD=1.\n\n"
            f"Đã quét:\n{roots}"
        )
    PRETRAINED_MODEL_SOURCE = PRETRAINED_MODEL_NAME
else:
    PRETRAINED_MODEL_SOURCE = str(PRETRAINED_MODEL_SOURCE)

live_log("03A configuration is ready.")
live_log(f"PRETRAINED_MODEL_SOURCE={PRETRAINED_MODEL_SOURCE}")
live_log(f"ALLOW_HF_DOWNLOAD={ALLOW_HF_DOWNLOAD}")
'''.strip(),
            )
            cell["source"] = src.splitlines(keepends=True)
            continue

        if "FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)" in src:
            src = src.replace(
                "FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_NAME)",
                "live_log(f\"Loading feature extractor from: {PRETRAINED_MODEL_SOURCE}\")\n"
                "FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL_SOURCE, local_files_only=not ALLOW_HF_DOWNLOAD)\n"
                "live_log(\"Feature extractor loaded.\")",
            )
            cell["source"] = src.splitlines(keepends=True)
            continue

        if "class RawBackboneMultiTaskSER" in src and "AutoModel.from_pretrained(model_name)" in src:
            src = src.replace(
                "        self.backbone = AutoModel.from_pretrained(model_name)\n",
                "        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=not ALLOW_HF_DOWNLOAD)\n",
            )
            src = src.replace(
                "    model = RawBackboneMultiTaskSER(PRETRAINED_MODEL_NAME, HIDDEN_DIM, DROPOUT).to(DEVICE)\n",
                "    model = RawBackboneMultiTaskSER(PRETRAINED_MODEL_SOURCE, HIDDEN_DIM, DROPOUT).to(DEVICE)\n",
            )
            cell["source"] = src.splitlines(keepends=True)
            continue


patch_nb(NB_03C, patch_03c)
patch_nb(NB_03A, patch_03a)
