from pathlib import Path

import nbformat


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
NB_PATH = ROOT / "06_w2v_based_models" / "03_Emotion2Vec RawAudio Backbone Finetune 5Fold 10Fold" / "03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"


CELL3 = """## Cách chạy khuyến nghị sau khi thấy run cũ quá chậm/yếu

Run cũ đang fine-tune last 4 layer trong 8 epoch và chạy nhiều fold ngay từ đầu. Với raw-audio backbone, cách đó rất tốn thời gian nhưng chưa đủ kiểm soát để biết model đang yếu vì hyperparameter, pooling, split hay do backbone.

Notebook này được chỉnh lại theo hai giai đoạn:

### Giai đoạn A - tune nhanh nhưng học kỹ hơn

Mặc định:

```text
RUN_MODE=tune
RUN_PROTOCOLS=5fold_session
MAX_FOLDS=1
EPOCHS=20
PATIENCE=6
MAX_SECONDS=4.0
```

Mục tiêu là xem 1 fold có vượt rõ baseline cũ không trước khi đốt GPU cho toàn bộ 5-fold/10-fold.

### Giai đoạn B - chạy full protocol

Khi cấu hình đã ổn, set:

```text
RUN_MODE=full
RUN_PROTOCOLS=5fold_session,10fold_speaker
MAX_FOLDS=0
EPOCHS=12
MAX_SECONDS=6.0
EVAL_TRAIN_SPLIT=1
```

Output `fusion_features` của train/val/test theo từng fold dùng để đưa sang notebook 04.

Các thay đổi so với bản cũ:

- Attention statistics pooling thay cho mean pooling.
- Class weights + label smoothing cho emotion classification.
- Warmup + cosine scheduler theo step.
- Early stopping thật sự bằng `PATIENCE`.
- DataLoader có `NUM_WORKERS` để giảm nghẽn khi đọc/resample wav.
"""


CELL7 = """AUDIO_DIR = find_audio_dir()
print("AUDIO_DIR:", AUDIO_DIR)
print("SPLIT_5FOLD_PATH:", SPLIT_5FOLD_PATH)
print("SPLIT_10FOLD_PATH:", SPLIT_10FOLD_PATH)

RUN_MODE = os.getenv("RUN_MODE", "tune").strip().lower()
IS_TUNE_MODE = RUN_MODE != "full"

PRETRAINED_MODEL_NAME = os.getenv("PRETRAINED_MODEL_NAME", "microsoft/wavlm-base-plus")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
MAX_SECONDS = float(os.getenv("MAX_SECONDS", "4.0" if IS_TUNE_MODE else "6.0"))
MAX_SAMPLES = int(SAMPLE_RATE * MAX_SECONDS)

EPOCHS = int(os.getenv("EPOCHS", "20" if IS_TUNE_MODE else "12"))
PATIENCE = int(os.getenv("PATIENCE", "6" if IS_TUNE_MODE else "4"))
MIN_DELTA = float(os.getenv("MIN_DELTA", "0.002"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "2"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))

LR_BACKBONE = float(os.getenv("LR_BACKBONE", "8e-6"))
LR_HEAD = float(os.getenv("LR_HEAD", "5e-4"))
MIN_LR_RATIO = float(os.getenv("MIN_LR_RATIO", "0.05"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.08"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.30"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "256"))
UNFREEZE_LAST_N = int(os.getenv("UNFREEZE_LAST_N", "4"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
USE_SCHEDULER = os.getenv("USE_SCHEDULER", "1") == "1"

MAX_FOLDS = int(os.getenv("MAX_FOLDS", "1" if IS_TUNE_MODE else "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session" if IS_TUNE_MODE else "5fold_session,10fold_speaker").split(",") if x.strip()]
EVAL_TRAIN_SPLIT = os.getenv("EVAL_TRAIN_SPLIT", "0" if IS_TUNE_MODE else "1") == "1"

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03a_pretrained_backbone_tuned")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:
    p.mkdir(parents=True, exist_ok=True)

print("Run mode:", RUN_MODE)
print("Pretrained model:", PRETRAINED_MODEL_NAME)
print("Output:", OUTPUT_DIR)
print({
    "MAX_SECONDS": MAX_SECONDS,
    "EPOCHS": EPOCHS,
    "PATIENCE": PATIENCE,
    "BATCH_SIZE": BATCH_SIZE,
    "GRAD_ACCUM": GRAD_ACCUM,
    "NUM_WORKERS": NUM_WORKERS,
    "LR_BACKBONE": LR_BACKBONE,
    "LR_HEAD": LR_HEAD,
    "UNFREEZE_LAST_N": UNFREEZE_LAST_N,
    "LABEL_SMOOTHING": LABEL_SMOOTHING,
    "RUN_PROTOCOLS": RUN_PROTOCOLS,
    "MAX_FOLDS": MAX_FOLDS,
    "EVAL_TRAIN_SPLIT": EVAL_TRAIN_SPLIT,
})
"""


CELL8 = """def vad_to_0_1(values):
    return np.clip((values.astype(np.float32) - 1.0) / 4.0, 0.0, 1.0)

def vad_from_0_1(values):
    return values.astype(np.float32) * 4.0 + 1.0

def concordance_ccc_torch(pred, target, eps=1e-8):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    pred_var = pred.var(dim=0, unbiased=False)
    target_var = target.var(dim=0, unbiased=False)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    return (2.0 * cov) / (pred_var + target_var + (pred_mean - target_mean).pow(2) + eps)

def concordance_ccc_np(pred, true, eps=1e-8):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    pred_mean = pred.mean(axis=0)
    true_mean = true.mean(axis=0)
    pred_var = pred.var(axis=0)
    true_var = true.var(axis=0)
    cov = ((pred - pred_mean) * (true - true_mean)).mean(axis=0)
    return (2.0 * cov) / (pred_var + true_var + (pred_mean - true_mean) ** 2 + eps)

def compute_metrics(y_true, y_pred, vad_true_01, vad_pred_01):
    vad_true_raw = vad_from_0_1(np.asarray(vad_true_01))
    vad_pred_raw = vad_from_0_1(np.asarray(vad_pred_01))
    ccc = concordance_ccc_np(vad_pred_raw, vad_true_raw)
    mae = np.abs(vad_pred_raw - vad_true_raw).mean(axis=0)
    rmse = np.sqrt(((vad_pred_raw - vad_true_raw) ** 2).mean(axis=0))
    return {
        "WA": float(accuracy_score(y_true, y_pred)),
        "UAR": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "CCC_valence": float(ccc[0]),
        "CCC_arousal": float(ccc[1]),
        "CCC_dominance": float(ccc[2]),
        "CCC_mean": float(np.mean(ccc)),
        "MAE_mean": float(np.mean(mae)),
        "RMSE_mean": float(np.mean(rmse)),
    }

def primary_score(metrics):
    return 0.40 * metrics["UAR"] + 0.20 * metrics["WA"] + 0.20 * metrics["Macro_F1"] + 0.20 * metrics["CCC_mean"]

def class_weights_for(df):
    labels = df["emotion_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def multitask_loss(outputs, emotion_true, vad_true, class_weights=None, ce_weight=1.0, mse_weight=0.35, ccc_weight=0.50):
    ce = F.cross_entropy(
        outputs["emotion_logits"],
        emotion_true,
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    mse = F.mse_loss(outputs["vad_pred"], vad_true)
    ccc_loss = (1.0 - concordance_ccc_torch(outputs["vad_pred"], vad_true)).mean()
    return ce_weight * ce + mse_weight * mse + ccc_weight * ccc_loss

def module_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

def load_module_state_dict(model, state_dict):
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)

def zip_output(output_dir):
    output_dir = Path(output_dir)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    print("Saved zip:", zip_path)
    return zip_path
"""


CELL10 = """class RawAudioDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True).copy()
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vad = row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
        return {
            "utterance_id": str(row["utterance_id"]),
            "train_sample_id": str(row["train_sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
            "split": str(row["split"]),
            "wav_path": str(resolve_wav_path(row)),
            "emotion_id": int(row["emotion_id"]),
            "vad": vad_to_0_1(vad),
        }

def collate_raw(batch):
    arrays = [load_audio_16k(Path(item["wav_path"])) for item in batch]
    encoded = FEATURE_EXTRACTOR(
        arrays,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SAMPLES,
    )
    out = {
        "input_values": encoded["input_values"],
        "attention_mask": encoded.get("attention_mask"),
        "emotion_id": torch.tensor([x["emotion_id"] for x in batch], dtype=torch.long),
        "vad": torch.tensor(np.stack([x["vad"] for x in batch]), dtype=torch.float32),
        "utterance_id": [x["utterance_id"] for x in batch],
        "train_sample_id": [x["train_sample_id"] for x in batch],
        "speaker_id": [x["speaker_id"] for x in batch],
        "session": [x["session"] for x in batch],
        "split": [x["split"] for x in batch],
    }
    return out

def make_loader(df, shuffle=False):
    kwargs = {
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
        "collate_fn": collate_raw,
    }
    if NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(RawAudioDataset(df), **kwargs)

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def forward_model(model, input_values, attention_mask=None, return_embedding=False):
    # DataParallel can fail on the final small batch when one GPU receives no samples.
    # For that case, fall back to the underlying module on the main device.
    if isinstance(model, nn.DataParallel) and input_values.size(0) < len(model.device_ids):
        return model.module(input_values, attention_mask, return_embedding=return_embedding)
    return model(input_values, attention_mask, return_embedding=return_embedding)
"""


CELL12 = """class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, dim, hidden=128, dropout=0.1):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hidden, attention_mask=None):
        scores = self.attn(hidden).squeeze(-1)
        if attention_mask is not None:
            mask = attention_mask
            if mask.shape[1] != hidden.shape[1]:
                mask = F.interpolate(mask.float().unsqueeze(1), size=hidden.shape[1], mode="nearest").squeeze(1)
            scores = scores.masked_fill(mask.to(hidden.device) <= 0, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (hidden * weights).sum(dim=1)
        var = ((hidden - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1).clamp(min=1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1), weights.squeeze(-1)

class RawBackboneMultiTaskSER(nn.Module):
    def __init__(self, model_name, hidden_dim=256, dropout=0.30):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = int(getattr(self.backbone.config, "hidden_size", 768))
        self.pool = AttentiveStatisticsPooling(h, hidden=max(128, h // 4), dropout=dropout * 0.5)
        self.shared = nn.Sequential(
            nn.LayerNorm(h * 2),
            nn.Dropout(dropout),
            nn.Linear(h * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.emotion_head = nn.Linear(hidden_dim, NUM_CLASSES)
        self.vad_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())

    def forward(self, input_values, attention_mask=None, return_embedding=False):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        pooled, attn_weights = self.pool(outputs.last_hidden_state, attention_mask)
        embedding = self.shared(pooled)
        out = {"emotion_logits": self.emotion_head(embedding), "vad_pred": self.vad_head(embedding)}
        if return_embedding:
            out["embedding"] = embedding
            out["attention_weights"] = attn_weights
        return out

def freeze_backbone(model, unfreeze_last_n):
    target = model.module if isinstance(model, nn.DataParallel) else model
    if unfreeze_last_n < 0:
        for p in target.backbone.parameters():
            p.requires_grad = True
        return "Full backbone fine-tuning"
    for p in target.backbone.parameters():
        p.requires_grad = False
    layers = getattr(getattr(target.backbone, "encoder", None), "layers", None)
    if layers is not None and unfreeze_last_n > 0:
        for layer in layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        return f"Frozen backbone except last {unfreeze_last_n} encoder layers"
    return "Backbone frozen"

def build_model():
    model = RawBackboneMultiTaskSER(PRETRAINED_MODEL_NAME, HIDDEN_DIM, DROPOUT).to(DEVICE)
    note = freeze_backbone(model, UNFREEZE_LAST_N)
    if USE_DATA_PARALLEL:
        model = nn.DataParallel(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(note)
    print(f"Trainable parameters: {trainable:,}/{total:,} ({trainable/max(total,1):.2%})")
    return model

def build_optimizer(model):
    backbone, heads = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        clean = name.replace("module.", "")
        if clean.startswith("backbone."):
            backbone.append(param)
        else:
            heads.append(param)
    groups = []
    if backbone:
        groups.append({"params": backbone, "lr": LR_BACKBONE})
    if heads:
        groups.append({"params": heads, "lr": LR_HEAD})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

def build_scheduler(optimizer, total_update_steps):
    if not USE_SCHEDULER:
        return None
    warmup_steps = max(1, int(total_update_steps * WARMUP_RATIO))
    total_update_steps = max(warmup_steps + 1, total_update_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-4, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
"""


CELL13 = """@torch.no_grad()
def evaluate(model, loader, split_name, class_weights=None):
    if len(loader.dataset) == 0:
        raise ValueError(f"Split `{split_name}` rỗng, không thể evaluate. Hãy kiểm tra mapping train/val/test của fold.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, embeddings, probs = [], [], [], [], [], []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = forward_model(model, batch["input_values"], batch.get("attention_mask"), return_embedding=True)
        loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights)
        prob = torch.softmax(outputs["emotion_logits"], dim=-1)
        pred = prob.argmax(dim=-1)
        y_true.extend(batch["emotion_id"].detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
        vad_true.append(batch["vad"].detach().cpu().numpy())
        vad_pred.append(outputs["vad_pred"].detach().cpu().numpy())
        embeddings.append(outputs["embedding"].detach().cpu().numpy())
        probs.append(prob.detach().cpu().numpy())
        for i, uid in enumerate(batch["utterance_id"]):
            rows.append({
                "utterance_id": uid,
                "train_sample_id": batch["train_sample_id"][i],
                "speaker_id": batch["speaker_id"][i],
                "session": batch["session"][i],
                "split": split_name,
            })
        total_loss += float(loss.detach().cpu())
        n_batches += 1
    if not vad_true:
        raise ValueError(f"Không có batch nào trong split `{split_name}`. DataLoader đang rỗng.")
    vad_true = np.concatenate(vad_true)
    vad_pred = np.concatenate(vad_pred)
    embeddings = np.concatenate(embeddings)
    probs = np.concatenate(probs)
    metrics = compute_metrics(y_true, y_pred, vad_true, vad_pred)
    metrics["loss"] = total_loss / max(n_batches, 1)
    pred_df = pd.DataFrame(rows)
    pred_df["true_emotion_id"] = y_true
    pred_df["pred_emotion_id"] = y_pred
    for i, name in EMOTION_ID_TO_NAME.items():
        pred_df[f"prob_{name}"] = probs[:, i]
    for j, name in enumerate(["valence", "arousal", "dominance"]):
        pred_df[f"true_{name}"] = vad_from_0_1(vad_true)[:, j]
        pred_df[f"pred_{name}"] = vad_from_0_1(vad_pred)[:, j]
    feature_npz = {
        "utterance_id": pred_df["utterance_id"].to_numpy(),
        "train_sample_id": pred_df["train_sample_id"].to_numpy(),
        "embedding": embeddings.astype(np.float32),
        "emotion_probs": probs.astype(np.float32),
        "vad_pred": vad_pred.astype(np.float32),
        "emotion_true": np.asarray(y_true, dtype=np.int64),
        "vad_true": vad_true.astype(np.float32),
    }
    return metrics, pred_df, feature_npz
"""


CELL14 = """def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)
    test_loader = make_loader(test_df, shuffle=False)

    model = build_model()
    optimizer = build_optimizer(model)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(GRAD_ACCUM, 1)))
    scheduler = build_scheduler(optimizer, updates_per_epoch * EPOCHS)
    scaler = make_grad_scaler(USE_AMP)
    class_weights = class_weights_for(train_df) if USE_CLASS_WEIGHTS else None
    if class_weights is not None:
        print("Class weights:", [round(float(x), 4) for x in class_weights.detach().cpu()])

    best_score, best_epoch, bad_epochs = -1e9, -1, 0
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch)
            with autocast_context(USE_AMP):
                outputs = forward_model(model, batch["input_values"], batch.get("attention_mask"), return_embedding=False)
                loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights) / GRAD_ACCUM
            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
            running += float(loss.detach().cpu()) * GRAD_ACCUM

        val_metrics, _, _ = evaluate(model, val_loader, "val", class_weights=class_weights)
        score = primary_score(val_metrics)
        current_lrs = [group["lr"] for group in optimizer.param_groups]
        row = {
            "protocol": protocol,
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "val_primary_score": score,
            "lr_min": min(current_lrs),
            "lr_max": max(current_lrs),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "seconds": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | "
            f"val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | "
            f"score={score:.4f} | lr={row['lr_min']:.2e}-{row['lr_max']:.2e}"
        )
        if score > best_score + MIN_DELTA:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model_state_dict": module_state_dict(model), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best score={best_score:.4f}")
                break

    checkpoint = torch.load(best_path, map_location=DEVICE)
    load_module_state_dict(model, checkpoint["model_state_dict"])
    split_outputs = {}
    split_loaders = [("val", val_loader), ("test", test_loader)]
    if EVAL_TRAIN_SPLIT:
        split_loaders.insert(0, ("train", train_loader))
    for split_name, loader in split_loaders:
        metrics, pred_df, feature_npz = evaluate(model, loader, split_name, class_weights=class_weights)
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_pretrained_features.npz", **feature_npz)
        split_outputs[split_name] = metrics
    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")
    result = {
        "protocol": protocol,
        "fold": fold_name,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        **split_outputs["test"],
    }
    print("Test:", {k: result[k] for k in ["WA", "UAR", "Macro_F1", "CCC_mean", "MAE_mean"]})
    return result
"""


CELL16 = """config = {
    "notebook": "03A pretrained raw-audio backbone fine-tuning",
    "run_mode": RUN_MODE,
    "pretrained_model_name": PRETRAINED_MODEL_NAME,
    "sample_rate": SAMPLE_RATE,
    "max_seconds": MAX_SECONDS,
    "epochs": EPOCHS,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "num_workers": NUM_WORKERS,
    "lr_backbone": LR_BACKBONE,
    "lr_head": LR_HEAD,
    "min_lr_ratio": MIN_LR_RATIO,
    "warmup_ratio": WARMUP_RATIO,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "hidden_dim": HIDDEN_DIM,
    "unfreeze_last_n": UNFREEZE_LAST_N,
    "label_smoothing": LABEL_SMOOTHING,
    "use_class_weights": USE_CLASS_WEIGHTS,
    "use_scheduler": USE_SCHEDULER,
    "eval_train_split": EVAL_TRAIN_SPLIT,
    "use_data_parallel": USE_DATA_PARALLEL,
    "n_gpus": N_GPUS,
    "run_protocols": RUN_PROTOCOLS,
    "max_folds": MAX_FOLDS,
    "architecture": "pretrained raw audio backbone + attentive statistics pooling + two-head emotion/VAD",
}
(OUTPUT_DIR / "03a_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    nb.cells[3].source = CELL3
    nb.cells[7].source = CELL7
    nb.cells[8].source = CELL8
    nb.cells[10].source = CELL10
    nb.cells[12].source = CELL12
    nb.cells[13].source = CELL13
    nb.cells[14].source = CELL14
    nb.cells[16].source = CELL16
    nbformat.write(nb, NB_PATH)
    print("Updated", NB_PATH)


if __name__ == "__main__":
    main()
