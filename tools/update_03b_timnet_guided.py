from pathlib import Path
import nbformat


NOTEBOOK = Path(
    r"D:\UTE\Speech Programming\Speech Project\06_w2v_based_models"
    r"\03_Emotion2Vec Downstream Finetune 5Fold 10Fold"
    r"\03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
)


def main():
    nb = nbformat.read(NOTEBOOK, as_version=4)

    nb.cells[0].source = """# 03B - TIM-Net-Guided Full 06D Acoustic Co-Attention Multi-Task SER 5-Fold + 10-Fold

Notebook này là bản làm lại của 03B để kết hợp các ý tưởng tốt từ 06D, Emonity và TIM-Net nhưng vẫn giữ đánh giá sạch theo fold.

Mục tiêu chính:

```text
Full 06D features
  -> TIM-Net-style temporal acoustic branch
  -> Spectral ResNet/SE log-Mel branch
  -> Stats MLP branch
  -> Emotion2Vec adapter branch
  -> Emotion2Vec-guided co-attention
  -> gated fusion
  -> two heads: emotion classification + VAD regression
```

Sáu ý tưởng được đưa vào notebook này:

1. Giữ baseline full 06D: `X_temporal`, `X_spectral`, `X_stats`, `X_e2v`.
2. Nâng temporal branch theo TIM-Net: bidirectional temporal stream + multi-scale dilated Conv1D + gated temporal blocks.
3. Nâng spectral branch theo Emonity-style 2D CNN: log-Mel/delta/delta-delta đi qua Residual SE ResNet branch riêng.
4. Training recipe có kiểm soát: class weights, label smoothing, AdamW, gradient clipping, optional cosine warm restarts.
5. Train-only augmentation: temporal masking/noise và SpecAugment nhẹ chỉ áp dụng trong train fold.
6. Vẫn chạy cả `5fold_session` và `10fold_speaker`, xuất bảng fold/summary và fusion features cho notebook 04.

Điểm quan trọng: notebook này không copy cách augment trước split của một số repo. Fold được tách trước, augmentation chỉ xảy ra trong `Dataset(train=True)`, nên validation/test không bị rò rỉ biến thể của cùng một sample.
"""

    nb.cells[2].source = """## Kiến trúc acoustic mới học từ 06D + TIM-Net + Emonity

### Branch A - TIM-Net-style temporal branch

Input:

```text
[batch, C_temporal, time]
```

`X_temporal` vẫn lấy từ full 06D: MFCC, delta MFCC, delta-delta MFCC, RMS, ZCR, centroid, bandwidth, rolloff, spectral contrast theo frame. Khác với 06D cũ, branch này không chỉ dùng CNN + BiLSTM mà chuyển sang temporal backbone kiểu TIM-Net:

```text
1x1 Conv projection
 -> forward temporal stream
 -> backward temporal stream
 -> dilated gated Conv1D blocks với dilation 1,2,4,...
 -> pooling theo từng scale
 -> learnable scale weighting
 -> z_temporal
```

Ý nghĩa: cảm xúc không chỉ nằm ở một frame mà nằm ở biến thiên theo thời gian. Multi-scale dilation giúp model thấy pattern ngắn, vừa và dài mà không cần làm model quá nặng.

### Branch B - Spectral ResNet/SE branch

Input:

```text
[batch, C_spectral, n_mels, time]
```

`X_spectral` gồm log-Mel, delta log-Mel và delta-delta log-Mel. Branch này được giữ như một nhánh riêng mạnh hơn:

```text
Residual SE Conv2D blocks
 -> downsample theo time-frequency
 -> adaptive pooling
 -> z_spectral
```

### Branch C - Stats branch

`X_stats` giữ các thống kê utterance-level. Đây là tín hiệu phụ có thể diễn giải, không được ép phải gánh toàn bộ bài toán.

### Branch D - Emotion2Vec branch

`X_e2v` là representation pretrained. Nó làm query để hỏi các acoustic tokens bằng co-attention.
"""

    nb.cells[3].source = """## Emotion2Vec-guided co-attention và gated fusion

Ta tạo các token:

```text
z_temporal  từ TIM-Net-style branch
z_spectral  từ Spectral ResNet branch
z_stats     từ Stats MLP branch
z_e2v       từ Emotion2Vec adapter
```

Emotion2Vec làm query, acoustic/stat tokens làm key/value:

$$
Q = z_{e2v}
$$

$$
K,V = [z_{temporal}, z_{spectral}, z_{stats}]
$$

$$
z_{context} = \\operatorname{Attention}(Q,K,V)
$$

Sau đó fusion dùng cả concat và gate:

```text
[z_temporal, z_spectral, z_e2v, z_context, z_stats]
 -> branch gate
 -> fusion MLP
 -> shared embedding
 -> emotion head + VAD head
```

Gate giúp model giảm ảnh hưởng của branch yếu thay vì concat cứng mọi tín hiệu.
"""

    nb.cells[7].source = """def vad_to_0_1(values):
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
    return 0.35 * metrics["UAR"] + 0.20 * metrics["WA"] + 0.20 * metrics["Macro_F1"] + 0.25 * metrics["CCC_mean"]

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

    nb.cells[9].source = """from contextlib import nullcontext

EPOCHS = int(os.getenv("EPOCHS", "90"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LR = float(os.getenv("LR", "6e-4"))
MIN_LR = float(os.getenv("MIN_LR", "1e-6"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "2e-4"))
DROPOUT = float(os.getenv("DROPOUT", "0.32"))
PATIENCE = int(os.getenv("PATIENCE", "14"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.06"))

USE_AUGMENTATION = os.getenv("USE_AUGMENTATION", "1") == "1"
TEMPORAL_AUG_PROB = float(os.getenv("TEMPORAL_AUG_PROB", "0.35"))
SPECTRAL_AUG_PROB = float(os.getenv("SPECTRAL_AUG_PROB", "0.45"))
TEMPORAL_NOISE_STD = float(os.getenv("TEMPORAL_NOISE_STD", "0.015"))
TIME_MASK_MAX = int(os.getenv("TIME_MASK_MAX", "36"))
FREQ_MASK_MAX = int(os.getenv("FREQ_MASK_MAX", "18"))

USE_CLASS_WEIGHTS = os.getenv("USE_CLASS_WEIGHTS", "1") == "1"
USE_BALANCED_SAMPLER = os.getenv("USE_BALANCED_SAMPLER", "0") == "1"
USE_SCHEDULER = os.getenv("USE_SCHEDULER", "1") == "1"
SCHEDULER_T0 = int(os.getenv("SCHEDULER_T0", "10"))
SCHEDULER_T_MULT = int(os.getenv("SCHEDULER_T_MULT", "2"))
USE_AMP = os.getenv("USE_AMP", "1") == "1"
AMP_ENABLED = bool(USE_AMP and DEVICE.type == "cuda")

TIMNET_HIDDEN = int(os.getenv("TIMNET_HIDDEN", "160"))
TIMNET_DILATIONS = int(os.getenv("TIMNET_DILATIONS", "8"))
TIMNET_KERNEL_SIZE = int(os.getenv("TIMNET_KERNEL_SIZE", "3"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "192"))
MAX_FOLDS = int(os.getenv("MAX_FOLDS", "0"))
RUN_PROTOCOLS = [x.strip() for x in os.getenv("RUN_PROTOCOLS", "5fold_session,10fold_speaker").split(",") if x.strip()]

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03b_timnet_guided_06d_coattention")).resolve()
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_DIR = OUTPUT_DIR / "reports"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FIGURE_DIR = OUTPUT_DIR / "figures"
for p_dir in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:
    p_dir.mkdir(parents=True, exist_ok=True)
print("Output:", OUTPUT_DIR)
print({
    "EPOCHS": EPOCHS,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "LABEL_SMOOTHING": LABEL_SMOOTHING,
    "USE_AUGMENTATION": USE_AUGMENTATION,
    "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,
    "USE_BALANCED_SAMPLER": USE_BALANCED_SAMPLER,
    "USE_SCHEDULER": USE_SCHEDULER,
    "AMP_ENABLED": AMP_ENABLED,
    "TIMNET_DILATIONS": TIMNET_DILATIONS,
})
"""

    nb.cells[12].source = """def compute_scalers(indices):
    scalers = {}
    scalers["temporal_mean"] = X_temporal[indices].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    scalers["temporal_std"] = (X_temporal[indices].std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
    scalers["spectral_mean"] = X_spectral[indices].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    scalers["spectral_std"] = (X_spectral[indices].std(axis=(0, 2, 3), keepdims=True) + 1e-6).astype(np.float32)
    scalers["stats_scaler"] = StandardScaler().fit(X_stats[indices])
    scalers["e2v_scaler"] = StandardScaler().fit(X_e2v[indices])
    return scalers

def augment_temporal(x):
    if random.random() > TEMPORAL_AUG_PROB:
        return x
    x = x.copy()
    frames = x.shape[-1]
    if frames > 8 and random.random() < 0.65:
        width = random.randint(4, min(TIME_MASK_MAX, frames))
        start = random.randint(0, max(0, frames - width))
        x[:, start:start + width] = 0
    if random.random() < 0.40 and TEMPORAL_NOISE_STD > 0:
        x += np.random.normal(0, TEMPORAL_NOISE_STD, size=x.shape).astype(np.float32)
    if frames > 8 and random.random() < 0.25:
        shift = random.randint(-min(8, frames // 10), min(8, frames // 10))
        x = np.roll(x, shift=shift, axis=-1)
    return x.astype(np.float32)

def augment_spectral(x):
    if random.random() > SPECTRAL_AUG_PROB:
        return x
    x = x.copy()
    _, n_mels, frames = x.shape
    if frames > 8 and random.random() < 0.75:
        width = random.randint(4, min(TIME_MASK_MAX, frames))
        start = random.randint(0, max(0, frames - width))
        x[:, :, start:start + width] = 0
    if n_mels > 8 and random.random() < 0.70:
        width = random.randint(3, min(FREQ_MASK_MAX, n_mels))
        start = random.randint(0, max(0, n_mels - width))
        x[:, start:start + width, :] = 0
    return x.astype(np.float32)

class Full06DDataset(Dataset):
    def __init__(self, df, scalers, train=False):
        self.df = df.reset_index(drop=True).copy()
        self.indices = np.asarray([feature_index[str(sid)] for sid in self.df["train_sample_id"].astype(str)], dtype=np.int64)
        self.scalers = scalers
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        i = self.indices[item]
        temporal = (X_temporal[i] - self.scalers["temporal_mean"][0]) / self.scalers["temporal_std"][0]
        spectral = (X_spectral[i] - self.scalers["spectral_mean"][0]) / self.scalers["spectral_std"][0]
        if self.train and USE_AUGMENTATION:
            temporal = augment_temporal(temporal)
            spectral = augment_spectral(spectral)
        stats = self.scalers["stats_scaler"].transform(X_stats[i:i+1]).astype(np.float32)[0]
        e2v = self.scalers["e2v_scaler"].transform(X_e2v[i:i+1]).astype(np.float32)[0]
        vad = vad_to_0_1(row[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32))
        return {
            "temporal": torch.tensor(temporal, dtype=torch.float32),
            "spectral": torch.tensor(spectral, dtype=torch.float32),
            "stats": torch.tensor(stats, dtype=torch.float32),
            "e2v": torch.tensor(e2v, dtype=torch.float32),
            "emotion_id": torch.tensor(int(row["emotion_id"]), dtype=torch.long),
            "vad": torch.tensor(vad, dtype=torch.float32),
            "utterance_id": str(row["utterance_id"]),
            "train_sample_id": str(row["train_sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "session": str(row["session"]),
        }

def collate_full06d(batch):
    return {
        "temporal": torch.stack([b["temporal"] for b in batch]),
        "spectral": torch.stack([b["spectral"] for b in batch]),
        "stats": torch.stack([b["stats"] for b in batch]),
        "e2v": torch.stack([b["e2v"] for b in batch]),
        "emotion_id": torch.stack([b["emotion_id"] for b in batch]),
        "vad": torch.stack([b["vad"] for b in batch]),
        "utterance_id": [b["utterance_id"] for b in batch],
        "train_sample_id": [b["train_sample_id"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "session": [b["session"] for b in batch],
    }

def make_balanced_sampler(df):
    labels = df["emotion_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

def make_loader(dataset, shuffle=False, sampler=None):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_full06d,
    )

def to_device(batch):
    return {k: (v.to(DEVICE, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

def autocast_context():
    if not AMP_ENABLED:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)

def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
"""

    nb.cells[13].source = """## Model blocks mới: 06D + TIM-Net temporal + Spectral ResNet + gated fusion

Phần này thay thế temporal branch cũ bằng branch học từ TIM-Net:

- `TemporalAwareBlock`: Conv1D có dilation, BatchNorm, GELU, SpatialDropout-style dropout và sigmoid gate.
- `TIMNetTemporalBranch`: chạy forward stream và backward stream, gom nhiều dilation scale bằng learnable weighting.
- `SpectralResNetBranch`: Residual SE Conv2D blocks cho log-Mel/delta/delta-delta.
- `EmotionGuidedCoAttention`: emotion2vec hỏi temporal/spectral/stats tokens.
- `BranchGatedFusion`: học gate cho từng branch trước khi fusion.
"""

    nb.cells[14].source = """class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, max(8, dim // 2)), nn.Tanh(), nn.Linear(max(8, dim // 2), 1))
    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1), weights

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=self.pad, bias=False)
    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., :-self.pad]
        return y

class TemporalAwareBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        gate = torch.sigmoid(self.net(x))
        return x * gate

class TIMNetTemporalBranch(nn.Module):
    def __init__(self, in_channels, hidden=160, out_dim=192, dilations=8, kernel_size=3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT * 0.35),
        )
        dilation_values = [2 ** i for i in range(dilations)]
        self.forward_blocks = nn.ModuleList([TemporalAwareBlock(hidden, kernel_size, d, DROPOUT * 0.35) for d in dilation_values])
        self.backward_blocks = nn.ModuleList([TemporalAwareBlock(hidden, kernel_size, d, DROPOUT * 0.35) for d in dilation_values])
        self.scale_score = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim), nn.GELU(), nn.Dropout(DROPOUT))

    def forward(self, x):
        fwd = self.input_proj(x)
        bwd = self.input_proj(torch.flip(x, dims=[-1]))
        scale_tokens = []
        for f_block, b_block in zip(self.forward_blocks, self.backward_blocks):
            fwd = f_block(fwd)
            bwd = b_block(bwd)
            token = (fwd + bwd).mean(dim=-1)
            scale_tokens.append(token)
        tokens = torch.stack(scale_tokens, dim=1)
        scale_weights = torch.softmax(self.scale_score(tokens), dim=1)
        pooled = (tokens * scale_weights).sum(dim=1)
        return self.proj(pooled), scale_weights.squeeze(-1)

class SE2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SE2D(out_ch),
        )
        self.shortcut = nn.Identity()
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))

class SpectralResNetBranch(nn.Module):
    def __init__(self, in_channels, out_dim=192):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.net = nn.Sequential(
            ResidualSEBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualSEBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualSEBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualSEBlock(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, out_dim), nn.GELU(), nn.Dropout(DROPOUT))
    def forward(self, x):
        return self.proj(self.net(self.stem(x)))

class MLPBranch(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
    def forward(self, x):
        return self.net(x)

class EmotionGuidedCoAttention(nn.Module):
    def __init__(self, dim=192, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=DROPOUT * 0.5)
        self.norm = nn.LayerNorm(dim)
    def forward(self, z_e2v, z_temporal, z_spectral, z_stats):
        q = z_e2v.unsqueeze(1)
        kv = torch.stack([z_temporal, z_spectral, z_stats], dim=1)
        context, weights = self.attn(q, kv, kv, need_weights=True)
        context = context.squeeze(1)
        return self.norm(z_e2v + context), weights

class BranchGatedFusion(nn.Module):
    def __init__(self, dim=192, n_tokens=5, hidden=384):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim * n_tokens, n_tokens), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(dim * n_tokens, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
    def forward(self, tokens):
        raw = torch.cat(tokens, dim=1)
        gates = self.gate(raw)
        gated = torch.cat([tok * gates[:, i:i+1] for i, tok in enumerate(tokens)], dim=1)
        return self.fusion(gated), gates

class Full06DCoAttentionMultiTaskSER(nn.Module):
    def __init__(self, temporal_dim, spectral_channels, stats_dim, e2v_dim, num_classes=4):
        super().__init__()
        self.temporal_branch = TIMNetTemporalBranch(
            temporal_dim,
            hidden=TIMNET_HIDDEN,
            out_dim=EMBED_DIM,
            dilations=TIMNET_DILATIONS,
            kernel_size=TIMNET_KERNEL_SIZE,
        )
        self.spectral_branch = SpectralResNetBranch(spectral_channels, out_dim=EMBED_DIM)
        self.e2v_branch = MLPBranch(e2v_dim, out_dim=EMBED_DIM, hidden=384)
        self.stats_branch = MLPBranch(stats_dim, out_dim=EMBED_DIM, hidden=384)
        self.co_attention = EmotionGuidedCoAttention(dim=EMBED_DIM, heads=4)
        self.fusion = BranchGatedFusion(dim=EMBED_DIM, n_tokens=5, hidden=384)
        self.emotion_head = nn.Linear(EMBED_DIM, num_classes)
        self.vad_head = nn.Sequential(nn.Linear(EMBED_DIM, 3), nn.Sigmoid())

    def forward(self, temporal, spectral, stats, e2v, return_embedding=False):
        z_t, temporal_scale_weights = self.temporal_branch(temporal)
        z_s = self.spectral_branch(spectral)
        z_e = self.e2v_branch(e2v)
        z_stats = self.stats_branch(stats)
        z_context, co_weights = self.co_attention(z_e, z_t, z_s, z_stats)
        fused, branch_gates = self.fusion([z_t, z_s, z_e, z_context, z_stats])
        out = {"emotion_logits": self.emotion_head(fused), "vad_pred": self.vad_head(fused)}
        if return_embedding:
            out["embedding"] = fused
            out["z_temporal"] = z_t
            out["z_spectral"] = z_s
            out["z_e2v"] = z_e
            out["z_stats"] = z_stats
            out["co_weights"] = co_weights
            out["temporal_scale_weights"] = temporal_scale_weights
            out["branch_gates"] = branch_gates
        return out

model_preview = Full06DCoAttentionMultiTaskSER(X_temporal.shape[1], X_spectral.shape[1], X_stats.shape[1], X_e2v.shape[1], NUM_CLASSES)
print("Parameters:", sum(p.numel() for p in model_preview.parameters()))
del model_preview
"""

    nb.cells[15].source = """@torch.no_grad()
def evaluate(model, loader, split_name, class_weights=None):
    if len(loader.dataset) == 0:
        raise ValueError(f"Split {split_name} rỗng.")
    model.eval()
    y_true, y_pred, vad_true, vad_pred, embeddings, probs = [], [], [], [], [], []
    rows = []
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = to_device(batch)
        outputs = model(batch["temporal"], batch["spectral"], batch["stats"], batch["e2v"], return_embedding=True)
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
        raise ValueError(f"Không có batch nào trong split {split_name}.")
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

def class_weights_for(df):
    labels = df["emotion_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def train_fold(protocol, fold_name, fold_df, seed):
    set_seed(seed)
    assert_fold_has_required_splits(protocol, fold_name, fold_df)
    train_df = fold_df[fold_df["split"] == "train"].reset_index(drop=True)
    val_df = fold_df[fold_df["split"] == "val"].reset_index(drop=True)
    test_df = fold_df[fold_df["split"] == "test"].reset_index(drop=True)
    print(f"\\n=== {protocol} | {fold_name} ===")
    print("Train/Val/Test:", len(train_df), len(val_df), len(test_df))

    train_indices = np.asarray([feature_index[str(sid)] for sid in train_df["train_sample_id"].astype(str)], dtype=np.int64)
    scalers = compute_scalers(train_indices)
    train_ds = Full06DDataset(train_df, scalers, train=True)
    val_ds = Full06DDataset(val_df, scalers, train=False)
    test_ds = Full06DDataset(test_df, scalers, train=False)
    sampler = make_balanced_sampler(train_df) if USE_BALANCED_SAMPLER else None
    train_loader = make_loader(train_ds, shuffle=sampler is None, sampler=sampler)
    val_loader = make_loader(val_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)

    model = Full06DCoAttentionMultiTaskSER(
        X_temporal.shape[1], X_spectral.shape[1], X_stats.shape[1], X_e2v.shape[1], NUM_CLASSES
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = None
    if USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT, eta_min=MIN_LR
        )
    scaler = make_grad_scaler()
    class_weights = class_weights_for(train_df) if USE_CLASS_WEIGHTS else None

    best_score, best_epoch, bad_epochs = -1e9, -1, 0
    best_path = MODEL_DIR / protocol / f"{fold_name}_best.pt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = to_device(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                outputs = model(batch["temporal"], batch["spectral"], batch["stats"], batch["e2v"])
                loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step((epoch - 1) + (step + 1) / max(len(train_loader), 1))
            running += float(loss.detach().cpu())

        val_metrics, _, _ = evaluate(model, val_loader, "val", class_weights=class_weights)
        score = primary_score(val_metrics)
        row = {
            "protocol": protocol,
            "fold": fold_name,
            "epoch": epoch,
            "train_loss": running / max(len(train_loader), 1),
            "lr": optimizer.param_groups[0]["lr"],
            "val_primary_score": score,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | "
            f"score={score:.4f} | lr={row['lr']:.2e}"
        )
        if score > best_score:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "best_epoch": best_epoch, "best_val_score": best_score}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=DEVICE)["model_state_dict"])
    split_outputs = {}
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        metrics, pred_df, feature_npz = evaluate(model, loader, split_name, class_weights=class_weights)
        pred_df.to_csv(REPORT_DIR / f"{protocol}_{fold_name}_{split_name}_predictions.csv", index=False, encoding="utf-8-sig")
        np.savez_compressed(FUSION_DIR / f"{protocol}_{fold_name}_{split_name}_coattention_features.npz", **feature_npz)
        split_outputs[split_name] = metrics
    pd.DataFrame(history).to_csv(REPORT_DIR / f"{protocol}_{fold_name}_history.csv", index=False, encoding="utf-8-sig")
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

    nb.cells[16].source = nb.cells[16].source.replace("03b_full_06d_coattention_results_by_fold.csv", "03b_timnet_guided_06d_results_by_fold.csv")
    nb.cells[16].source = nb.cells[16].source.replace("03b_full_06d_coattention_summary.csv", "03b_timnet_guided_06d_summary.csv")

    nb.cells[17].source = """config = {
    "notebook": "03B TIM-Net-guided full 06D acoustic co-attention multi-task model",
    "feature_cache_path": str(FEATURE_CACHE_PATH),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "min_lr": MIN_LR,
    "weight_decay": WEIGHT_DECAY,
    "dropout": DROPOUT,
    "patience": PATIENCE,
    "label_smoothing": LABEL_SMOOTHING,
    "use_augmentation": USE_AUGMENTATION,
    "temporal_aug_prob": TEMPORAL_AUG_PROB,
    "spectral_aug_prob": SPECTRAL_AUG_PROB,
    "use_class_weights": USE_CLASS_WEIGHTS,
    "use_balanced_sampler": USE_BALANCED_SAMPLER,
    "use_scheduler": USE_SCHEDULER,
    "use_amp": USE_AMP,
    "run_protocols": RUN_PROTOCOLS,
    "architecture": {
        "temporal": "TIM-Net-style bidirectional multi-scale dilated gated Conv1D branch",
        "spectral": "Residual SE Spectral ResNet branch on log-Mel/delta/delta-delta",
        "stats": "MLP branch projected to shared token dimension",
        "emotion2vec": "MLP adapter used as co-attention query",
        "fusion": "Emotion2Vec-guided co-attention over temporal/spectral/stats tokens + branch-gated fusion",
        "heads": "emotion classification + VAD regression",
    },
    "reference_ideas": [
        "06D full multi-branch feature cache",
        "TIM-Net temporal-aware bidirectional multi-scale dilated Conv1D",
        "Emonity-style separate spectral CNN branch, but without augmentation leakage",
        "train-only temporal/spec augmentation",
        "label smoothing/class weights/scheduler",
        "5-fold session and 10-fold speaker evaluation",
    ],
}
(OUTPUT_DIR / "03b_timnet_guided_06d_run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
zip_output(OUTPUT_DIR)
"""

    nbformat.write(nb, NOTEBOOK)
    print(f"Updated {NOTEBOOK}")


if __name__ == "__main__":
    main()
