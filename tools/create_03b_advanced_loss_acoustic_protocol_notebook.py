import json
from pathlib import Path


ROOT = Path(r"D:\UTE\Speech Programming\Speech Project")
FOLDER = ROOT / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold"
BASE_NOTEBOOK = FOLDER / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb"
OUT_NOTEBOOK = FOLDER / "03B_Advanced_Loss_Acoustic_Training_Protocol.ipynb"


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise RuntimeError(f"Pattern not found:\n{old[:500]}")
    return src.replace(old, new, 1)


nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))

overview = md(
    r"""
# 03B Advanced: loss tuning + stronger acoustic branch + training protocol

Notebook này kế thừa toàn bộ pipeline của `03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold.ipynb`, nhưng thêm ba nhóm cải tiến có kiểm soát:

1. **Tune loss multi-task**
   - Giữ chế độ fixed-weight cũ để so sánh.
   - Thêm `LOSS_MODE=uncertainty`, học trọng số loss theo uncertainty thay vì phải dò tay hoàn toàn.
   - Vẫn giữ `CE + CCC + MSE + VAD variance regularization`.

2. **Nâng acoustic branch**
   - Temporal branch dùng multi-scale Conv1D kernels `3/5/7` để học biến thiên ngắn và dài của MFCC/prosody.
   - Thêm Squeeze-and-Excitation cho temporal/spectral CNN để học kênh acoustic nào quan trọng hơn.
   - Thay mean pooling bằng attentive statistics pooling cho audio/text tokens, giữ cả weighted mean và weighted std.

3. **Tối ưu training protocol**
   - Warmup + cosine decay giữ nguyên từ bản 03B.
   - Gradient clipping chuyển thành hyperparameter `MAX_GRAD_NORM`.
   - RMM có warmup, tránh mask acoustic quá sớm vì VAD phụ thuộc nhiều vào audio.
   - Log thêm uncertainty weights nếu dùng `LOSS_MODE=uncertainty`.

## Vì sao chọn các kỹ thuật này?

| Nhóm | Kỹ thuật | Tham khảo | Lý do đưa vào 03B |
|---|---|---|---|
| Multi-task loss | Uncertainty weighting | [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) | Emotion là classification, VAD là regression; hai loss khác scale nên fixed weight dễ lệch task. |
| Multi-task loss | GradNorm | [GradNorm](https://arxiv.org/abs/1711.02257) | Bài báo chỉ ra multi-task cần cân bằng tốc độ học giữa các task; notebook này chưa cài GradNorm full, nhưng dùng làm định hướng so sánh sau. |
| Multi-task loss | PCGrad | [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) | Nếu emotion và VAD gradient xung đột, PCGrad là hướng tiếp theo. Notebook này ưu tiên uncertainty vì nhẹ hơn. |
| Acoustic branch | Squeeze-and-Excitation | [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | Acoustic feature nhiều kênh; SE giúp model tự chọn kênh hữu ích thay vì coi mọi kênh ngang nhau. |
| Acoustic pooling | Attentive statistics pooling | [Attentive Statistics Pooling for Deep Speaker Embedding](https://arxiv.org/abs/1803.10963) | Speech emotion/VAD phụ thuộc biến thiên theo thời gian; weighted mean + std tốt hơn mean pooling đơn giản. |
| Acoustic robustness | SpecAugment | [SpecAugment](https://arxiv.org/abs/1904.08779) | Gợi ý ablation sau cho spectral features; bản notebook này chưa bật augmentation mặc định để tránh đổi quá nhiều cùng lúc. |
| Training protocol | Warm restarts / cosine schedule | [SGDR](https://arxiv.org/abs/1608.03983) | Cosine schedule giúp ổn định training; bản 03B đang dùng warmup + cosine decay, có thể thử restart sau. |

## Bảng tham khảo chi tiết

| Bài báo / nguồn | Nhóm kỹ thuật | Đã dùng trong notebook này? | Vị trí trong notebook | Kỳ vọng cải thiện | Rủi ro cần theo dõi |
|---|---|---:|---|---|---|
| [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115) | Multi-task loss balancing | Có | `MultiTaskLossBalancer`, `LOSS_MODE=uncertainty` | Giảm việc emotion CE áp đảo VAD regression; tự học trọng số loss theo độ khó task. | Nếu uncertainty học lệch, một task có thể bị giảm trọng số quá mạnh. Cần xem `loss_w_ce`, `loss_w_ccc`, `loss_w_mse` trong history. |
| [GradNorm: Gradient Normalization for Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257) | Multi-task gradient balancing | Chưa cài full | Định hướng sau ablation | Cân bằng tốc độ học giữa emotion và VAD bằng gradient norm. | Cài đặt phức tạp hơn vì phải tách backward từng task. |
| [Gradient Surgery for Multi-Task Learning](https://arxiv.org/abs/2001.06782) | Conflict-aware multi-task optimization | Chưa cài full | Định hướng sau ablation | Nếu emotion và VAD gradient xung đột, PCGrad có thể giảm ảnh hưởng tiêu cực giữa task. | Tốn thêm compute và code phức tạp hơn. |
| [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) | Channel attention | Có | `SEBlock1D`, `SEBlock2D`, `USE_SE_BLOCK=1` | Acoustic có nhiều kênh MFCC/log-Mel/prosody; SE giúp model tự tăng/giảm kênh quan trọng. | Nếu data ít, SE có thể overfit nhẹ; cần theo dõi gap train/val. |
| [Attentive Statistics Pooling for Deep Speaker Embedding](https://arxiv.org/abs/1803.10963) | Attention pooling cho speech | Có | `TokenAttentiveStatisticsPooling`, `USE_ATTENTIVE_POOLING=1` | Speech emotion phụ thuộc đoạn nổi bật, không nên chỉ mean pooling. Weighted mean + std giữ thông tin biến thiên. | Attention có thể tập trung sai nếu branch acoustic chưa học ổn. |
| [SpecAugment](https://arxiv.org/abs/1904.08779) | Data augmentation cho spectral feature | Chưa bật mặc định | Hướng ablation sau | Tăng robustness cho log-Mel/spectral branch. | Có thể làm giảm điểm nếu augmentation quá mạnh với IEMOCAP nhỏ. |
| [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | Training schedule | Một phần | Warmup + cosine schedule hiện tại | Giúp learning rate giảm mềm, training ổn định hơn fixed LR. | Warm restart chưa bật vì cần ablation riêng. |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Self-attention / cross-attention nền tảng | Có | `TransformerEncoderLayer`, `MultiheadAttention` bridge | Cho phép token acoustic và text tương tác linh hoạt hơn concat vector. | Nếu bridge quá mạnh, text có thể lấn VAD; vì vậy VAD head đang dùng `acoustic_heavy`. |

## Giải thích từng kỹ thuật được thêm

### 1. Uncertainty weighting cho multi-task loss

Vấn đề của 03B là có hai loại task khác nhau:

| Task | Output | Loss chính | Metric chính |
|---|---|---|---|
| Emotion classification | 4 lớp emotion | Cross-entropy | WA, UAR, Macro-F1 |
| VAD regression | valence, arousal, dominance | CCC + MSE | CCC, MAE |

Nếu dùng loss cố định:

```text
L = w_ce * L_ce + w_ccc * L_ccc + w_mse * L_mse
```

ta phải tự dò `w_ce`, `w_ccc`, `w_mse`. Uncertainty weighting học các trọng số này qua tham số `log_vars`:

```text
L_total = sum(exp(-s_i) * L_i + s_i)
```

Trong notebook:

```text
LOSS_MODE=uncertainty
```

sẽ bật `MultiTaskLossBalancer`. Trong history sẽ có:

```text
loss_w_ce
loss_w_ccc
loss_w_mse
```

để xem model đang ưu tiên task nào.

### 2. Multi-scale Temporal Conv1D

Acoustic temporal features không chỉ có một loại biến thiên. Pitch/energy có thể thay đổi rất nhanh, còn prosody/rhythm thay đổi chậm hơn. Vì vậy temporal branch dùng ba kernel song song:

```text
kernel 3: biến thiên ngắn
kernel 5: biến thiên trung bình
kernel 7: biến thiên dài hơn
```

Sau đó concat và project lại về `hidden_dim`.

| Flag | Ý nghĩa |
|---|---|
| `ACOUSTIC_BRANCH_MODE=advanced` | bật acoustic branch nâng cấp |
| `USE_MULTISCALE_TEMPORAL=1` | bật Conv1D đa kernel |

### 3. Squeeze-and-Excitation cho acoustic channels

SE block học trọng số theo kênh:

```text
channel_weight = sigmoid(MLP(global_average_pool(x)))
x_out = x * channel_weight
```

Với acoustic branch, điều này giúp model học được kênh nào quan trọng hơn, ví dụ MFCC, energy, pitch, log-Mel region, hoặc spectral pattern.

| Block | Vai trò |
|---|---|
| `SEBlock1D` | dùng cho temporal Conv1D |
| `SEBlock2D` | dùng cho spectral Conv2D |
| `USE_SE_BLOCK=1` | bật/tắt SE |

### 4. Attentive Statistics Pooling

Bản cũ lấy mean pooling token:

```text
audio_pool = mean(audio_tokens)
```

Bản advanced dùng attention để học token nào quan trọng hơn:

```text
alpha_t = softmax(score(h_t))
mean = sum(alpha_t * h_t)
std = sqrt(sum(alpha_t * (h_t - mean)^2))
pool = MLP([mean, std])
```

Lý do: với speech emotion, một đoạn ngắn có thể chứa tín hiệu cảm xúc rất mạnh. Mean pooling dễ làm loãng tín hiệu đó.

### 5. RMM warmup và acoustic-aware VAD

VAD, đặc biệt arousal và dominance, phụ thuộc nhiều vào audio/prosody. Vì vậy bản advanced giữ nguyên ý tưởng RMM nhưng không mask modality ngay từ đầu:

```text
RMM_WARMUP_EPOCHS=8
```

Sau warmup, RMM chủ yếu mask text:

```text
RMM_TEXT_PROB=0.90
```

Mục tiêu là ép model không phụ thuộc quá mức vào transcript, đồng thời không phá acoustic branch quá sớm.

### 6. Training protocol cần theo dõi

| Tín hiệu log | Cách đọc |
|---|---|
| `val_UAR` tăng nhưng `val_CCC` giảm | emotion đang thắng VAD, cần tăng ưu tiên VAD hoặc giảm text dominance |
| `VAD_pred_std_mean` rất nhỏ so với `VAD_true_std_mean` | VAD collapse, cần kiểm scale/loss |
| `loss_w_ce` quá lớn, `loss_w_ccc` quá nhỏ | uncertainty đang ưu tiên classification quá nhiều |
| train loss giảm nhưng val CCC giảm | overfit acoustic/fusion, cần dropout/weight decay/early stopping |
| CCC dominance thấp hơn valence/arousal | dominance cần acoustic/prosody mạnh hơn, có thể thử `VAD_REPRESENTATION=acoustic_only` |

## Cấu hình mặc định của bản advanced

```text
LOSS_MODE=uncertainty
ACOUSTIC_BRANCH_MODE=advanced
USE_ATTENTIVE_POOLING=1
USE_SE_BLOCK=1
USE_MULTISCALE_TEMPORAL=1
MAX_GRAD_NORM=1.0
RMM_WARMUP_EPOCHS=8
RMM_START=0.15
RMM_TEXT_PROB=0.90
```

Gợi ý chạy so sánh công bằng:

```text
1. Chạy notebook 03B cũ lấy baseline.
2. Chạy notebook này cùng split 5-fold.
3. So sánh UAR, Macro-F1, CCC_mean, CCC_valence/arousal/dominance, MAE.
4. Nếu CCC tăng nhưng UAR giảm nhẹ, cân nhắc chỉnh PRIMARY_UAR_WEIGHT / PRIMARY_CCC_WEIGHT.
```
"""
)

# Add overview after the title/intro cells.
nb["cells"].insert(2, overview)


advanced_classes = r"""
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not USE_SE_BLOCK:
            return x
        w = self.net(x).unsqueeze(-1)
        return x * w

class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not USE_SE_BLOCK:
            return x
        w = self.net(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class TemporalBranch(nn.Module):
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        self.advanced = ACOUSTIC_BRANCH_MODE == "advanced" and USE_MULTISCALE_TEMPORAL
        if self.advanced:
            branch_dim = max(32, hidden // 3)
            self.branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_dim, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_dim),
                    nn.GELU(),
                    nn.Dropout(dropout * 0.35),
                )
                for k in [3, 5, 7]
            ])
            self.project = nn.Sequential(
                nn.Conv1d(branch_dim * 3, hidden, kernel_size=1),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                SEBlock1D(hidden),
                nn.Dropout(dropout * 0.4),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout * 0.4),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
                nn.MaxPool1d(2),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                SEBlock1D(hidden),
                nn.Dropout(dropout * 0.5),
                nn.MaxPool1d(2),
            )

    def forward(self, x):
        if self.advanced:
            x = torch.cat([branch(x) for branch in self.branches], dim=1)
            return self.project(x).transpose(1, 2)
        return self.net(x).transpose(1, 2)

class SpectralBranch(nn.Module):
    def __init__(self, in_channels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            SEBlock2D(32),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout * 0.3),
            nn.Conv2d(32, hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            SEBlock2D(hidden // 2),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(dropout * 0.3),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            SEBlock2D(hidden),
            nn.Dropout2d(dropout * 0.25),
        )

    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=2)
        return h.transpose(1, 2)

class VectorToken(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x).unsqueeze(1)

class TokenAttentiveStatisticsPooling(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.Tanh(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(hidden // 2, 1),
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
        )

    def forward(self, tokens, mask=None):
        if not USE_ATTENTIVE_POOLING:
            if mask is None:
                return tokens.mean(dim=1)
            return (tokens * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        scores = self.score(tokens).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        mean = (tokens * weights).sum(dim=1)
        var = ((tokens - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return self.proj(torch.cat([mean, std], dim=-1))

class FrozenTextBranch(nn.Module):
    def __init__(self, model_name, hidden, dropout):
        super().__init__()
        live_log(f"Loading text backbone from: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=not ALLOW_HF_DOWNLOAD)
        live_log("Loaded text backbone.")
        for p in self.backbone.parameters():
            p.requires_grad = TEXT_LR > 0
        if TEXT_LR <= 0:
            self.backbone.eval()
        text_hidden = int(self.backbone.config.hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(text_hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, input_ids, attention_mask):
        if TEXT_LR <= 0:
            with torch.no_grad():
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state), attention_mask.bool()

class AcousticTextBridgeFusionSER(nn.Module):
    def __init__(self, temporal_channels, spectral_channels, stats_dim, e2v_dim, hidden=256, heads=8, bridge_tokens=16, dropout=0.3):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("HIDDEN_DIM phải chia hết cho NUM_HEADS.")
        self.temporal = TemporalBranch(temporal_channels, hidden, dropout)
        self.spectral = SpectralBranch(spectral_channels, hidden, dropout)
        self.stats = VectorToken(stats_dim, hidden, dropout)
        self.e2v = VectorToken(e2v_dim, hidden, dropout)
        self.text = FrozenTextBranch(TEXT_MODEL_SOURCE, hidden, dropout)

        audio_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        text_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.audio_self_attention = nn.TransformerEncoder(audio_layer, num_layers=1)
        self.text_self_attention = nn.TransformerEncoder(text_layer, num_layers=1)

        self.audio_pooler = TokenAttentiveStatisticsPooling(hidden, dropout)
        self.text_pooler = TokenAttentiveStatisticsPooling(hidden, dropout)

        self.bridge_audio_query = nn.Parameter(torch.randn(bridge_tokens, hidden) * 0.02)
        self.bridge_text_query = nn.Parameter(torch.randn(bridge_tokens, hidden) * 0.02)
        self.audio_key_text_value = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.text_key_audio_value = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)

        self.concat_fusion = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(dropout))
        self.bridge_fusion = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.Sigmoid())
        self.vad_acoustic_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.vad_fused_fusion = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.emotion_head = nn.Linear(hidden, 4)
        self.vad_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout * 0.25), nn.Linear(hidden // 2, 3), nn.Sigmoid())

    @staticmethod
    def _pad_tokens_to_same_length(audio, text, text_mask):
        audio_mask = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        target = max(audio.shape[1], text.shape[1])

        def pad_token(x):
            if x.shape[1] == target:
                return x
            return F.pad(x, (0, 0, 0, target - x.shape[1]), value=0.0)

        def pad_mask(mask):
            if mask.shape[1] == target:
                return mask
            return F.pad(mask, (0, target - mask.shape[1]), value=False)

        return pad_token(audio), pad_mask(audio_mask), pad_token(text), pad_mask(text_mask)

    def acoustic_tokens(self, batch):
        return torch.cat([
            self.temporal(batch["temporal"]),
            self.spectral(batch["spectral"]),
            self.stats(batch["stats"]),
            self.e2v(batch["e2v"]),
        ], dim=1)

    def forward(self, batch, mask_modality=None, return_embedding=False):
        audio = self.acoustic_tokens(batch)
        text, text_mask = self.text(batch["input_ids"], batch["attention_mask"])

        if mask_modality == "acoustic":
            audio = torch.zeros_like(audio)
        elif mask_modality == "text":
            text = torch.zeros_like(text)

        audio = self.audio_self_attention(audio)
        text = self.text_self_attention(text, src_key_padding_mask=~text_mask)
        audio_pool = self.audio_pooler(audio)
        text_pool = self.text_pooler(text, text_mask)

        if FUSION_MODE == "acoustic_only":
            z = audio_pool
            attn_a, attn_t = None, None
        elif FUSION_MODE == "text_only":
            z = text_pool
            attn_a, attn_t = None, None
        elif FUSION_MODE == "acoustic_text_concat":
            z = self.concat_fusion(torch.cat([audio_pool, text_pool], dim=-1))
            attn_a, attn_t = None, None
        else:
            audio_pad, audio_mask, text_pad, text_mask_pad = self._pad_tokens_to_same_length(audio, text, text_mask)
            qa = self.bridge_audio_query.unsqueeze(0).expand(audio.size(0), -1, -1)
            qt = self.bridge_text_query.unsqueeze(0).expand(audio.size(0), -1, -1)
            e_audio, attn_a = self.audio_key_text_value(
                query=qa,
                key=audio_pad,
                value=text_pad,
                key_padding_mask=~audio_mask,
                need_weights=return_embedding,
            )
            e_text, attn_t = self.text_key_audio_value(
                query=qt,
                key=text_pad,
                value=audio_pad,
                key_padding_mask=~text_mask_pad,
                need_weights=return_embedding,
            )
            bridge = 0.5 * (e_audio.mean(dim=1) + e_text.mean(dim=1))
            gate = self.gate(torch.cat([audio_pool, text_pool], dim=-1))
            z = self.bridge_fusion(bridge + gate * audio_pool + (1.0 - gate) * text_pool)

        if VAD_REPRESENTATION == "acoustic_heavy":
            z_vad = self.vad_acoustic_fusion(torch.cat([audio_pool, z], dim=-1))
        elif VAD_REPRESENTATION == "fused_text_audio":
            z_vad = self.vad_fused_fusion(torch.cat([z, text_pool], dim=-1))
        elif VAD_REPRESENTATION == "acoustic_only":
            z_vad = audio_pool
        else:
            z_vad = z

        out = {"emotion_logits": self.emotion_head(z), "vad_pred": self.vad_head(z_vad)}
        if return_embedding:
            out["embedding"] = z
            out["vad_embedding"] = z_vad
            out["attn_audio_query"] = attn_a
            out["attn_text_query"] = attn_t
        return out
"""


advanced_loss = r"""
def class_weights_for(df):
    labels = df["emotion_id"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=4).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

class MultiTaskLossBalancer(nn.Module):
    def __init__(self, mode="fixed"):
        super().__init__()
        self.mode = mode
        if mode == "uncertainty":
            # emotion CE, VAD CCC loss, VAD MSE loss
            self.log_vars = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self.register_parameter("log_vars", None)

    def forward(self, ce, ccc_loss, mse, vad_var_loss):
        if self.mode == "uncertainty":
            losses = torch.stack([ce, ccc_loss, mse])
            precision = torch.exp(-self.log_vars)
            balanced = (precision * losses + self.log_vars).sum()
            return balanced + VAD_VAR_WEIGHT * vad_var_loss
        return CE_WEIGHT * ce + CCC_WEIGHT * ccc_loss + MSE_WEIGHT * mse + VAD_VAR_WEIGHT * vad_var_loss

    def summary(self):
        if self.mode != "uncertainty" or self.log_vars is None:
            return {}
        with torch.no_grad():
            weights = torch.exp(-self.log_vars.detach()).cpu().numpy()
        return {
            "loss_w_ce": float(weights[0]),
            "loss_w_ccc": float(weights[1]),
            "loss_w_mse": float(weights[2]),
        }

def task_loss_components(outputs, y, vad_true, class_weights=None):
    ce = F.cross_entropy(outputs["emotion_logits"], y, weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    mse = F.mse_loss(outputs["vad_pred"], vad_true)
    ccc = concordance_ccc_torch(outputs["vad_pred"], vad_true)
    ccc_loss = 1.0 - ccc.mean()
    pred_std = outputs["vad_pred"].std(dim=0, unbiased=False)
    true_std = vad_true.std(dim=0, unbiased=False).detach()
    var_floor = (0.5 * true_std).clamp_min(1e-4)
    vad_var_loss = F.relu(var_floor - pred_std).pow(2).mean()
    return ce, ccc_loss, mse, vad_var_loss

def multitask_loss(outputs, y, vad_true, class_weights=None, loss_balancer=None):
    ce, ccc_loss, mse, vad_var_loss = task_loss_components(outputs, y, vad_true, class_weights=class_weights)
    if loss_balancer is None:
        return CE_WEIGHT * ce + CCC_WEIGHT * ccc_loss + MSE_WEIGHT * mse + VAD_VAR_WEIGHT * vad_var_loss
    return loss_balancer(ce, ccc_loss, mse, vad_var_loss)

def rmm_mask_for_epoch(epoch):
    if not USE_RMM:
        return None
    if epoch <= RMM_WARMUP_EPOCHS:
        return None
    progress = min(1.0, max(0.0, (epoch - 1) / max(1, EPOCHS - 1)))
    p = RMM_START * 0.5 * (1.0 + math.cos(math.pi * progress))
    if p < RMM_MIN_THRESHOLD:
        return None
    if random.random() >= p:
        return None
    return "text" if random.random() < RMM_TEXT_PROB else "acoustic"

def make_grad_scaler():
    return GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))
"""


for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell["source"])

    if "RUN_MODE = os.getenv" in src and "TEXT_MODEL_NAME" in src and "VAD_RAW_MIN" in src:
        src = replace_once(src, 'EPOCHS = int(os.getenv("EPOCHS", "70" if not IS_TUNE_MODE else "5"))', 'EPOCHS = int(os.getenv("EPOCHS", "90" if not IS_TUNE_MODE else "8"))')
        src = replace_once(src, 'PATIENCE = int(os.getenv("PATIENCE", "10" if not IS_TUNE_MODE else "2"))', 'PATIENCE = int(os.getenv("PATIENCE", "14" if not IS_TUNE_MODE else "3"))')
        src = replace_once(src, 'OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03b_acoustic_text_bridge_fusion")).resolve()', 'OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_03b_advanced_loss_acoustic_protocol")).resolve()')
        src = replace_once(src, 'RMM_START = float(os.getenv("RMM_START", "0.20"))', 'RMM_START = float(os.getenv("RMM_START", "0.15"))')
        src = replace_once(src, 'RMM_WARMUP_EPOCHS = int(os.getenv("RMM_WARMUP_EPOCHS", "5"))', 'RMM_WARMUP_EPOCHS = int(os.getenv("RMM_WARMUP_EPOCHS", "8"))')
        src = replace_once(
            src,
            'VAD_REPRESENTATION = os.getenv("VAD_REPRESENTATION", "acoustic_heavy").strip().lower()\n',
            'VAD_REPRESENTATION = os.getenv("VAD_REPRESENTATION", "acoustic_heavy").strip().lower()\n'
            'LOSS_MODE = os.getenv("LOSS_MODE", "uncertainty").strip().lower()\n'
            'ACOUSTIC_BRANCH_MODE = os.getenv("ACOUSTIC_BRANCH_MODE", "advanced").strip().lower()\n'
            'USE_ATTENTIVE_POOLING = os.getenv("USE_ATTENTIVE_POOLING", "1") == "1"\n'
            'USE_SE_BLOCK = os.getenv("USE_SE_BLOCK", "1") == "1"\n'
            'USE_MULTISCALE_TEMPORAL = os.getenv("USE_MULTISCALE_TEMPORAL", "1") == "1"\n'
            'MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))\n',
        )
        src = replace_once(
            src,
            '    "VAD_REPRESENTATION": VAD_REPRESENTATION,\n',
            '    "VAD_REPRESENTATION": VAD_REPRESENTATION,\n'
            '    "LOSS_MODE": LOSS_MODE,\n'
            '    "ACOUSTIC_BRANCH_MODE": ACOUSTIC_BRANCH_MODE,\n'
            '    "USE_ATTENTIVE_POOLING": USE_ATTENTIVE_POOLING,\n'
            '    "USE_SE_BLOCK": USE_SE_BLOCK,\n'
            '    "USE_MULTISCALE_TEMPORAL": USE_MULTISCALE_TEMPORAL,\n'
            '    "MAX_GRAD_NORM": MAX_GRAD_NORM,\n',
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "class TemporalBranch" in src and "class AcousticTextBridgeFusionSER" in src:
        cell["source"] = advanced_classes.splitlines(keepends=True)
        continue

    if "def class_weights_for" in src and "def multitask_loss" in src and "def rmm_mask_for_epoch" in src:
        cell["source"] = advanced_loss.splitlines(keepends=True)
        continue

    if "def evaluate(model, loader" in src and "multitask_loss(outputs" in src:
        src = src.replace(
            "def evaluate(model, loader, class_weights=None, return_features=False):",
            "def evaluate(model, loader, class_weights=None, loss_balancer=None, return_features=False):",
        )
        src = src.replace(
            'loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights)',
            'loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights, loss_balancer=loss_balancer)',
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "def train_one_fold" in src and "optimizer = torch.optim.AdamW(param_groups)" in src:
        src = replace_once(
            src,
            '    live_log(f"Model ready. Trainable params: {trainable_params:,}/{total_params:,} ({trainable_params/max(total_params,1):.2%})")\n\n'
            '    text_params, other_params = [], []',
            '    live_log(f"Model ready. Trainable params: {trainable_params:,}/{total_params:,} ({trainable_params/max(total_params,1):.2%})")\n'
            '    loss_balancer = MultiTaskLossBalancer(LOSS_MODE).to(DEVICE)\n'
            '    live_log(f"Loss mode: {LOSS_MODE}; acoustic mode: {ACOUSTIC_BRANCH_MODE}; attentive_pooling={USE_ATTENTIVE_POOLING}; se={USE_SE_BLOCK}")\n\n'
            '    text_params, other_params = [], []',
        )
        src = replace_once(
            src,
            '    if text_params:\n'
            '        param_groups.append({"params": text_params, "lr": TEXT_LR, "weight_decay": WEIGHT_DECAY})\n'
            '    optimizer = torch.optim.AdamW(param_groups)',
            '    if text_params:\n'
            '        param_groups.append({"params": text_params, "lr": TEXT_LR, "weight_decay": WEIGHT_DECAY})\n'
            '    if any(p.requires_grad for p in loss_balancer.parameters()):\n'
            '        param_groups.append({"params": list(loss_balancer.parameters()), "lr": LR, "weight_decay": 0.0})\n'
            '    optimizer = torch.optim.AdamW(param_groups)',
        )
        src = src.replace(
            'loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights) / GRAD_ACCUM',
            'loss = multitask_loss(outputs, batch["emotion_id"], batch["vad"], class_weights=class_weights, loss_balancer=loss_balancer) / GRAD_ACCUM',
        )
        src = src.replace(
            'torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)',
            'torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) + list(loss_balancer.parameters()) if p.requires_grad], MAX_GRAD_NORM)',
        )
        src = src.replace(
            'val_metrics, _, _ = evaluate(model, val_loader, class_weights=class_weights)',
            'val_metrics, _, _ = evaluate(model, val_loader, class_weights=class_weights, loss_balancer=loss_balancer)',
        )
        src = replace_once(
            src,
            '            **{f"val_{k}": v for k, v in val_metrics.items()},\n'
            '        }',
            '            **{f"val_{k}": v for k, v in val_metrics.items()},\n'
            '            **loss_balancer.summary(),\n'
            '        }',
        )
        src = replace_once(
            src,
            '            torch.save({"model_state_dict": model.state_dict(), "config": config, "best_score": float(best_score)}, best_path)',
            '            torch.save({"model_state_dict": model.state_dict(), "loss_balancer_state_dict": loss_balancer.state_dict(), "config": config, "best_score": float(best_score)}, best_path)',
        )
        src = replace_once(
            src,
            '    model.load_state_dict(checkpoint["model_state_dict"])\n',
            '    model.load_state_dict(checkpoint["model_state_dict"])\n'
            '    if "loss_balancer_state_dict" in checkpoint:\n'
            '        loss_balancer.load_state_dict(checkpoint["loss_balancer_state_dict"])\n',
        )
        src = src.replace(
            'metrics, pred_df, feature_npz = evaluate(model, loader, class_weights=class_weights, return_features=True)',
            'metrics, pred_df, feature_npz = evaluate(model, loader, class_weights=class_weights, loss_balancer=loss_balancer, return_features=True)',
        )
        src = src.replace(
            '    del model\n',
            '    del model, loss_balancer\n',
        )
        cell["source"] = src.splitlines(keepends=True)
        continue


OUT_NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUT_NOTEBOOK)
