from collections import Counter, defaultdict
from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "01&02_Data_and_DataProcessing" / "ser_processed" / "metadata.csv"
OUT = ROOT / "Midterm_Param" / "06D_midterm_assets" / "domain_dataset_evidence.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DATASETS = ["RAVDESS", "CREMA-D", "TESS", "SAVEE"]
EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "disgust"]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def quantiles(values):
    values = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not values:
        return 0, 0, 0, 0, 0

    def q(p):
        pos = (len(values) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return values[lo]
        return values[lo] * (hi - pos) + values[hi] * (pos - lo)

    return min(values), q(0.25), q(0.5), q(0.75), max(values)


rows = []
with METADATA.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("emotion") in EMOTIONS and row.get("dataset") in DATASETS:
            rows.append(row)

dataset_counts = Counter(r["dataset"] for r in rows)
emotion_counts = defaultdict(Counter)
durations = defaultdict(list)
rms_values = defaultdict(list)
rms_key = "rms_raw_file"
for r in rows:
    ds = r["dataset"]
    emotion_counts[ds][r["emotion"]] += 1
    try:
        durations[ds].append(float(r.get("duration", 0)))
    except ValueError:
        pass
    try:
        rms_values[ds].append(float(r.get(rms_key, r.get("rms", 0))))
    except ValueError:
        pass

W, H = 1800, 1180
img = Image.new("RGB", (W, H), "#f8fafc")
draw = ImageDraw.Draw(img)

INK = "#102a43"
MUTED = "#64748b"
GRID = "#dbe3ef"
BLUE = "#2563eb"
GREEN = "#059669"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
TEAL = "#0891b2"
COLORS = ["#3b82f6", "#14b8a6", "#f97316", "#8b5cf6"]

title_f = font(44, True)
subtitle_f = font(24)
head_f = font(28, True)
body_f = font(20)
small_f = font(16)
tiny_f = font(13)

draw.text((55, 34), "Domain / Dataset Evidence for 06D SER", fill=INK, font=title_f)
draw.text(
    (58, 90),
    "Same 6-emotion task, but each corpus has different sample counts, label balance, duration and energy distribution.",
    fill=MUTED,
    font=subtitle_f,
)


def panel(x, y, w, h, title):
    draw.rounded_rectangle((x, y, x + w, y + h), radius=24, fill="#ffffff", outline="#cbd5e1", width=2)
    draw.text((x + 24, y + 18), title, fill=INK, font=head_f)


def map_y(value, vmin, vmax, y_top, y_bottom):
    if vmax == vmin:
        return (y_top + y_bottom) / 2
    return y_bottom - (value - vmin) / (vmax - vmin) * (y_bottom - y_top)


# Panel 1: sample counts
p1 = (55, 155, 805, 430)
panel(*p1, "A. Samples by dataset")
x, y, w, h = p1
plot_left, plot_top, plot_right, plot_bottom = x + 75, y + 75, x + w - 35, y + h - 70
max_count = max(dataset_counts.values())
draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRID, width=2)
draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=GRID, width=2)
bar_w = 90
gap = (plot_right - plot_left - len(DATASETS) * bar_w) / (len(DATASETS) + 1)
for i, ds in enumerate(DATASETS):
    bx = plot_left + gap + i * (bar_w + gap)
    val = dataset_counts[ds]
    by = map_y(val, 0, max_count, plot_top, plot_bottom)
    draw.rounded_rectangle((bx, by, bx + bar_w, plot_bottom), radius=10, fill=COLORS[i], outline=COLORS[i])
    draw.text((bx + 12, by - 28), str(val), fill=INK, font=body_f)
    draw.text((bx - 5, plot_bottom + 14), ds, fill=INK, font=body_f)
draw.text((plot_left - 62, plot_top + 110), "utterances", fill=MUTED, font=small_f)
draw.text((plot_left + 135, plot_bottom + 46), "x-axis: dataset / domain; y-axis: number of utterances", fill=MUTED, font=small_f)

# Panel 2: heatmap
p2 = (940, 155, 805, 430)
panel(*p2, "B. Emotion distribution by dataset")
x, y, w, h = p2
hm_left, hm_top = x + 140, y + 90
cell_w, cell_h = 135, 42
max_cell = max(emotion_counts[ds][emo] for ds in DATASETS for emo in EMOTIONS)
for j, ds in enumerate(DATASETS):
    draw.text((hm_left + j * cell_w + 18, hm_top - 36), ds, fill=INK, font=small_f)
for i, emo in enumerate(EMOTIONS):
    draw.text((x + 30, hm_top + i * cell_h + 12), emo, fill=INK, font=small_f)
    for j, ds in enumerate(DATASETS):
        val = emotion_counts[ds][emo]
        intensity = int(240 - (val / max_cell) * 150)
        fill = f"#{intensity:02x}{min(255, intensity + 25):02x}ff"
        x0 = hm_left + j * cell_w
        y0 = hm_top + i * cell_h
        draw.rectangle((x0, y0, x0 + cell_w - 6, y0 + cell_h - 5), fill=fill, outline="#cbd5e1")
        draw.text((x0 + 42, y0 + 11), str(val), fill=INK, font=small_f)
draw.text((hm_left + 20, hm_top + len(EMOTIONS) * cell_h + 20), "x-axis: dataset; y-axis: emotion label; cell color/count: samples", fill=MUTED, font=small_f)


def draw_boxplot(panel_rect, title, data, y_label, color):
    panel(*panel_rect, title)
    x, y, w, h = panel_rect
    left, top, right, bottom = x + 95, y + 85, x + w - 45, y + h - 75
    all_values = [v for ds in DATASETS for v in data[ds]]
    vmin, vmax = min(all_values), max(all_values)
    pad = (vmax - vmin) * 0.08 if vmax > vmin else 1
    vmin, vmax = max(0, vmin - pad), vmax + pad
    draw.line((left, bottom, right, bottom), fill=GRID, width=2)
    draw.line((left, top, left, bottom), fill=GRID, width=2)
    draw.text((left - 70, top + 115), y_label, fill=MUTED, font=small_f)
    step = (right - left) / len(DATASETS)
    for i, ds in enumerate(DATASETS):
        vals = data[ds]
        mn, q1, med, q3, mx = quantiles(vals)
        cx = left + step * (i + 0.5)
        y_mn = map_y(mn, vmin, vmax, top, bottom)
        y_q1 = map_y(q1, vmin, vmax, top, bottom)
        y_med = map_y(med, vmin, vmax, top, bottom)
        y_q3 = map_y(q3, vmin, vmax, top, bottom)
        y_mx = map_y(mx, vmin, vmax, top, bottom)
        bw = 76
        draw.line((cx, y_mx, cx, y_q3), fill=color, width=4)
        draw.line((cx, y_q1, cx, y_mn), fill=color, width=4)
        draw.rectangle((cx - bw / 2, y_q3, cx + bw / 2, y_q1), fill="#e0f2fe", outline=color, width=3)
        draw.line((cx - bw / 2, y_med, cx + bw / 2, y_med), fill=ORANGE, width=4)
        draw.line((cx - 26, y_mn, cx + 26, y_mn), fill=color, width=3)
        draw.line((cx - 26, y_mx, cx + 26, y_mx), fill=color, width=3)
        draw.text((cx - 42, bottom + 14), ds, fill=INK, font=small_f)
    draw.text((left + 85, bottom + 46), "x-axis: dataset / domain; y-axis: feature value distribution", fill=MUTED, font=small_f)


draw_boxplot((55, 640, 805, 430), "C. Duration distribution by dataset", durations, "seconds", BLUE)
draw_boxplot((940, 640, 805, 430), "D. RMS energy distribution by dataset", rms_values, "RMS", TEAL)

draw.text(
    (65, 1112),
    "Interpretation: dataset/domain shift means the model may learn recording style, speaker pool or duration/energy patterns, not only emotion. This motivates strict and per-dataset evaluation.",
    fill=INK,
    font=body_f,
)

img.save(OUT)
print(OUT)
