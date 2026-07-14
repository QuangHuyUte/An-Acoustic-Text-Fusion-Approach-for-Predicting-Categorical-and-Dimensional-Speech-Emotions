from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Midterm_Param" / "06D_midterm_assets" / "bilstm_temporal_context_diagram.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


W, H = 1800, 760
img = Image.new("RGB", (W, H), "#f7fbff")
draw = ImageDraw.Draw(img)

INK = "#12263f"
BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#059669"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"
MUTED = "#64748b"

title_f = font(48, True)
sub_f = font(26)
head_f = font(28, True)
body_f = font(22)
small_f = font(18)

draw.text((55, 35), "BiLSTM trong Branch A: học ngữ cảnh cảm xúc hai chiều", fill=INK, font=title_f)
draw.text(
    (58, 94),
    "Input là chuỗi acoustic frames [B, 132, T]. Forward LSTM đọc từ trái sang phải, backward LSTM đọc từ phải sang trái.",
    fill=MUTED,
    font=sub_f,
)

# Input feature strip
x0, y0 = 80, 210
draw.rounded_rectangle((x0, y0, x0 + 260, y0 + 350), radius=28, fill="#eff6ff", outline=BLUE, width=4)
draw.text((x0 + 32, y0 + 28), "X_temporal", fill=BLUE, font=head_f)
draw.text((x0 + 38, y0 + 72), "[132, T]", fill=INK, font=body_f)
features = ["MFCC", "delta", "delta2", "RMS", "ZCR", "spectral"]
for i, name in enumerate(features):
    yy = y0 + 125 + i * 34
    draw.rounded_rectangle((x0 + 36, yy, x0 + 220, yy + 22), radius=8, fill="#dbeafe", outline="#93c5fd")
    draw.text((x0 + 52, yy - 1), name, fill=INK, font=small_f)

# Time frames
frame_y = 260
frame_x = 455
frame_w = 84
gap = 26
for i in range(8):
    x = frame_x + i * (frame_w + gap)
    draw.rounded_rectangle((x, frame_y, x + frame_w, frame_y + 74), radius=16, fill="#ffffff", outline="#93c5fd", width=3)
    draw.text((x + 22, frame_y + 20), f"t{i+1}", fill=BLUE, font=head_f)
draw.text((frame_x + 110, frame_y - 45), "Temporal frames", fill=INK, font=head_f)

# Forward row
fw_y = 410
for i in range(8):
    x = frame_x + i * (frame_w + gap)
    draw.rounded_rectangle((x, fw_y, x + frame_w, fw_y + 64), radius=16, fill="#ecfdf5", outline=GREEN, width=3)
    draw.text((x + 15, fw_y + 16), "FW", fill=GREEN, font=head_f)
    if i < 7:
        draw.line((x + frame_w + 4, fw_y + 32, x + frame_w + gap - 6, fw_y + 32), fill=GREEN, width=5)
        draw.polygon(
            [(x + frame_w + gap - 6, fw_y + 32), (x + frame_w + gap - 18, fw_y + 24), (x + frame_w + gap - 18, fw_y + 40)],
            fill=GREEN,
        )
draw.text((frame_x - 5, fw_y - 38), "Forward context: quá khứ -> hiện tại", fill=GREEN, font=head_f)

# Backward row
bw_y = 525
for i in range(8):
    x = frame_x + i * (frame_w + gap)
    draw.rounded_rectangle((x, bw_y, x + frame_w, bw_y + 64), radius=16, fill="#f5f3ff", outline=PURPLE, width=3)
    draw.text((x + 15, bw_y + 16), "BW", fill=PURPLE, font=head_f)
    if i > 0:
        draw.line((x - gap + 8, bw_y + 32, x - 4, bw_y + 32), fill=PURPLE, width=5)
        draw.polygon([(x - gap + 8, bw_y + 32), (x - gap + 20, bw_y + 24), (x - gap + 20, bw_y + 40)], fill=PURPLE)
draw.text((frame_x - 5, bw_y - 38), "Backward context: tương lai -> hiện tại", fill=PURPLE, font=head_f)

# Connector input to frames
draw.line((x0 + 260, y0 + 180, frame_x - 35, frame_y + 37), fill=BLUE, width=5)
draw.polygon([(frame_x - 35, frame_y + 37), (frame_x - 55, frame_y + 26), (frame_x - 55, frame_y + 48)], fill=BLUE)

# Output box
out_x, out_y = 1430, 305
draw.rounded_rectangle((out_x, out_y, out_x + 300, out_y + 220), radius=30, fill="#fff7ed", outline=ORANGE, width=4)
draw.text((out_x + 36, out_y + 30), "BiLSTM output", fill=ORANGE, font=head_f)
draw.text((out_x + 45, out_y + 78), "[B, T/2, 192]", fill=INK, font=body_f)
draw.text((out_x + 38, out_y + 124), "concat(FW, BW)", fill=INK, font=body_f)
draw.text((out_x + 38, out_y + 160), "-> Attention pooling", fill=INK, font=body_f)

draw.line((frame_x + 8 * (frame_w + gap) - gap + 10, 445, out_x - 35, out_y + 90), fill=ORANGE, width=5)
draw.line((frame_x + 8 * (frame_w + gap) - gap + 10, 557, out_x - 35, out_y + 130), fill=ORANGE, width=5)
draw.polygon([(out_x - 35, out_y + 90), (out_x - 55, out_y + 78), (out_x - 55, out_y + 102)], fill=ORANGE)
draw.polygon([(out_x - 35, out_y + 130), (out_x - 55, out_y + 118), (out_x - 55, out_y + 142)], fill=ORANGE)

# Bottom explanation
draw.rounded_rectangle((72, 625, W - 72, 724), radius=24, fill="#ffffff", outline="#cbd5e1", width=2)
draw.text(
    (105, 644),
    "Vì sao hữu ích cho SER: cảm xúc cần ngữ cảnh cả câu, không chỉ một frame riêng lẻ.",
    fill=INK,
    font=body_f,
)
draw.text(
    (105, 682),
    "BiLSTM giúp mỗi frame biết giọng đã thay đổi trước đó và sẽ tiếp tục thay đổi phía sau như thế nào.",
    fill=INK,
    font=body_f,
)

img.save(OUT)
print(OUT)
