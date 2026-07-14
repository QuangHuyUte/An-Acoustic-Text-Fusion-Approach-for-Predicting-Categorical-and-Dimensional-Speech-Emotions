import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\ASUS\.codex\attachments\ffcd5f62-8580-49b5-8b72-ebf013940a11\pasted-text.txt")
OUTPUT = ROOT / "Midterm_Param" / "Corrected_YouTube_Subtitles_Minimal_Timestamp_Format.txt"


def repair_mojibake(text: str) -> str:
    sample = text[:1000]
    if not any(token in sample for token in ("Ã", "Ä", "áº", "á»", "Æ")):
        return text
    # YouTube/exported text was UTF-8 Vietnamese decoded as Windows-1252.
    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeEncodeError:
        return text


REPLACEMENTS = [
    # Project title / names.
    (r"Speemotion\s+Recognition", "Speech Emotion Recognition"),
    (r"Speech emotion recognition", "Speech Emotion Recognition"),
    (r"speech emotion recognition", "Speech Emotion Recognition"),
    (r"presentation feedback\s+system", "presentation feedback system"),
    (r"Bùi em Huy", "Bùi Quang Huy"),
    (r"Nguyễn Tào Hư", "Nguyễn Tài Huy"),
    (r"Nguyễn Tào Huy", "Nguyễn Tài Huy"),
    (r"Nguyễn Minh Cường", "Nguyễn Minh Cường"),

    # Datasets.
    (r"raft gradi test of cv", "RAVDESS, CREMA-D, TESS và SAVEE"),
    (r"Rest, Gramad, test và Savi", "RAVDESS, CREMA-D, TESS và SAVEE"),
    (r"Rest, Gramad, TESS và Savi", "RAVDESS, CREMA-D, TESS và SAVEE"),
    (r"Rest", "RAVDESS"),
    (r"Gramad", "CREMA-D"),
    (r"\bSavi\b", "SAVEE"),
    (r"\bCV\b", "SAVEE"),
    (r"\bGrapnet\b", "CREMA-D"),
    (r"\bRevides\b", "RAVDESS"),
    (r"test và save", "TESS và SAVEE"),
    (r"test save", "TESS SAVEE"),
    (r"tập save", "tập SAVEE"),
    (r"\bsave\b", "SAVEE"),
    (r"\bID\b", "RAVDESS"),
    (r"test dataset", "TESS dataset"),

    # Labels and metrics.
    (r"neutral happy, set\s+and fear discus", "neutral, happy, sad, angry, fear và disgust"),
    (r"neutral happy, sad\s+and fear discus", "neutral, happy, sad, angry, fear và disgust"),
    (r"nhạng", "nhãn"),
    (r"\bset\b", "sad"),
    (r"\bdiscus\b", "disgust"),
    (r"\bFe\b", "fear"),
    (r"\bFE\b", "fear"),
    (r"accurcy", "accuracy"),
    (r"validay", "validation"),
    (r"confusion magic", "confusion matrix"),
    (r"marco", "macro"),

    # Data processing.
    (r"go và objective", "goal và objective"),
    (r"\bgo của", "goal của"),
    (r"trùng hóa dữ liệu", "chuẩn hóa dữ liệu"),
    (r"chiếc suất", "trích xuất"),
    (r"chiếch suất", "trích xuất"),
    (r"chiếch xuất", "trích xuất"),
    (r"chế xuất", "trích xuất"),
    (r"chích xuất", "trích xuất"),
    (r"chất xuất", "trích xuất"),
    (r"trích suất", "trích xuất"),
    (r"file gâm", "file ghi âm"),
    (r"file ghi hoặc là audio", "file ghi âm hoặc là audio"),
    (r"data lickage", "data leakage"),
    (r"lickage", "leakage"),
    (r"orimentation", "augmentation"),
    (r"Feature scaling s split", "Feature scaling sau split"),

    # Features.
    (r"\bpy n speaking rate\b", "pitch và speaking rate"),
    (r"post pattern", "pause pattern"),
    (r"\bMSCC\b", "MFCC"),
    (r"\bmscc\b", "MFCC"),
    (r"\bMSSC\b", "MFCC"),
    (r"\bMFC\b", "MFCC"),
    (r"\bmfc\b", "MFCC"),
    (r"\bFCC\b", "MFCC"),
    (r"\bFSCC\b", "MFCC"),
    (r"\bSCC\b", "MFCC"),
    (r"\bMSC\b", "MFCC"),
    (r"\bmsc\b", "MFCC"),
    (r"\bmcc\b", "MFCC"),
    (r"\bMCC\b", "MFCC"),
    (r"Delta to Delta", "delta-delta"),
    (r"delta to delta", "delta-delta"),
    (r"Delta data", "delta-delta"),
    (r"delta data", "delta-delta"),
    (r"Delta Delta", "delta-delta"),
    (r"delta Delta", "delta-delta"),
    (r"Delta MFCC", "delta MFCC"),
    (r"MFCC, Delta, Delta, Delta", "MFCC, delta MFCC, delta-delta MFCC"),
    (r"delta Delta MFCC", "delta-delta MFCC"),
    (r"delta delta\s+MFCC", "delta-delta MFCC"),
    (r"log mail", "log-Mel"),
    (r"logil", "log-Mel"),
    (r"lockmail Mel", "log-Mel"),
    (r"lockmail", "log-Mel"),
    (r"log-Mel Mel", "log-Mel"),
    (r"log-Mel\s+Mel", "log-Mel"),
    (r"LC mail", "log-Mel"),
    (r"\bmail\b", "Mel"),
    (r"\bDroma\b", "chroma"),
    (r"\bdrama\b", "chroma"),
    (r"\bDroma\b", "chroma"),
    (r"\bMc learning\b", "machine learning"),
    (r"đặc chân", "đặc trưng"),
    (r"đặt chân", "đặc trưng"),
    (r"điệ trưng", "đặc trưng"),
    (r"âm sách", "âm sắc"),
    (r"màu dọc", "màu giọng"),
    (r"giải tầng", "dải tần"),
    (r"chom một chroma", "chroma"),
    (r"chong biểu", "trong biểu"),
    (r"độ giả audio", "độ dài audio"),
    (r"thanh đạ", "thang đo"),
    (r"x sáng", "ít sáng"),
    (r"trạng thái tỉnh", "trạng thái tĩnh"),
    (r"thái tỉnh", "trạng thái tĩnh"),
    (r"lượng tỉnh", "năng lượng tĩnh"),
    (r"trương MFCC", "đặc trưng MFCC"),
    (r"mailbin", "Mel bin"),
    (r"Smog RAM", "spectrogram"),
    (r"way form", "waveform"),
    (r"WF form", "waveform"),

    # Models and algorithms.
    (r"\bonin\b", "1D-CNN"),
    (r"\bonein\b", "1D-CNN"),
    (r"\bonecinin\b", "1D-CNN"),
    (r"\boncinin\b", "1D-CNN"),
    (r"\bone desinin gru\b", "1D-CNN-GRU"),
    (r"\bone CNN\b", "1D-CNN"),
    (r"\bone dcn\b", "1D-CNN"),
    (r"\bone sin\b", "1D-CNN"),
    (r"\bOne dimension\b", "one-dimensional"),
    (r"\bone dimension\b", "one-dimensional"),
    (r"\bltin\b", "LSTM"),
    (r"\bCN\b", "CNN"),
    (r"\bCNNN\b", "CNN"),
    (r"\bLDM\b", "LSTM"),
    (r"\bLCDM\b", "LSTM"),
    (r"\bLSDM\b", "LSTM"),
    (r"\bUSDM\b", "LSTM"),
    (r"\bGRW\b", "GRU"),
    (r"\bGRWU\b", "GRU"),
    (r"\bGRW\b", "GRU"),
    (r"do view fusion", "dual-view fusion"),
    (r"example learning", "ensemble learning"),
    (r"weight average", "weighted average"),
    (r"CR search", "grid search"),
    (r"time straight", "time stretch"),
    (r"pit shift", "pitch shift"),
    (r"addictive white noise", "additive white Gaussian noise"),
    (r"nhiễu trắng Gion", "nhiễu trắng Gaussian"),
    (r"global average pulling", "global average pooling"),
    (r"sigmo", "sigmoid"),
    (r"s and exit ation", "Squeeze-and-Excitation"),
    (r"\bSQ\b", "Squeeze"),
    (r"channel special attention", "channel-spatial attention"),
    (r"special attention", "spatial attention"),
    (r"extension", "attention"),
    (r"onec nan", "1D-CNN"),
    (r"file tool lại", "fine-tune lại"),
    (r"support vet sin", "Support Vector Machine"),
    (r"support vet", "Support Vector Machine"),
    (r"support vine", "Support Vector Machine"),
    (r"attention pulling", "attention pooling"),
    (r"neuronetwork", "neural network"),
    (r"Speaker One", "speaker 1"),
    (r"speaker one", "speaker 1"),
    (r"emotion to dipension", "emotion2vec"),
    (r"avsvm", "RBF-SVM"),
    (r"gom example", "gom ensemble"),
    (r"example là stacking", "ensemble là stacking"),

    # Protocols / splits.
    (r"còn ba random", "combined random"),
    (r"combine random", "combined random"),
    (r"combine st", "combined strict"),
    (r"ba stck", "combined strict"),
    (r"phần chen", "phần train"),
    (r"tập chen", "tập train"),

    # Common transcript mistakes.
    (r"thiyết trình", "thuyết trình"),
    (r"tiếp trình", "thuyết trình"),
    (r"chứ trình", "thuyết trình"),
    (r"chức trình", "thuyết trình"),
    (r"thứ trình", "thuyết trình"),
    (r"thuyết trầm", "thuyết trình"),
    (r"midterom", "midterm"),
    (r"datet", "dataset"),
    (r"bộ form", "performance"),
    (r"phment microphone", "format/microphone"),
    (r"AMS ZCR", "RMS, ZCR"),
    (r"\bAMS\b", "RMS"),
    (r"bờ phong", "performance"),
    (r"tổng huyết hóa", "tổng quát hóa"),
    (r"tổng quét hóa", "tổng quát hóa"),
    (r"phân tích giải đoán", "phân loại"),
    (r"bị ver gần nhau", "bị nhầm gần nhau"),
    (r"fear với sá", "fear với sad"),
    (r"Phi với SAS", "fear với sad"),
    (r"phi với Happy", "fear với happy"),
    (r"Phi với Happy", "fear với happy"),
    (r"vecơ", "vector"),
]


def apply_replacements(text: str) -> str:
    for pattern, repl in REPLACEMENTS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE if pattern.islower() else 0)
    text = re.sub(r"log-Mel\s+Mel", "log-Mel", text)
    final_replacements = {
        "one CNN": "1D-CNN",
        "one dcn": "1D-CNN",
        "one sin": "1D-CNN",
        "One dimension": "one-dimensional",
        "one dimension": "one-dimensional",
        "attention pulling": "attention pooling",
        "support vet": "Support Vector Machine",
        "neuronetwork": "neural network",
        "Speaker One": "speaker 1",
        "speaker one": "speaker 1",
    }
    for old, new in final_replacements.items():
        text = text.replace(old, new)
    return text


def main():
    raw = SOURCE.read_text(encoding="utf-8")
    repaired = repair_mojibake(raw)
    fixed = apply_replacements(repaired)
    OUTPUT.write_text(fixed, encoding="utf-8")
    print("Wrote", OUTPUT)


if __name__ == "__main__":
    main()
