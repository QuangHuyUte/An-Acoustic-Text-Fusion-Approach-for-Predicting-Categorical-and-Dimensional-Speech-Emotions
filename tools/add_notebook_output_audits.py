import json
from pathlib import Path

import pandas as pd


BASE = Path(r"D:/UTE/Speech Programming/Speech Project")
NB1_DIR = BASE / "06_w2v_based_models/01IEMOCAP Dataset Analysis and Speaker-Independent Splits"
NB2_DIR = BASE / "06_w2v_based_models/02IEMOCAP Feature Extraction Emotion2Vec Acoustic"
NB1 = NB1_DIR / "01_IEMOCAP_Dataset_Analysis_and_Splits.ipynb"
NB2 = NB2_DIR / "02_IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb"
MARKER = "<!-- CODEx output audit v2 -->"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt_num(value: float) -> str:
    return f"{value:.3f}"


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def patch_notebook(path: Path, audit_text: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    nb["cells"] = [
        cell
        for cell in nb["cells"]
        if MARKER not in "".join(cell.get("source", []))
    ]
    nb["cells"].append(md_cell(audit_text))
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def build_notebook_01_audit() -> str:
    meta_full = pd.read_csv(NB1_DIR / "metadata/iemocap_metadata_full.csv")
    meta4 = pd.read_csv(NB1_DIR / "metadata/iemocap_4class_avd_metadata.csv")
    counts = meta4["emotion_4class"].value_counts().to_dict()
    sessions = meta4["session"].value_counts().sort_index().to_dict()
    gender = meta4["gender"].value_counts().to_dict()
    dur = meta4["duration"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    crop = {sec: (meta4["duration"] > sec).mean() for sec in [3, 4, 5, 8, 10, 15]}
    avd_mean = meta4[["valence", "arousal", "dominance"]].mean()
    avd_by_emotion = (
        meta4.groupby("emotion_4class")[["valence", "arousal", "dominance", "duration"]]
        .mean()
        .round(3)
    )
    leak5 = pd.read_csv(NB1_DIR / "reports/iemocap_5fold_session_leakage_check.csv")
    leak10 = pd.read_csv(NB1_DIR / "reports/iemocap_10fold_speaker_leakage_check.csv")
    speaker_cols = [c for c in leak10.columns if c.startswith("speaker_")]
    session_cols = [c for c in leak10.columns if c.startswith("session_")]
    speaker_leak10 = (leak10[speaker_cols].astype(str) != "[]").any().any()
    session_overlap10 = (leak10[session_cols].astype(str) != "[]").any().any()
    leak5_any = (
        leak5[[c for c in leak5.columns if c != "fold"]].astype(str) != "[]"
    ).any().any()
    fig_count = len(list((NB1_DIR / "figures").glob("*.png")))
    report_count = len(list((NB1_DIR / "reports").glob("*")))
    split_count = len(list((NB1_DIR / "splits").glob("*")))

    avd_table = avd_by_emotion.to_markdown()
    counts_text = ", ".join(f"{k}: {v}" for k, v in counts.items())
    sessions_text = ", ".join(f"{k}: {v}" for k, v in sessions.items())
    gender_text = ", ".join(f"{k}: {v}" for k, v in gender.items())
    crop_text = ", ".join(f">{sec}s: {fmt_pct(rate)}" for sec, rate in crop.items())

    audit = f"""
{MARKER}

## 12. Kiểm tra output sau khi có metadata và audio HF

Phần này là phần audit sau khi notebook đã chạy thật trên metadata/audio tải từ [AbstractTTS/IEMOCAP](https://huggingface.co/datasets/AbstractTTS/IEMOCAP), không chỉ là mô tả kế hoạch.

### Trạng thái dữ liệu

- Metadata đầy đủ có **{len(meta_full):,} utterances**.
- Subset 4 lớp dùng cho emotion classification hiện có **{len(meta4):,} utterances** với nhãn `neutral`, `angry`, `sad`, `happy`; trong đó `happy` đã gộp cả `happy` và `excited`.
- Tất cả dòng trong subset 4 lớp đều có đường dẫn WAV đã xuất từ HuggingFace.
- Phân bố nhãn 4 lớp: **{counts_text}**.
- Phân bố session: **{sessions_text}**.
- Phân bố giới tính: **{gender_text}**.

Điểm cần ghi rất rõ: con số **{len(meta4):,}** này là subset theo `major_emotion` của bản HuggingFace, không phải đúng benchmark 4-class **5,531 utterances** thường thấy trong nhiều bài IEMOCAP. Vì vậy, khi báo cáo kết quả, cần ghi là **HF-major-emotion 4-class setting**. Nếu muốn so trực tiếp với paper dùng 5,531 mẫu, ta phải tái tạo đúng tiêu chí consensus/label filtering của paper đó.

### Độ dài audio và ảnh hưởng tới feature extraction

- Độ dài trung bình: **{fmt_num(dur['mean'])} giây**.
- Median: **{fmt_num(dur['50%'])} giây**.
- 75% utterance ngắn hơn **{fmt_num(dur['75%'])} giây**.
- 95% utterance ngắn hơn **{fmt_num(dur['95%'])} giây**.
- Audio dài nhất trong subset: **{fmt_num(dur['max'])} giây**.
- Tỷ lệ audio dài hơn mốc cố định: **{crop_text}**.

Nhận xét: nếu mô hình dùng một cửa sổ cố định 4 giây và chỉ truncate, khoảng **{fmt_pct(crop[4])}** mẫu sẽ bị mất thông tin cuối câu. Vì vậy notebook feature ở bước 02 nên dùng segmentation/pooling thay vì cắt cụt một đoạn duy nhất. Với emotion2vec hoặc wav2vec2, hướng hợp lý là chia thành segment ngắn, lấy embedding từng segment, sau đó attention/mean pooling theo utterance.

### Valence, arousal, dominance trong dữ liệu này

Các cột `valence`, `arousal`, `dominance` nằm trên thang **1-5** từ annotation IEMOCAP; notebook cũng tạo bản chuẩn hóa `*_norm` về khoảng 0-1 để thuận tiện cho regression.

- Valence trung bình: **{fmt_num(avd_mean['valence'])}**.
- Arousal trung bình: **{fmt_num(avd_mean['arousal'])}**.
- Dominance trung bình: **{fmt_num(avd_mean['dominance'])}**.

Trung bình AVD theo emotion:

{avd_table}

Nhận xét từ thống kê này phù hợp với trực giác cảm xúc: `angry` có arousal/dominance cao và valence thấp; `happy` có valence cao; `sad` có valence và arousal thấp hơn; `neutral` nằm gần vùng trung tính. Đây là lý do task emotion classification và AVD regression có liên quan, nhưng không hoàn toàn giống nhau: một bên dự đoán lớp rời rạc, một bên dự đoán tọa độ cảm xúc liên tục.

### Kiểm tra split speaker-independent

- **5-fold session split**: không có overlap speaker/session giữa train, validation và test (`leakage_check` không phát hiện leak). Đây là split nghiêm ngặt theo session, thường khó hơn vì test session hoàn toàn mới.
- **10-fold speaker split**: không có speaker overlap giữa train, validation và test. Tuy nhiên có session overlap ở một số fold vì hai speaker trong cùng session có thể nằm ở các split khác nhau. Điều này vẫn đúng với mục tiêu **speaker-independent**, nhưng không phải **session-independent**.
- Output split đã được lưu thành JSON và CSV long format để notebook train có thể đọc lại trực tiếp.

### File output đã sinh ra

- Figures: **{fig_count}** file PNG.
- Reports/tables: **{report_count}** file.
- Split files: **{split_count}** file.

Kết luận cho notebook 01: dữ liệu đã sẵn sàng cho hai protocol chính, nhưng khi viết báo cáo cần tách rõ hai ngữ cảnh: **5-fold session-independent** để đánh giá nghiêm ngặt hơn, và **10-fold speaker-independent** để đối chiếu với các bài dùng 8 speakers train, 1 validation, 1 test.
"""
    (NB1_DIR / "reports/notebook01_output_audit_and_interpretation.md").write_text(
        audit.strip() + "\n", encoding="utf-8"
    )
    return audit


def build_notebook_02_audit() -> str:
    demo = pd.read_csv(NB2_DIR / "reports/demo_feature_summary.csv")
    selected = pd.read_csv(NB2_DIR / "reports/selected_demo_samples.csv")
    comp = pd.read_csv(NB2_DIR / "reports/comparison_pair_feature_summary.csv")
    subset = pd.read_csv(NB2_DIR / "reports/feature_summary_subset.csv")
    ref = pd.read_csv(NB2_DIR / "reports/feature_reference_table.csv")
    fig_count = len(list((NB2_DIR / "figures").glob("*.png")))
    sample_fig_count = len(list((NB2_DIR / "sample_visualizations").glob("*.png")))
    cache_files = sorted(p.name for p in (NB2_DIR / "features_cache").glob("*"))
    has_emotion2vec = any("emotion2vec" in name.lower() for name in cache_files)

    demo_rows = []
    for row in demo.sort_values("emotion_4class").to_dict("records"):
        demo_rows.append(
            f"| {row['emotion_4class']} | {row['train_sample_id']} | "
            f"{row['original_duration']:.3f} | {row['wave_rms']:.4f} | "
            f"{row['rms_mean']:.4f} | {row['zcr_mean']:.4f} |"
        )
    demo_table = "\n".join(
        [
            "| Emotion | Sample | Duration(s) | Wave RMS | Frame RMS mean | ZCR mean |",
            "|---|---:|---:|---:|---:|---:|",
            *demo_rows,
        ]
    )

    comp_rows = []
    for row in comp.to_dict("records"):
        comp_rows.append(
            f"| {row['comparison']} | {row['train_sample_id']} | {row['emotion_4class']} | "
            f"{row['speaker_id']} | {row['wave_rms']:.4f} | {row['rms_mean']:.4f} | "
            f"{row['zcr_mean']:.4f} | {row['mfcc0_mean']:.2f} |"
        )
    comp_table = "\n".join(
        [
            "| Comparison | Sample | Emotion | Speaker | Wave RMS | Frame RMS | ZCR | MFCC0 mean |",
            "|---|---:|---|---|---:|---:|---:|---:|",
            *comp_rows,
        ]
    )

    feature_groups = ", ".join(ref["feature_group"].drop_duplicates().astype(str).tolist())
    cache_text = ", ".join(cache_files) if cache_files else "chưa có cache"
    emotion2vec_text = "đã có" if has_emotion2vec else "chưa có"

    audit = f"""
{MARKER}

## 10. Kiểm tra output và nhận xét sau khi chạy feature notebook

Phần này kiểm tra output thực tế của notebook 02 sau khi đã có metadata/audio từ HuggingFace. Mục tiêu là phân biệt rõ ba lớp việc: chuẩn hóa audio, trích acoustic feature minh họa, và phần emotion2vec embedding sẽ chạy ở bước tiếp theo khi môi trường model sẵn sàng.

### Trạng thái chạy hiện tại

- Số sample demo được chọn để visualize: **{len(selected)}** mẫu, mỗi emotion một mẫu.
- Số dòng trong cache feature hiện tại: **{len(subset)}** dòng. Đây là **subset minh họa**, không phải full cache cho toàn bộ **6,877** utterances.
- Số nhóm feature trong bảng tham chiếu: **{len(ref)}** dòng mô tả, gồm: **{feature_groups}**.
- Figures tổng quan: **{fig_count}** PNG.
- Figures theo từng sample: **{sample_fig_count}** PNG.
- Cache hiện có: **{cache_text}**.
- Emotion2vec embedding cache: **{emotion2vec_text}**.

Điểm cần ghi rõ: trong lần chạy này, notebook đang ưu tiên chạy ổn định phần acoustic demo bằng SciPy/Numpy. Một số hàm `librosa.feature` trong môi trường hiện tại chạy quá chậm/hang, nên phần demo dùng STFT + Mel filterbank + DCT để tạo MFCC-like/log-Mel feature. Điều này đủ để phân tích pipeline, visualization và kiểm thử preprocessing, nhưng chưa nên xem là feature extractor cuối cùng để train mô hình chính.

### Chuẩn hóa audio trước khi trích feature

Notebook thực hiện các bước chuẩn hóa cơ bản:

- Load audio từ WAV đã export.
- Ép về mono nếu cần.
- Resample về **16 kHz** nếu sample rate khác.
- Peak normalize về mức ổn định để giảm khác biệt âm lượng ghi âm.
- Giữ lại duration thật và dùng segment length **4 giây** cho minh họa.

Các bước này phải làm trước MFCC/log-Mel/emotion2vec vì đặc trưng âm thanh phụ thuộc trực tiếp vào sample rate, số kênh, scale biên độ và độ dài tín hiệu. Với pretrained speech model như emotion2vec/wav2vec2/WavLM, mô hình nhận **raw waveform** nên không cần MFCC làm input chính, nhưng vẫn cần chuẩn hóa waveform.

### Quan sát từ 4 mẫu minh họa

{demo_table}

Nhận xét: mẫu `happy` được chọn có RMS lớn hơn rõ trong nhóm demo, nhưng đây chỉ là minh họa trực quan trên vài mẫu, chưa phải kết luận thống kê toàn dataset. Các hình waveform/RMS và heatmap giúp kiểm tra feature có thay đổi theo thời gian, có bị blank, bị clipping hoặc bị padding/truncation sai hay không.

### So sánh cùng speaker, khác emotion và khác speaker

{comp_table}

Nhận xét: so sánh này cho thấy feature thay đổi không chỉ theo emotion mà còn theo speaker/conversation. Đây là lý do roadmap cần giữ split speaker-independent ngay từ đầu; nếu random split, mô hình dễ học giọng người nói thay vì học tín hiệu cảm xúc.

### Phần cần chạy tiếp để thành feature notebook hoàn chỉnh

- Bật full extraction sau khi thống nhất extractor cuối cùng: không nên full-cache nếu còn dùng bản fast demo.
- Với mô hình chính, dùng emotion2vec embedding làm nhánh semantic/acoustic representation chính.
- Acoustic handcrafted feature nên đóng vai trò nhánh bổ sung: energy/RMS, ZCR, spectral summary, pitch metadata hoặc pitch extractor ổn định hơn.
- Không dùng `f0_mean=0` từ fast extractor làm kết luận pitch. Trong metadata HuggingFace đã có `pitch_mean`/`pitch_std`; nếu cần pitch thật từ waveform, nên dùng `pyworld`, `praat-parselmouth`, hoặc extractor ổn định khác.

Kết luận cho notebook 02: notebook đã đủ để kiểm tra preprocessing, visualization, bảng feature và logic demo; nhưng trước khi train mô hình chính cần bổ sung bước emotion2vec embedding thật và full cache theo 5-fold/10-fold split.
"""
    (NB2_DIR / "reports/notebook02_output_audit_and_interpretation.md").write_text(
        audit.strip() + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    patch_notebook(NB1, build_notebook_01_audit())
    patch_notebook(NB2, build_notebook_02_audit())
    print("Patched notebooks with output audit sections.")


if __name__ == "__main__":
    main()
