import json
from pathlib import Path


BASE = Path(r"D:/UTE/Speech Programming/Speech Project")
NB1 = BASE / "06_w2v_based_models/01IEMOCAP Dataset Analysis and Speaker-Independent Splits/01_IEMOCAP_Dataset_Analysis_and_Splits.ipynb"
NB2 = BASE / "06_w2v_based_models/02IEMOCAP Feature Extraction Emotion2Vec Acoustic/02_IEMOCAP_Feature_Extraction_Emotion2Vec_Acoustic.ipynb"


NB1_SUMMARY_CODE = r'''
if metadata.empty or train_metadata.empty:
    display(Markdown("Chưa thể tạo final report vì metadata rỗng."))
else:
    duration_desc = train_metadata["duration"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    crop_rates = {sec: (train_metadata["duration"] > sec).mean() for sec in [3, 4, 5, 8, 10, 15]}
    avd_mean = train_metadata[["valence", "arousal", "dominance"]].mean()
    avd_by_emotion = (
        train_metadata.groupby("emotion_4class")[["valence", "arousal", "dominance", "duration"]]
        .mean()
        .round(3)
    )
    leak5_path = REPORT_DIR / "iemocap_5fold_session_leakage_check.csv"
    leak10_path = REPORT_DIR / "iemocap_10fold_speaker_leakage_check.csv"
    leak5_df = pd.read_csv(leak5_path) if leak5_path.exists() else pd.DataFrame()
    leak10_df = pd.read_csv(leak10_path) if leak10_path.exists() else pd.DataFrame()
    leak5_any = False
    if not leak5_df.empty:
        leak5_any = (leak5_df[[c for c in leak5_df.columns if c != "fold"]].astype(str) != "[]").any().any()
    speaker_leak10 = False
    session_overlap10 = False
    if not leak10_df.empty:
        speaker_cols = [c for c in leak10_df.columns if c.startswith("speaker_")]
        session_cols = [c for c in leak10_df.columns if c.startswith("session_")]
        speaker_leak10 = (leak10_df[speaker_cols].astype(str) != "[]").any().any()
        session_overlap10 = (leak10_df[session_cols].astype(str) != "[]").any().any()

    report_lines = [
        "# IEMOCAP Dataset Analysis Summary",
        "",
        "## Data source and scope",
        "",
        "- Source used in this notebook: AbstractTTS/IEMOCAP on HuggingFace.",
        f"- Full metadata utterances: **{len(metadata):,}**.",
        f"- 4-class train-ready utterances: **{len(train_metadata):,}**.",
        f"- Sessions: **{train_metadata['session'].nunique()}**.",
        f"- Speakers: **{train_metadata['speaker_id'].nunique()}**.",
        f"- Conversations: **{train_metadata['conversation_id'].nunique()}**.",
        f"- WAV found rate in full metadata: **{metadata['wav_found'].mean():.4f}**.",
        "",
        "Important: this 4-class subset is built from the HuggingFace `major_emotion` style labels. It has "
        f"**{len(train_metadata):,}** samples, so it is not automatically the same as the canonical IEMOCAP 4-class "
        "benchmark with about 5,531 utterances used by many papers. Results should be reported as "
        "**HF-major-emotion 4-class setting** unless we reproduce the exact paper filtering rule.",
        "",
        "## 4-Class Emotion Counts",
        "",
    ]
    for emotion, count in train_metadata["emotion_4class"].value_counts().sort_index().items():
        report_lines.append(f"- {emotion}: **{count:,}**")

    report_lines += [
        "",
        "## Duration and segmentation risk",
        "",
        f"- Mean duration: **{duration_desc['mean']:.3f}s**.",
        f"- Median duration: **{duration_desc['50%']:.3f}s**.",
        f"- 75th percentile: **{duration_desc['75%']:.3f}s**.",
        f"- 95th percentile: **{duration_desc['95%']:.3f}s**.",
        f"- Max duration: **{duration_desc['max']:.3f}s**.",
        "",
        "Crop/truncation risk if a fixed-length input is used:",
    ]
    for sec, rate in crop_rates.items():
        report_lines.append(f"- Longer than {sec}s: **{rate * 100:.2f}%**")

    report_lines += [
        "",
        "Interpretation: a single 4-second truncation would discard the tail of about "
        f"**{crop_rates[4] * 100:.2f}%** of 4-class utterances. For emotion2vec/wav2vec2-style models, a safer design is "
        "segment-level encoding followed by utterance-level pooling or attention.",
        "",
        "## Valence, arousal, dominance",
        "",
        "The AVD labels are continuous annotation scores on a 1-5 scale. This notebook also keeps normalized "
        "`*_norm` columns on a 0-1 scale for regression.",
        "",
        f"- Mean valence: **{avd_mean['valence']:.3f}**.",
        f"- Mean arousal: **{avd_mean['arousal']:.3f}**.",
        f"- Mean dominance: **{avd_mean['dominance']:.3f}**.",
        "",
        "Average AVD by emotion:",
        "",
        avd_by_emotion.to_markdown(),
        "",
        "Interpretation: angry has low valence and high arousal/dominance; happy has the highest valence; sad has lower "
        "valence/arousal; neutral stays near the center. This supports the multi-task idea, but emotion classification "
        "and AVD regression should still be evaluated with different metrics.",
        "",
        "## Speaker-independent split checks",
        "",
        f"- 5-fold session split leakage detected: **{leak5_any}**.",
        f"- 10-fold speaker split speaker leakage detected: **{speaker_leak10}**.",
        f"- 10-fold speaker split has session overlap: **{session_overlap10}**.",
        "",
        "Interpretation: the 5-fold protocol is stricter at the session level. The 10-fold protocol is speaker-independent "
        "because test speakers do not appear in train/validation, but it can still share the same session context through "
        "the other speaker.",
        "",
        "## Output files",
        "",
        f"- Metadata full: `{METADATA_DIR / 'iemocap_metadata_full.csv'}`",
        f"- Metadata 4-class + AVD: `{METADATA_DIR / 'iemocap_4class_avd_metadata.csv'}`",
        f"- 5-fold split JSON: `{SPLIT_DIR / 'iemocap_5fold_session.json'}`",
        f"- 10-fold split JSON: `{SPLIT_DIR / 'iemocap_10fold_speaker.json'}`",
        f"- Figures generated: **{len(list(FIGURE_DIR.glob('*.png')))}**.",
        f"- Report/table files generated: **{len(list(REPORT_DIR.glob('*')))}**.",
        "",
        "## Final note for the next notebook",
        "",
        "Notebook 02 should not blindly truncate all utterances to 4 seconds. It should preserve segment-level information "
        "and later aggregate features at utterance level for 5-fold and 10-fold training.",
    ]

    report_path = REPORT_DIR / "IEMOCAP_dataset_analysis_summary.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    audit_path = REPORT_DIR / "notebook01_output_audit_and_interpretation.md"
    audit_path.write_text("\n".join(report_lines), encoding="utf-8")
    display(Markdown("\n".join(report_lines)))
    print("Saved report:", report_path)
    print("Saved audit:", audit_path)
'''


NB2_SUMMARY_CODE = r'''
from pathlib import Path

selected_path = REPORT_DIR / "selected_demo_samples.csv"
demo_path = REPORT_DIR / "demo_feature_summary.csv"
comparison_path = REPORT_DIR / "comparison_pair_feature_summary.csv"
feature_subset_path = REPORT_DIR / "feature_summary_subset.csv"

selected_df = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()
demo_df = pd.read_csv(demo_path) if demo_path.exists() else pd.DataFrame()
comparison_df = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
feature_subset_df = pd.read_csv(feature_subset_path) if feature_subset_path.exists() else pd.DataFrame()
cache_files = sorted(p.name for p in CACHE_DIR.glob("*")) if "CACHE_DIR" in globals() else []
emotion2vec_cache_exists = any("emotion2vec" in name.lower() for name in cache_files)
sample_vis_dir = SAMPLE_DIR if "SAMPLE_DIR" in globals() else Path("sample_visualizations")

report_lines = [
    "# Feature Extraction Notebook 02 Summary",
    "",
    "## Current execution status",
    "",
    f"- Metadata path: `{METADATA_PATH}`",
    f"- Metadata available: **{METADATA_PATH.exists()}**.",
    f"- Rows with existing wav: **{len(audio_metadata) if 'audio_metadata' in globals() else 0:,}**.",
    f"- Selected demo samples: **{len(selected_df)}**.",
    f"- Feature cache rows in this run: **{len(feature_subset_df)}**.",
    f"- Cache files: **{', '.join(cache_files) if cache_files else 'none'}**.",
    f"- Emotion2vec embedding cache exists: **{emotion2vec_cache_exists}**.",
    f"- Target sample rate: **{TARGET_SR} Hz**.",
    f"- Segment seconds: **{SEGMENT_SECONDS}**.",
    f"- Segment overlap: **{SEGMENT_OVERLAP}**.",
    f"- Peak normalize: **{PEAK_NORMALIZE}**.",
    f"- RMS normalize: **{RMS_NORMALIZE}**.",
    "",
    "Important: this run intentionally keeps feature extraction small for inspection. The current cache is a demo/subset "
    "cache, not the final full feature cache for all 6,877 utterances.",
    "",
    "## Preprocessing meaning",
    "",
    "Before handcrafted features or pretrained speech embeddings, every waveform should be loaded, converted to mono, "
    "resampled to 16 kHz when needed, amplitude-normalized, and segmented/padded consistently. Pretrained models such "
    "as emotion2vec/wav2vec2/WavLM consume raw waveform, so MFCC/log-Mel are not required as their input, but the waveform "
    "normalization still matters.",
    "",
    "## Feature groups and intended role",
    "",
]
for row in feature_reference_rows:
    report_lines.append(f"- **{row['feature_group']}**: {row['used_by_future_model']}")

report_lines += [
    "",
    "## Observations from demo samples",
    "",
]
if not demo_df.empty:
    demo_table = demo_df[
        ["emotion_4class", "train_sample_id", "original_duration", "wave_rms", "rms_mean", "zcr_mean"]
    ].sort_values("emotion_4class").round(4)
    report_lines += [
        demo_table.to_markdown(index=False),
        "",
        "The selected happy sample has visibly higher RMS in this tiny demo set. This is useful as a visual sanity check, "
        "but it is not yet a dataset-level conclusion.",
        "",
    ]
else:
    report_lines.append("- Demo feature table was not generated.")

report_lines += [
    "## Same-speaker / different-speaker checks",
    "",
]
if not comparison_df.empty:
    comparison_table = comparison_df[
        ["comparison", "train_sample_id", "emotion_4class", "speaker_id", "wave_rms", "rms_mean", "zcr_mean", "mfcc0_mean"]
    ].round(4)
    report_lines += [
        comparison_table.to_markdown(index=False),
        "",
        "These comparisons show why random split is risky: handcrafted acoustic values change with speaker and conversation, "
        "not only emotion. The next training notebooks should keep 5-fold and 10-fold speaker-independent protocols.",
        "",
    ]
else:
    report_lines.append("- Comparison table was not generated.")

report_lines += [
    "## Caveats before model training",
    "",
    "- The fast extractor uses SciPy/Numpy STFT + Mel filterbank + DCT for MFCC-like/log-Mel inspection because some "
    "`librosa.feature` calls were too slow in this environment.",
    "- Do not use `f0_mean=0` from the fast extractor as pitch evidence. Use HuggingFace metadata pitch fields or a stable "
    "pitch extractor such as pyworld/praat-parselmouth later.",
    "- Do not silently replace failed emotion2vec embeddings with zeros. Missing embeddings should stop the extraction step.",
    "- Full extraction should be enabled only after the final extractor is chosen.",
    "",
    "## Generated visual outputs",
    "",
    f"- Overview figures: **{len(list(FIGURE_DIR.glob('*.png')))}**.",
    f"- Sample visualizations: **{len(list(sample_vis_dir.glob('*.png')))}**.",
    "",
    "## Next step",
    "",
    "The next implementation step is real emotion2vec embedding extraction for the full 4-class IEMOCAP subset, then "
    "segment/utterance pooling aligned with both 5-fold session-independent and 10-fold speaker-independent splits.",
]

summary_path = REPORT_DIR / "02_feature_extraction_summary.md"
summary_path.write_text("\n".join(report_lines), encoding="utf-8")
audit_path = REPORT_DIR / "notebook02_output_audit_and_interpretation.md"
audit_path.write_text("\n".join(report_lines), encoding="utf-8")
display(Markdown("\n".join(report_lines)))
print("Saved:", summary_path)
print("Saved audit:", audit_path)
'''


def replace_summary_cell(path: Path, marker: str, new_code: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    replaced = False
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if marker in src:
            cell["source"] = [line + "\n" for line in new_code.strip().splitlines()]
            replaced = True
            break
    if not replaced:
        raise RuntimeError(f"Could not find summary cell marker {marker} in {path}")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    replace_summary_cell(NB1, "IEMOCAP_dataset_analysis_summary.md", NB1_SUMMARY_CODE)
    replace_summary_cell(NB2, "02_feature_extraction_summary.md", NB2_SUMMARY_CODE)
    print("Updated summary report cells.")


if __name__ == "__main__":
    main()
