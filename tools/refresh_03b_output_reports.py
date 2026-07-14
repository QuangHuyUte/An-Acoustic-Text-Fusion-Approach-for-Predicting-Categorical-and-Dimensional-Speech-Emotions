import json
import zipfile
from pathlib import Path

import pandas as pd


BASE = Path(r"D:\UTE\Speech Programming\Speech Project")
RUN_DIR = BASE / "06_w2v_based_models" / "03B_CoAttention_Emotion2Vec_Acoustic_MultiTask_5_10Fold"
OUTPUT_DIR = RUN_DIR / "output"
REPORT_DIR = OUTPUT_DIR / "reports"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
FUSION_DIR = OUTPUT_DIR / "fusion_features"
FUSION_MODE = "acoustic_text_bridge_rmm"


def paper_metric(metric: str, group: pd.DataFrame) -> str:
    mean = group[metric].mean()
    std = group[metric].std(ddof=0)
    if metric == "MAE_mean":
        return f"{mean:.4f} +/- {std:.4f}"
    if metric.startswith("CCC"):
        return f"{mean:.3f} +/- {std:.3f}"
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


results_path = REPORT_DIR / f"03B_{FUSION_MODE}_results_by_fold.csv"
config_path = REPORT_DIR / f"03B_{FUSION_MODE}_config.json"
reference_path = REPORT_DIR / "03B_reference_and_ablation_table.csv"

results_df = pd.read_csv(results_path)
config = json.loads(config_path.read_text(encoding="utf-8"))
reference_df = pd.read_csv(reference_path)

metrics = [
    "WA",
    "UAR",
    "Macro_F1",
    "CCC_mean",
    "CCC_valence",
    "CCC_arousal",
    "CCC_dominance",
    "MAE_mean",
]

summary_rows = []
for (protocol, fusion_mode), group in results_df.groupby(["protocol", "fusion_mode"]):
    row = {"protocol": protocol, "fusion_mode": fusion_mode, "folds": len(group)}
    for metric in metrics:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=0)
        row[f"{metric}_paper"] = paper_metric(metric, group)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = REPORT_DIR / f"03B_{FUSION_MODE}_summary_paper_style.csv"
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

report_md = REPORT_DIR / f"03B_{FUSION_MODE}_report.md"
with report_md.open("w", encoding="utf-8") as f:
    f.write("# 03B Acoustic + Transcript Bridge Fusion Report\n\n")
    f.write("## Config\n\n")
    for k, v in config.items():
        f.write(f"- `{k}`: `{v}`\n")
    f.write("\n## Summary\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n\n## Ablation / Reference\n\n")
    f.write(reference_df.to_markdown(index=False))
    f.write("\n\n## Interpretation Guide\n\n")
    f.write("- So sánh `acoustic_only` với `acoustic_text_bridge_rmm` để kiểm tra transcript có bổ sung không.\n")
    f.write("- So sánh `acoustic_text_concat` với `acoustic_text_bridge` để kiểm tra bridge-token fusion có tốt hơn nối vector không.\n")
    f.write("- So sánh `acoustic_text_bridge` với `acoustic_text_bridge_rmm` để kiểm tra RMM có làm model cân bằng modality hơn không.\n")
    f.write("\n## Quick Read\n\n")
    f.write("- VAD đã được train ở thang normalized 0-1 và prediction được xuất ngược về thang 1-5.\n")
    f.write("- `VAD_pred_std_mean` trong results/predictions không còn collapse về 0, nên lỗi scale VAD trước đó đã được xử lý.\n")

zip_path = RUN_DIR / f"03B_{FUSION_MODE}_outputs.zip"
if zip_path.exists():
    zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for folder in [REPORT_DIR, FIGURE_DIR, MODEL_DIR, FUSION_DIR]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(OUTPUT_DIR).as_posix())

print("summary", summary_path)
print("report", report_md)
print("zip", zip_path, zip_path.stat().st_size)
