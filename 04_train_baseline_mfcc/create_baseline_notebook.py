import nbformat as nbf
from pathlib import Path
from textwrap import dedent


NOTEBOOK_PATH = Path(__file__).with_name("04_train_baseline_mfcc_ml.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb.cells = [
    md(
        """
        # 04. Baseline SER - MFCC + Logistic Regression / SVM / Random Forest

        Notebook này train các baseline truyền thống cho Speech Emotion Recognition (SER) dựa trên feature vector đã trích xuất ở bước Data Processing.

        Mục tiêu chính:

        - Tạo mốc so sánh trước khi train CNN/log-Mel.
        - Train và so sánh Logistic Regression, SVM và Random Forest.
        - Báo cáo accuracy, macro-F1, classification report, confusion matrix.
        - Phân tích performance theo từng dataset: CREMA-D, RAVDESS, TESS.
        - Lưu model, metric, hình ảnh và file đóng gói để tải về.

        Input bắt buộc:

        - `metadata.csv`
        - `baseline_features.npz`
        - `processing_config.json` nếu có

        Notebook tự tìm các file này trong Kaggle input/working hoặc trong workspace local.
        """
    ),
    md(
        """
        ## 1. Cài đặt và import thư viện

        Cell dưới kiểm tra các thư viện cần thiết. Kaggle thường đã có `numpy`, `pandas`, `sklearn`, `matplotlib`, `seaborn`, `joblib`. Nếu thiếu thư viện nhẹ, notebook sẽ cài bổ sung.
        """
    ),
    code(
        """
        import sys
        import subprocess
        import importlib.util

        required = {
            "numpy": "numpy",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "joblib": "joblib",
        }

        missing = [pkg for module, pkg in required.items() if importlib.util.find_spec(module) is None]
        if missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

        import json
        import time
        import shutil
        import subprocess
        from pathlib import Path

        import joblib
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns

        from IPython.display import display
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            classification_report,
            confusion_matrix,
        )

        sns.set_theme(style="whitegrid")
        RANDOM_STATE = 42
        np.random.seed(RANDOM_STATE)
        print("Libraries ready.")
        """
    ),
    md(
        """
        ## 2. Tự tìm processed dataset/artifact

        Notebook ưu tiên tìm artifact theo thứ tự:

        1. Kaggle working: `/kaggle/working/**/ser_processed`
        2. Kaggle input: `/kaggle/input/**/ser_processed`
        3. Kaggle input root có trực tiếp `metadata.csv` và `baseline_features.npz`
        4. Workspace local hiện tại, ví dụ `01&02_Data_and_DataProcessing/ser_processed`

        Nếu Kaggle dataset của bạn chứa folder `ser_processed`, không cần sửa path bằng tay.
        """
    ),
    code(
        """
        def looks_like_processed_dir(path: Path) -> bool:
            return (path / "metadata.csv").exists() and (path / "baseline_features.npz").exists()


        def find_processed_dir() -> Path:
            cwd = Path.cwd().resolve()
            candidates = []

            explicit_candidates = [
                cwd / "ser_processed",
                cwd / "Speech_Project" / "ser_processed",
                cwd / "01&02_Data_and_DataProcessing" / "ser_processed",
                cwd.parent / "01&02_Data_and_DataProcessing" / "ser_processed",
                cwd.parent / "Data_and_DataProcessing" / "ser_processed",
                Path("/kaggle/working/Speech_Project/ser_processed"),
                Path("/kaggle/working/ser_processed"),
            ]
            candidates.extend(explicit_candidates)

            search_roots = [
                Path("/kaggle/working"),
                Path("/kaggle/input"),
                cwd,
                cwd.parent,
            ]

            for root in search_roots:
                if root.exists():
                    candidates.extend(root.glob("**/ser_processed"))
                    candidates.extend(p.parent for p in root.glob("**/baseline_features.npz"))

            seen = set()
            unique_candidates = []
            for candidate in candidates:
                try:
                    resolved = candidate.resolve()
                except Exception:
                    resolved = candidate
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    unique_candidates.append(resolved)

            valid = [p for p in unique_candidates if looks_like_processed_dir(p)]
            if not valid:
                checked = "\\n".join(str(p) for p in unique_candidates[:40])
                raise FileNotFoundError(
                    "Không tìm thấy processed artifact gồm metadata.csv và baseline_features.npz. "
                    "Hãy chắc chắn Kaggle input hoặc local workspace có folder ser_processed.\\n"
                    f"Đã kiểm tra một số path:\\n{checked}"
                )

            # Prefer writable Kaggle/local working outputs over read-only input only when artifacts exist there.
            valid = sorted(valid, key=lambda p: ("/kaggle/input" in str(p), len(str(p))))
            return valid[0]


        PROCESSED_DIR = find_processed_dir()

        if Path("/kaggle/working").exists():
            PROJECT_ROOT = Path("/kaggle/working/Speech_Project")
        else:
            PROJECT_ROOT = Path.cwd().resolve()

        OUTPUT_DIR = PROJECT_ROOT / "04_train_baseline_mfcc" / "outputs"
        MODEL_DIR = OUTPUT_DIR / "models"
        FIGURE_DIR = OUTPUT_DIR / "figures"
        REPORT_DIR = OUTPUT_DIR / "reports"
        PRED_DIR = OUTPUT_DIR / "predictions"

        for directory in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, REPORT_DIR, PRED_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        print("PROCESSED_DIR:", PROCESSED_DIR)
        print("OUTPUT_DIR:", OUTPUT_DIR)
        """
    ),
    md(
        """
        ## 3. Load metadata, feature và config

        Feature dùng trong notebook này là `X_scaled` từ `baseline_features.npz`. Scaler đã được fit trên train split ở notebook Data Processing, nên tránh được data leakage.
        """
    ),
    code(
        """
        METADATA_PATH = PROCESSED_DIR / "metadata.csv"
        FEATURE_PATH = PROCESSED_DIR / "baseline_features.npz"
        CONFIG_PATH = PROCESSED_DIR / "processing_config.json"

        metadata = pd.read_csv(METADATA_PATH)
        features = np.load(FEATURE_PATH, allow_pickle=True)

        if CONFIG_PATH.exists():
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        else:
            config = {}

        print("Metadata shape:", metadata.shape)
        print("Feature arrays:", features.files)
        print("Config:", json.dumps(config, indent=2, ensure_ascii=False))

        display(metadata.head())
        """
    ),
    md(
        """
        ## 4. Kiểm tra chất lượng input

        Các kiểm tra bắt buộc:

        - Số dòng metadata khớp số vector feature.
        - Không duplicate `sample_id`.
        - Không duplicate `dataset + source_filename` nếu có cột `source_filename`.
        - Split có đủ train/validation/test.
        - Không overlap speaker giữa các split.
        """
    ),
    code(
        """
        X = features["X_scaled"].astype(np.float32)
        y = features["y"].astype(str)
        feature_sample_ids = features["sample_id"].astype(str)
        feature_splits = features["split"].astype(str)

        assert len(metadata) == len(X) == len(y) == len(feature_sample_ids), "Metadata và feature không khớp số dòng."
        assert metadata["sample_id"].astype(str).tolist() == feature_sample_ids.tolist(), "sample_id metadata và npz không cùng thứ tự."
        assert metadata["split"].astype(str).tolist() == feature_splits.tolist(), "split metadata và npz không cùng thứ tự."
        assert metadata["sample_id"].duplicated().sum() == 0, "sample_id bị duplicate."

        if "source_filename" in metadata.columns:
            duplicate_source = metadata.duplicated(["dataset", "source_filename"]).sum()
        else:
            duplicate_source = metadata.assign(
                source_filename=metadata["filepath"].map(lambda x: Path(str(x)).name.lower())
            ).duplicated(["dataset", "source_filename"]).sum()

        print("Duplicate dataset + filename:", duplicate_source)
        print("Split counts:")
        display(metadata["split"].value_counts().rename_axis("split").reset_index(name="samples"))

        print("Dataset x split:")
        display(pd.crosstab(metadata["dataset"], metadata["split"]))

        print("Emotion x split:")
        display(pd.crosstab(metadata["emotion"], metadata["split"]))

        speaker_overlap = {}
        for s1, s2 in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            a = set(metadata.loc[metadata["split"] == s1, "speaker_id"].astype(str))
            b = set(metadata.loc[metadata["split"] == s2, "speaker_id"].astype(str))
            speaker_overlap[f"{s1}-{s2}"] = len(a & b)

        print("Speaker overlap:", speaker_overlap)
        assert all(v == 0 for v in speaker_overlap.values()), "Có speaker overlap giữa các split."
        assert duplicate_source == 0, "Còn duplicate dataset + filename."
        """
    ),
    md(
        """
        ## 5. Chuẩn bị train/validation/test

        Ta dùng:

        - `train`: fit model.
        - `validation`: chọn model tốt nhất theo macro-F1.
        - `test`: báo cáo kết quả cuối cùng.

        Macro-F1 được ưu tiên vì các emotion không cân bằng hoàn toàn.
        """
    ),
    code(
        """
        label_order = ["neutral", "happy", "sad", "angry", "fear", "disgust"]
        label_order = [label for label in label_order if label in sorted(set(y))]

        train_mask = feature_splits == "train"
        val_mask = feature_splits == "validation"
        test_mask = feature_splits == "test"

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        meta_train = metadata.loc[train_mask].reset_index(drop=True)
        meta_val = metadata.loc[val_mask].reset_index(drop=True)
        meta_test = metadata.loc[test_mask].reset_index(drop=True)

        print("Train:", X_train.shape, "Validation:", X_val.shape, "Test:", X_test.shape)
        print("Labels:", label_order)
        """
    ),
    md(
        """
        ## 6. Khai báo baseline models

        Models được train:

        1. **Logistic Regression**: baseline tuyến tính, nhanh, dễ kiểm tra sanity.
        2. **SVM RBF**: baseline truyền thống mạnh cho MFCC/statistical features.
        3. **Random Forest**: baseline phi tuyến, dễ giải thích feature-independent hơn neural network.

        Nếu Kaggle/CPU quá chậm, có thể đặt `RUN_RBF_SVM = False`, nhưng bản báo cáo nên có SVM nếu chạy được.
        """
    ),
    code(
        """
        RUN_LOGISTIC_REGRESSION = True
        RUN_RBF_SVM = True
        RUN_RANDOM_FOREST = True

        models = {}

        if RUN_LOGISTIC_REGRESSION:
            models["logistic_regression"] = LogisticRegression(
                max_iter=3000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        if RUN_RBF_SVM:
            models["svm_rbf"] = SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                cache_size=1000,
                random_state=RANDOM_STATE,
            )

        if RUN_RANDOM_FOREST:
            models["random_forest"] = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        print("Models:", list(models.keys()))
        """
    ),
    md(
        """
        ## 7. Train và evaluate models

        Mỗi model được đánh giá trên validation và test. Kết quả lưu vào:

        - `reports/metrics_summary.csv`
        - `reports/classification_report_*.csv`
        - `predictions/predictions_*.csv`
        - `models/*.pkl`
        """
    ),
    code(
        """
        def compute_metrics(y_true, y_pred):
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            }


        def evaluate_split(model_name, split_name, y_true, y_pred, meta):
            row = {
                "model": model_name,
                "split": split_name,
                "n_samples": len(y_true),
                **compute_metrics(y_true, y_pred),
            }

            pred_df = meta[["sample_id", "dataset", "speaker_id", "emotion", "split"]].copy()
            pred_df["y_true"] = y_true
            pred_df["y_pred"] = y_pred
            pred_df["correct"] = pred_df["y_true"] == pred_df["y_pred"]
            pred_df.to_csv(PRED_DIR / f"predictions_{model_name}_{split_name}.csv", index=False)

            report = classification_report(
                y_true,
                y_pred,
                labels=label_order,
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).T
            report_df.to_csv(REPORT_DIR / f"classification_report_{model_name}_{split_name}.csv")
            return row, pred_df


        metrics_rows = []
        trained_models = {}
        prediction_cache = {}

        for model_name, model in models.items():
            print(f"\\n=== Training {model_name} ===")
            start = time.perf_counter()
            model.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            print(f"Training time: {train_time:.2f}s")

            model_path = MODEL_DIR / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            trained_models[model_name] = model

            for split_name, X_split, y_split, meta_split in [
                ("validation", X_val, y_val, meta_val),
                ("test", X_test, y_test, meta_test),
            ]:
                pred_start = time.perf_counter()
                y_pred = model.predict(X_split)
                inference_time = time.perf_counter() - pred_start
                row, pred_df = evaluate_split(model_name, split_name, y_split, y_pred, meta_split)
                row["train_time_sec"] = train_time
                row["inference_time_sec"] = inference_time
                row["inference_ms_per_sample"] = (inference_time / max(len(y_split), 1)) * 1000
                metrics_rows.append(row)
                prediction_cache[(model_name, split_name)] = pred_df
                print(split_name, {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})

        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(REPORT_DIR / "metrics_summary.csv", index=False)
        display(metrics_df.sort_values(["split", "macro_f1"], ascending=[True, False]))

        best_row = (
            metrics_df[metrics_df["split"] == "validation"]
            .sort_values("macro_f1", ascending=False)
            .iloc[0]
        )
        best_model_name = best_row["model"]
        best_model_path = MODEL_DIR / "best_baseline_model.pkl"
        joblib.dump(trained_models[best_model_name], best_model_path)

        label_info = {
            "label_order": label_order,
            "best_model": best_model_name,
            "selection_metric": "validation_macro_f1",
            "processed_dir": str(PROCESSED_DIR),
        }
        (MODEL_DIR / "baseline_label_info.json").write_text(
            json.dumps(label_info, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Best baseline model:", best_model_name)
        print("Saved:", best_model_path)
        """
    ),
    md(
        """
        ## 8. Visualize model comparison

        Biểu đồ này dùng để đưa vào báo cáo: so sánh accuracy và macro-F1 giữa các baseline.
        """
    ),
    code(
        """
        plot_df = metrics_df.melt(
            id_vars=["model", "split"],
            value_vars=["accuracy", "macro_f1", "weighted_f1"],
            var_name="metric",
            value_name="score",
        )

        plt.figure(figsize=(11, 5))
        sns.barplot(data=plot_df, x="model", y="score", hue="metric")
        plt.title("Baseline model comparison")
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        comparison_path = FIGURE_DIR / "baseline_model_comparison.png"
        plt.savefig(comparison_path, dpi=180)
        plt.show()

        plt.figure(figsize=(9, 5))
        sns.barplot(data=metrics_df, x="model", y="macro_f1", hue="split")
        plt.title("Validation/Test Macro-F1 by baseline model")
        plt.ylim(0, 1)
        plt.xticks(rotation=20)
        plt.tight_layout()
        macro_path = FIGURE_DIR / "baseline_macro_f1_by_split.png"
        plt.savefig(macro_path, dpi=180)
        plt.show()

        print("Saved figures:", comparison_path, macro_path)
        """
    ),
    md(
        """
        ## 9. Confusion matrix

        Confusion matrix giúp phân tích emotion nào bị nhầm. Đây là phần rất quan trọng trong báo cáo SER.
        """
    ),
    code(
        """
        for model_name in models:
            pred_df = prediction_cache[(model_name, "test")]
            cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"], labels=label_order)
            cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_order, yticklabels=label_order, ax=axes[0])
            axes[0].set_title(f"{model_name} - Test confusion matrix")
            axes[0].set_xlabel("Predicted")
            axes[0].set_ylabel("True")

            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=label_order, yticklabels=label_order, ax=axes[1])
            axes[1].set_title(f"{model_name} - Normalized confusion matrix")
            axes[1].set_xlabel("Predicted")
            axes[1].set_ylabel("True")

            plt.tight_layout()
            out_path = FIGURE_DIR / f"confusion_matrix_{model_name}_test.png"
            plt.savefig(out_path, dpi=180)
            plt.show()
            print("Saved:", out_path)
        """
    ),
    md(
        """
        ## 10. Per-dataset performance

        Phân tích theo dataset giúp trả lời câu hỏi: model học tốt trên dataset nào, generalization có ổn không, TESS có làm kết quả bị lệch không.
        """
    ),
    code(
        """
        per_dataset_rows = []

        for model_name in models:
            pred_df = prediction_cache[(model_name, "test")]
            for dataset_name, part in pred_df.groupby("dataset"):
                per_dataset_rows.append({
                    "model": model_name,
                    "dataset": dataset_name,
                    "n_samples": len(part),
                    **compute_metrics(part["y_true"], part["y_pred"]),
                })

        per_dataset_df = pd.DataFrame(per_dataset_rows)
        per_dataset_df.to_csv(REPORT_DIR / "per_dataset_metrics_test.csv", index=False)
        display(per_dataset_df.sort_values(["model", "macro_f1"], ascending=[True, False]))

        plt.figure(figsize=(10, 5))
        sns.barplot(data=per_dataset_df, x="dataset", y="macro_f1", hue="model")
        plt.title("Test Macro-F1 by dataset")
        plt.ylim(0, 1)
        plt.tight_layout()
        dataset_path = FIGURE_DIR / "per_dataset_macro_f1_test.png"
        plt.savefig(dataset_path, dpi=180)
        plt.show()
        print("Saved:", dataset_path)
        """
    ),
    md(
        """
        ## 11. Per-class F1

        Biểu đồ F1 theo từng emotion cho biết emotion nào khó nhất. Phần này nên đưa vào report cùng confusion matrix.
        """
    ),
    code(
        """
        per_class_rows = []

        for model_name in models:
            report_path = REPORT_DIR / f"classification_report_{model_name}_test.csv"
            report_df = pd.read_csv(report_path, index_col=0)
            for emotion in label_order:
                if emotion in report_df.index:
                    per_class_rows.append({
                        "model": model_name,
                        "emotion": emotion,
                        "precision": report_df.loc[emotion, "precision"],
                        "recall": report_df.loc[emotion, "recall"],
                        "f1_score": report_df.loc[emotion, "f1-score"],
                        "support": report_df.loc[emotion, "support"],
                    })

        per_class_df = pd.DataFrame(per_class_rows)
        per_class_df.to_csv(REPORT_DIR / "per_class_metrics_test.csv", index=False)
        display(per_class_df)

        plt.figure(figsize=(12, 5))
        sns.barplot(data=per_class_df, x="emotion", y="f1_score", hue="model", order=label_order)
        plt.title("Test F1-score by emotion")
        plt.ylim(0, 1)
        plt.tight_layout()
        class_path = FIGURE_DIR / "per_class_f1_test.png"
        plt.savefig(class_path, dpi=180)
        plt.show()
        print("Saved:", class_path)
        """
    ),
    md(
        """
        ## 12. Error analysis nhanh

        Cell này hiển thị một số mẫu dự đoán sai của model tốt nhất. Khi viết báo cáo, có thể dùng phần này để phân tích limitation.
        """
    ),
    code(
        """
        best_pred_df = prediction_cache[(best_model_name, "test")]
        errors = best_pred_df[~best_pred_df["correct"]].copy()

        print("Best model:", best_model_name)
        print("Test errors:", len(errors), "/", len(best_pred_df))

        display(
            errors[["sample_id", "dataset", "speaker_id", "y_true", "y_pred"]]
            .head(30)
        )

        error_table = pd.crosstab(errors["y_true"], errors["y_pred"]).reindex(index=label_order, columns=label_order, fill_value=0)
        display(error_table)
        error_table.to_csv(REPORT_DIR / f"error_crosstab_{best_model_name}_test.csv")
        """
    ),
    md(
        """
        ## 13. Lưu summary cho report

        Cell này tạo một file JSON tóm tắt kết quả quan trọng nhất để dùng trong report/slide.
        """
    ),
    code(
        """
        test_best = metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == "test")].iloc[0].to_dict()
        val_best = metrics_df[(metrics_df["model"] == best_model_name) & (metrics_df["split"] == "validation")].iloc[0].to_dict()

        summary = {
            "best_model": best_model_name,
            "validation": val_best,
            "test": test_best,
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
            "labels": label_order,
            "feature_dim": int(X.shape[1]),
            "models_trained": list(models.keys()),
        }

        summary_path = REPORT_DIR / "baseline_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Saved:", summary_path)
        """
    ),
    md(
        """
        ## 14. Đóng gói output để tải về

        Kaggle thường hỗ trợ tải `.zip` tốt nhất. Nếu môi trường có lệnh `rar`, cell này sẽ tạo thêm `.rar`; nếu không có, notebook sẽ tự fallback sang `.zip`.

        Output nằm trong:

        - Kaggle: `/kaggle/working/baseline_mfcc_results.zip`
        - Local: folder project hiện tại
        """
    ),
    code(
        """
        archive_base = PROJECT_ROOT / "baseline_mfcc_results"
        zip_path = archive_base.with_suffix(".zip")

        if zip_path.exists():
            zip_path.unlink()

        shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(OUTPUT_DIR.parent),
            base_dir=OUTPUT_DIR.name,
        )
        print("Created ZIP:", zip_path)

        rar_exe = shutil.which("rar")
        rar_path = archive_base.with_suffix(".rar")

        if rar_exe:
            if rar_path.exists():
                rar_path.unlink()
            try:
                subprocess.run(
                    [rar_exe, "a", "-r", str(rar_path), OUTPUT_DIR.name],
                    cwd=str(OUTPUT_DIR.parent),
                    check=True,
                )
                print("Created RAR:", rar_path)
            except Exception as exc:
                print("RAR creation failed, ZIP is still available:", exc)
        else:
            print("RAR executable not found. ZIP package is available instead.")

        print("Package size MB:", round(zip_path.stat().st_size / 1024 / 1024, 2))
        """
    ),
    md(
        """
        ## 15. Checklist sau khi chạy

        Sau khi chạy notebook, kiểm tra các file sau:

        - `models/best_baseline_model.pkl`
        - `models/logistic_regression.pkl`
        - `models/svm_rbf.pkl`
        - `models/random_forest.pkl`
        - `reports/metrics_summary.csv`
        - `reports/per_dataset_metrics_test.csv`
        - `reports/per_class_metrics_test.csv`
        - `figures/baseline_model_comparison.png`
        - `figures/confusion_matrix_*_test.png`
        - `baseline_mfcc_results.zip`

        Nếu SVM RBF quá chậm trên Kaggle CPU, tắt `RUN_RBF_SVM` rồi chạy lại để có Logistic Regression và Random Forest trước.
        """
    ),
]

nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
