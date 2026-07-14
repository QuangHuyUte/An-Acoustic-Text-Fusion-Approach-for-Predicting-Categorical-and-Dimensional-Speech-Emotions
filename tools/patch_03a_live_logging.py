import json
from pathlib import Path


NOTEBOOK = Path(
    r"06_w2v_based_models\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold"
    r"\03A_Emotion2Vec_Pretrained_RawAudio_Backbone_Finetune_5_10Fold.ipynb"
)


def replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise SystemExit(f"Pattern not found:\n{old[:400]}")
    return src.replace(old, new, 1)


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

for cell in nb["cells"]:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))

    if "OUTPUT_DIR = Path(os.getenv" in src and "MODEL_DIR = OUTPUT_DIR / \"models\"" in src:
        src = replace_once(
            src,
            "for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:\n"
            "    p.mkdir(parents=True, exist_ok=True)\n\n"
            "print(\"Run mode:\", RUN_MODE)\n",
            "for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:\n"
            "    p.mkdir(parents=True, exist_ok=True)\n\n"
            "LIVE_LOG_PATH = REPORT_DIR / \"03A_live_training_log.txt\"\n"
            "LOG_EVERY_STEPS = int(os.getenv(\"LOG_EVERY_STEPS\", \"25\"))\n\n"
            "def live_log(message):\n"
            "    stamp = time.strftime(\"%Y-%m-%d %H:%M:%S\")\n"
            "    text = f\"[{stamp}] {message}\"\n"
            "    print(text, flush=True)\n"
            "    with LIVE_LOG_PATH.open(\"a\", encoding=\"utf-8\") as f:\n"
            "        f.write(text + \"\\n\")\n\n"
            "print(\"Run mode:\", RUN_MODE)\n",
        )
        src = replace_once(
            src,
            "    \"EVAL_TRAIN_SPLIT\": EVAL_TRAIN_SPLIT,\n"
            "})\n",
            "    \"EVAL_TRAIN_SPLIT\": EVAL_TRAIN_SPLIT,\n"
            "    \"LOG_EVERY_STEPS\": LOG_EVERY_STEPS,\n"
            "})\n"
            "live_log(\"03A configuration is ready.\")\n",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "class RawBackboneMultiTaskSER" in src and "AutoModel.from_pretrained(model_name)" in src:
        src = replace_once(
            src,
            "        self.backbone = AutoModel.from_pretrained(model_name)\n",
            "        live_log(f\"Loading raw-audio pretrained backbone: {model_name}\")\n"
            "        self.backbone = AutoModel.from_pretrained(model_name)\n"
            "        live_log(\"Raw-audio pretrained backbone loaded.\")\n",
        )
        src = replace_once(
            src,
            "    print(note)\n"
            "    print(f\"Trainable parameters: {trainable:,}/{total:,} ({trainable/max(total,1):.2%})\")\n",
            "    live_log(note)\n"
            "    live_log(f\"Trainable parameters: {trainable:,}/{total:,} ({trainable/max(total,1):.2%})\")\n",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "def evaluate(model, loader, split_name, class_weights=None):" in src:
        src = replace_once(
            src,
            "    total_loss, n_batches = 0.0, 0\n"
            "    for batch in loader:\n",
            "    total_loss, n_batches = 0.0, 0\n"
            "    eval_start = time.time()\n"
            "    live_log(f\"Evaluate {split_name}: batches={len(loader)}\")\n"
            "    for batch_idx, batch in enumerate(loader, start=1):\n",
        )
        src = replace_once(
            src,
            "        total_loss += float(loss.detach().cpu())\n"
            "        n_batches += 1\n",
            "        total_loss += float(loss.detach().cpu())\n"
            "        n_batches += 1\n"
            "        if batch_idx == 1 or batch_idx % LOG_EVERY_STEPS == 0 or batch_idx == len(loader):\n"
            "            live_log(f\"Evaluate {split_name}: batch {batch_idx}/{len(loader)} elapsed={time.time()-eval_start:.1f}s\")\n",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "def train_fold(protocol, fold_name, fold_df, seed):" in src:
        src = replace_once(
            src,
            "    print(f\"\\n=== {protocol} | {fold_name} ===\")\n"
            "    print(\"Train/Val/Test:\", len(train_df), len(val_df), len(test_df))\n\n"
            "    train_loader = make_loader(train_df, shuffle=True)\n"
            "    val_loader = make_loader(val_df, shuffle=False)\n"
            "    test_loader = make_loader(test_df, shuffle=False)\n\n"
            "    model = build_model()\n",
            "    live_log(f\"=== {protocol} | {fold_name} ===\")\n"
            "    live_log(f\"Train/Val/Test: {len(train_df)} {len(val_df)} {len(test_df)}\")\n\n"
            "    live_log(\"Build raw-audio DataLoaders\")\n"
            "    train_loader = make_loader(train_df, shuffle=True)\n"
            "    val_loader = make_loader(val_df, shuffle=False)\n"
            "    test_loader = make_loader(test_df, shuffle=False)\n"
            "    live_log(f\"DataLoaders ready. Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}\")\n\n"
            "    live_log(\"Build raw-audio pretrained model\")\n"
            "    model = build_model()\n",
        )
        src = replace_once(
            src,
            "    if class_weights is not None:\n"
            "        print(\"Class weights:\", [round(float(x), 4) for x in class_weights.detach().cpu()])\n",
            "    if class_weights is not None:\n"
            "        live_log(\"Class weights: \" + str([round(float(x), 4) for x in class_weights.detach().cpu()]))\n",
        )
        src = replace_once(
            src,
            "    for epoch in range(1, EPOCHS + 1):\n"
            "        model.train()\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        running = 0.0\n"
            "        start = time.time()\n"
            "        for step, batch in enumerate(train_loader, start=1):\n",
            "    live_log(\"Optimizer, scheduler, and AMP scaler are ready. Start training loop.\")\n\n"
            "    for epoch in range(1, EPOCHS + 1):\n"
            "        model.train()\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        running = 0.0\n"
            "        start = time.time()\n"
            "        live_log(f\"Epoch {epoch:03d} started. Train batches={len(train_loader)}\")\n"
            "        for step, batch in enumerate(train_loader, start=1):\n",
        )
        src = replace_once(
            src,
            "            running += float(loss.detach().cpu()) * GRAD_ACCUM\n\n"
            "        val_metrics, _, _ = evaluate(model, val_loader, \"val\", class_weights=class_weights)\n",
            "            running += float(loss.detach().cpu()) * GRAD_ACCUM\n"
            "            if step == 1 or step % LOG_EVERY_STEPS == 0 or step == len(train_loader):\n"
            "                live_log(\n"
            "                    f\"Epoch {epoch:03d} step {step}/{len(train_loader)} \"\n"
            "                    f\"loss={running/max(step,1):.4f} elapsed={time.time()-start:.1f}s\"\n"
            "                )\n\n"
            "        live_log(f\"Epoch {epoch:03d} training done. Start validation.\")\n"
            "        val_metrics, _, _ = evaluate(model, val_loader, \"val\", class_weights=class_weights)\n",
        )
        src = replace_once(
            src,
            "        history.append(row)\n"
            "        print(\n"
            "            f\"Epoch {epoch:02d} | train_loss={row['train_loss']:.4f} | \"\n"
            "            f\"val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | \"\n"
            "            f\"score={score:.4f} | lr={row['lr_min']:.2e}-{row['lr_max']:.2e}\"\n"
            "        )\n",
            "        history.append(row)\n"
            "        pd.DataFrame(history).to_csv(REPORT_DIR / f\"{protocol}_{fold_name}_history_live.csv\", index=False, encoding=\"utf-8-sig\")\n"
            "        live_log(\n"
            "            f\"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | \"\n"
            "            f\"val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | \"\n"
            "            f\"score={score:.4f} | lr={row['lr_min']:.2e}-{row['lr_max']:.2e}\"\n"
            "        )\n",
        )
        src = replace_once(
            src,
            "            torch.save({\"model_state_dict\": module_state_dict(model), \"best_epoch\": best_epoch, \"best_val_score\": best_score}, best_path)\n",
            "            torch.save({\"model_state_dict\": module_state_dict(model), \"best_epoch\": best_epoch, \"best_val_score\": best_score}, best_path)\n"
            "            live_log(f\"Saved new best checkpoint: epoch={epoch}, score={best_score:.4f}, path={best_path}\")\n",
        )
        src = replace_once(
            src,
            "                print(f\"Early stopping at epoch {epoch}; best epoch={best_epoch}, best score={best_score:.4f}\")\n",
            "                live_log(f\"Early stopping at epoch {epoch}; best epoch={best_epoch}, best score={best_score:.4f}\")\n",
        )
        src = replace_once(
            src,
            "    checkpoint = torch.load(best_path, map_location=DEVICE)\n"
            "    load_module_state_dict(model, checkpoint[\"model_state_dict\"])\n",
            "    live_log(f\"Load best checkpoint from {best_path}\")\n"
            "    checkpoint = torch.load(best_path, map_location=DEVICE)\n"
            "    load_module_state_dict(model, checkpoint[\"model_state_dict\"])\n",
        )
        src = replace_once(
            src,
            "    for split_name, loader in split_loaders:\n"
            "        metrics, pred_df, feature_npz = evaluate(model, loader, split_name, class_weights=class_weights)\n",
            "    for split_name, loader in split_loaders:\n"
            "        live_log(f\"Evaluate/export split={split_name}\")\n"
            "        metrics, pred_df, feature_npz = evaluate(model, loader, split_name, class_weights=class_weights)\n",
        )
        src = replace_once(
            src,
            "    print(\"Test:\", {k: result[k] for k in [\"WA\", \"UAR\", \"Macro_F1\", \"CCC_mean\", \"MAE_mean\"]})\n",
            "    live_log(\"Test: \" + str({k: result[k] for k in [\"WA\", \"UAR\", \"Macro_F1\", \"CCC_mean\", \"MAE_mean\"]}))\n",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "all_results = []" in src and "results_df = pd.DataFrame(all_results)" in src:
        src = replace_once(
            src,
            "        all_results.append(train_fold(protocol, fold, df[df[\"fold\"] == fold].reset_index(drop=True), SEED + idx))\n",
            "        result = train_fold(protocol, fold, df[df[\"fold\"] == fold].reset_index(drop=True), SEED + idx)\n"
            "        all_results.append(result)\n"
            "        pd.DataFrame(all_results).to_csv(REPORT_DIR / \"03A_results_live.csv\", index=False, encoding=\"utf-8-sig\")\n"
            "        live_log(f\"Finished fold {fold}. Live results saved.\")\n",
        )
        src = replace_once(
            src,
            "print(\"Total seconds:\", round(time.time() - start_all, 2))",
            "live_log(f\"Total seconds: {round(time.time() - start_all, 2)}\")",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK)
