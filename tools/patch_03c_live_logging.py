import json
from pathlib import Path


NOTEBOOK = Path(
    r"06_w2v_based_models\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold"
    r"\03C_Transcript_Pretrained_Text_MultiTask_5_10Fold.ipynb"
)


def replace_once(src: str, old: str, new: str) -> str:
    if old not in src:
        raise SystemExit(f"Pattern not found:\n{old[:300]}")
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
            "print({",
            "for p in [MODEL_DIR, REPORT_DIR, FUSION_DIR, FIGURE_DIR]:\n"
            "    p.mkdir(parents=True, exist_ok=True)\n\n"
            "LIVE_LOG_PATH = REPORT_DIR / \"03C_live_training_log.txt\"\n"
            "LOG_EVERY_STEPS = int(os.getenv(\"LOG_EVERY_STEPS\", \"25\"))\n\n"
            "def live_log(message):\n"
            "    stamp = time.strftime(\"%Y-%m-%d %H:%M:%S\")\n"
            "    text = f\"[{stamp}] {message}\"\n"
            "    print(text, flush=True)\n"
            "    with LIVE_LOG_PATH.open(\"a\", encoding=\"utf-8\") as f:\n"
            "        f.write(text + \"\\n\")\n\n"
            "print({",
        )
        src = replace_once(
            src,
            "    \"OUTPUT_DIR\": str(OUTPUT_DIR),\n"
            "})",
            "    \"OUTPUT_DIR\": str(OUTPUT_DIR),\n"
            "    \"LOG_EVERY_STEPS\": LOG_EVERY_STEPS,\n"
            "})\n"
            "live_log(\"03C configuration is ready.\")",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)" in src:
        src = replace_once(
            src,
            "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)\n",
            "live_log(f\"Loading tokenizer: {TEXT_MODEL_NAME}\")\n"
            "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)\n"
            "live_log(\"Tokenizer loaded.\")\n",
        )
        src = replace_once(
            src,
            "sample_batch = next(iter(make_loader(metadata.head(4), shuffle=False)))\n"
            "print({k: (v.shape if hasattr(v, \"shape\") else v[:2]) for k, v in sample_batch.items()})",
            "live_log(\"Building a small sample batch for tokenizer/data-loader sanity check...\")\n"
            "sample_batch = next(iter(make_loader(metadata.head(4), shuffle=False)))\n"
            "live_log(str({k: (v.shape if hasattr(v, \"shape\") else v[:2]) for k, v in sample_batch.items()}))",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "class TranscriptMultiTaskSER" in src and "AutoModel.from_pretrained(model_name)" in src:
        src = replace_once(
            src,
            "        self.backbone = AutoModel.from_pretrained(model_name)\n",
            "        live_log(f\"Loading text backbone: {model_name}\")\n"
            "        self.backbone = AutoModel.from_pretrained(model_name)\n"
            "        live_log(\"Text backbone loaded.\")\n",
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
            "    test_loader = make_loader(test_df, shuffle=False)\n",
            "    live_log(f\"=== {protocol} | {fold_name} ===\")\n"
            "    live_log(f\"Train/Val/Test: {len(train_df)} {len(val_df)} {len(test_df)}\")\n\n"
            "    live_log(\"Build DataLoaders\")\n"
            "    train_loader = make_loader(train_df, shuffle=True)\n"
            "    val_loader = make_loader(val_df, shuffle=False)\n"
            "    test_loader = make_loader(test_df, shuffle=False)\n"
            "    live_log(f\"DataLoaders ready. Train batches={len(train_loader)}, Val batches={len(val_loader)}, Test batches={len(test_loader)}\")\n",
        )
        src = replace_once(
            src,
            "    model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)\n",
            "    live_log(f\"Build TranscriptMultiTaskSER with backbone={TEXT_MODEL_NAME}\")\n"
            "    model = TranscriptMultiTaskSER(TEXT_MODEL_NAME, num_classes=4, dropout=DROPOUT).to(DEVICE)\n",
        )
        src = replace_once(
            src,
            "    scaler = GradScaler(\"cuda\", enabled=amp_enabled())\n\n"
            "    best_score = -1e9\n"
            "    best_state = None\n"
            "    best_epoch = 0\n"
            "    stale = 0\n"
            "    history = []\n",
            "    scaler = GradScaler(\"cuda\", enabled=amp_enabled())\n"
            "    live_log(\"Optimizer, scheduler, and AMP scaler are ready. Start training loop.\")\n\n"
            "    best_score = -1e9\n"
            "    best_epoch = 0\n"
            "    stale = 0\n"
            "    history = []\n"
            "    best_path = MODEL_DIR / f\"{protocol}_{fold_name}_best.pt\"\n",
        )
        src = replace_once(
            src,
            "    for epoch in range(1, EPOCHS + 1):\n"
            "        model.train()\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        total_loss, n_steps = 0.0, 0\n"
            "        for step, batch in enumerate(train_loader, start=1):\n",
            "    for epoch in range(1, EPOCHS + 1):\n"
            "        model.train()\n"
            "        optimizer.zero_grad(set_to_none=True)\n"
            "        total_loss, n_steps = 0.0, 0\n"
            "        epoch_start = time.time()\n"
            "        live_log(f\"Epoch {epoch:03d} started. Train batches={len(train_loader)}\")\n"
            "        for step, batch in enumerate(train_loader, start=1):\n",
        )
        src = replace_once(
            src,
            "            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM\n"
            "            n_steps += 1\n\n"
            "        val_metrics, _ = evaluate_model(model, val_loader, class_weight=class_weight)\n",
            "            total_loss += float(loss.detach().cpu()) * GRAD_ACCUM\n"
            "            n_steps += 1\n"
            "            if step == 1 or step % LOG_EVERY_STEPS == 0 or step == len(train_loader):\n"
            "                live_log(\n"
            "                    f\"Epoch {epoch:03d} step {step}/{len(train_loader)} \"\n"
            "                    f\"loss={total_loss/max(n_steps,1):.4f} elapsed={time.time()-epoch_start:.1f}s\"\n"
            "                )\n\n"
            "        live_log(f\"Epoch {epoch:03d} training done. Start validation.\")\n"
            "        val_metrics, _ = evaluate_model(model, val_loader, class_weight=class_weight)\n",
        )
        src = replace_once(
            src,
            "        history.append(row)\n"
            "        print(f\"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}\")\n\n"
            "        if score > best_score + MIN_DELTA:\n"
            "            best_score = score\n"
            "            best_state = {k: v.detach().cpu().clone() for k, v in state_dict_clean(model).items()}\n"
            "            best_epoch = epoch\n"
            "            stale = 0\n",
            "        history.append(row)\n"
            "        pd.DataFrame(history).to_csv(REPORT_DIR / f\"{protocol}_{fold_name}_history_live.csv\", index=False, encoding=\"utf-8-sig\")\n"
            "        live_log(f\"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_UAR={val_metrics['UAR']:.4f} | val_CCC={val_metrics['CCC_mean']:.4f} | score={score:.4f}\")\n\n"
            "        if score > best_score + MIN_DELTA:\n"
            "            best_score = score\n"
            "            if best_path.exists():\n"
            "                best_path.unlink()\n"
            "            torch.save(state_dict_clean(model), best_path)\n"
            "            live_log(f\"Saved new best checkpoint: epoch={epoch}, score={best_score:.4f}, path={best_path}\")\n"
            "            best_epoch = epoch\n"
            "            stale = 0\n",
        )
        src = replace_once(
            src,
            "                print(\"Early stopping\")\n",
            "                live_log(\"Early stopping\")\n",
        )
        src = replace_once(
            src,
            "    load_state_dict_clean(model, best_state)\n"
            "    test_metrics, _ = evaluate_model(model, test_loader, class_weight=class_weight)\n",
            "    live_log(f\"Load best checkpoint from {best_path}\")\n"
            "    best_state = torch.load(best_path, map_location=\"cpu\")\n"
            "    load_state_dict_clean(model, best_state)\n"
            "    live_log(\"Start final test evaluation\")\n"
            "    test_metrics, _ = evaluate_model(model, test_loader, class_weight=class_weight)\n",
        )
        src = replace_once(
            src,
            "    print(\"Test:\", {k: result[k] for k in [\"WA\", \"UAR\", \"Macro_F1\", \"CCC_mean\", \"MAE_mean\"]})\n\n"
            "    torch.save(best_state, MODEL_DIR / f\"{protocol}_{fold_name}_best.pt\")\n",
            "    live_log(\"Test: \" + str({k: result[k] for k in [\"WA\", \"UAR\", \"Macro_F1\", \"CCC_mean\", \"MAE_mean\"]}))\n\n",
        )
        src = replace_once(
            src,
            "    for split_name, loader in export_splits:\n"
            "        _, feature_npz = evaluate_model(model, loader, class_weight=class_weight, return_features=True)\n",
            "    for split_name, loader in export_splits:\n"
            "        live_log(f\"Evaluate/export split={split_name}\")\n"
            "        _, feature_npz = evaluate_model(model, loader, class_weight=class_weight, return_features=True)\n",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

    if "all_results = []" in src and "results_df = pd.DataFrame(all_results)" in src:
        src = replace_once(
            src,
            "        fold_df = split_df[split_df[\"fold\"] == fold].reset_index(drop=True)\n"
            "        all_results.append(train_fold(protocol, fold, fold_df, SEED + idx))\n",
            "        fold_df = split_df[split_df[\"fold\"] == fold].reset_index(drop=True)\n"
            "        result = train_fold(protocol, fold, fold_df, SEED + idx)\n"
            "        all_results.append(result)\n"
            "        pd.DataFrame(all_results).to_csv(REPORT_DIR / \"03C_results_live.csv\", index=False, encoding=\"utf-8-sig\")\n"
            "        live_log(f\"Finished fold {fold}. Live results saved.\")\n",
        )
        src = replace_once(
            src,
            "print(\"Total seconds:\", round(time.time() - start, 2))",
            "live_log(f\"Total seconds: {round(time.time() - start, 2)}\")",
        )
        cell["source"] = src.splitlines(keepends=True)
        continue

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(NOTEBOOK)
