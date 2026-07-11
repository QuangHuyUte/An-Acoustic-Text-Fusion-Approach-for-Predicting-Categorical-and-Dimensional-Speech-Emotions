"""Training skeleton for TED Talk Ratings style models.

This file documents the expected batch structure and provides a minimal train
step. It intentionally leaves dataset loading open because the public code/data
link in the arXiv PDF is not directly available from the paper copy in this
workspace.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch
from torch import nn
from torch.optim import Adam

from models_ted_talk_ratings import TEDModelConfig, TextProsodyFusionModel, WordSequenceLSTM


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_items = 0

    for batch in batches:
        labels = batch["labels"].to(device).float()
        optimizer.zero_grad()

        if "prosody" in batch:
            logits = model(
                batch["tokens"].to(device),
                batch["sentence_mask"].to(device),
                batch["word_mask"].to(device),
                batch["prosody"].to(device),
                batch["prosody_mask"].to(device),
            )
        else:
            logits = model(
                batch["tokens"].to(device),
                batch["sentence_mask"].to(device),
                batch["word_mask"].to(device),
            )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        total_items += labels.size(0)

    return total_loss / max(total_items, 1)


def build_model(args: argparse.Namespace) -> nn.Module:
    config = TEDModelConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_labels=args.num_labels,
        dropout=args.dropout,
    )
    if args.model == "word_lstm":
        return WordSequenceLSTM(config)
    if args.model == "text_prosody":
        return TextProsodyFusionModel(config)
    raise ValueError(f"Unsupported model: {args.model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["word_lstm", "text_prosody"], default="word_lstm")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--embedding-dim", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-labels", type=int, default=14)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(model)
    print("Dataset loading is intentionally project-specific.")
    print("Expected batch keys:")
    print("  tokens: [batch, num_sentences, num_words]")
    print("  sentence_mask: [batch, num_sentences]")
    print("  word_mask: [batch, num_sentences, num_words]")
    print("  labels: [batch, 14]")
    print("  optional prosody: [batch, num_sentences, time_steps, 8]")
    print("  optional prosody_mask: [batch, num_sentences]")
    print(f"Optimizer ready: {optimizer.__class__.__name__} on {device}")


if __name__ == "__main__":
    main()

