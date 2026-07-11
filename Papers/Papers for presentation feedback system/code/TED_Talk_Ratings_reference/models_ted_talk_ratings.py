"""Reference TED Talk Ratings models.

These modules are intentionally compact and paper-oriented. They are not
the official implementation from the paper. They provide a clean starting
point for reproducing the Word Sequence LSTM and prosody-fusion ideas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class TEDModelConfig:
    vocab_size: int
    embedding_dim: int = 300
    hidden_dim: int = 256
    num_labels: int = 14
    dropout: float = 0.3
    padding_idx: int = 0
    prosody_dim: int = 8
    prosody_channels: tuple[int, int, int, int] = (16, 16, 32, 64)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    total = (values * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1.0)
    return total / denom


class WordSequenceLSTM(nn.Module):
    """Sentence-level word sequence LSTM with talk-level pooling.

    Expected input shape:
        tokens: [batch, num_sentences, num_words]
        sentence_mask: [batch, num_sentences]
        word_mask: [batch, num_sentences, num_words]
    """

    def __init__(self, config: TEDModelConfig, embedding_weight: torch.Tensor | None = None):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.padding_idx,
        )
        if embedding_weight is not None:
            self.embedding.weight.data.copy_(embedding_weight)

        self.sentence_encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_dim, config.num_labels)

    def encode_sentences(self, tokens: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_sentences, num_words = tokens.shape
        flat_tokens = tokens.reshape(batch_size * num_sentences, num_words)
        flat_mask = word_mask.reshape(batch_size * num_sentences, num_words)

        embedded = self.embedding(flat_tokens)
        outputs, _ = self.sentence_encoder(embedded)
        sentence_vectors = masked_mean(outputs, flat_mask, dim=1)
        return sentence_vectors.reshape(batch_size, num_sentences, self.config.hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        sentence_mask: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentence_vectors = self.encode_sentences(tokens, word_mask)
        talk_vector = masked_mean(sentence_vectors, sentence_mask, dim=1)
        logits = self.classifier(self.dropout(talk_vector))
        return logits


class ProsodyCNN(nn.Module):
    """1D CNN over sentence-level prosody sequences.

    Expected input shape:
        prosody: [batch, num_sentences, time_steps, prosody_dim]
        prosody_mask: [batch, num_sentences]

    The TED paper uses an 8D prosody signal sampled at 10 Hz and four Conv1D
    layers with receptive field 3, ending in a 64D vector.
    """

    def __init__(self, config: TEDModelConfig):
        super().__init__()
        channels = config.prosody_channels
        layers: list[nn.Module] = []
        in_channels = config.prosody_dim
        for index, out_channels in enumerate(channels):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            if index in (1, 2):
                layers.append(nn.MaxPool1d(kernel_size=2))
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.output_dim = channels[-1]

    def forward(self, prosody: torch.Tensor, prosody_mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_sentences, time_steps, prosody_dim = prosody.shape
        flat = prosody.reshape(batch_size * num_sentences, time_steps, prosody_dim)
        flat = flat.transpose(1, 2)
        encoded = self.conv(flat)
        encoded = encoded.max(dim=-1).values
        encoded = encoded.reshape(batch_size, num_sentences, self.output_dim)
        return masked_mean(encoded, prosody_mask, dim=1)


class TextProsodyFusionModel(nn.Module):
    """Word Sequence LSTM plus prosody CNN fusion.

    This is a practical simplification of the paper's Dependency TreeLSTM +
    Prosody CNN setup. It keeps the audio/text fusion idea while avoiding the
    implementation complexity of batching dependency trees.
    """

    def __init__(self, config: TEDModelConfig, embedding_weight: torch.Tensor | None = None):
        super().__init__()
        self.text_model = WordSequenceLSTM(config, embedding_weight=embedding_weight)
        self.prosody_encoder = ProsodyCNN(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim + self.prosody_encoder.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_labels),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        sentence_mask: torch.Tensor,
        word_mask: torch.Tensor,
        prosody: torch.Tensor,
        prosody_mask: torch.Tensor,
    ) -> torch.Tensor:
        sentence_vectors = self.text_model.encode_sentences(tokens, word_mask)
        text_vector = masked_mean(sentence_vectors, sentence_mask, dim=1)
        prosody_vector = self.prosody_encoder(prosody, prosody_mask)
        fused = torch.cat([text_vector, prosody_vector], dim=-1)
        return self.classifier(self.dropout(fused))

