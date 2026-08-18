from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .features import FEATURE_DIM


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = FEATURE_DIM
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.25
    num_classes: int = 7


class ActionLSTM(nn.Module):
    """A causal LSTM: predictions never use future video frames."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm(inputs)
        return self.head(sequence[:, -1])

    def checkpoint(self, class_names: list[str]) -> dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "model_config": asdict(self.config),
            "class_names": class_names,
        }


def load_model(path: str, device: torch.device) -> tuple[ActionLSTM, list[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = ActionLSTM(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, list(checkpoint["class_names"])

