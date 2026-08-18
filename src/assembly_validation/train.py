from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import ActionLSTM, ModelConfig


class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, str]], class_to_id: dict[str, int]) -> None:
        self.rows = rows
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with np.load(row["sample_path"]) as sample:
            features = sample["features"].astype(np.float32)
        return torch.from_numpy(features), torch.tensor(self.class_to_id[row["label"]])


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = total_correct = total_items = 0
    with torch.no_grad():
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            total_loss += float(loss_fn(logits, targets)) * len(targets)
            total_correct += int((logits.argmax(1) == targets).sum())
            total_items += len(targets)
    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_index(Path(args.index))
    if not rows:
        raise ValueError("Processed index is empty")
    class_names = sorted({row["label"] for row in rows})
    class_to_id = {name: index for index, name in enumerate(class_names)}
    train_rows = [row for row in rows if row["split"].lower() == "train"]
    val_rows = [row for row in rows if row["split"].lower() in {"val", "validation"}]
    if not train_rows or not val_rows:
        raise ValueError("Index must contain non-empty train and val splits")

    train_loader = DataLoader(
        WindowDataset(train_rows, class_to_id),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        WindowDataset(val_rows, class_to_id), batch_size=args.batch_size, shuffle=False
    )
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if args.device else default_device)
    config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        dropout=args.dropout,
        num_classes=len(class_names),
    )
    model = ActionLSTM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(f"epoch={epoch:03d} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.checkpoint(class_names), output)

    metrics = {
        "best_val_accuracy": best_accuracy,
        "classes": class_names,
        "device": str(device),
        "history": history,
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train causal LSTM action classifier")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
