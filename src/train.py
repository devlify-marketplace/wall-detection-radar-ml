import argparse
import os

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils import RadarSequenceDataset, build_dataset
from model import CNN_LSTM_HumanDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM model on IWR6843 radar sequences"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--augment", action="store_true")
    return parser.parse_args()


def collate_fn(batch):
    samples = []
    labels = []
    for item, label in batch:
        samples.append(torch.tensor(item, dtype=torch.float32))
        labels.append(torch.tensor(label, dtype=torch.float32))
    return torch.stack(samples), torch.stack(labels).view(-1, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_files, val_files = build_dataset(args.data_dir)

    if not train_files:
        raise FileNotFoundError(f"No training data found in {args.data_dir}")

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    train_dataset = RadarSequenceDataset(
        train_files,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        augment=args.augment,
    )
    val_dataset = RadarSequenceDataset(
        val_files,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Model
    model = CNN_LSTM_HumanDetector(in_channels=1, hidden_size=64).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * sequences.size(0)

        train_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)

                logits = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item() * sequences.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print(f"Training complete. Best model: {best_path}")


if __name__ == "__main__":
    main()
