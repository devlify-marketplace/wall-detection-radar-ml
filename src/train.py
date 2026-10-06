import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from data_utils import RadarDataset, build_dataset
from model import RadarCNN
from preprocess_radar import preprocess_radar_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Train a radar-based wall human detector.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing class folders.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--image_size", type=int, default=64, help="Input image size.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saved models.")
    return parser.parse_args()

def collate_fn(batch):
    """Custom collate function for DataLoader."""
    images = []
    labels = []
    for img, label in batch:
        images.append(torch.tensor(img, dtype=torch.float32))
        labels.append(torch.tensor(label, dtype=torch.float32))
    return torch.stack(images), torch.stack(labels).view(-1, 1)

def main():
    args = parse_args()

    train_files, val_files = build_dataset(args.data_dir)

    if not train_files:
        raise FileNotFoundError(f"No training data found in {args.data_dir}")

    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")

    train_dataset = RadarDataset(train_files, image_size=args.image_size)
    val_dataset = RadarDataset(val_files, image_size=args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = RadarCNN(in_channels=1, num_classes=1).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "best_model.pth")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
        acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    print(f"Training complete. Best model saved to {best_path}")

if __name__ == "__main__":
    main()
