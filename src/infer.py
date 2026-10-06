import argparse

import numpy as np
import torch

from model import RadarCNN
from data_utils import load_radar_sample
from preprocess_radar import preprocess_radar_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a radar sample.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model.")
    parser.add_argument("--input", type=str, required=True, help="Input radar sample (.npz)")
    parser.add_argument("--image_size", type=int, default=64, help="Input size expected by the model.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading sample from {args.input}...")
    sample, label = load_radar_sample(args.input)
    processed = preprocess_radar_sample(sample, image_size=args.image_size)

    print(f"Loading model from {args.model_path}...")
    model = RadarCNN(in_channels=1, num_classes=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    x = torch.tensor(processed, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

    print(f"\n--- Results ---")
    print(f"Ground truth label: {label}")
    print(f"Predicted label: {pred}")
    print(f"Probability (human present): {prob:.4f}")

if __name__ == "__main__":
    main()
