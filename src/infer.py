import argparse

import numpy as np
import torch

from data_utils import load_sequence_sample
from model import CNN_LSTM_HumanDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on a radar sequence")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sample
    print(f"Loading {args.input}...")
    frames, label, metadata = load_sequence_sample(args.input)
    frames = np.asarray(frames, dtype=np.float32)

    # Normalize
    frames_min = np.min(frames)
    frames_max = np.max(frames)
    if frames_max > frames_min:
        frames = (frames - frames_min) / (frames_max - frames_min)

    # Adjust sequence length
    if frames.shape[0] < args.sequence_length:
        pad = np.repeat(frames[-1:], args.sequence_length - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    elif frames.shape[0] > args.sequence_length:
        start = max(0, (frames.shape[0] - args.sequence_length) // 2)
        frames = frames[start:start + args.sequence_length]

    # Add channel dimension
    frames = frames[:, None, :, :]

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = CNN_LSTM_HumanDetector(in_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # Inference
    x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= args.confidence_threshold else 0

    print(f"\n--- Results ---")
    print(f"Ground truth: {label}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {prob:.4f}")
    print(f"Detection: {'HUMAN DETECTED' if pred == 1 else 'NO HUMAN'}")


if __name__ == "__main__":
    main()
