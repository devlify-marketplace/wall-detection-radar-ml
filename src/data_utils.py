import os
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset
import cv2


def load_sequence_sample(path: str):
    """Load a radar sequence sample from .npz file."""
    data = np.load(path, allow_pickle=True)
    frames = data["frames"]
    label = int(data["label"])
    metadata = data["metadata"].item() if "metadata" in data else {}
    return frames, label, metadata


def build_dataset(data_dir: str) -> Tuple[List[str], List[str]]:
    """Build training and validation file lists."""
    labels = ["0", "1"]
    train_files = []
    val_files = []

    for cls in labels:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue

        files = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.endswith(".npz")
        ])

        if len(files) == 0:
            continue

        # 80/20 split
        split_idx = max(1, int(len(files) * 0.8))
        train_files.extend(files[:split_idx])
        val_files.extend(files[split_idx:])

    return train_files, val_files


class RadarSequenceDataset(Dataset):
    """PyTorch Dataset for radar sequences."""
    
    def __init__(
        self,
        file_list: List[str],
        sequence_length: int = 10,
        image_size: int = 64,
        augment: bool = False,
    ):
        self.file_list = file_list
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        frames, label, _ = load_sequence_sample(path)

        frames = np.asarray(frames, dtype=np.float32)

        if frames.ndim == 2:
            frames = frames[None, :, :]

        # Adjust sequence length
        if frames.shape[0] < self.sequence_length:
            # Repeat last frame
            pad_needed = self.sequence_length - frames.shape[0]
            pad = np.repeat(frames[-1:], pad_needed, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        elif frames.shape[0] > self.sequence_length:
            # Take centered window
            start = max(0, (frames.shape[0] - self.sequence_length) // 2)
            frames = frames[start:start + self.sequence_length]

        # Normalize
        frames = self._normalize_frames(frames)
        
        # Resize
        frames = self._resize_frames(frames, self.image_size)
        
        # Augmentation (optional)
        if self.augment:
            frames = self._augment(frames)
        
        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        frames = frames[:, None, :, :]

        return frames, float(label)

    def _normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        frames = frames.astype(np.float32)
        frames_min = np.min(frames)
        frames_max = np.max(frames)
        if frames_max > frames_min:
            frames = (frames - frames_min) / (frames_max - frames_min)
        return frames

    def _resize_frames(self, frames: np.ndarray, image_size: int) -> np.ndarray:
        resized = []
        for frame in frames:
            frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            resized.append(frame)
        return np.stack(resized, axis=0)
    
    def _augment(self, frames: np.ndarray) -> np.ndarray:
        """Light augmentation: random noise, slight rotation."""
        # Add Gaussian noise
        noise = np.random.normal(0, 0.01, frames.shape)
        frames = frames + noise
        frames = np.clip(frames, 0, 1)
        return frames
