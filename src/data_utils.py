import os
from typing import List, Tuple, Dict, Any

import numpy as np
from torch.utils.data import Dataset
import cv2

def load_radar_sample(path: str) -> Tuple[np.ndarray, int]:
    """Load a radar sample from an .npz file."""
    data = np.load(path, allow_pickle=True)
    sample = data["sample"]
    label = int(data["label"])
    return sample, label

def build_dataset(data_dir: str) -> Tuple[List[str], List[str]]:
    """Build lists of training and validation file paths from a directory structure."""
    labels = ["0", "1"]
    train_files = []
    val_files = []
    
    for cls in labels:
        class_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        
        files = sorted([os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(".npz")])
        
        if len(files) == 0:
            continue
        
        # Simple split: 80/20
        split_idx = max(1, int(len(files) * 0.8))
        train_files.extend(files[:split_idx])
        val_files.extend(files[split_idx:])
    
    return train_files, val_files

class RadarDataset(Dataset):
    """PyTorch Dataset for radar samples."""
    
    def __init__(self, file_list: List[str], image_size: int = 64, transform=None):
        self.file_list = file_list
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        path = self.file_list[idx]
        sample, label = load_radar_sample(path)
        arr = np.asarray(sample, dtype=np.float32)

        # If the sample is already 2D, use it directly.
        # If it is a 1D vector, reshape to a square image.
        if arr.ndim == 1:
            arr = np.reshape(arr, (self.image_size, self.image_size))
        elif arr.ndim == 2:
            pass
        else:
            # Reduce high-dimensional data to a 2D image-like representation
            arr = arr[0]

        # Resize to a fixed size for model input
        arr = self._resize_image(arr, self.image_size)

        # Normalize
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = arr.astype(np.float32)

        if arr.ndim == 2:
            arr = arr[None, :, :]  # CxHxW

        if self.transform is not None:
            arr = self.transform(arr)

        return arr, float(label)

    def _resize_image(self, arr: np.ndarray, size: int) -> np.ndarray:
        """Resize array to specified size."""
        if arr.shape[0] == size and arr.shape[1] == size:
            return arr

        arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        return arr
