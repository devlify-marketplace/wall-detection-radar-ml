from typing import Optional

import numpy as np
import cv2

def preprocess_radar_sample(sample: np.ndarray, image_size: int = 64, normalize: bool = True) -> np.ndarray:
    """Preprocess a radar sample into a 2D representation.
    
    Args:
        sample: Raw radar data (1D, 2D, or higher dimensional)
        image_size: Target size for the output image
        normalize: Whether to normalize the output
    
    Returns:
        Preprocessed radar image with shape (1, H, W)
    """
    arr = np.asarray(sample, dtype=np.float32)

    if arr.ndim == 1:
        arr = arr.reshape((image_size, image_size))

    if arr.ndim == 2:
        # Use FFT magnitude to create a radar-like image
        fft_mag = np.abs(np.fft.fft2(arr))
        fft_mag = np.fft.fftshift(fft_mag)
        arr = fft_mag
    elif arr.ndim >= 3:
        # Take first frame for a simple starter pipeline
        arr = arr[0]

    # Convert to a simple image-like representation
    arr = np.abs(arr)
    if normalize:
        arr = arr - np.min(arr)
        max_val = np.max(arr)
        if max_val > 0:
            arr = arr / max_val
    arr = arr.astype(np.float32)

    # Resize if larger than expected
    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)

    return arr[None, :, :]
