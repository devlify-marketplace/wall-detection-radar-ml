"""Convert raw IWR6843 radar data to range-Doppler sequences."""

import argparse
import struct
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftshift
from scipy.ndimage import uniform_filter1d
import cv2
from tqdm import tqdm


class IWR6843Processor:
    """
    Process raw IWR6843 ADC data into range-Doppler images.
    """
    
    def __init__(
        self,
        num_tx: int = 3,
        num_rx: int = 4,
        num_chirps: int = 64,
        num_adc_samples: int = 256,
        num_doppler_bins: int = 64,
    ):
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_chirps = num_chirps
        self.num_adc_samples = num_adc_samples
        self.num_doppler_bins = num_doppler_bins
    
    def parse_raw_frame(self, raw_bytes: bytes) -> np.ndarray:
        """
        Parse raw ADC bytes into IQ complex array.
        Format: [chirps][samples][tx][rx] of int16 real and imaginary.
        """
        # Interpret as int16 values
        data = np.frombuffer(raw_bytes, dtype=np.int16)
        num_samples = len(data) // 2
        
        # Convert to complex IQ
        iq_data = data[0::2] + 1j * data[1::2]
        
        # Reshape to [chirps, samples, tx, rx]
        expected_size = self.num_chirps * self.num_adc_samples * self.num_tx * self.num_rx
        if len(iq_data) != expected_size:
            # Pad or trim
            if len(iq_data) < expected_size:
                iq_data = np.pad(iq_data, (0, expected_size - len(iq_data)))
            else:
                iq_data = iq_data[:expected_size]
        
        radar_cube = iq_data.reshape(
            self.num_chirps,
            self.num_adc_samples,
            self.num_tx,
            self.num_rx
        )
        return radar_cube
    
    def compute_range_doppler(self, radar_cube: np.ndarray) -> np.ndarray:
        """
        Compute range-Doppler heatmap from radar cube.
        
        Steps:
        1. Sum across TX and RX (MIMO combination)
        2. FFT across samples for range
        3. FFT across chirps for Doppler
        """
        # Sum across TX and RX antennas
        rd_cube = radar_cube.sum(axis=2).sum(axis=2)  # [chirps, samples]
        
        # Range FFT (axis 1)
        range_fft = fft(rd_cube, axis=1)
        range_mag = np.abs(range_fft[:, :self.num_adc_samples // 2])
        
        # Doppler FFT (axis 0)
        doppler_fft = fft(range_mag, axis=0, n=self.num_doppler_bins)
        doppler_mag = np.abs(fftshift(doppler_fft, axes=0))
        
        return doppler_mag
    
    def remove_clutter(self, rd_map: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Simple clutter removal using background subtraction.
        """
        # Moving average along range dimension
        bg = uniform_filter1d(rd_map, size=kernel_size, axis=1)
        clutter_removed = rd_map - bg
        clutter_removed = np.maximum(clutter_removed, 0)
        return clutter_removed
    
    def normalize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1].
        """
        img = np.abs(img).astype(np.float32)
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)
        return img
    
    def process_frame(self, raw_bytes: bytes) -> np.ndarray:
        """
        Full processing pipeline: raw -> range-Doppler -> clutter removal -> normalized.
        """
        radar_cube = self.parse_raw_frame(raw_bytes)
        rd_map = self.compute_range_doppler(radar_cube)
        rd_map = self.remove_clutter(rd_map)
        rd_map = self.normalize_image(rd_map)
        return rd_map


def parse_args():
    parser = argparse.ArgumentParser(description="Process raw IWR6843 data to range-Doppler sequences")
    parser.add_argument("--input", type=str, required=True, help="Input raw binary file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for .npz sequences")
    parser.add_argument("--label", type=int, required=True, help="Label (0=no human, 1=human)")
    parser.add_argument("--sequence_length", type=int, default=10, help="Frames per sequence")
    parser.add_argument("--image_size", type=int, default=64, help="Output image size")
    parser.add_argument("--num_tx", type=int, default=3)
    parser.add_argument("--num_rx", type=int, default=4)
    parser.add_argument("--num_chirps", type=int, default=64)
    parser.add_argument("--num_adc_samples", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    
    processor = IWR6843Processor(
        num_tx=args.num_tx,
        num_rx=args.num_rx,
        num_chirps=args.num_chirps,
        num_adc_samples=args.num_adc_samples,
    )
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate frame size in bytes
    frame_size = (
        args.num_chirps
        * args.num_adc_samples
        * args.num_tx
        * args.num_rx
        * 4  # 2 bytes real + 2 bytes imaginary
    )
    
    # Read raw file
    with open(input_path, 'rb') as f:
        raw_data = f.read()
    
    num_frames = len(raw_data) // frame_size
    print(f"Processing {num_frames} frames from {input_path}")
    
    # Process frames
    processed_frames = []
    for i in tqdm(range(num_frames)):
        start = i * frame_size
        end = start + frame_size
        frame_bytes = raw_data[start:end]
        
        rd_map = processor.process_frame(frame_bytes)
        rd_map = cv2.resize(rd_map, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        processed_frames.append(rd_map)
    
    # Group into sequences
    num_sequences = len(processed_frames) // args.sequence_length
    print(f"Creating {num_sequences} sequences")
    
    for seq_idx in tqdm(range(num_sequences)):
        start = seq_idx * args.sequence_length
        end = start + args.sequence_length
        frames_seq = np.stack(processed_frames[start:end])
        
        output_file = output_dir / f"sequence_{seq_idx:04d}.npz"
        np.savez_compressed(
            output_file,
            frames=frames_seq,
            label=args.label,
            metadata={"sequence_idx": seq_idx},
        )
    
    print(f"Saved sequences to {output_dir}")


if __name__ == "__main__":
    main()
