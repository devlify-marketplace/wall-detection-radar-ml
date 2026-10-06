"""Real-time human detection using IWR6843 radar and trained CNN-LSTM model."""

import argparse
import time
from collections import deque

import numpy as np
import torch

from radar_reader import IWR6843Reader
from radar_processing import IWR6843Processor
from model import CNN_LSTM_HumanDetector


class LiveDetector:
    """Real-time detection pipeline."""
    
    def __init__(
        self,
        model_path: str,
        port: str,
        sequence_length: int = 10,
        image_size: int = 64,
        confidence_threshold: float = 0.5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = CNN_LSTM_HumanDetector(in_channels=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Initialize radar reader and processor
        self.reader = IWR6843Reader(port)
        self.processor = IWR6843Processor()
        
        # Sequence buffer
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.frame_buffer = deque(maxlen=sequence_length)
        
        print(f"Model loaded from {model_path}")
        print(f"Connected to radar on {port}")
        print(f"Ready for real-time detection")
    
    def process_frame(self, raw_frame: dict) -> np.ndarray:
        """Process a single raw radar frame."""
        if raw_frame is None:
            return None
        
        iq_data = raw_frame["iq_data"]
        frame_bytes = iq_data.tobytes()
        
        # Process to range-Doppler
        rd_map = self.processor.process_frame(frame_bytes)
        
        return rd_map
    
    def normalize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Normalize frame sequence."""
        frames = np.asarray(frames, dtype=np.float32)
        frames_min = np.min(frames)
        frames_max = np.max(frames)
        if frames_max > frames_min:
            frames = (frames - frames_min) / (frames_max - frames_min)
        return frames
    
    def infer(self) -> tuple:
        """Run inference on current frame buffer."""
        if len(self.frame_buffer) < self.sequence_length:
            return None, None
        
        frames = np.stack(list(self.frame_buffer))
        frames = self.normalize_frames(frames)
        frames = frames[:, None, :, :]  # Add channel dimension
        
        x = torch.tensor(frames, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= self.confidence_threshold else 0
        
        return pred, prob
    
    def run(self, fps: int = 10, duration: int = 0):
        """Run live detection loop.
        
        Args:
            fps: Frames per second to process
            duration: Duration in seconds (0 = infinite)
        """
        start_time = time.time()
        frame_count = 0
        detection_count = 0
        
        try:
            while True:
                if duration > 0 and time.time() - start_time > duration:
                    break
                
                # Read frame
                raw_frame = self.reader.read_frame()
                if raw_frame is None:
                    continue
                
                # Process frame
                rd_map = self.process_frame(raw_frame)
                if rd_map is None:
                    continue
                
                # Resize
                import cv2
                rd_map = cv2.resize(
                    rd_map, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
                )
                
                # Add to buffer
                self.frame_buffer.append(rd_map)
                frame_count += 1
                
                # Infer when buffer is full
                if len(self.frame_buffer) == self.sequence_length:
                    pred, prob = self.infer()
                    if pred is not None:
                        status = "HUMAN DETECTED" if pred == 1 else "NO HUMAN"
                        if pred == 1:
                            detection_count += 1
                        print(
                            f"Frame {frame_count} | "
                            f"{status} | "
                            f"Confidence: {prob:.4f}"
                        )
                
                # Rate control
                time.sleep(1.0 / fps)
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.reader.close()
            elapsed = time.time() - start_time
            print(f"\n--- Summary ---")
            print(f"Frames processed: {frame_count}")
            print(f"Detections: {detection_count}")
            print(f"Elapsed time: {elapsed:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time human detection with IWR6843")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model path")
    parser.add_argument("--port", type=str, required=True, help="Radar serial port")
    parser.add_argument("--fps", type=int, default=10, help="Processing fps")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds (0=infinite)")
    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    
    detector = LiveDetector(
        model_path=args.model_path,
        port=args.port,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        confidence_threshold=args.confidence_threshold,
    )
    
    detector.run(fps=args.fps, duration=args.duration)


if __name__ == "__main__":
    main()
