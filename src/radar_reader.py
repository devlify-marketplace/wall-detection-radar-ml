"""Read raw IWR6843 radar data from UART and save to binary file."""

import argparse
import struct
import time
from pathlib import Path

import numpy as np
import serial
from tqdm import tqdm


class IWR6843Reader:
    """
    Read raw ADC data from TI IWR6843 mmWave radar via UART.
    
    Frame structure:
    - Magic bytes: [0x0102, 0x0304, 0x0506, 0x0708]
    - Frame header: [version, packet_len, packet_num, chirp_margin, frame_margin]
    - ADC data: complex IQ samples
    """
    
    MAGIC_WORD = [0x0102, 0x0304, 0x0506, 0x0708]
    MAGIC_WORD_LEN = 4
    FRAME_HEADER_SIZE = 20  # bytes
    
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.serial = None
        self.connect()
    
    def connect(self):
        try:
            self.serial = serial.Serial(self.port, self.baud, timeout=5)
            print(f"Connected to {self.port} at {self.baud} baud")
            time.sleep(2)  # Give radar time to initialize
        except Exception as e:
            raise RuntimeError(f"Failed to connect: {e}")
    
    def read_frame(self):
        """Read a single radar frame from UART."""
        if not self.serial:
            return None
        
        try:
            # Read magic words
            magic = []
            for _ in range(self.MAGIC_WORD_LEN):
                word = struct.unpack('>H', self.serial.read(2))[0]
                magic.append(word)
            
            if magic != self.MAGIC_WORD:
                return None
            
            # Read frame header
            header_data = self.serial.read(self.FRAME_HEADER_SIZE)
            if len(header_data) < self.FRAME_HEADER_SIZE:
                return None
            
            version, packet_len, packet_num, chirp_margin, frame_margin = struct.unpack(
                '>HHHBB',
                header_data[:8]
            )
            
            # Read ADC data
            adc_len = packet_len - self.FRAME_HEADER_SIZE
            adc_data = self.serial.read(adc_len)
            
            if len(adc_data) < adc_len:
                return None
            
            # Parse IQ samples (2-byte real + 2-byte imaginary)
            num_samples = adc_len // 4
            samples = struct.unpack(f'>{num_samples * 2}h', adc_data)
            iq_data = np.array([complex(samples[i], samples[i+1]) for i in range(0, len(samples), 2)])
            
            return {
                'version': version,
                'packet_num': packet_num,
                'iq_data': iq_data,
                'adc_len': adc_len,
            }
        except Exception as e:
            print(f"Error reading frame: {e}")
            return None
    
    def close(self):
        if self.serial:
            self.serial.close()
            print("Disconnected from radar")


def parse_args():
    parser = argparse.ArgumentParser(description="Read IWR6843 radar data from UART")
    parser.add_argument("--port", type=str, required=True, help="Serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--output", type=str, required=True, help="Output binary file")
    parser.add_argument("--duration", type=int, default=30, help="Recording duration in seconds")
    parser.add_argument("--fps", type=int, default=10, help="Expected frames per second")
    return parser.parse_args()


def main():
    args = parse_args()
    
    reader = IWR6843Reader(args.port, args.baud)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    frames_collected = []
    expected_frames = args.duration * args.fps
    
    print(f"Collecting {expected_frames} frames...")
    
    try:
        with tqdm(total=expected_frames) as pbar:
            start_time = time.time()
            while time.time() - start_time < args.duration:
                frame = reader.read_frame()
                if frame is not None:
                    frames_collected.append(frame)
                    pbar.update(1)
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        reader.close()
    
    # Save frames
    if frames_collected:
        with open(output_path, 'wb') as f:
            for frame in frames_collected:
                frame_data = frame['iq_data'].tobytes()
                f.write(frame_data)
        print(f"Saved {len(frames_collected)} frames to {output_path}")
    else:
        print("No frames collected")


if __name__ == "__main__":
    main()
