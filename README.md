# Human Detection Behind Walls with TI IWR6843 mmWave Radar and CNN-LSTM

This project implements a practical human detection system for behind-wall sensing using TI IWR6843 mmWave radar and deep learning.

## Overview

The system follows a realistic TI IWR6843 radar processing pipeline:

1. Capture raw radar ADC data from TI IWR6843
2. Process chirp data into range and Doppler domains
3. Generate range-Doppler heatmaps
4. Remove static clutter and background noise
5. Build sequences of radar frames
6. Use CNN-LSTM model to classify human presence

## Hardware

Required:
- TI IWR6843 mmWave Radar EVM
- TI mmWave Radar AWR6843 or IWR6843AOPEVM evaluation board
- UART to USB cable for data communication

Optional:
- NVIDIA Jetson Nano / Xavier / Orin for deployment
- Thermal camera for validation

## TI IWR6843 Specifications

- Frequency: 60 GHz
- Range: 0.7m to 100m (configurable)
- Doppler: ±70 m/s
- Angular resolution: MIMO antenna array
- Output: Raw ADC data or range-Doppler cubes
- Interface: UART, SPI

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── iwr6843_config.txt
├── data/
│   ├── raw/
│   │   └── *.bin (raw radar files)
│   ├── processed/
│   │   └── train/
│   │       ├── 0/
│   │       └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
├── src/
│   ├── __init__.py
│   ├── radar_reader.py
│   ├── radar_processing.py
│   ├── data_utils.py
│   ├── model.py
│   ├── train.py
│   ├── infer.py
│   └── live_inference.py
├── checkpoints/
└── logs/
```

## TI IWR6843 Data Format

The radar outputs raw ADC data in binary format:
- Each frame contains complex IQ samples
- Structure: [chirps][samples per chirp][TX channels][RX channels]
- Common configuration:
  - Num chirps per frame: 64-128
  - Samples per chirp: 256-512
  - TX antennas: 3
  - RX antennas: 4

## Data collection

### 1) Configure IWR6843

Edit `config/iwr6843_config.txt` with your settings:

```
%  IWR6843 Configuration
mmwaveVersion = 0

platform = xwr6843

ChannelCfg 0 1 0 1 1 1 3 2 0
AdcCfg 2 1
AdcbufCfg -1 0 1 0 1
ProfileCfg 0 60 7 7 60 59 0 59.2 -33 40 0 0 159 0 0 1
ChirpCfg 0 0 0 0 0 0 0 1 0 0 0 1
FrameCfg 0 2 64 100 50 1 0
LoopBackCfg 0
RxDcRmCfg 1 0 64
```

### 2) Capture data

Use TI's mmWave SDK or a custom Python script to read UART and save binary files.

### 3) Process and label

Convert raw binary to `.npz` sequences with labels.

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Connect IWR6843 and collect data

```bash
python src/radar_reader.py --port /dev/ttyUSB0 --baud 115200 --output data/raw/session_001.bin
```

### 3) Process raw radar data

```bash
python src/radar_processing.py --input data/raw/session_001.bin --output data/processed/train/1/ --label 1
```

### 4) Train model

```bash
python src/train.py --data_dir data/processed --epochs 30 --batch_size 8
```

### 5) Live inference

```bash
python src/live_inference.py --model_path checkpoints/best_model.pth --port /dev/ttyUSB0
```

## Radar preprocessing for IWR6843

The preprocessing pipeline converts raw ADC data to range-Doppler images:

1. **Frame extraction** from raw binary stream
2. **ADC to IQ conversion** (complex baseband)
3. **Range FFT** (1D FFT along fast-time)
4. **Doppler FFT** (1D FFT along slow-time, across chirps)
5. **Magnitude extraction** (abs of complex range-Doppler matrix)
6. **Clutter removal** (static background subtraction)
7. **Log scaling** and normalization
8. **Image resizing** to fixed input size (64x64)

## Model

CNN-LSTM for temporal sequence classification:
- CNN: extracts spatial features from each range-Doppler frame
- LSTM: captures temporal motion patterns
- Output: human present / not present

## Configuration

Key IWR6843 parameters:
- `num_tx`: number of transmit antennas (1-3)
- `num_rx`: number of receive antennas (1-4)
- `num_chirps_per_frame`: chirps per frame (64, 128)
- `num_adc_samples`: samples per chirp (256, 512)
- `range_resolution`: depends on bandwidth (typical: 0.044m)
- `doppler_resolution`: depends on chirp rate (typical: 0.1 m/s)

## Realistic limitations

- Drywall: excellent penetration, 3-5m range for motion detection
- Brick: moderate penetration, 1-2m range
- Concrete: poor penetration, <1m range
- Metal reinforcement: severe reflections, reduced performance
- Static person: harder to detect than moving person
- Environmental clutter: reflections from furniture, other walls

## Next steps

1. Collect 10-20 minutes of radar data per scenario
2. Tune preprocessing (FFT windows, clutter removal)
3. Train baseline model
4. Validate on unseen wall types
5. Deploy on Jetson with TensorRT
6. Add activity classification (standing, walking, subtle motion)

## Deployment

For production inference on Jetson:
```bash
python src/live_inference.py --model_path checkpoints/best_model.pth --port /dev/ttyUSB0 --fps 10 --confidence_threshold 0.7
```

## References

- TI IWR6843 User's Guide: https://www.ti.com/lit/ug/tihu629/tihu629.pdf
- mmWave SDK: https://www.ti.com/tool/MMWAVE_SDK
- Through-wall imaging research: radar + clutter suppression techniques

## License

MIT
