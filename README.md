# Human Detection Behind Walls with mmWave Radar + CNN

This project is a practical starter architecture for detecting a human behind a wall using radar data and a convolutional neural network (CNN).

## Goal

Detect whether a human is present behind an obstacle using:
- mmWave radar
- signal preprocessing
- range-Doppler visualization
- CNN classification

## System design

- Sensor: mmWave radar (TI IWR6843 / AWR6843 preferred)
- Signal processing: FFT-based range and Doppler extraction
- Feature representation: 2D range-Doppler image
- ML model: CNN binary classifier
- Output: human present / absent

## Recommended hardware

- TI IWR6843 or AWR6843 mmWave radar
- Jetson Nano / Xavier / Orin for deployment
- Optional thermal camera for validation and debugging

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train/
│   │   ├── 0/
│   │   └── 1/
│   └── val/
│       ├── 0/
│       └── 1/
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── preprocess_radar.py
│   ├── model.py
│   ├── train.py
│   └── infer.py
└── checkpoints/
```

## Dataset format

Use `.npz` files with a simple structure:

```python
{
    "sample": np.ndarray,   # shape: (H, W) or (chirps, samples) or (frames, ...)
    "label": 0 or 1         # 0 = no human, 1 = human
}
```

Examples:
- `data/train/0/sample_001.npz`
- `data/train/1/sample_002.npz`

You can also store multiple samples as separate `.npz` files inside class folders.

## Data pipeline

1. Collect radar sweeps under empty-room and human-present conditions.
2. Preprocess the raw radar matrix.
3. Convert it to a 2D range-Doppler-style image.
4. Train a CNN to classify the image.
5. Run inference on new radar observations.

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare dataset

Create folders:

```bash
mkdir -p data/train/0 data/train/1 data/val/0 data/val/1
```

Add `.npz` files to each folder.

### 3) Train

```bash
python src/train.py --data_dir data --epochs 20 --batch_size 16 --image_size 64
```

### 4) Inference

```bash
python src/infer.py --model_path checkpoints/best_model.pth --input data/val/1/sample_001.npz
```

## Model

The default model is a compact CNN:
- Conv -> ReLU -> BatchNorm
- Conv -> ReLU -> BatchNorm
- MaxPool
- Conv -> ReLU -> BatchNorm
- Global average pooling
- Linear sigmoid / binary classifier

## Important notes

- This is a startup-friendly baseline. Real-world performance depends heavily on wall material, sensor setup, and clutter.
- Through-wall detection is much easier with drywall and thin partitions than with thick concrete or metal reinforcement.
- It is strongly recommended to collect a dataset from the exact environment where the system will be used.

## Next improvements

- Add Doppler-time sequences instead of a single frame
- Use CNN + LSTM for motion classification
- Add background subtraction and clutter removal
- Train on multiple wall types
- Deploy on Jetson using TensorRT
- Fuse radar + thermal data

## License

MIT
