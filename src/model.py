import torch
import torch.nn as nn


class RadarCNN(nn.Module:
    """CNN feature extractor for radar range-Doppler images."""
    
    def __init__(self, in_channels=1, hidden_size=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.hidden_size = hidden_size

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return x


class CNN_LSTM_HumanDetector(nn.Module):
    """CNN-LSTM for temporal human detection behind walls."""
    
    def __init__(self, in_channels=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.cnn = RadarCNN(in_channels, hidden_size)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (B, T, C, H, W) - batch, time, channels, height, width
        Returns:
            logits: (B, 1)
        """
        batch_size, seq_len, c, h, w = x.size()
        feature_frames = []

        # Extract CNN features for each frame
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            feat = self.cnn(frame)   # (B, 64)
            feature_frames.append(feat.unsqueeze(1))

        # Stack features
        features = torch.cat(feature_frames, dim=1)  # (B, T, 64)

        # LSTM on feature sequence
        lstm_out, (h_n, c_n) = self.lstm(features)  # (B, T, hidden)
        last_hidden = lstm_out[:, -1, :]  # (B, hidden)

        # Classification
        logits = self.classifier(last_hidden)  # (B, 1)
        return logits
