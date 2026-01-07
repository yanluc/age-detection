import torch
import torch.nn as nn


class AgeRegressionCNN(nn.Module):
    """
    Medium-size CNN for age regression on 200x200 RGB images,
    modified to be TensorFlow Lite Micro–friendly:

      - Only uses ops that map cleanly to TFLite Micro kernels:
        Conv2D + ReLU + MaxPool2D + AvgPool2D + FullyConnected + Sigmoid.
      - No GroupNorm (it becomes reduction ops like SUM/MEAN that your
        current ESP32 TFLM build doesn't implement).
      - No AdaptiveAvgPool2d (we use fixed AvgPool2d with kernel=12).

    With base_channels=32:
      - params ~= 1.2M -> ~4.8 MB as float32, ~1.2 MB as int8.
    """

    def __init__(self, base_channels: int = 4):
        super(AgeRegressionCNN, self).__init__()

        def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
            # Two convs per stage with ReLU, then 2x downsampling.
            # All layers are standard ops that TFLite Micro supports.
            return nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),

                nn.Conv2d(
                    out_ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),

                nn.MaxPool2d(2, 2),
            )

        # Spatial sizes for 200×200 input with 4x MaxPool(2,2):
        # 200 -> 100 -> 50 -> 25 -> 12 (integer division)
        self.features = nn.Sequential(
            conv_block(3, base_channels),
            conv_block(base_channels, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 4),
            conv_block(base_channels * 4, base_channels * 8),
        )

        # Fixed global average pooling: feature map is 12x12, so pool with 12x12
        # to get (N, C, 1, 1). This should map to AVERAGE_POOL_2D in TFLite.
        self.global_pool = nn.AvgPool2d(kernel_size=12, stride=1)

        # Fully connected regressor head (same structure as before)
        self.regressor = nn.Sequential(
            nn.Flatten(),                           # (N, C, 1, 1) -> (N, C)
            nn.Linear(base_channels * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),

            nn.Linear(64, 1),
            nn.Sigmoid(),  # normalized age ∈ [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        # (N, 1) -> (N,)
        return x.squeeze(1)
