# infer.py
import os
from PIL import Image

import torch
import torchvision.transforms as T

from model import AgeRegressionCNN  # uses your model.py


# -------------------------
# Hard-coded configuration
# -------------------------
IMAGE_PATH = "face.png"          # default input image
CHECKPOINT_PATH = "checkpoint.pt"

IMG_SIZE = 200
BASE_CHANNELS = 10                # must match training (train.py uses base_channels=4)

AGE_MIN = 1.0
AGE_MAX = 90.0
AGE_RANGE = AGE_MAX - AGE_MIN

# Same normalization as train.py
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)

    # train.py saves: {"model_state": ..., "optimizer_state": ..., "epoch": ..., "best_loss": ...}
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        # fallback if someone saved only state_dict
        state = ckpt

    model.load_state_dict(state, strict=True)


def preprocess_image(image_path: str) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")

    tfm = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )

    x = tfm(img)          # (3, H, W)
    x = x.unsqueeze(0)    # (1, 3, H, W)
    return x


def denormalize_age(y_norm: float) -> float:
    # Model predicts normalized age in [0,1]
    return (y_norm * AGE_RANGE) + AGE_MIN


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AgeRegressionCNN(base_channels=BASE_CHANNELS).to(device)
    model.eval()

    load_checkpoint_into_model(model, CHECKPOINT_PATH, device)

    x = preprocess_image(IMAGE_PATH).to(device)

    with torch.inference_mode():
        y = model(x)  # shape: (N,)
        y_norm = float(torch.clamp(y[0], 0.0, 1.0).item())

    age_years = denormalize_age(y_norm)

    print(f"Image: {IMAGE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device.type}")
    print(f"Pred (normalized [0..1]): {y_norm:.6f}")
    print(f"Pred age (years): {age_years:.3f}")


if __name__ == "__main__":
    main()
