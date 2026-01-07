import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from model import AgeRegressionCNN

BASE_CHANNELS = 10 # the most important const denoting the network size, 20 is about max for esp32-s3 but it makes the infer time about 15s
CHECKPOINT_PATH = "checkpoint.pt"
torch.backends.cudnn.benchmark = True  # helps on GPU/ROCm

AGE_MIN = 1.0
AGE_MAX = 90.0
AGE_RANGE = AGE_MAX - AGE_MIN

# -------- Augmentation knobs (safe defaults) --------
IMG_SIZE = 200
RESIZE_SHORT = 220  # resize shorter side before crop

USE_MIXUP = True
MIXUP_P = 0.30
MIXUP_ALPHA = 0.2
# ---------------------------------------------------

import builtins
_original_print = builtins.print  # save the real print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    with open("train_log.txt", "a", encoding="utf-8") as f:
        _original_print(*args, file=f, **kwargs)

# ============================================================
# Dataset
# ============================================================
class PreloadedAgeDataset(Dataset):
    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor, transform=None):
        self.X = X_tensor
        self.y = y_tensor
        self.transform = transform

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int):
        img = self.X[idx]
        label = self.y[idx]

        # X stored as CHW float32 with 0-255 values.
        # Convert to PIL without going through numpy (less overhead).
        if self.transform is not None:
            img_u8 = img.to(torch.uint8)
            img_pil = TF.to_pil_image(img_u8)
            img = self.transform(img_pil)

        return img, label


# ============================================================
# Checkpoint helpers
# ============================================================
def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_loss: float) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_loss": best_loss,
        },
        CHECKPOINT_PATH,
    )
    print(f"💾 Saved checkpoint at epoch {epoch} ✅")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    if not os.path.exists(CHECKPOINT_PATH):
        print("⚠️ No checkpoint found. Starting fresh.")
        return 0, float("inf")

    data = torch.load(CHECKPOINT_PATH, map_location=device)
    try:
        model.load_state_dict(data["model_state"])
        optimizer.load_state_dict(data["optimizer_state"])
        print(f"🔄 Restored checkpoint from epoch {data['epoch']} (best loss: {data['best_loss']:.6f})")
        return int(data["epoch"]), float(data["best_loss"])
    except Exception as e:
        print(f"🚫 Could not fully load checkpoint (maybe architecture changed?): {e}")
        print("   🆕 Training will start from scratch.")
        return 0, float("inf")


# ============================================================
# Data loading
# ============================================================
def load_training_data(root_dir: str = "./face_age"):
    print("📦 Loading training data...")
    input_files, outputs = [], []

    for age in sorted(os.listdir(root_dir)):
        age_dir = os.path.join(root_dir, age)
        if not os.path.isdir(age_dir):
            continue
        try:
            age_int = int(age)
        except ValueError:
            continue

        for file in os.listdir(age_dir):
            input_files.append(os.path.join(age_dir, file))
            outputs.append(age_int)

    if not input_files:
        raise RuntimeError(f"❌ No images found under {root_dir!r}")

    imgs = [np.array(Image.open(f).convert("RGB")) for f in input_files]
    X = torch.tensor(np.array(imgs), dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(outputs, dtype=torch.float32)

    # Normalize ages to [0, 1] for training; model output is also [0, 1].
    y_norm = (y - AGE_MIN) / AGE_RANGE
    print(f"📸 Loaded {len(X)} images successfully.")
    return X, y_norm.view(-1, 1)


# ============================================================
# MixUp (optional)
# ============================================================
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float):
    if alpha <= 0.0 or x.size(0) < 2:
        return x, y
    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample().to(device=x.device, dtype=x.dtype)
    index = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[index]
    y_mix = lam * y + (1.0 - lam) * y[index]
    return x_mix, y_mix


# ============================================================
# Training function
# ============================================================
def train(model: AgeRegressionCNN, training_data, epochs: int, batch_size: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train, X_test, y_train, y_test = training_data

    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    # Strong-but-safe augmentation for face age regression:
    # - Mild crop/scale jitter (biggest win)
    # - Mild affine (rot/translate/scale/shear) with non-black fill (avoid corner artifacts)
    # - Photometric jitter + occasional blur
    # - RandomErasing (simulates occlusions)
    train_transforms = T.Compose(
        [
            T.Resize(RESIZE_SHORT, interpolation=InterpolationMode.BILINEAR),
            T.RandomResizedCrop(
                IMG_SIZE,
                scale=(0.88, 1.00),
                ratio=(0.92, 1.08),
                interpolation=InterpolationMode.BILINEAR,
            ),
            T.RandomHorizontalFlip(p=0.5),

            T.RandomApply(
                [
                    T.RandomAffine(
                        degrees=8,
                        translate=(0.04, 0.04),
                        scale=(0.95, 1.05),
                        shear=(-4, 4, -2, 2),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=128,  # important: avoid black-corner shortcut
                    )
                ],
                p=0.8,
            ),

            T.RandomApply(
                [
                    T.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                        saturation=0.12,
                        hue=0.04,
                    )
                ],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.05),

            T.RandomApply(
                [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.10,
            ),

            T.ToTensor(),
            normalize,

            T.RandomErasing(
                p=0.25,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )

    # Deterministic preprocessing for validation.
    test_transforms = T.Compose(
        [
            T.Resize(RESIZE_SHORT, interpolation=InterpolationMode.BILINEAR),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            normalize,
        ]
    )

    train_dataset = PreloadedAgeDataset(X_train, y_train, transform=train_transforms)
    test_dataset = PreloadedAgeDataset(X_test, y_test, transform=test_transforms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Device info
    if device.type == "cuda":
        print(f"🧠 Using GPU: {torch.cuda.get_device_name(0)} 🚀")
    else:
        print("🧠 Using CPU 🐢")
    print(f"📊 Train: {len(train_dataset)} | Test: {len(test_dataset)} | Base channels: {BASE_CHANNELS}")

    # Loss on normalized ages in [0, 1]
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    start_epoch, best_eval_loss = load_checkpoint(model, optimizer, device)

    print("🚀 Starting training...")

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0
        total_train_mae_years = 0.0

        # Training loop
        for X_batch, y_batch in tqdm(
            train_loader,
            desc=f"🏋️ Epoch {epoch}/{epochs}",
            ncols=100,
            leave=False,
        ):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device).view(-1)

            # Optional MixUp (often improves regression generalization)
            if USE_MIXUP and (torch.rand(()) < MIXUP_P):
                X_batch, y_batch = mixup_batch(X_batch, y_batch, alpha=MIXUP_ALPHA)

            optimizer.zero_grad()
            preds = model(X_batch)

            # Clamp just in case; Sigmoid already keeps this in [0, 1].
            preds = preds.clamp(0.0, 1.0)

            loss = loss_fn(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(preds - y_batch)).item()
                total_train_mae_years += batch_mae * AGE_RANGE

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_mae_years = total_train_mae_years / len(train_loader)

        # Evaluation
        model.eval()
        total_eval_loss = 0.0
        total_eval_mae_years = 0.0
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device).view(-1)

                preds = model(X_batch)
                preds = preds.clamp(0.0, 1.0)

                loss = loss_fn(preds, y_batch)
                total_eval_loss += loss.item()
                batch_mae = torch.mean(torch.abs(preds - y_batch)).item()
                total_eval_mae_years += batch_mae * AGE_RANGE

        avg_eval_loss = total_eval_loss / len(test_loader)
        avg_eval_mae_years = total_eval_mae_years / len(test_loader)

        scheduler.step(avg_eval_loss)
        epoch_time = time.time() - epoch_start

        print(
            f"\n📅 Epoch {epoch:03d}/{epochs}\n"
            f"   🔹 Train loss: {avg_train_loss:.5f}\n"
            f"   🔹 Train MAE:  {avg_train_mae_years:.2f}y\n"
            f"   🔸 Eval loss:  {avg_eval_loss:.5f}\n"
            f"   🔸 Eval MAE:   {avg_eval_mae_years:.2f}y\n"
            f"   ⚙️  LR: {optimizer.param_groups[0]['lr']:.2e}\n"
            f"   ⏱️  Time: {epoch_time:.1f}s\n"
        )

        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            save_checkpoint(model, optimizer, epoch, best_eval_loss)
        else:
            print("📉 No improvement this epoch.")


# ============================================================
# Entry point
# ============================================================
def main():
    if torch.cuda.is_available():
        print(f"🔧 Device: GPU {torch.cuda.get_device_name(0)}")
    else:
        print("🔧 Device: CPU")

    X, y = load_training_data()
    data_split = train_test_split(X, y, test_size=0.2, random_state=42)

    model = AgeRegressionCNN(base_channels=BASE_CHANNELS)
    train(model, data_split, epochs=512, batch_size=64, lr=1e-3)


if __name__ == "__main__":
    main()
