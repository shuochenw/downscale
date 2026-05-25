# =========================================================
# user settings
# =========================================================
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 0.0
RANDOM_SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-7
NUM_RESBLK = 8
NUM_FEATURES = 64
SCALE = 4

INPUT_PATH = "/projects/sds-lab/Shuochen/downscaling/old/gcm_2deg_conus.pth"
TARGET_PATH = "/projects/sds-lab/Shuochen/downscaling/old/rcm_0.5deg_conus.pth"
SAVE_ROOT = "/projects/sds-lab/Shuochen/downscaling/old/trained_models/SRResNet_gcm_2deg_to_rcm_0.5deg"

VAL_FRACTION = 0.2


# =========================================================
# imports
# =========================================================
import os
import copy
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SRResNet


# =========================================================
# setup
# =========================================================
os.makedirs(SAVE_ROOT, exist_ok=True)

best_ckpt_path = os.path.join(SAVE_ROOT, "best_model.pth")
summary_path = os.path.join(SAVE_ROOT, "training_summary.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# helpers
# =========================================================
def ensure_nchw(tensor, name):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)

    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [N, C, H, W] or [N, H, W]. Got {tensor.shape}")

    return tensor.contiguous()


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values.")


# =========================================================
# load tensors
# =========================================================
X = ensure_nchw(torch.load(INPUT_PATH).float(), "X")
y = ensure_nchw(torch.load(TARGET_PATH).float(), "y")

if X.shape[0] != y.shape[0]:
    raise ValueError(f"Sample mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}")

if y.shape[-2] != SCALE * X.shape[-2]:
    raise ValueError(f"Height mismatch: {y.shape[-2]} != {SCALE} * {X.shape[-2]}")

if y.shape[-1] != SCALE * X.shape[-1]:
    raise ValueError(f"Width mismatch: {y.shape[-1]} != {SCALE} * {X.shape[-1]}")

assert_finite(X, "X")
assert_finite(y, "y")

print("Loaded shapes:")
print("X:", X.shape)
print("y:", y.shape)


# =========================================================
# chronological train/validation split
# =========================================================
n_samples = X.shape[0]
val_samples = int(round(n_samples * VAL_FRACTION))

if val_samples <= 0 or val_samples >= n_samples:
    raise ValueError(
        f"Invalid validation split: n_samples={n_samples}, "
        f"VAL_FRACTION={VAL_FRACTION}, val_samples={val_samples}"
    )

train_samples = n_samples - val_samples

X_train = X[:train_samples]
y_train = y[:train_samples]

X_val = X[train_samples:]
y_val = y[train_samples:]

print("Split shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)


# =========================================================
# normalization from training set only
# =========================================================
X_mean = X_train.mean()
X_std = X_train.std()
y_mean = y_train.mean()
y_std = y_train.std()

if X_std.item() == 0:
    X_std = torch.tensor(1.0, dtype=X_train.dtype)

if y_std.item() == 0:
    y_std = torch.tensor(1.0, dtype=y_train.dtype)

X_train_n = ((X_train - X_mean) / X_std).contiguous()
X_val_n = ((X_val - X_mean) / X_std).contiguous()

y_train_n = ((y_train - y_mean) / y_std).contiguous()
y_val_n = ((y_val - y_mean) / y_std).contiguous()

print("Normalization:")
print("X_mean:", X_mean.item())
print("X_std :", X_std.item())
print("y_mean:", y_mean.item())
print("y_std :", y_std.item())


# =========================================================
# datasets
# =========================================================
training_set = TensorDataset(X_train_n, y_train_n)
validation_set = TensorDataset(X_val_n, y_val_n, y_val)

train_dataloader = DataLoader(
    training_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_dataloader = DataLoader(
    validation_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# =========================================================
# model
# =========================================================
model = SRResNet(
    num_resblk=NUM_RESBLK,
    num_features=NUM_FEATURES,
    input_channels=X.shape[1],
    output_channels=y.shape[1],
    scale=SCALE,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

scheduler = ReduceLROnPlateau(
    optimizer,
    patience=15,
    factor=0.5,
)

loss_fn = nn.MSELoss()


# =========================================================
# train
# =========================================================
train_loss_list = []
val_loss_list = []

best_val_loss = float("inf")
best_epoch = -1
epochs_no_improve = 0

y_mean_dev = y_mean.to(device)
y_std_dev = y_std.to(device)

for epoch in range(EPOCHS):
    model.train()
    train_loss_sum = 0.0
    train_samples_seen = 0

    for Xn, yn in train_dataloader:
        Xn = Xn.to(device)
        yn = yn.to(device)

        y_pred_n = model(Xn)
        loss = loss_fn(y_pred_n, yn)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = Xn.shape[0]
        train_loss_sum += loss.item() * batch_size
        train_samples_seen += batch_size

    train_loss = train_loss_sum / train_samples_seen
    train_loss_list.append(train_loss)

    model.eval()
    val_loss_sum = 0.0
    val_samples_seen = 0
    val_physical_sse = 0.0
    val_physical_count = 0

    with torch.no_grad():
        for Xn, yn, y_raw in val_dataloader:
            Xn = Xn.to(device)
            yn = yn.to(device)
            y_raw = y_raw.to(device)

            y_pred_n = model(Xn)
            loss = loss_fn(y_pred_n, yn)

            y_pred = y_pred_n * y_std_dev + y_mean_dev
            se = (y_pred - y_raw) ** 2

            batch_size = Xn.shape[0]
            val_loss_sum += loss.item() * batch_size
            val_samples_seen += batch_size
            val_physical_sse += se.sum().item()
            val_physical_count += se.numel()

    val_loss = val_loss_sum / val_samples_seen
    val_physical_mse = val_physical_sse / val_physical_count
    val_loss_list.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save(
            {
                "epoch": best_epoch,
                "model_name": "SRResNet",
                "input_path": INPUT_PATH,
                "target_path": TARGET_PATH,
                "model_hparams": {
                    "num_resblk": NUM_RESBLK,
                    "num_features": NUM_FEATURES,
                    "input_channels": X.shape[1],
                    "output_channels": y.shape[1],
                    "scale": SCALE,
                },
                "training_hparams": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "epochs": EPOCHS,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_delta": MIN_DELTA,
                    "loss": "mse",
                    "val_fraction": VAL_FRACTION,
                },
                "model_state": copy.deepcopy(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "train_loss_list": train_loss_list,
                "val_loss_list": val_loss_list,
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
            },
            best_ckpt_path,
        )

    else:
        epochs_no_improve += 1

    print(
        f"Epoch {epoch:03d} | "
        f"Train MSE: {train_loss:.6f} | "
        f"Val MSE: {val_loss:.6f} | "
        f"Val physical MSE: {val_physical_mse:.6f} | "
        f"Best Val: {best_val_loss:.6f} | "
        f"Best Epoch: {best_epoch:03d} | "
        f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
        f"lr: {optimizer.param_groups[0]['lr']:.6e}"
    )

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping at epoch {epoch:03d}; "
            f"best epoch was {best_epoch:03d}"
        )
        break


# =========================================================
# save summary
# =========================================================
summary = {
    "best_val_loss": best_val_loss,
    "best_epoch": best_epoch,
    "best_checkpoint_path": best_ckpt_path,
    "input_path": INPUT_PATH,
    "target_path": TARGET_PATH,
    "save_root": SAVE_ROOT,
    "model_hparams": {
        "num_resblk": NUM_RESBLK,
        "num_features": NUM_FEATURES,
        "input_channels": X.shape[1],
        "output_channels": y.shape[1],
        "scale": SCALE,
    },
    "training_hparams": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "loss": "mse",
        "val_fraction": VAL_FRACTION,
    },
    "normalization": {
        "X_mean": X_mean.item(),
        "X_std": X_std.item(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
    },
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print("\nTraining finished.")
print(f"Best validation MSE: {best_val_loss:.6f}")
print(f"Best epoch: {best_epoch}")
print("Best checkpoint:", best_ckpt_path)
print("Summary saved to:", summary_path)
