# =========================================================
# user settings
# =========================================================
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 0.0
RANDOM_SEED = 42

BATCH_SIZE = 256
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
NUM_RESBLK = 8
NUM_FEATURES = 64

rcm_var = "tmean"
gcm_name = "CanESM2"
rcm_name = "CanRCM4"
grid = "NAM-44i"
factor = 4

DATA_ROOT = "/projects/sds-lab/Shuochen/downscaling/CORDEX"


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
# paths
# =========================================================
exp_folder_name = os.path.join(
    DATA_ROOT,
    f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.GCM_to_HR_RCM",
)

input_file = "low_res.pth"
target_file = "high_res.pth"

save_root = os.path.join(
    exp_folder_name,
    "trained_models",
    "SRResNet",
    "GCM_low_res_climate_only_plain_mse",
)

os.makedirs(save_root, exist_ok=True)

best_ckpt_path = os.path.join(save_root, "best_model.pth")
summary_path = os.path.join(save_root, "training_summary.json")


# =========================================================
# setup
# =========================================================
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
def match_sample_dim(tensor, n_samples, name):
    if tensor.shape[0] == n_samples:
        return tensor

    if tensor.shape[0] == 1:
        print(f"{name} has one sample; expanding to {n_samples} samples.")
        return tensor.expand(n_samples, -1, -1, -1)

    raise ValueError(
        f"{name} has incompatible sample dimension: "
        f"{tensor.shape[0]} vs expected {n_samples}"
    )


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values.")


# =========================================================
# load data
# =========================================================
X = torch.load(os.path.join(exp_folder_name, input_file)).float()
y = torch.load(os.path.join(exp_folder_name, target_file)).float()

if X.shape[0] != y.shape[0]:
    raise ValueError(f"Sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")

if y.shape[-2] != factor * X.shape[-2]:
    raise ValueError(
        f"Height mismatch: HR height {y.shape[-2]} != "
        f"{factor} * LR height {X.shape[-2]}"
    )

if y.shape[-1] != factor * X.shape[-1]:
    raise ValueError(
        f"Width mismatch: HR width {y.shape[-1]} != "
        f"{factor} * LR width {X.shape[-1]}"
    )

assert_finite(X, "X")
assert_finite(y, "y")

print("Loaded shapes:")
print("X      :", X.shape)
print("y      :", y.shape)


# =========================================================
# train/validation split
# =========================================================
if gcm_name == "CanESM2":
    start_year = 1950
    train_end_year = 2005
    val_start_year = 2081

    train_end_idx = (train_end_year - start_year + 1) * 365
    val_start_idx = (val_start_year - start_year) * 365

elif gcm_name == "EC-EARTH":
    def is_leap(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def days_between_years(start_year, end_year_inclusive):
        total = 0
        for year in range(start_year, end_year_inclusive + 1):
            total += 366 if is_leap(year) else 365
        return total

    start_year = 1951
    train_end_year = 2005
    val_start_year = 2081

    train_end_idx = days_between_years(start_year, train_end_year)
    val_start_idx = days_between_years(start_year, val_start_year - 1)

else:
    raise ValueError(f"Year split is not defined for gcm_name={gcm_name}")

X_train = X[:train_end_idx]
y_train = y[:train_end_idx]

X_val = X[val_start_idx:]
y_val = y[val_start_idx:]

print("Split shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_val  :", X_val.shape)
print("y_val  :", y_val.shape)


# =========================================================
# normalization
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

print("Normalization:")
print("X_mean:", X_mean.item())
print("X_std :", X_std.item())
print("y_mean:", y_mean.item())
print("y_std :", y_std.item())


# =========================================================
# datasets
# =========================================================
training_set = TensorDataset(X_train_n, y_train_n)
validation_set = TensorDataset(X_val_n, y_val)

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
    input_channels=1,
    output_channels=1,
    scale=factor,
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
    train_samples = 0

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
        train_samples += batch_size

    train_loss = train_loss_sum / train_samples
    train_loss_list.append(train_loss)

    model.eval()
    val_sse = 0.0
    val_count = 0

    with torch.no_grad():
        for Xn, y_raw in val_dataloader:
            Xn = Xn.to(device)
            y_raw = y_raw.to(device)

            y_pred_n = model(Xn)
            y_pred = y_pred_n * y_std_dev + y_mean_dev

            se = (y_pred - y_raw) ** 2
            val_sse += se.sum().item()
            val_count += se.numel()

    val_loss = val_sse / val_count
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
                "input_file": input_file,
                "target_file": target_file,
                "model_hparams": {
                    "num_resblk": NUM_RESBLK,
                    "num_features": NUM_FEATURES,
                    "input_channels": 1,
                    "output_channels": 1,
                    "scale": factor,
                },
                "training_hparams": {
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "epochs": EPOCHS,
                    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                    "min_delta": MIN_DELTA,
                    "loss": "train_full_image_normalized_mse__val_full_image_physical_mse",
                    "rcm_var": rcm_var,
                    "gcm_name": gcm_name,
                    "rcm_name": rcm_name,
                    "grid": grid,
                    "train_end_year": train_end_year,
                    "val_start_year": val_start_year,
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
        f"Train: {train_loss:.6f} | "
        f"Val physical MSE: {val_loss:.6f} | "
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
    "input_file": input_file,
    "target_file": target_file,
    "model_hparams": {
        "num_resblk": NUM_RESBLK,
        "num_features": NUM_FEATURES,
        "input_channels": 1,
        "output_channels": 1,
        "scale": factor,
    },
    "training_hparams": {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "loss": "train_full_image_normalized_mse__val_full_image_physical_mse",
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
