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

rcm_var = "tas"
gcm_name = "CanESM2"
rcm_name = "RCA4"
grid = "NAM-44i"
rcm_product = "raw"
factor = 4

DATA_ROOT = "/projects/sds-lab/Shuochen/downscaling/preprocessed"


# =========================================================
# imports
# =========================================================
import os
import copy
import json
import random

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import SRResNet

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# paths
# =========================================================
exp_folder_name = os.path.join(
    DATA_ROOT,
    f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.{rcm_product}.GCM_RCM",
)

input_file = "low_res.pth"
baseline_file = "high_res_gcm.pth"
target_file = "high_res.pth"
hr_mask_file = "high_res_mask.pth"

save_root = os.path.join(
    exp_folder_name,
    "trained_models",
    "SRResNet",
    "GCM_low_res_high_res_gcm_residual_to_HR_RCM_land_val",
)

os.makedirs(save_root, exist_ok=True)

best_ckpt_path = os.path.join(save_root, "best_model.pth")
summary_path = os.path.join(save_root, "training_summary.json")
validation_plot_path = os.path.join(save_root, "validation_sample_prediction.png")


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
baseline = torch.load(os.path.join(exp_folder_name, baseline_file)).float()
y = torch.load(os.path.join(exp_folder_name, target_file)).float()
hr_mask = torch.load(os.path.join(exp_folder_name, hr_mask_file)).float()

baseline = match_sample_dim(baseline, y.shape[0], "baseline")
hr_mask = match_sample_dim(hr_mask, y.shape[0], "hr_mask")
hr_mask = (hr_mask > 0.5).float()

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

if y.shape != hr_mask.shape:
    raise ValueError(f"Target shape {y.shape} does not match HR mask shape {hr_mask.shape}")

if y.shape != baseline.shape:
    raise ValueError(
        f"Target shape {y.shape} does not match HR GCM baseline shape {baseline.shape}"
    )

assert_finite(X, "X")
assert_finite(baseline, "baseline")
assert_finite(y, "y")
assert_finite(hr_mask, "hr_mask")

print("Loaded shapes:")
print("X       :", X.shape)
print("baseline:", baseline.shape)
print("y       :", y.shape)
print("hr_mask :", hr_mask.shape)


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
baseline_train = baseline[:train_end_idx]
y_train = y[:train_end_idx]

X_val = X[val_start_idx:]
baseline_val = baseline[val_start_idx:]
y_val = y[val_start_idx:]
hr_mask_val = hr_mask[val_start_idx:]

print("Split shapes:")
print("X_train        :", X_train.shape)
print("baseline_train :", baseline_train.shape)
print("y_train        :", y_train.shape)
print("X_val          :", X_val.shape)
print("baseline_val   :", baseline_val.shape)
print("y_val          :", y_val.shape)
print("hr_mask_val    :", hr_mask_val.shape)


# =========================================================
# residual target and normalization
# =========================================================
# HR GCM is used as a physical baseline. The network learns:
#   HR RCM residual = HR RCM - HR GCM baseline
residual_train = (y_train - baseline_train).contiguous()

X_mean = X_train.mean()
X_std = X_train.std()
residual_mean = residual_train.mean()
residual_std = residual_train.std()

if X_std.item() == 0:
    X_std = torch.tensor(1.0, dtype=X_train.dtype)

if residual_std.item() == 0:
    residual_std = torch.tensor(1.0, dtype=residual_train.dtype)

X_train_in = ((X_train - X_mean) / X_std).contiguous()
X_val_in = ((X_val - X_mean) / X_std).contiguous()

y_train_n = ((residual_train - residual_mean) / residual_std).contiguous()

print("Residual setup:")
print("baseline_file:", baseline_file)
print("X_mean:", X_mean.item())
print("X_std :", X_std.item())
print("residual_mean:", residual_mean.item())
print("residual_std :", residual_std.item())
print("X_train_in:", X_train_in.shape)
print("X_val_in  :", X_val_in.shape)
print("y_train_n :", y_train_n.shape)


# =========================================================
# datasets
# =========================================================
training_set = TensorDataset(X_train_in, y_train_n)
validation_set = TensorDataset(X_val_in, y_val, baseline_val, hr_mask_val)

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

residual_mean_dev = residual_mean.to(device)
residual_std_dev = residual_std.to(device)

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
        for Xn, y_raw, baseline_raw, mask_raw in val_dataloader:
            Xn = Xn.to(device)
            y_raw = y_raw.to(device)
            baseline_raw = baseline_raw.to(device)
            mask_raw = mask_raw.to(device)

            y_pred_n = model(Xn)
            y_pred_residual = y_pred_n * residual_std_dev + residual_mean_dev
            y_pred = baseline_raw + y_pred_residual

            mask_raw = (mask_raw > 0.5).to(dtype=y_pred.dtype)
            se = ((y_pred - y_raw) ** 2) * mask_raw
            val_sse += se.sum().item()
            val_count += mask_raw.sum().item()

    if val_count <= 0:
        raise ValueError("Validation mask has no land pixels.")

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
                "baseline_file": baseline_file,
                "hr_mask_file": hr_mask_file,
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
                    "loss": "train_normalized_residual_mse__val_land_only_physical_mse",
                    "input_channels_description": "channel0=normalized_masked_LR_GCM",
                    "residual_baseline": (
                        "high_res_gcm.pth is added in physical space; "
                        "model predicts HR_RCM_minus_HR_GCM residual"
                    ),
                    "validation_mask": "high_res_mask.pth is used only for land-only validation loss",
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
                "residual_mean": residual_mean,
                "residual_std": residual_std,
            },
            best_ckpt_path,
        )

    else:
        epochs_no_improve += 1

    print(
        f"Epoch {epoch:03d} | "
        f"Train: {train_loss:.6f} | "
        f"Val land physical MSE: {val_loss:.6f} | "
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
    "validation_sample_plot_path": validation_plot_path,
    "input_file": input_file,
    "baseline_file": baseline_file,
    "hr_mask_file": hr_mask_file,
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
        "loss": "train_normalized_residual_mse__val_land_only_physical_mse",
        "input_channels_description": "channel0=normalized_masked_LR_GCM",
        "residual_baseline": (
            "high_res_gcm.pth is added in physical space; "
            "model predicts HR_RCM_minus_HR_GCM residual"
        ),
        "validation_mask": "high_res_mask.pth is used only for land-only validation loss",
    },
    "normalization": {
        "X_mean": X_mean.item(),
        "X_std": X_std.item(),
        "residual_mean": residual_mean.item(),
        "residual_std": residual_std.item(),
    },
}


# =========================================================
# plot one validation sample
# =========================================================
if len(validation_set) == 0:
    raise ValueError("Validation set is empty; cannot create validation sample plot.")

if os.path.exists(best_ckpt_path):
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
else:
    print("Best checkpoint not found; plotting the final model state.")

model.eval()

with torch.no_grad():
    X_sample = X_val_in[0:1].to(device)
    y_true = y_val[0, 0].cpu().numpy()
    baseline_sample = baseline_val[0:1].to(device)
    mask = (hr_mask_val[0, 0] > 0.5).cpu().numpy()

    y_pred_n = model(X_sample)
    y_pred_residual = y_pred_n * residual_std_dev + residual_mean_dev
    y_pred = (baseline_sample + y_pred_residual)[0, 0].cpu().numpy()

y_error = y_pred - y_true

y_true_plot = np.where(mask, y_true, np.nan)
y_pred_plot = np.where(mask, y_pred, np.nan)
y_error_plot = np.where(mask, y_error, np.nan)

value_pixels = np.concatenate(
    [
        y_true_plot[np.isfinite(y_true_plot)],
        y_pred_plot[np.isfinite(y_pred_plot)],
    ]
)

if value_pixels.size == 0:
    raise ValueError("Validation sample mask has no land pixels.")

vmin = float(value_pixels.min())
vmax = float(value_pixels.max())

error_pixels = y_error_plot[np.isfinite(y_error_plot)]
error_lim = float(np.max(np.abs(error_pixels))) if error_pixels.size > 0 else 1.0

if error_lim == 0.0:
    error_lim = 1.0

fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

im0 = axes[0].imshow(y_true_plot, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("Ground truth")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(y_pred_plot, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title("Model output")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(
    y_error_plot,
    origin="lower",
    cmap="RdBu_r",
    vmin=-error_lim,
    vmax=error_lim,
)
axes[2].set_title("Error: output - truth")
axes[2].set_xticks([])
axes[2].set_yticks([])
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle("First validation sample, land pixels only")
fig.savefig(validation_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Validation sample plot saved to:", validation_plot_path)

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=4)

print("\nTraining finished.")
print(f"Best validation MSE: {best_val_loss:.6f}")
print(f"Best epoch: {best_epoch}")
print("Best checkpoint:", best_ckpt_path)
print("Summary saved to:", summary_path)
