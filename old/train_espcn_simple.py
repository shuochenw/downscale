# =========================================================
# user settings
# =========================================================
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 0.0
RANDOM_SEED = 42

BATCH_SIZE = 32
LEARNING_RATE = 2e-4

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
import json
import random

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import ESPCNx4

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
target_file = "high_res.pth"

save_root = os.path.join(
    exp_folder_name,
    "trained_models",
    "ESPCN",
    "low_res_to_high_res_full_image_mse",
)

os.makedirs(save_root, exist_ok=True)

best_ckpt_path = os.path.join(save_root, "best_model.pth")
summary_path = os.path.join(save_root, "training_summary.json")
validation_plot_path = os.path.join(save_root, "validation_sample_prediction.png")
validation_summary_plot_path = os.path.join(save_root, "validation_set_summary.png")
validation_rmse_plot_path = os.path.join(save_root, "validation_set_rmse.png")


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
def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values.")


def safe_std(tensor):
    std = tensor.std()
    if std.item() == 0:
        return torch.tensor(1.0, dtype=tensor.dtype)
    return std


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
print("X:", X.shape)
print("y:", y.shape)


# =========================================================
# train/validation split
# =========================================================
if gcm_name == "CanESM2":
    start_year = 1951
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

if X_train.shape[0] == 0 or X_val.shape[0] == 0:
    raise ValueError("Train or validation split is empty.")


# =========================================================
# normalization
# =========================================================
X_mean = X_train.mean()
X_std = safe_std(X_train)
y_mean = y_train.mean()
y_std = safe_std(y_train)

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
validation_set = TensorDataset(X_val_n, y_val_n)

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
model = ESPCNx4().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
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
val_physical_loss_list = []

best_val_loss = float("inf")
best_epoch = -1
epochs_no_improve = 0

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
    val_loss_sum = 0.0
    val_physical_loss_sum = 0.0
    val_samples = 0

    with torch.no_grad():
        for Xn, yn in val_dataloader:
            Xn = Xn.to(device)
            yn = yn.to(device)

            y_pred_n = model(Xn)
            loss = loss_fn(y_pred_n, yn)
            y_pred = y_pred_n * y_std.to(device) + y_mean.to(device)
            y_true = yn * y_std.to(device) + y_mean.to(device)
            physical_loss = loss_fn(y_pred, y_true)

            batch_size = Xn.shape[0]
            val_loss_sum += loss.item() * batch_size
            val_physical_loss_sum += physical_loss.item() * batch_size
            val_samples += batch_size

    val_loss = val_loss_sum / val_samples
    val_physical_loss = val_physical_loss_sum / val_samples
    val_loss_list.append(val_loss)
    val_physical_loss_list.append(val_physical_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        best_epoch = epoch
        epochs_no_improve = 0

        torch.save(
            {
                "epoch": best_epoch,
                "model_name": "ESPCNx4",
                "input_file": input_file,
                "target_file": target_file,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_loss_list": train_loss_list,
                "val_loss_list": val_loss_list,
                "val_physical_loss_list": val_physical_loss_list,
                "X_mean": X_mean,
                "X_std": X_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "config": {
                    "rcm_var": rcm_var,
                    "gcm_name": gcm_name,
                    "rcm_name": rcm_name,
                    "grid": grid,
                    "rcm_product": rcm_product,
                    "factor": factor,
                    "loss": "full_image_mse",
                    "input": "low_res.pth only",
                    "target": "high_res.pth only",
                    "regularization": None,
                },
            },
            best_ckpt_path,
        )
    else:
        epochs_no_improve += 1

    print(
        f"Epoch {epoch:03d} | "
        f"Train full-image MSE: {train_loss:.6f} | "
        f"Val full-image MSE: {val_loss:.6f} | "
        f"Val physical MSE: {val_physical_loss:.6f} | "
        f"Best Val: {best_val_loss:.6f} | "
        f"Best Epoch: {best_epoch:03d} | "
        f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
        f"lr: {optimizer.param_groups[0]['lr']:.6e}"
    )

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f"Early stopping at epoch {epoch}.")
        break


# =========================================================
# summary
# =========================================================
summary = {
    "model_name": "ESPCNx4",
    "exp_folder_name": exp_folder_name,
    "save_root": save_root,
    "best_checkpoint": best_ckpt_path,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "validation_sample_plot_path": validation_plot_path,
    "validation_summary_plot_path": validation_summary_plot_path,
    "validation_rmse_plot_path": validation_rmse_plot_path,
    "input_file": input_file,
    "target_file": target_file,
    "loss": "full_image_mse",
    "regularization": None,
    "train_loss_list": train_loss_list,
    "val_loss_list": val_loss_list,
    "val_physical_loss_list": val_physical_loss_list,
    "normalization": {
        "X_mean": X_mean.item(),
        "X_std": X_std.item(),
        "y_mean": y_mean.item(),
        "y_std": y_std.item(),
    },
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Training summary saved to:", summary_path)


# =========================================================
# plot one validation sample
# =========================================================
if len(validation_set) == 0:
    raise ValueError("Validation set is empty; cannot create validation sample plot.")

checkpoint = torch.load(best_ckpt_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    X_sample = X_val_n[0:1].to(device)
    y_true = y_val[0, 0].cpu().numpy()

    y_pred_n = model(X_sample)
    y_pred = (y_pred_n * y_std.to(device) + y_mean.to(device))[0, 0].cpu().numpy()

error = y_pred - y_true

value_pixels = np.concatenate([
    y_true[np.isfinite(y_true)].ravel(),
    y_pred[np.isfinite(y_pred)].ravel(),
])

if value_pixels.size == 0:
    raise ValueError("No finite pixels found for validation plot.")

vmin = float(value_pixels.min())
vmax = float(value_pixels.max())
err_abs = float(np.nanmax(np.abs(error)))
if err_abs == 0:
    err_abs = 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[0].set_title("Target high_res")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
axes[1].set_title("ESPCN prediction")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(error, origin="lower", cmap="coolwarm", vmin=-err_abs, vmax=err_abs)
axes[2].set_title("Prediction - target")
axes[2].set_xticks([])
axes[2].set_yticks([])
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle("First validation sample, full image")
fig.savefig(validation_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Validation sample plot saved to:", validation_plot_path)


# =========================================================
# plot entire validation set summary
# =========================================================
sum_true = None
sum_pred = None
sum_error = None
sum_sq_error = None
total_samples = 0

with torch.no_grad():
    for Xn, yn in val_dataloader:
        Xn = Xn.to(device)
        yn = yn.to(device)

        y_pred_n = model(Xn)
        y_pred_batch = (y_pred_n * y_std.to(device) + y_mean.to(device)).cpu()
        y_true_batch = (yn * y_std.to(device) + y_mean.to(device)).cpu()
        error_batch = y_pred_batch - y_true_batch

        batch_sum_true = y_true_batch.sum(dim=0)
        batch_sum_pred = y_pred_batch.sum(dim=0)
        batch_sum_error = error_batch.sum(dim=0)
        batch_sum_sq_error = (error_batch ** 2).sum(dim=0)

        if sum_true is None:
            sum_true = batch_sum_true
            sum_pred = batch_sum_pred
            sum_error = batch_sum_error
            sum_sq_error = batch_sum_sq_error
        else:
            sum_true += batch_sum_true
            sum_pred += batch_sum_pred
            sum_error += batch_sum_error
            sum_sq_error += batch_sum_sq_error

        total_samples += Xn.shape[0]

if total_samples <= 0:
    raise ValueError("Validation set is empty; cannot create validation summary plot.")

mean_true = (sum_true / total_samples)[0].numpy()
mean_pred = (sum_pred / total_samples)[0].numpy()
bias = (mean_true - mean_pred)
rmse = torch.sqrt(sum_sq_error / total_samples)[0].numpy()

summary_values = np.concatenate([
    mean_true[np.isfinite(mean_true)].ravel(),
    mean_pred[np.isfinite(mean_pred)].ravel(),
])

if summary_values.size == 0:
    raise ValueError("No finite pixels found for validation summary plot.")

summary_vmin = float(summary_values.min())
summary_vmax = float(summary_values.max())
bias_abs = float(np.nanmax(np.abs(bias)))
if bias_abs == 0:
    bias_abs = 1.0

rmse_vmax = float(np.nanmax(rmse))
if rmse_vmax == 0:
    rmse_vmax = 1.0

fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

im0 = axes[0].imshow(mean_true, origin="lower", cmap="viridis", vmin=summary_vmin, vmax=summary_vmax)
axes[0].set_title("Mean target")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(mean_pred, origin="lower", cmap="viridis", vmin=summary_vmin, vmax=summary_vmax)
axes[1].set_title("Mean prediction")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(bias, origin="lower", cmap="coolwarm", vmin=-bias_abs, vmax=bias_abs)
axes[2].set_title("Bias: target - prediction")
axes[2].set_xticks([])
axes[2].set_yticks([])
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.suptitle("Entire validation set mean fields, full image")
fig.savefig(validation_summary_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Validation summary plot saved to:", validation_summary_plot_path)

fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
im = ax.imshow(rmse, origin="lower", cmap="magma", vmin=0.0, vmax=rmse_vmax)
ax.set_title("Validation set RMSE")
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("Entire validation set, full image")
fig.savefig(validation_rmse_plot_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Validation RMSE plot saved to:", validation_rmse_plot_path)
print("Best checkpoint saved to:", best_ckpt_path)
print(f"Best validation MSE: {best_val_loss:.6f}")
