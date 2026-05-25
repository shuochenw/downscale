# =========================================================
# imports
# =========================================================
import os
import copy
import json
import argparse

import torch
import optuna
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models import ViTDownscaler


# =========================================================
# argument parser
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--rcm_var",
    type=str,
    choices=["tmean", "tmax", "tmin", "prec"],
    required=True,
)

parser.add_argument(
    "--gcm_name",
    type=str,
    required=True,
)

parser.add_argument(
    "--rcm_name",
    type=str,
    required=True,
)

parser.add_argument(
    "--grid",
    type=str,
    default="NAM-44i",
)

parser.add_argument(
    "--factor",
    type=int,
    default=4,
)

parser.add_argument(
    "--input_source",
    type=str,
    choices=["gcm", "coarse"],
    default="gcm",
    help="'gcm' uses low_res.pth; 'coarse' uses coarse_4x.pth",
)

parser.add_argument(
    "--input_channels",
    type=int,
    default=3,
    help=(
        "Input channels for ViTDownscaler. Default 3: "
        "LR climate input, LR land-sea mask, LR elevation."
    ),
)

parser.add_argument(
    "--alpha_coarse",
    type=float,
    default=0.5,
    help="Weight for full-image coarse RCM consistency loss.",
)

parser.add_argument(
    "--n_trials",
    type=int,
    default=3000,
)

parser.add_argument(
    "--epochs",
    type=int,
    default=300,
)

parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=30,
)

parser.add_argument(
    "--min_delta",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--random_seed",
    type=int,
    default=42,
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
)

parser.add_argument(
    "--model_name",
    type=str,
    choices=["ViTDownscaler"],
    default="ViTDownscaler",
)

args = parser.parse_args()


# =========================================================
# settings from args
# =========================================================
rcm_var = args.rcm_var
gcm_name = args.gcm_name
rcm_name = args.rcm_name
grid = args.grid
factor = args.factor

INPUT_SOURCE = args.input_source
INPUT_CHANNELS = args.input_channels
ALPHA_COARSE = args.alpha_coarse

N_TRIALS = args.n_trials
EPOCHS = args.epochs
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
MIN_DELTA = args.min_delta
RANDOM_SEED = args.random_seed
BATCH_SIZE = args.batch_size

model_name = args.model_name

print("Running with:")
print("rcm_var =", rcm_var)
print("gcm_name =", gcm_name)
print("rcm_name =", rcm_name)
print("grid =", grid)
print("factor =", factor)
print("input_source =", INPUT_SOURCE)
print("input_channels =", INPUT_CHANNELS)
print("alpha_coarse =", ALPHA_COARSE)
print("n_trials =", N_TRIALS)
print("epochs =", EPOCHS)
print("early_stopping_patience =", EARLY_STOPPING_PATIENCE)
print("batch_size =", BATCH_SIZE)
print("model_name =", model_name)


# =========================================================
# data folder
# =========================================================
exp_folder_name = (
    f"/projects/sds-lab/Shuochen/downscaling/CORDEX/"
    f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.GCM_to_HR_RCM/"
)


# =========================================================
# device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================================================
# choose input file
# =========================================================
if INPUT_SOURCE == "gcm":
    input_file = "low_res.pth"
    input_mask_file = "low_res_mask.pth"
    input_elev_file = "low_res_elevation.pth"
    input_tag = "GCM_low_res"

elif INPUT_SOURCE == "coarse":
    input_file = f"coarse_{factor}x.pth"
    input_mask_file = f"coarse_{factor}x_mask.pth"
    input_elev_file = f"coarse_{factor}x_elevation.pth"
    input_tag = f"coarsened_RCM_{factor}x"

else:
    raise ValueError(
        "INPUT_SOURCE must be either 'gcm' or 'coarse'. "
        f"Got: {INPUT_SOURCE}"
    )


# =========================================================
# coarse consistency target files
# =========================================================
coarse_target_file = f"coarse_{factor}x.pth"
coarse_target_mask_file = f"coarse_{factor}x_mask.pth"


# =========================================================
# save paths
# =========================================================
input_setup_tag = f"{input_tag}_with_mask_elevation"
loss_setup_tag = "full_image_loss_no_ocean_constraint"

save_root = os.path.join(
    exp_folder_name,
    "trained_models",
    model_name,
    input_setup_tag,
    loss_setup_tag,
)

os.makedirs(save_root, exist_ok=True)

study_summary_path = os.path.join(save_root, "optuna_best_params.json")
all_trials_summary_path = os.path.join(save_root, "optuna_all_trials_summary.json")

print("Input source:", INPUT_SOURCE)
print("Input file:", input_file)
print("Input mask file:", input_mask_file)
print("Input elevation file:", input_elev_file)
print("Coarse consistency target file:", coarse_target_file)
print("Coarse consistency mask file:", coarse_target_mask_file)
print("Input channels:", INPUT_CHANNELS)
print("Loss setup:", loss_setup_tag)
print("All trial checkpoints will be saved to:", save_root)


# =========================================================
# helper functions
# =========================================================
def match_sample_dim(tensor, n_samples, name):
    """
    Allows either:
      [N, C, H, W] full-time tensor
    or:
      [1, C, H, W] static tensor.

    Returns tensor with first dimension compatible with N.
    """
    if tensor.shape[0] == n_samples:
        return tensor

    if tensor.shape[0] == 1:
        print(f"{name} has one sample; expanding to {n_samples} samples.")
        return tensor.expand(n_samples, -1, -1, -1)

    raise ValueError(
        f"{name} has incompatible sample dimension: "
        f"{tensor.shape[0]} vs expected {n_samples}"
    )


def full_image_mse_from_sums(pred, target):
    """
    Return squared-error sum and number of elements.
    Useful for validation so the final MSE is not biased by last-batch size.
    """
    se = (pred - target) ** 2
    return se.sum().item(), se.numel()


# =========================================================
# load tensors
# =========================================================
X = torch.load(os.path.join(exp_folder_name, input_file))
X_mask = torch.load(os.path.join(exp_folder_name, input_mask_file))
X_elev = torch.load(os.path.join(exp_folder_name, input_elev_file))

y = torch.load(os.path.join(exp_folder_name, "high_res.pth"))
mask_hr = torch.load(os.path.join(exp_folder_name, "high_res_mask.pth"))

coarse_path = os.path.join(exp_folder_name, coarse_target_file)
coarse_mask_path = os.path.join(exp_folder_name, coarse_target_mask_file)

if not os.path.exists(coarse_path):
    raise FileNotFoundError(
        f"Coarse RCM consistency target not found: {coarse_path}"
    )

if not os.path.exists(coarse_mask_path):
    raise FileNotFoundError(
        f"Coarse RCM consistency mask not found: {coarse_mask_path}"
    )

y_coarse = torch.load(coarse_path)
mask_coarse = torch.load(coarse_mask_path)

print("Loaded shapes:")
print("X          :", X.shape)
print("X_mask     :", X_mask.shape)
print("X_elev     :", X_elev.shape)
print("y          :", y.shape)
print("mask_hr    :", mask_hr.shape)
print("y_coarse   :", y_coarse.shape)
print("mask_coarse:", mask_coarse.shape)


# =========================================================
# shape checks and static/full expansion
# =========================================================
X_mask = match_sample_dim(X_mask, X.shape[0], "X_mask")
X_elev = match_sample_dim(X_elev, X.shape[0], "X_elev")

mask_hr = match_sample_dim(mask_hr, y.shape[0], "mask_hr")
y_coarse = match_sample_dim(y_coarse, X.shape[0], "y_coarse")
mask_coarse = match_sample_dim(mask_coarse, X.shape[0], "mask_coarse")

if y.shape != mask_hr.shape:
    raise ValueError(
        f"Shape mismatch: y shape {y.shape} does not match HR mask shape {mask_hr.shape}"
    )

if X.shape != X_mask.shape:
    raise ValueError(
        f"Shape mismatch: X shape {X.shape} does not match X_mask shape {X_mask.shape}"
    )

if X.shape != X_elev.shape:
    raise ValueError(
        f"Shape mismatch: X shape {X.shape} does not match X_elev shape {X_elev.shape}"
    )

if y_coarse.shape != mask_coarse.shape:
    raise ValueError(
        f"Shape mismatch: y_coarse shape {y_coarse.shape} "
        f"does not match mask_coarse shape {mask_coarse.shape}"
    )

if y_coarse.shape != X.shape:
    raise ValueError(
        f"Shape mismatch: y_coarse shape {y_coarse.shape} "
        f"must match LR input shape {X.shape}"
    )

if X.shape[0] != y.shape[0]:
    raise ValueError(
        f"Sample mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples"
    )

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

# Masks are used only as input channels.
# They are NOT used in the loss.
mask_hr = mask_hr.float()
mask_hr = (mask_hr > 0.5).float()

X_mask = X_mask.float()
X_mask = (X_mask > 0.5).float()

mask_coarse = mask_coarse.float()
mask_coarse = (mask_coarse > 0.5).float()

# Elevation should already be normalized in preprocessing.
X_elev = X_elev.float()
X_elev = torch.nan_to_num(X_elev, nan=0.0, posinf=0.0, neginf=0.0)

# Climate tensors
X = X.float()
y = y.float()
y_coarse = y_coarse.float()

print("After checks:")
print("X          :", X.shape)
print("X_mask     :", X_mask.shape)
print("X_elev     :", X_elev.shape)
print("y          :", y.shape)
print("mask_hr    :", mask_hr.shape)
print("y_coarse   :", y_coarse.shape)
print("mask_coarse:", mask_coarse.shape)


# =========================================================
# split train/val by year
# train: historical period
# val: future period
# =========================================================
if gcm_name == "CanESM2":
    # CanESM2/CORDEX preprocessing currently gives 1950-2100,
    # assuming no leap years in the aligned dataset.
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
        for y in range(start_year, end_year_inclusive + 1):
            total += 366 if is_leap(y) else 365
        return total

    start_year = 1951
    train_end_year = 2005
    val_start_year = 2081

    train_end_idx = days_between_years(start_year, train_end_year)
    val_start_idx = days_between_years(start_year, val_start_year - 1)

else:
    raise ValueError(f"Year split not defined for gcm_name={gcm_name}")

print("train_end_idx =", train_end_idx)
print("val_start_idx =", val_start_idx)
print("Total samples  =", X.shape[0])


# =========================================================
# create train/val tensors
# =========================================================
X_train = X[:train_end_idx]
X_mask_train = X_mask[:train_end_idx]
X_elev_train = X_elev[:train_end_idx]

y_train = y[:train_end_idx]
mask_train = mask_hr[:train_end_idx]

y_coarse_train = y_coarse[:train_end_idx]
mask_coarse_train = mask_coarse[:train_end_idx]

X_val = X[val_start_idx:]
X_mask_val = X_mask[val_start_idx:]
X_elev_val = X_elev[val_start_idx:]

y_val = y[val_start_idx:]
mask_val = mask_hr[val_start_idx:]

y_coarse_val = y_coarse[val_start_idx:]
mask_coarse_val = mask_coarse[val_start_idx:]

print("Train/Val shapes before model input construction:")
print("X_train          :", X_train.shape)
print("X_mask_train     :", X_mask_train.shape)
print("X_elev_train     :", X_elev_train.shape)
print("y_train          :", y_train.shape)
print("mask_train       :", mask_train.shape)
print("y_coarse_train   :", y_coarse_train.shape)
print("mask_coarse_train:", mask_coarse_train.shape)

print("X_val            :", X_val.shape)
print("X_mask_val       :", X_mask_val.shape)
print("X_elev_val       :", X_elev_val.shape)
print("y_val            :", y_val.shape)
print("mask_val         :", mask_val.shape)
print("y_coarse_val     :", y_coarse_val.shape)
print("mask_coarse_val  :", mask_coarse_val.shape)


# =========================================================
# normalization stats from training set only
# Full-image normalization:
#   - No land-only normalization
#   - No ocean masking in normalization
#   - No enforced zero over ocean after normalization
# =========================================================
X_mean = X_train.mean()
X_std = X_train.std()
X_max = X_train.max()

y_mean = y_train.mean()
y_std = y_train.std()
y_max = y_train.max()

if X_std.item() == 0:
    X_std = torch.tensor(1.0, dtype=X_train.dtype)

if y_std.item() == 0:
    y_std = torch.tensor(1.0, dtype=y_train.dtype)

print("Full-image normalization stats:")
print("X_mean:", X_mean.item())
print("X_std :", X_std.item())
print("X_max :", X_max.item())
print("y_mean:", y_mean.item())
print("y_std :", y_std.item())
print("y_max :", y_max.item())


# =========================================================
# normalize climate variable and build model inputs
# =========================================================
X_train_clim_n = (X_train - X_mean) / X_std
X_val_clim_n = (X_val - X_mean) / X_std

# Do NOT force ocean/invalid pixels to zero after normalization.

y_train_n = (y_train - y_mean) / y_std
y_val_n = (y_val - y_mean) / y_std

# Coarse RCM target normalized using same HR target statistics.
y_coarse_train_n = (y_coarse_train - y_mean) / y_std
y_coarse_val_n = (y_coarse_val - y_mean) / y_std

# ViT input channels:
# channel 0: normalized LR climate variable
# channel 1: LR land-sea mask
# channel 2: LR elevation
X_train_in = torch.cat(
    [X_train_clim_n, X_mask_train, X_elev_train],
    dim=1,
).contiguous()

X_val_in = torch.cat(
    [X_val_clim_n, X_mask_val, X_elev_val],
    dim=1,
).contiguous()

input_channels_description = (
    "ViTDownscaler input channels: "
    "channel0=normalized_LR_climate_input; "
    "channel1=LR_land_sea_mask; "
    "channel2=LR_elevation. "
    "The model upsamples these LR channels to HR grid before patch embedding."
)

print("Train/Val shapes after model input construction:")
print("X_train_in:", X_train_in.shape)
print("X_val_in  :", X_val_in.shape)

if X_train_in.shape[1] != INPUT_CHANNELS:
    raise ValueError(
        f"Expected {INPUT_CHANNELS} input channels, got {X_train_in.shape[1]}"
    )

if X_val_in.shape[1] != INPUT_CHANNELS:
    raise ValueError(
        f"Expected {INPUT_CHANNELS} input channels, got {X_val_in.shape[1]}"
    )


# =========================================================
# losses
# =========================================================
loss_fn = nn.MSELoss(reduction="mean")


# =========================================================
# datasets
# Masks are kept only as model inputs.
# They are not used in the loss.
# =========================================================
training_set = TensorDataset(
    X_train_in,
    y_train_n,
    y_coarse_train_n,
)

validation_set = TensorDataset(
    X_val_in,
    y_val,
    y_coarse_val,
)


# =========================================================
# fixed image sizes
# =========================================================
HR_IMG_SIZE = (y.shape[-2], y.shape[-1])
LR_IMG_SIZE = (X.shape[-2], X.shape[-1])

print("LR image size:", LR_IMG_SIZE)
print("HR image size:", HR_IMG_SIZE)


# =========================================================
# objective function
# =========================================================
def objective(trial):
    # -----------------------------------------------------
    # hyperparameters to search
    # -----------------------------------------------------
    batch_size = trial.suggest_categorical("batch_size", [BATCH_SIZE])

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        3e-4,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    patch_size = trial.suggest_categorical(
        "patch_size",
        [2, 4],
    )

    embed_dim = trial.suggest_categorical(
        "embed_dim",
        [64, 128, 192, 256],
    )

    depth = trial.suggest_int(
        "depth",
        2,
        8,
        step=2,
    )

    num_heads = trial.suggest_categorical(
        "num_heads",
        [4, 8],
    )

    mlp_ratio = trial.suggest_categorical(
        "mlp_ratio",
        [2.0, 4.0],
    )

    dropout = trial.suggest_categorical(
        "dropout",
        [0.0, 0.05, 0.1],
    )

    decoder_features = trial.suggest_categorical(
        "decoder_features",
        [32, 64, 128],
    )

    # Skip invalid head/embed combinations
    if embed_dim % num_heads != 0:
        raise optuna.TrialPruned()

    # Skip invalid patch sizes for this HR image
    if HR_IMG_SIZE[0] % patch_size != 0 or HR_IMG_SIZE[1] % patch_size != 0:
        raise optuna.TrialPruned()

    # -----------------------------------------------------
    # paths for this trial
    # -----------------------------------------------------
    trial_ckpt_path = os.path.join(save_root, f"trial_{trial.number:03d}.pth")

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------
    model = ViTDownscaler(
        in_channels=INPUT_CHANNELS,
        out_channels=1,
        hr_img_size=HR_IMG_SIZE,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        decoder_features=decoder_features,
    ).to(device)

    # -----------------------------------------------------
    # dataloaders
    # -----------------------------------------------------
    train_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
    )

    # -----------------------------------------------------
    # training setup
    # -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=15,
        factor=0.5,
    )

    train_loss_list = []
    train_hr_loss_list = []
    train_coarse_loss_list = []

    val_loss_list = []
    val_coarse_loss_list = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    y_mean_dev = y_mean.to(device)
    y_std_dev = y_std.to(device)

    # -----------------------------------------------------
    # train loop
    # -----------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0.0
        train_hr_loss = 0.0
        train_coarse_loss = 0.0

        for Xn, yn, y_coarse_n in train_dataloader:
            Xn = Xn.to(device)
            yn = yn.to(device)
            y_coarse_n = y_coarse_n.to(device)

            # normalized HR prediction
            y_pred_n = model(Xn)

            # full-image HR loss in normalized space
            loss_hr = loss_fn(y_pred_n, yn)

            # full-image coarse RCM consistency loss in normalized space
            y_pred_coarse_n = F.interpolate(
                y_pred_n,
                size=y_coarse_n.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            loss_coarse = loss_fn(y_pred_coarse_n, y_coarse_n)

            loss = loss_hr + ALPHA_COARSE * loss_coarse

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_hr_loss += loss_hr.item()
            train_coarse_loss += loss_coarse.item()

        train_loss /= len(train_dataloader)
        train_hr_loss /= len(train_dataloader)
        train_coarse_loss /= len(train_dataloader)

        train_loss_list.append(train_loss)
        train_hr_loss_list.append(train_hr_loss)
        train_coarse_loss_list.append(train_coarse_loss)

        # -------------------------------------------------
        # validation in physical space
        # No ocean masking and no enforced zero.
        # Use full-image physical MSE.
        # -------------------------------------------------
        model.eval()

        val_sse = 0.0
        val_numel = 0

        val_coarse_sse = 0.0
        val_coarse_numel = 0

        with torch.no_grad():
            for Xn, y_raw, y_coarse_raw in val_dataloader:
                Xn = Xn.to(device)
                y_raw = y_raw.to(device)
                y_coarse_raw = y_coarse_raw.to(device)

                # normalized prediction
                y_pred_n = model(Xn)

                # physical prediction
                y_pred = y_pred_n * y_std_dev + y_mean_dev

                # full-image physical validation MSE
                sse, numel = full_image_mse_from_sums(y_pred, y_raw)
                val_sse += sse
                val_numel += numel

                # full-image physical coarse validation MSE
                y_pred_coarse = F.interpolate(
                    y_pred,
                    size=y_coarse_raw.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                sse_c, numel_c = full_image_mse_from_sums(
                    y_pred_coarse,
                    y_coarse_raw,
                )

                val_coarse_sse += sse_c
                val_coarse_numel += numel_c

        val_loss = val_sse / val_numel
        val_coarse_loss = val_coarse_sse / val_coarse_numel

        val_loss_list.append(val_loss)
        val_coarse_loss_list.append(val_coarse_loss)

        scheduler.step(val_loss)

        # report to Optuna for pruning
        trial.report(val_loss, epoch)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_val_loss", best_val_loss)

        if trial.should_prune():
            trial.set_user_attr("status", "pruned")
            raise optuna.TrialPruned()

        # -------------------------------------------------
        # save best model for this trial
        # -------------------------------------------------
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val_loss", best_val_loss)
            trial.set_user_attr("checkpoint_path", trial_ckpt_path)

            torch.save(
                {
                    "trial_number": trial.number,
                    "epoch": best_epoch,
                    "model_name": model.__class__.__name__,
                    "model_choice": model_name,
                    "input_source": INPUT_SOURCE,
                    "input_file": input_file,
                    "input_mask_file": input_mask_file,
                    "input_elevation_file": input_elev_file,
                    "target_file": "high_res.pth",
                    "target_mask_file": "high_res_mask.pth",
                    "coarse_target_file": coarse_target_file,
                    "coarse_target_mask_file": coarse_target_mask_file,
                    "model_hparams": {
                        "input_channels": INPUT_CHANNELS,
                        "output_channels": 1,
                        "hr_img_size": HR_IMG_SIZE,
                        "lr_img_size": LR_IMG_SIZE,
                        "patch_size": patch_size,
                        "embed_dim": embed_dim,
                        "depth": depth,
                        "num_heads": num_heads,
                        "mlp_ratio": mlp_ratio,
                        "dropout": dropout,
                        "decoder_features": decoder_features,
                        "factor": factor,
                    },
                    "training_hparams": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "alpha_coarse": ALPHA_COARSE,
                        "epochs": EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "min_delta": MIN_DELTA,
                        "loss": (
                            "train_full_image_hr_mse_normalized "
                            "+ alpha_coarse * full_image_coarse_rcm_consistency_normalized"
                        ),
                        "validation_loss": "val_full_image_physical_mse",
                        "validation_coarse_loss": (
                            "downsampled_prediction_vs_coarse_RCM_full_image_physical_mse"
                        ),
                        "input_channels_description": input_channels_description,
                        "ocean_constraint": "none; prediction is not multiplied by HR mask",
                        "normalization": "full_image_training_stats; no land-only normalization",
                        "factor": factor,
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
                    "train_loss": train_loss,
                    "train_hr_loss": train_hr_loss,
                    "train_coarse_loss": train_coarse_loss,
                    "val_loss": val_loss,
                    "val_coarse_loss": val_coarse_loss,
                    "train_loss_list": train_loss_list,
                    "train_hr_loss_list": train_hr_loss_list,
                    "train_coarse_loss_list": train_coarse_loss_list,
                    "val_loss_list": val_loss_list,
                    "val_coarse_loss_list": val_coarse_loss_list,
                    "X_mean": X_mean,
                    "X_std": X_std,
                    "X_max": X_max,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "y_max": y_max,
                },
                trial_ckpt_path,
            )

        else:
            epochs_no_improve += 1

        # print(
        #     f"Trial {trial.number:03d} | "
        #     f"Epoch {epoch:03d} | "
        #     f"Model: {model_name} | "
        #     f"Input: {INPUT_SOURCE} | "
        #     f"Patch: {patch_size} | "
        #     f"Embed: {embed_dim} | "
        #     f"Depth: {depth} | "
        #     f"Heads: {num_heads} | "
        #     f"Train Total: {train_loss:.6f} | "
        #     f"Train HR Full: {train_hr_loss:.6f} | "
        #     f"Train Coarse Full: {train_coarse_loss:.6f} | "
        #     f"Val Full Physical: {val_loss:.6f} | "
        #     f"Val Coarse Full Physical: {val_coarse_loss:.6f} | "
        #     f"Best Val: {best_val_loss:.6f} | "
        #     f"Best Epoch: {best_epoch:03d} | "
        #     f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
        #     f"lr: {optimizer.param_groups[0]['lr']:.6e}"
        # )

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered for trial {trial.number:03d} "
                f"at epoch {epoch:03d} | best epoch: {best_epoch:03d}"
            )
            break

    trial.set_user_attr("status", "completed")
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("checkpoint_path", trial_ckpt_path)

    return best_val_loss


# =========================================================
# create and run study
# =========================================================
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=5,
    ),
)

study.optimize(objective, n_trials=N_TRIALS)


# =========================================================
# save best params summary
# =========================================================
best_trial = study.best_trial

best_summary = {
    "best_trial_number": best_trial.number,
    "best_value": best_trial.value,
    "best_params": best_trial.params,
    "best_epoch": best_trial.user_attrs.get("best_epoch", None),
    "best_checkpoint_path": best_trial.user_attrs.get(
        "checkpoint_path",
        os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
    ),
    "input_source": INPUT_SOURCE,
    "input_file": input_file,
    "input_mask_file": input_mask_file,
    "input_elevation_file": input_elev_file,
    "target_file": "high_res.pth",
    "target_mask_file": "high_res_mask.pth",
    "coarse_target_file": coarse_target_file,
    "coarse_target_mask_file": coarse_target_mask_file,
    "alpha_coarse": ALPHA_COARSE,
    "input_channels": INPUT_CHANNELS,
    "input_channels_description": input_channels_description,
    "loss_setup": loss_setup_tag,
    "loss": (
        "train_full_image_hr_mse_normalized "
        "+ alpha_coarse * full_image_coarse_rcm_consistency_normalized"
    ),
    "validation_loss": "val_full_image_physical_mse",
    "ocean_constraint": "none",
    "normalization": "full_image_training_stats",
    "model_name": model_name,
    "rcm_var": rcm_var,
    "gcm_name": gcm_name,
    "rcm_name": rcm_name,
    "grid": grid,
    "factor": factor,
    "lr_img_size": LR_IMG_SIZE,
    "hr_img_size": HR_IMG_SIZE,
}

with open(study_summary_path, "w") as f:
    json.dump(best_summary, f, indent=4)


# =========================================================
# save all trial summaries
# =========================================================
all_trials_summary = []

for t in study.trials:
    trial_info = {
        "trial_number": t.number,
        "state": str(t.state),
        "value": t.value if t.value is not None else None,
        "params": t.params,
        "best_epoch": t.user_attrs.get("best_epoch", None),
        "best_val_loss": t.user_attrs.get("best_val_loss", None),
        "checkpoint_path": t.user_attrs.get(
            "checkpoint_path",
            os.path.join(save_root, f"trial_{t.number:03d}.pth"),
        ),
        "input_source": INPUT_SOURCE,
        "input_file": input_file,
        "input_mask_file": input_mask_file,
        "input_elevation_file": input_elev_file,
        "coarse_target_file": coarse_target_file,
        "coarse_target_mask_file": coarse_target_mask_file,
        "alpha_coarse": ALPHA_COARSE,
        "input_channels": INPUT_CHANNELS,
        "loss_setup": loss_setup_tag,
        "model_name": model_name,
        "rcm_var": rcm_var,
        "gcm_name": gcm_name,
        "rcm_name": rcm_name,
        "grid": grid,
        "factor": factor,
        "lr_img_size": LR_IMG_SIZE,
        "hr_img_size": HR_IMG_SIZE,
    }

    all_trials_summary.append(trial_info)

with open(all_trials_summary_path, "w") as f:
    json.dump(all_trials_summary, f, indent=4)


# =========================================================
# final logs
# =========================================================
print("\nOptuna study finished.")
print(f"Model name: {model_name}")
print(f"Input source: {INPUT_SOURCE}")
print(f"Input file: {input_file}")
print(f"Input mask file: {input_mask_file}")
print(f"Input elevation file: {input_elev_file}")
print(f"Coarse target file: {coarse_target_file}")
print(f"Coarse target mask file: {coarse_target_mask_file}")
print(f"Alpha coarse: {ALPHA_COARSE}")
print(f"Input channels: {INPUT_CHANNELS}")
print(f"Loss setup: {loss_setup_tag}")
print(f"LR image size: {LR_IMG_SIZE}")
print(f"HR image size: {HR_IMG_SIZE}")
print(f"Best trial: {best_trial.number}")
print(f"Best value: {best_trial.value:.6f}")
print(f"Best epoch: {best_trial.user_attrs.get('best_epoch', None)}")
print("Best params:", best_trial.params)

print(
    "Best checkpoint:",
    best_trial.user_attrs.get(
        "checkpoint_path",
        os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
    ),
)

print("Best summary saved to:", study_summary_path)
print("All trial summaries saved to:", all_trials_summary_path)
print("All per-trial checkpoints saved under:", save_root)