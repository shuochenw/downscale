# =========================================================
# imports
# =========================================================
import os
import copy
import json
import argparse
import random

import torch
import optuna
import numpy as np

from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import UNet, ESPCNx4, SRResNet, SRResNet_HR_Aux, ClimateRCAN


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
        "Input channels for original SRResNet. Default 3: "
        "LR climate input, LR land-sea mask, LR elevation. "
        "For SRResNet_HR_Aux, this is automatically set to 1."
    ),
)

parser.add_argument(
    "--hr_aux_channels",
    type=int,
    default=2,
    help="HR auxiliary channels for SRResNet_HR_Aux. Default 2: HR mask + HR elevation.",
)

parser.add_argument(
    "--hr_elevation_file",
    type=str,
    default="high_res_elevation.pth",
    help="High-resolution elevation tensor used by SRResNet_HR_Aux.",
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
    "--data_root",
    type=str,
    default="/projects/sds-lab/Shuochen/downscaling/CORDEX",
    help="Root folder containing preprocessed CORDEX experiment folders.",
)

parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Optional checkpoint/output directory. Defaults inside the experiment folder.",
)

parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="DataLoader worker processes.",
)

parser.add_argument(
    "--pin_memory",
    action="store_true",
    help="Pin host memory for faster CPU-to-GPU batches.",
)

parser.add_argument(
    "--amp",
    action="store_true",
    help="Use CUDA automatic mixed precision.",
)

parser.add_argument(
    "--grad_clip_norm",
    type=float,
    default=1.0,
    help="Clip gradient norm. Use <= 0 to disable.",
)

parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Request deterministic CUDA kernels where PyTorch supports them.",
)

parser.add_argument(
    "--log_every",
    type=int,
    default=10,
    help="Print one training line every N epochs. Use <= 0 to disable epoch logs.",
)

parser.add_argument(
    "--study_name",
    type=str,
    default=None,
    help="Optional Optuna study name. Useful with --optuna_storage.",
)

parser.add_argument(
    "--optuna_storage",
    type=str,
    default=None,
    help="Optional Optuna storage URL, e.g. sqlite:///study.db, for resumable studies.",
)

parser.add_argument(
    "--resume_study",
    action="store_true",
    help="Resume an existing Optuna study with the same name/storage if present.",
)

parser.add_argument(
    "--model_name",
    type=str,
    choices=["SRResNet", "SRResNet_HR_Aux", "ClimateRCAN"],
    default="ClimateRCAN",
    help=(
        "SRResNet: original model with LR climate + LR mask + LR elevation as input. "
        "SRResNet_HR_Aux: LR climate input, then HR mask/elevation fused after upsampling. "
        "ClimateRCAN: stronger no-BatchNorm channel-attention SR model with a learned "
        "bilinear skip branch."
    ),
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
HR_AUX_CHANNELS = args.hr_aux_channels
HR_ELEVATION_FILE = args.hr_elevation_file
ALPHA_COARSE = args.alpha_coarse

# SRResNet and ClimateRCAN use LR climate + LR mask + LR elevation.
# HR-aux SRResNet uses only LR climate as the main model input.
if args.model_name == "SRResNet_HR_Aux":
    INPUT_CHANNELS = 1
else:
    INPUT_CHANNELS = args.input_channels

N_TRIALS = args.n_trials
EPOCHS = args.epochs
EARLY_STOPPING_PATIENCE = args.early_stopping_patience
MIN_DELTA = args.min_delta
RANDOM_SEED = args.random_seed
BATCH_SIZE = args.batch_size
DATA_ROOT = args.data_root
SAVE_DIR = args.save_dir
NUM_WORKERS = args.num_workers
PIN_MEMORY = args.pin_memory
USE_AMP = args.amp
GRAD_CLIP_NORM = args.grad_clip_norm
DETERMINISTIC = args.deterministic
LOG_EVERY = args.log_every
STUDY_NAME = args.study_name
OPTUNA_STORAGE = args.optuna_storage
RESUME_STUDY = args.resume_study

model_name = args.model_name

print("Running with:")
print("rcm_var =", rcm_var)
print("gcm_name =", gcm_name)
print("rcm_name =", rcm_name)
print("grid =", grid)
print("factor =", factor)
print("input_source =", INPUT_SOURCE)
print("input_channels =", INPUT_CHANNELS)
print("hr_aux_channels =", HR_AUX_CHANNELS)
print("hr_elevation_file =", HR_ELEVATION_FILE)
print("alpha_coarse =", ALPHA_COARSE)
print("n_trials =", N_TRIALS)
print("epochs =", EPOCHS)
print("early_stopping_patience =", EARLY_STOPPING_PATIENCE)
print("batch_size =", BATCH_SIZE)
print("data_root =", DATA_ROOT)
print("save_dir =", SAVE_DIR)
print("num_workers =", NUM_WORKERS)
print("pin_memory =", PIN_MEMORY)
print("amp =", USE_AMP)
print("grad_clip_norm =", GRAD_CLIP_NORM)
print("deterministic =", DETERMINISTIC)
print("log_every =", LOG_EVERY)
print("study_name =", STUDY_NAME)
print("optuna_storage =", OPTUNA_STORAGE)
print("resume_study =", RESUME_STUDY)
print("model_name =", model_name)


# =========================================================
# data folder
# This should match the preprocessing output folder
# =========================================================
exp_folder_name = (
    os.path.join(
        DATA_ROOT,
        f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.GCM_to_HR_RCM",
    )
)


# =========================================================
# device
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = not DETERMINISTIC

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True, warn_only=True)


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
# Added loss_setup_tag so this full-image-loss experiment does not
# overwrite the older masked-loss experiment.
# =========================================================
if model_name == "SRResNet_HR_Aux":
    input_setup_tag = f"{input_tag}_with_HR_mask_elevation_aux"
else:
    input_setup_tag = f"{input_tag}_with_mask_elevation"

loss_setup_tag = "full_image_loss_no_ocean_constraint"

save_root = os.path.join(
    exp_folder_name,
    "trained_models",
    model_name,
    input_setup_tag,
    loss_setup_tag,
)

if SAVE_DIR is not None:
    save_root = SAVE_DIR

os.makedirs(save_root, exist_ok=True)

study_summary_path = os.path.join(save_root, "optuna_best_params.json")
all_trials_summary_path = os.path.join(save_root, "optuna_all_trials_summary.json")

print("Input source:", INPUT_SOURCE)
print("Input file:", input_file)
print("Input mask file:", input_mask_file)
print("Input elevation file:", input_elev_file)
print("HR elevation file:", HR_ELEVATION_FILE)
print("Coarse consistency target file:", coarse_target_file)
print("Coarse consistency mask file:", coarse_target_mask_file)
print("Input channels:", INPUT_CHANNELS)
print("Loss setup:", loss_setup_tag)
print("All trial checkpoints will be saved to:", save_root)


# =========================================================
# helper function
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


def assert_finite_tensor(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def make_worker_init_fn(seed):
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


# =========================================================
# load tensors
# =========================================================
X = torch.load(os.path.join(exp_folder_name, input_file))
X_mask = torch.load(os.path.join(exp_folder_name, input_mask_file))

y = torch.load(os.path.join(exp_folder_name, "high_res.pth"))
mask_hr = torch.load(os.path.join(exp_folder_name, "high_res_mask.pth"))

# Coarse RCM target for coarse consistency loss
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

# LR elevation is needed by original SRResNet.
# HR elevation is needed by SRResNet_HR_Aux.
X_elev = None
X_hr_elev = None

if model_name in ["SRResNet", "ClimateRCAN"]:
    X_elev = torch.load(os.path.join(exp_folder_name, input_elev_file))

if model_name == "SRResNet_HR_Aux":
    hr_elev_path = os.path.join(exp_folder_name, HR_ELEVATION_FILE)
    if not os.path.exists(hr_elev_path):
        raise FileNotFoundError(
            f"SRResNet_HR_Aux requires HR elevation, but this file was not found: {hr_elev_path}"
        )
    X_hr_elev = torch.load(hr_elev_path)

print("Loaded shapes:")
print("X          :", X.shape)
print("X_mask     :", X_mask.shape)

if X_elev is not None:
    print("X_elev     :", X_elev.shape)

if X_hr_elev is not None:
    print("X_hr_elev  :", X_hr_elev.shape)

print("y          :", y.shape)
print("mask_hr    :", mask_hr.shape)
print("y_coarse   :", y_coarse.shape)
print("mask_coarse:", mask_coarse.shape)


# =========================================================
# shape checks and static/full expansion
# =========================================================
X_mask = match_sample_dim(X_mask, X.shape[0], "X_mask")
mask_hr = match_sample_dim(mask_hr, y.shape[0], "mask_hr")
y_coarse = match_sample_dim(y_coarse, X.shape[0], "y_coarse")
mask_coarse = match_sample_dim(mask_coarse, X.shape[0], "mask_coarse")

if X_elev is not None:
    X_elev = match_sample_dim(X_elev, X.shape[0], "X_elev")

if X_hr_elev is not None:
    X_hr_elev = match_sample_dim(X_hr_elev, y.shape[0], "X_hr_elev")

if y.shape != mask_hr.shape:
    raise ValueError(
        f"Shape mismatch: y shape {y.shape} does not match HR mask shape {mask_hr.shape}"
    )

if X.shape != X_mask.shape:
    raise ValueError(
        f"Shape mismatch: X shape {X.shape} does not match X_mask shape {X_mask.shape}"
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

if X_elev is not None and X.shape != X_elev.shape:
    raise ValueError(
        f"Shape mismatch: X shape {X.shape} does not match X_elev shape {X_elev.shape}"
    )

if X_hr_elev is not None and y.shape != X_hr_elev.shape:
    raise ValueError(
        f"Shape mismatch: y shape {y.shape} does not match X_hr_elev shape {X_hr_elev.shape}"
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

# masks are still used as input channels / HR auxiliary channels,
# but they are NOT used in the loss.
mask_hr = mask_hr.float()
mask_hr = (mask_hr > 0.5).float()

X_mask = X_mask.float()
X_mask = (X_mask > 0.5).float()

mask_coarse = mask_coarse.float()
mask_coarse = (mask_coarse > 0.5).float()

# elevation should already be normalized in preprocessing
if X_elev is not None:
    X_elev = X_elev.float()
    X_elev = torch.nan_to_num(X_elev, nan=0.0, posinf=0.0, neginf=0.0)

if X_hr_elev is not None:
    X_hr_elev = X_hr_elev.float()
    X_hr_elev = torch.nan_to_num(X_hr_elev, nan=0.0, posinf=0.0, neginf=0.0)

# climate tensors
X = X.float()
y = y.float()
y_coarse = y_coarse.float()

for tensor_name, tensor in [
    ("X", X),
    ("X_mask", X_mask),
    ("y", y),
    ("mask_hr", mask_hr),
    ("y_coarse", y_coarse),
    ("mask_coarse", mask_coarse),
]:
    assert_finite_tensor(tensor, tensor_name)

if X_elev is not None:
    assert_finite_tensor(X_elev, "X_elev")

if X_hr_elev is not None:
    assert_finite_tensor(X_hr_elev, "X_hr_elev")

print("After checks:")
print("X          :", X.shape)
print("X_mask     :", X_mask.shape)

if X_elev is not None:
    print("X_elev     :", X_elev.shape)

if X_hr_elev is not None:
    print("X_hr_elev  :", X_hr_elev.shape)

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

y_train = y[:train_end_idx]
mask_train = mask_hr[:train_end_idx]

y_coarse_train = y_coarse[:train_end_idx]
mask_coarse_train = mask_coarse[:train_end_idx]

X_val = X[val_start_idx:]
X_mask_val = X_mask[val_start_idx:]

y_val = y[val_start_idx:]
mask_val = mask_hr[val_start_idx:]

y_coarse_val = y_coarse[val_start_idx:]
mask_coarse_val = mask_coarse[val_start_idx:]

if model_name in ["SRResNet", "ClimateRCAN"]:
    X_elev_train = X_elev[:train_end_idx]
    X_elev_val = X_elev[val_start_idx:]
else:
    X_elev_train = None
    X_elev_val = None

if model_name == "SRResNet_HR_Aux":
    X_hr_elev_train = X_hr_elev[:train_end_idx]
    X_hr_elev_val = X_hr_elev[val_start_idx:]
else:
    X_hr_elev_train = None
    X_hr_elev_val = None

print("Train/Val shapes before model input construction:")
print("X_train          :", X_train.shape)
print("X_mask_train     :", X_mask_train.shape)

if X_elev_train is not None:
    print("X_elev_train     :", X_elev_train.shape)

if X_hr_elev_train is not None:
    print("X_hr_elev_train  :", X_hr_elev_train.shape)

print("y_train          :", y_train.shape)
print("mask_train       :", mask_train.shape)
print("y_coarse_train   :", y_coarse_train.shape)
print("mask_coarse_train:", mask_coarse_train.shape)

print("X_val            :", X_val.shape)
print("X_mask_val       :", X_mask_val.shape)

if X_elev_val is not None:
    print("X_elev_val       :", X_elev_val.shape)

if X_hr_elev_val is not None:
    print("X_hr_elev_val    :", X_hr_elev_val.shape)

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

# avoid divide-by-zero
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

# target normalization
y_train_n = (y_train - y_mean) / y_std
y_val_n = (y_val - y_mean) / y_std

# coarse RCM target normalized using same HR target statistics.
# Do NOT mask coarse target.
y_coarse_train_n = (y_coarse_train - y_mean) / y_std
y_coarse_val_n = (y_coarse_val - y_mean) / y_std

if model_name in ["SRResNet", "ClimateRCAN"]:
    # Original model input:
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

    X_train_hr_aux = None
    X_val_hr_aux = None

    input_channels_description = (
        f"{model_name} input channels: "
        "channel0=normalized_LR_climate_input; "
        "channel1=LR_land_sea_mask; "
        "channel2=LR_elevation"
    )

elif model_name == "SRResNet_HR_Aux":
    # HR-aux model input:
    # main input is only the normalized LR climate variable.
    # HR auxiliary input is concatenated after upsampling inside the model.
    X_train_in = X_train_clim_n.contiguous()
    X_val_in = X_val_clim_n.contiguous()

    X_train_hr_aux = torch.cat(
        [mask_train, X_hr_elev_train],
        dim=1,
    ).contiguous()

    X_val_hr_aux = torch.cat(
        [mask_val, X_hr_elev_val],
        dim=1,
    ).contiguous()

    input_channels_description = (
        "SRResNet_HR_Aux main input: channel0=normalized_LR_climate_input; "
        "HR auxiliary channels after upsampling: "
        "channel0=HR_land_sea_mask; channel1=HR_elevation"
    )

else:
    raise ValueError(f"Unsupported model_name: {model_name}")

print("Train/Val shapes after model input construction:")
print("X_train_in:", X_train_in.shape)
print("X_val_in  :", X_val_in.shape)

if X_train_hr_aux is not None:
    print("X_train_hr_aux:", X_train_hr_aux.shape)

if X_val_hr_aux is not None:
    print("X_val_hr_aux  :", X_val_hr_aux.shape)

if X_train_in.shape[1] != INPUT_CHANNELS:
    raise ValueError(
        f"Expected {INPUT_CHANNELS} input channels, got {X_train_in.shape[1]}"
    )

if X_val_in.shape[1] != INPUT_CHANNELS:
    raise ValueError(
        f"Expected {INPUT_CHANNELS} input channels, got {X_val_in.shape[1]}"
    )

if model_name == "SRResNet_HR_Aux":
    if X_train_hr_aux.shape[1] != HR_AUX_CHANNELS:
        raise ValueError(
            f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
            f"got {X_train_hr_aux.shape[1]}"
        )

    if X_val_hr_aux.shape[1] != HR_AUX_CHANNELS:
        raise ValueError(
            f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
            f"got {X_val_hr_aux.shape[1]}"
        )


# =========================================================
# losses
# =========================================================
loss_fn = nn.MSELoss(reduction="mean")


# =========================================================
# datasets
# Masks are kept only as model inputs / auxiliary inputs.
# They are not used in the loss.
# =========================================================
if model_name in ["SRResNet", "ClimateRCAN"]:
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

elif model_name == "SRResNet_HR_Aux":
    training_set = TensorDataset(
        X_train_in,
        X_train_hr_aux,
        y_train_n,
        y_coarse_train_n,
    )

    validation_set = TensorDataset(
        X_val_in,
        X_val_hr_aux,
        y_val,
        y_coarse_val,
    )

else:
    raise ValueError(f"Unsupported model_name: {model_name}")


# =========================================================
# objective function
# =========================================================
def objective(trial):
    # -----------------------------------------------------
    # hyperparameters to search
    # -----------------------------------------------------
    batch_size = trial.suggest_categorical("batch_size", [BATCH_SIZE])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True)
    num_resblk = trial.suggest_int("num_resblk", 4, 16, step=2)
    num_features = trial.suggest_categorical("num_features", [32, 64, 96, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # -----------------------------------------------------
    # paths for this trial
    # -----------------------------------------------------
    trial_ckpt_path = os.path.join(save_root, f"trial_{trial.number:03d}.pth")

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------
    if model_name == "SRResNet":
        model = SRResNet(
            num_resblk=num_resblk,
            num_features=num_features,
            input_channels=INPUT_CHANNELS,
            output_channels=1,
            scale=factor,
        ).to(device)

    elif model_name == "ClimateRCAN":
        model = ClimateRCAN(
            num_resblk=num_resblk,
            num_features=num_features,
            input_channels=INPUT_CHANNELS,
            output_channels=1,
            scale=factor,
        ).to(device)

    elif model_name == "SRResNet_HR_Aux":
        model = SRResNet_HR_Aux(
            num_resblk=num_resblk,
            num_features=num_features,
            input_channels=INPUT_CHANNELS,
            output_channels=1,
            hr_aux_channels=HR_AUX_CHANNELS,
            scale=factor,
        ).to(device)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    # -----------------------------------------------------
    # dataloaders
    # -----------------------------------------------------
    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED + trial.number)
    dataloader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY and device.type == "cuda",
        "worker_init_fn": make_worker_init_fn(RANDOM_SEED + trial.number * 1000),
        "generator": generator,
    }

    if NUM_WORKERS > 0:
        dataloader_kwargs["persistent_workers"] = True

    train_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_dataloader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
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

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device.type == "cuda")

    def step_optimizer(loss):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

        scaler.step(optimizer)
        scaler.update()

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
        train_samples = 0

        if model_name in ["SRResNet", "ClimateRCAN"]:
            for Xn, yn, y_coarse_n in train_dataloader:
                Xn = Xn.to(device, non_blocking=True)
                yn = yn.to(device, non_blocking=True)
                y_coarse_n = y_coarse_n.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
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

                step_optimizer(loss)

                batch_samples = Xn.shape[0]
                train_samples += batch_samples
                train_loss += loss.item() * batch_samples
                train_hr_loss += loss_hr.item() * batch_samples
                train_coarse_loss += loss_coarse.item() * batch_samples

        elif model_name == "SRResNet_HR_Aux":
            for Xn, X_hr_aux, yn, y_coarse_n in train_dataloader:
                Xn = Xn.to(device, non_blocking=True)
                X_hr_aux = X_hr_aux.to(device, non_blocking=True)
                yn = yn.to(device, non_blocking=True)
                y_coarse_n = y_coarse_n.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                    # normalized HR prediction
                    y_pred_n = model(Xn, X_hr_aux)

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

                step_optimizer(loss)

                batch_samples = Xn.shape[0]
                train_samples += batch_samples
                train_loss += loss.item() * batch_samples
                train_hr_loss += loss_hr.item() * batch_samples
                train_coarse_loss += loss_coarse.item() * batch_samples

        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        train_loss /= train_samples
        train_hr_loss /= train_samples
        train_coarse_loss /= train_samples

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
            if model_name in ["SRResNet", "ClimateRCAN"]:
                for Xn, y_raw, y_coarse_raw in val_dataloader:
                    Xn = Xn.to(device, non_blocking=True)
                    y_raw = y_raw.to(device, non_blocking=True)
                    y_coarse_raw = y_coarse_raw.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                        # normalized prediction
                        y_pred_n = model(Xn)

                    # physical prediction
                    y_pred = y_pred_n.float() * y_std_dev + y_mean_dev

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

            elif model_name == "SRResNet_HR_Aux":
                for Xn, X_hr_aux, y_raw, y_coarse_raw in val_dataloader:
                    Xn = Xn.to(device, non_blocking=True)
                    X_hr_aux = X_hr_aux.to(device, non_blocking=True)
                    y_raw = y_raw.to(device, non_blocking=True)
                    y_coarse_raw = y_coarse_raw.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=USE_AMP and device.type == "cuda"):
                        # normalized prediction
                        y_pred_n = model(Xn, X_hr_aux)

                    # physical prediction
                    y_pred = y_pred_n.float() * y_std_dev + y_mean_dev

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

            else:
                raise ValueError(f"Unsupported model_name: {model_name}")

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
                    "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
                    "target_file": "high_res.pth",
                    "target_mask_file": "high_res_mask.pth",
                    "coarse_target_file": coarse_target_file,
                    "coarse_target_mask_file": coarse_target_mask_file,
                    "model_hparams": {
                        "num_resblk": num_resblk,
                        "num_features": num_features,
                        "input_channels": INPUT_CHANNELS,
                        "output_channels": 1,
                        "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
                        "scale": factor,
                    },
                    "training_hparams": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "alpha_coarse": ALPHA_COARSE,
                        "amp": USE_AMP,
                        "grad_clip_norm": GRAD_CLIP_NORM,
                        "num_workers": NUM_WORKERS,
                        "pin_memory": PIN_MEMORY,
                        "random_seed": RANDOM_SEED,
                        "deterministic": DETERMINISTIC,
                        "epochs": EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "min_delta": MIN_DELTA,
                        "loss": (
                            "train_full_image_hr_mse_normalized "
                            "+ alpha_coarse * full_image_coarse_rcm_consistency_normalized"
                        ),
                        "validation_loss": "val_full_image_physical_mse",
                        "validation_coarse_loss": "downsampled_prediction_vs_coarse_RCM_full_image_physical_mse",
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
                    "scaler_state": scaler.state_dict(),
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

        if LOG_EVERY > 0 and (epoch == 0 or (epoch + 1) % LOG_EVERY == 0):
            print(
                f"Trial {trial.number:03d} | "
                f"Epoch {epoch:03d} | "
                f"Model: {model_name} | "
                f"Input: {INPUT_SOURCE} | "
                f"Train Total: {train_loss:.6f} | "
                f"Train HR Full: {train_hr_loss:.6f} | "
                f"Train Coarse Full: {train_coarse_loss:.6f} | "
                f"Val Full Physical: {val_loss:.6f} | "
                f"Val Coarse Full Physical: {val_coarse_loss:.6f} | "
                f"Best Val: {best_val_loss:.6f} | "
                f"Best Epoch: {best_epoch:03d} | "
                f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
                f"lr: {optimizer.param_groups[0]['lr']:.6e}"
            )

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
    study_name=STUDY_NAME,
    storage=OPTUNA_STORAGE,
    load_if_exists=RESUME_STUDY,
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
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
    "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
    "target_file": "high_res.pth",
    "target_mask_file": "high_res_mask.pth",
    "coarse_target_file": coarse_target_file,
    "coarse_target_mask_file": coarse_target_mask_file,
    "alpha_coarse": ALPHA_COARSE,
    "input_channels": INPUT_CHANNELS,
    "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
    "data_root": DATA_ROOT,
    "save_root": save_root,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
    "amp": USE_AMP,
    "grad_clip_norm": GRAD_CLIP_NORM,
    "random_seed": RANDOM_SEED,
    "deterministic": DETERMINISTIC,
    "study_name": STUDY_NAME,
    "optuna_storage": OPTUNA_STORAGE,
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
        "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
        "coarse_target_file": coarse_target_file,
        "coarse_target_mask_file": coarse_target_mask_file,
        "alpha_coarse": ALPHA_COARSE,
        "input_channels": INPUT_CHANNELS,
        "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
        "amp": USE_AMP,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "random_seed": RANDOM_SEED,
        "loss_setup": loss_setup_tag,
        "model_name": model_name,
        "rcm_var": rcm_var,
        "gcm_name": gcm_name,
        "rcm_name": rcm_name,
        "grid": grid,
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
print(f"HR elevation file: {HR_ELEVATION_FILE if model_name == 'SRResNet_HR_Aux' else None}")
print(f"Coarse target file: {coarse_target_file}")
print(f"Coarse target mask file: {coarse_target_mask_file}")
print(f"Alpha coarse: {ALPHA_COARSE}")
print(f"Input channels: {INPUT_CHANNELS}")
print(f"HR aux channels: {HR_AUX_CHANNELS if model_name == 'SRResNet_HR_Aux' else None}")
print(f"Loss setup: {loss_setup_tag}")
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









# # =========================================================
# # imports
# # =========================================================
# import os
# import copy
# import json
# import argparse

# import torch
# import optuna
# import numpy as np

# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# from models import UNet, ESPCNx4, SRResNet, SRResNet_HR_Aux


# # =========================================================
# # argument parser
# # =========================================================
# parser = argparse.ArgumentParser()

# parser.add_argument(
#     "--rcm_var",
#     type=str,
#     choices=["tmean", "tmax", "tmin", "prec"],
#     required=True,
# )

# parser.add_argument(
#     "--gcm_name",
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--rcm_name",
#     type=str,
#     required=True,
# )

# parser.add_argument(
#     "--grid",
#     type=str,
#     default="NAM-44i",
# )

# parser.add_argument(
#     "--factor",
#     type=int,
#     default=4,
# )

# parser.add_argument(
#     "--input_source",
#     type=str,
#     choices=["gcm", "coarse"],
#     default="gcm",
#     help="'gcm' uses low_res.pth; 'coarse' uses coarse_4x.pth",
# )

# parser.add_argument(
#     "--input_channels",
#     type=int,
#     default=3,
#     help=(
#         "Input channels for original SRResNet. Default 3: "
#         "LR climate input, LR land-sea mask, LR elevation. "
#         "For SRResNet_HR_Aux, this is automatically set to 1."
#     ),
# )

# parser.add_argument(
#     "--hr_aux_channels",
#     type=int,
#     default=2,
#     help="HR auxiliary channels for SRResNet_HR_Aux. Default 2: HR mask + HR elevation.",
# )

# parser.add_argument(
#     "--hr_elevation_file",
#     type=str,
#     default="high_res_elevation.pth",
#     help="High-resolution elevation tensor used by SRResNet_HR_Aux.",
# )

# parser.add_argument(
#     "--alpha_coarse",
#     type=float,
#     default=0.5,
#     help="Weight for coarse RCM consistency loss.",
# )

# parser.add_argument(
#     "--n_trials",
#     type=int,
#     default=3000,
# )

# parser.add_argument(
#     "--epochs",
#     type=int,
#     default=300,
# )

# parser.add_argument(
#     "--early_stopping_patience",
#     type=int,
#     default=30,
# )

# parser.add_argument(
#     "--min_delta",
#     type=float,
#     default=0.0,
# )

# parser.add_argument(
#     "--random_seed",
#     type=int,
#     default=42,
# )

# parser.add_argument(
#     "--batch_size",
#     type=int,
#     default=256,
# )

# parser.add_argument(
#     "--model_name",
#     type=str,
#     choices=["SRResNet", "SRResNet_HR_Aux"],
#     default="SRResNet",
#     help=(
#         "SRResNet: original model with LR climate + LR mask + LR elevation as input. "
#         "SRResNet_HR_Aux: LR climate input, then HR mask/elevation fused after upsampling."
#     ),
# )

# args = parser.parse_args()


# # =========================================================
# # settings from args
# # =========================================================
# rcm_var = args.rcm_var
# gcm_name = args.gcm_name
# rcm_name = args.rcm_name
# grid = args.grid
# factor = args.factor

# INPUT_SOURCE = args.input_source
# HR_AUX_CHANNELS = args.hr_aux_channels
# HR_ELEVATION_FILE = args.hr_elevation_file
# ALPHA_COARSE = args.alpha_coarse

# # Original SRResNet uses LR climate + LR mask + LR elevation.
# # HR-aux SRResNet uses only LR climate as the main model input.
# if args.model_name == "SRResNet_HR_Aux":
#     INPUT_CHANNELS = 1
# else:
#     INPUT_CHANNELS = args.input_channels

# N_TRIALS = args.n_trials
# EPOCHS = args.epochs
# EARLY_STOPPING_PATIENCE = args.early_stopping_patience
# MIN_DELTA = args.min_delta
# RANDOM_SEED = args.random_seed
# BATCH_SIZE = args.batch_size

# model_name = args.model_name

# print("Running with:")
# print("rcm_var =", rcm_var)
# print("gcm_name =", gcm_name)
# print("rcm_name =", rcm_name)
# print("grid =", grid)
# print("factor =", factor)
# print("input_source =", INPUT_SOURCE)
# print("input_channels =", INPUT_CHANNELS)
# print("hr_aux_channels =", HR_AUX_CHANNELS)
# print("hr_elevation_file =", HR_ELEVATION_FILE)
# print("alpha_coarse =", ALPHA_COARSE)
# print("n_trials =", N_TRIALS)
# print("epochs =", EPOCHS)
# print("early_stopping_patience =", EARLY_STOPPING_PATIENCE)
# print("batch_size =", BATCH_SIZE)
# print("model_name =", model_name)


# # =========================================================
# # data folder
# # This should match the preprocessing output folder
# # =========================================================
# exp_folder_name = (
#     f"/projects/sds-lab/Shuochen/downscaling/CORDEX/"
#     f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.GCM_to_HR_RCM/"
# )


# # =========================================================
# # device
# # =========================================================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# torch.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(RANDOM_SEED)


# # =========================================================
# # choose input file
# # =========================================================
# if INPUT_SOURCE == "gcm":
#     input_file = "low_res.pth"
#     input_mask_file = "low_res_mask.pth"
#     input_elev_file = "low_res_elevation.pth"
#     input_tag = "GCM_low_res"

# elif INPUT_SOURCE == "coarse":
#     input_file = f"coarse_{factor}x.pth"
#     input_mask_file = f"coarse_{factor}x_mask.pth"
#     input_elev_file = f"coarse_{factor}x_elevation.pth"
#     input_tag = f"coarsened_RCM_{factor}x"

# else:
#     raise ValueError(
#         "INPUT_SOURCE must be either 'gcm' or 'coarse'. "
#         f"Got: {INPUT_SOURCE}"
#     )


# # =========================================================
# # coarse consistency target files
# # =========================================================
# coarse_target_file = f"coarse_{factor}x.pth"
# coarse_target_mask_file = f"coarse_{factor}x_mask.pth"


# # =========================================================
# # save paths
# # =========================================================
# if model_name == "SRResNet_HR_Aux":
#     input_setup_tag = f"{input_tag}_with_HR_mask_elevation_aux"
# else:
#     input_setup_tag = f"{input_tag}_with_mask_elevation"

# save_root = os.path.join(
#     exp_folder_name,
#     "trained_models",
#     model_name,
#     input_setup_tag,
# )

# os.makedirs(save_root, exist_ok=True)

# study_summary_path = os.path.join(save_root, "optuna_best_params.json")
# all_trials_summary_path = os.path.join(save_root, "optuna_all_trials_summary.json")

# print("Input source:", INPUT_SOURCE)
# print("Input file:", input_file)
# print("Input mask file:", input_mask_file)
# print("Input elevation file:", input_elev_file)
# print("HR elevation file:", HR_ELEVATION_FILE)
# print("Coarse consistency target file:", coarse_target_file)
# print("Coarse consistency mask file:", coarse_target_mask_file)
# print("Input channels:", INPUT_CHANNELS)
# print("All trial checkpoints will be saved to:", save_root)


# # =========================================================
# # helper function
# # =========================================================
# def match_sample_dim(tensor, n_samples, name):
#     """
#     Allows either:
#       [N, C, H, W] full-time tensor
#     or:
#       [1, C, H, W] static tensor.

#     Returns tensor with first dimension compatible with N.
#     """
#     if tensor.shape[0] == n_samples:
#         return tensor

#     if tensor.shape[0] == 1:
#         print(f"{name} has one sample; expanding to {n_samples} samples.")
#         return tensor.expand(n_samples, -1, -1, -1)

#     raise ValueError(
#         f"{name} has incompatible sample dimension: "
#         f"{tensor.shape[0]} vs expected {n_samples}"
#     )


# # =========================================================
# # load tensors
# # =========================================================
# X = torch.load(os.path.join(exp_folder_name, input_file))
# X_mask = torch.load(os.path.join(exp_folder_name, input_mask_file))

# y = torch.load(os.path.join(exp_folder_name, "high_res.pth"))
# mask_hr = torch.load(os.path.join(exp_folder_name, "high_res_mask.pth"))

# # Coarse RCM target for coarse consistency loss
# coarse_path = os.path.join(exp_folder_name, coarse_target_file)
# coarse_mask_path = os.path.join(exp_folder_name, coarse_target_mask_file)

# if not os.path.exists(coarse_path):
#     raise FileNotFoundError(
#         f"Coarse RCM consistency target not found: {coarse_path}"
#     )

# if not os.path.exists(coarse_mask_path):
#     raise FileNotFoundError(
#         f"Coarse RCM consistency mask not found: {coarse_mask_path}"
#     )

# y_coarse = torch.load(coarse_path)
# mask_coarse = torch.load(coarse_mask_path)

# # LR elevation is needed by original SRResNet.
# # HR elevation is needed by SRResNet_HR_Aux.
# X_elev = None
# X_hr_elev = None

# if model_name == "SRResNet":
#     X_elev = torch.load(os.path.join(exp_folder_name, input_elev_file))

# if model_name == "SRResNet_HR_Aux":
#     hr_elev_path = os.path.join(exp_folder_name, HR_ELEVATION_FILE)
#     if not os.path.exists(hr_elev_path):
#         raise FileNotFoundError(
#             f"SRResNet_HR_Aux requires HR elevation, but this file was not found: {hr_elev_path}"
#         )
#     X_hr_elev = torch.load(hr_elev_path)

# print("Loaded shapes:")
# print("X          :", X.shape)
# print("X_mask     :", X_mask.shape)
# if X_elev is not None:
#     print("X_elev     :", X_elev.shape)
# if X_hr_elev is not None:
#     print("X_hr_elev  :", X_hr_elev.shape)
# print("y          :", y.shape)
# print("mask_hr    :", mask_hr.shape)
# print("y_coarse   :", y_coarse.shape)
# print("mask_coarse:", mask_coarse.shape)


# # =========================================================
# # shape checks and static/full expansion
# # =========================================================
# X_mask = match_sample_dim(X_mask, X.shape[0], "X_mask")
# mask_hr = match_sample_dim(mask_hr, y.shape[0], "mask_hr")
# y_coarse = match_sample_dim(y_coarse, X.shape[0], "y_coarse")
# mask_coarse = match_sample_dim(mask_coarse, X.shape[0], "mask_coarse")

# if X_elev is not None:
#     X_elev = match_sample_dim(X_elev, X.shape[0], "X_elev")

# if X_hr_elev is not None:
#     X_hr_elev = match_sample_dim(X_hr_elev, y.shape[0], "X_hr_elev")

# if y.shape != mask_hr.shape:
#     raise ValueError(
#         f"Shape mismatch: y shape {y.shape} does not match HR mask shape {mask_hr.shape}"
#     )

# if X.shape != X_mask.shape:
#     raise ValueError(
#         f"Shape mismatch: X shape {X.shape} does not match X_mask shape {X_mask.shape}"
#     )

# if y_coarse.shape != mask_coarse.shape:
#     raise ValueError(
#         f"Shape mismatch: y_coarse shape {y_coarse.shape} "
#         f"does not match mask_coarse shape {mask_coarse.shape}"
#     )

# if y_coarse.shape != X.shape:
#     raise ValueError(
#         f"Shape mismatch: y_coarse shape {y_coarse.shape} "
#         f"must match LR input shape {X.shape}"
#     )

# if X_elev is not None and X.shape != X_elev.shape:
#     raise ValueError(
#         f"Shape mismatch: X shape {X.shape} does not match X_elev shape {X_elev.shape}"
#     )

# if X_hr_elev is not None and y.shape != X_hr_elev.shape:
#     raise ValueError(
#         f"Shape mismatch: y shape {y.shape} does not match X_hr_elev shape {X_hr_elev.shape}"
#     )

# if X.shape[0] != y.shape[0]:
#     raise ValueError(
#         f"Sample mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples"
#     )

# if y.shape[-2] != factor * X.shape[-2]:
#     raise ValueError(
#         f"Height mismatch: HR height {y.shape[-2]} != "
#         f"{factor} * LR height {X.shape[-2]}"
#     )

# if y.shape[-1] != factor * X.shape[-1]:
#     raise ValueError(
#         f"Width mismatch: HR width {y.shape[-1]} != "
#         f"{factor} * LR width {X.shape[-1]}"
#     )

# # ensure masks are float and binary
# mask_hr = mask_hr.float()
# mask_hr = (mask_hr > 0.5).float()

# X_mask = X_mask.float()
# X_mask = (X_mask > 0.5).float()

# mask_coarse = mask_coarse.float()
# mask_coarse = (mask_coarse > 0.5).float()

# # elevation should already be normalized in preprocessing
# if X_elev is not None:
#     X_elev = X_elev.float()
#     X_elev = torch.nan_to_num(X_elev, nan=0.0, posinf=0.0, neginf=0.0)

# if X_hr_elev is not None:
#     X_hr_elev = X_hr_elev.float()
#     X_hr_elev = torch.nan_to_num(X_hr_elev, nan=0.0, posinf=0.0, neginf=0.0)

# # climate tensors
# X = X.float()
# y = y.float()
# y_coarse = y_coarse.float()

# print("After checks:")
# print("X          :", X.shape)
# print("X_mask     :", X_mask.shape)
# if X_elev is not None:
#     print("X_elev     :", X_elev.shape)
# if X_hr_elev is not None:
#     print("X_hr_elev  :", X_hr_elev.shape)
# print("y          :", y.shape)
# print("mask_hr    :", mask_hr.shape)
# print("y_coarse   :", y_coarse.shape)
# print("mask_coarse:", mask_coarse.shape)


# # =========================================================
# # split train/val by year
# # train: historical period
# # val: future period
# # =========================================================
# if gcm_name == "CanESM2":
#     # CanESM2/CORDEX preprocessing currently gives 1950-2100,
#     # assuming no leap years in the aligned dataset.
#     start_year = 1950
#     train_end_year = 2005
#     val_start_year = 2081

#     train_end_idx = (train_end_year - start_year + 1) * 365
#     val_start_idx = (val_start_year - start_year) * 365

# elif gcm_name == "EC-EARTH":
#     def is_leap(year):
#         return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

#     def days_between_years(start_year, end_year_inclusive):
#         total = 0
#         for y in range(start_year, end_year_inclusive + 1):
#             total += 366 if is_leap(y) else 365
#         return total

#     start_year = 1951
#     train_end_year = 2005
#     val_start_year = 2081

#     train_end_idx = days_between_years(start_year, train_end_year)
#     val_start_idx = days_between_years(start_year, val_start_year - 1)

# else:
#     raise ValueError(f"Year split not defined for gcm_name={gcm_name}")

# print("train_end_idx =", train_end_idx)
# print("val_start_idx =", val_start_idx)
# print("Total samples  =", X.shape[0])


# # =========================================================
# # create train/val tensors
# # =========================================================
# X_train = X[:train_end_idx]
# X_mask_train = X_mask[:train_end_idx]

# y_train = y[:train_end_idx]
# mask_train = mask_hr[:train_end_idx]

# y_coarse_train = y_coarse[:train_end_idx]
# mask_coarse_train = mask_coarse[:train_end_idx]

# X_val = X[val_start_idx:]
# X_mask_val = X_mask[val_start_idx:]

# y_val = y[val_start_idx:]
# mask_val = mask_hr[val_start_idx:]

# y_coarse_val = y_coarse[val_start_idx:]
# mask_coarse_val = mask_coarse[val_start_idx:]

# if model_name == "SRResNet":
#     X_elev_train = X_elev[:train_end_idx]
#     X_elev_val = X_elev[val_start_idx:]
# else:
#     X_elev_train = None
#     X_elev_val = None

# if model_name == "SRResNet_HR_Aux":
#     X_hr_elev_train = X_hr_elev[:train_end_idx]
#     X_hr_elev_val = X_hr_elev[val_start_idx:]
# else:
#     X_hr_elev_train = None
#     X_hr_elev_val = None

# print("Train/Val shapes before model input construction:")
# print("X_train          :", X_train.shape)
# print("X_mask_train     :", X_mask_train.shape)
# if X_elev_train is not None:
#     print("X_elev_train     :", X_elev_train.shape)
# if X_hr_elev_train is not None:
#     print("X_hr_elev_train  :", X_hr_elev_train.shape)
# print("y_train          :", y_train.shape)
# print("mask_train       :", mask_train.shape)
# print("y_coarse_train   :", y_coarse_train.shape)
# print("mask_coarse_train:", mask_coarse_train.shape)

# print("X_val            :", X_val.shape)
# print("X_mask_val       :", X_mask_val.shape)
# if X_elev_val is not None:
#     print("X_elev_val       :", X_elev_val.shape)
# if X_hr_elev_val is not None:
#     print("X_hr_elev_val    :", X_hr_elev_val.shape)
# print("y_val            :", y_val.shape)
# print("mask_val         :", mask_val.shape)
# print("y_coarse_val     :", y_coarse_val.shape)
# print("mask_coarse_val  :", mask_coarse_val.shape)


# # =========================================================
# # normalization stats from training set only
# # Only normalize the climate variable channel.
# # Do NOT normalize land-sea masks.
# # Elevation is assumed already normalized by preprocessing.
# # =========================================================
# X_train_valid = X_train[X_mask_train > 0.5]

# if X_train_valid.numel() == 0:
#     raise ValueError("No valid points found in input mask.")

# X_mean = X_train_valid.mean()
# X_std = X_train_valid.std()
# X_max = X_train_valid.max()

# y_train_land = y_train[mask_train > 0.5]

# if y_train_land.numel() == 0:
#     raise ValueError("No land points found in HR training mask.")

# y_mean = y_train_land.mean()
# y_std = y_train_land.std()
# y_max = y_train_land.max()

# # avoid divide-by-zero
# if X_std.item() == 0:
#     X_std = torch.tensor(1.0, dtype=X_train.dtype)

# if y_std.item() == 0:
#     y_std = torch.tensor(1.0, dtype=y_train.dtype)

# print("Normalization stats:")
# print("X_mean:", X_mean.item())
# print("X_std :", X_std.item())
# print("X_max :", X_max.item())
# print("y_mean:", y_mean.item())
# print("y_std :", y_std.item())
# print("y_max :", y_max.item())


# # =========================================================
# # normalize climate variable and build model inputs
# # =========================================================
# X_train_clim_n = (X_train - X_mean) / X_std
# X_val_clim_n = (X_val - X_mean) / X_std

# # force invalid LR input pixels to zero after normalization
# X_train_clim_n = X_train_clim_n * X_mask_train
# X_val_clim_n = X_val_clim_n * X_mask_val

# # target normalization
# y_train_n = (y_train - y_mean) / y_std
# y_val_n = (y_val - y_mean) / y_std

# # coarse RCM target normalized using the same HR target statistics.
# # This matches the normalized model output before physical denormalization.
# y_coarse_train_n = (y_coarse_train - y_mean) / y_std
# y_coarse_val_n = (y_coarse_val - y_mean) / y_std

# y_coarse_train_n = y_coarse_train_n * mask_coarse_train
# y_coarse_val_n = y_coarse_val_n * mask_coarse_val

# if model_name == "SRResNet":
#     # Original model input:
#     # channel 0: normalized LR climate variable
#     # channel 1: LR land-sea mask
#     # channel 2: LR elevation
#     X_train_in = torch.cat(
#         [X_train_clim_n, X_mask_train, X_elev_train],
#         dim=1,
#     ).contiguous()

#     X_val_in = torch.cat(
#         [X_val_clim_n, X_mask_val, X_elev_val],
#         dim=1,
#     ).contiguous()

#     X_train_hr_aux = None
#     X_val_hr_aux = None

#     input_channels_description = (
#         "SRResNet input channels: "
#         "channel0=normalized_LR_climate_input; "
#         "channel1=LR_land_sea_mask; "
#         "channel2=LR_elevation"
#     )

# elif model_name == "SRResNet_HR_Aux":
#     # HR-aux model input:
#     # main input is only the normalized LR climate variable.
#     # HR auxiliary input is concatenated after upsampling inside the model.
#     X_train_in = X_train_clim_n.contiguous()
#     X_val_in = X_val_clim_n.contiguous()

#     X_train_hr_aux = torch.cat(
#         [mask_train, X_hr_elev_train],
#         dim=1,
#     ).contiguous()

#     X_val_hr_aux = torch.cat(
#         [mask_val, X_hr_elev_val],
#         dim=1,
#     ).contiguous()

#     input_channels_description = (
#         "SRResNet_HR_Aux main input: channel0=normalized_LR_climate_input; "
#         "HR auxiliary channels after upsampling: "
#         "channel0=HR_land_sea_mask; channel1=HR_elevation"
#     )

# else:
#     raise ValueError(f"Unsupported model_name: {model_name}")

# print("Train/Val shapes after model input construction:")
# print("X_train_in:", X_train_in.shape)
# print("X_val_in  :", X_val_in.shape)
# if X_train_hr_aux is not None:
#     print("X_train_hr_aux:", X_train_hr_aux.shape)
# if X_val_hr_aux is not None:
#     print("X_val_hr_aux  :", X_val_hr_aux.shape)

# if X_train_in.shape[1] != INPUT_CHANNELS:
#     raise ValueError(
#         f"Expected {INPUT_CHANNELS} input channels, got {X_train_in.shape[1]}"
#     )

# if X_val_in.shape[1] != INPUT_CHANNELS:
#     raise ValueError(
#         f"Expected {INPUT_CHANNELS} input channels, got {X_val_in.shape[1]}"
#     )

# if model_name == "SRResNet_HR_Aux":
#     if X_train_hr_aux.shape[1] != HR_AUX_CHANNELS:
#         raise ValueError(
#             f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
#             f"got {X_train_hr_aux.shape[1]}"
#         )
#     if X_val_hr_aux.shape[1] != HR_AUX_CHANNELS:
#         raise ValueError(
#             f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
#             f"got {X_val_hr_aux.shape[1]}"
#         )


# # =========================================================
# # losses
# # =========================================================
# def masked_mse(pred, target, mask, eps=1e-8):
#     """
#     pred, target, mask: [B, C, H, W]
#     mask = 1 over land, 0 over ocean.
#     Returns mean squared error over land pixels only.
#     """
#     se = (pred - target) ** 2
#     se = se * mask
#     return se.sum() / (mask.sum() + eps)


# loss_fn = nn.MSELoss()


# # =========================================================
# # datasets
# # =========================================================
# if model_name == "SRResNet":
#     training_set = TensorDataset(
#         X_train_in,
#         y_train_n,
#         y_train,
#         mask_train,
#         y_coarse_train_n,
#         mask_coarse_train,
#     )

#     validation_set = TensorDataset(
#         X_val_in,
#         y_val_n,
#         y_val,
#         mask_val,
#         y_coarse_val,
#         mask_coarse_val,
#     )

# elif model_name == "SRResNet_HR_Aux":
#     training_set = TensorDataset(
#         X_train_in,
#         X_train_hr_aux,
#         y_train_n,
#         y_train,
#         mask_train,
#         y_coarse_train_n,
#         mask_coarse_train,
#     )

#     validation_set = TensorDataset(
#         X_val_in,
#         X_val_hr_aux,
#         y_val_n,
#         y_val,
#         mask_val,
#         y_coarse_val,
#         mask_coarse_val,
#     )

# else:
#     raise ValueError(f"Unsupported model_name: {model_name}")


# # =========================================================
# # objective function
# # =========================================================
# def objective(trial):
#     # -----------------------------------------------------
#     # hyperparameters to search
#     # -----------------------------------------------------
#     batch_size = trial.suggest_categorical("batch_size", [BATCH_SIZE])
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True)
#     num_resblk = trial.suggest_int("num_resblk", 4, 16, step=2)
#     num_features = trial.suggest_categorical("num_features", [32, 64, 96, 128])
#     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

#     # -----------------------------------------------------
#     # paths for this trial
#     # -----------------------------------------------------
#     trial_ckpt_path = os.path.join(save_root, f"trial_{trial.number:03d}.pth")

#     # -----------------------------------------------------
#     # model
#     # -----------------------------------------------------
#     if model_name == "SRResNet":
#         model = SRResNet(
#             num_resblk=num_resblk,
#             num_features=num_features,
#             input_channels=INPUT_CHANNELS,
#             output_channels=1,
#             scale=factor,
#         ).to(device)

#     elif model_name == "SRResNet_HR_Aux":
#         model = SRResNet_HR_Aux(
#             num_resblk=num_resblk,
#             num_features=num_features,
#             input_channels=INPUT_CHANNELS,
#             output_channels=1,
#             hr_aux_channels=HR_AUX_CHANNELS,
#             scale=factor,
#         ).to(device)

#     else:
#         raise ValueError(f"Unsupported model_name: {model_name}")

#     # -----------------------------------------------------
#     # dataloaders
#     # -----------------------------------------------------
#     train_dataloader = DataLoader(
#         training_set,
#         batch_size=batch_size,
#         shuffle=True,
#     )

#     val_dataloader = DataLoader(
#         validation_set,
#         batch_size=batch_size,
#         shuffle=False,
#     )

#     # -----------------------------------------------------
#     # training setup
#     # -----------------------------------------------------
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay,
#     )

#     scheduler = ReduceLROnPlateau(
#         optimizer,
#         patience=15,
#         factor=0.5,
#     )

#     train_loss_list = []
#     train_hr_loss_list = []
#     train_coarse_loss_list = []

#     val_loss_list = []
#     val_coarse_loss_list = []

#     best_val_loss = float("inf")
#     best_epoch = -1
#     epochs_no_improve = 0

#     y_mean_dev = y_mean.to(device)
#     y_std_dev = y_std.to(device)

#     # -----------------------------------------------------
#     # train loop
#     # -----------------------------------------------------
#     for epoch in range(EPOCHS):
#         model.train()

#         train_loss = 0.0
#         train_hr_loss = 0.0
#         train_coarse_loss = 0.0

#         if model_name == "SRResNet":
#             for Xn, yn, y_raw, mask_raw, y_coarse_n, mask_coarse_raw in train_dataloader:
#                 Xn = Xn.to(device)
#                 yn = yn.to(device)
#                 mask_raw = mask_raw.to(device)

#                 y_coarse_n = y_coarse_n.to(device)
#                 mask_coarse_raw = mask_coarse_raw.to(device)

#                 # normalized HR prediction
#                 y_pred_n = model(Xn)

#                 # HR land-only loss
#                 loss_hr = masked_mse(y_pred_n, yn, mask_raw)

#                 # coarse RCM consistency loss
#                 y_pred_coarse_n = F.interpolate(
#                     y_pred_n,
#                     size=y_coarse_n.shape[-2:],
#                     mode="bilinear",
#                     align_corners=False,
#                 )

#                 loss_coarse = masked_mse(
#                     y_pred_coarse_n,
#                     y_coarse_n,
#                     mask_coarse_raw,
#                 )

#                 loss = loss_hr + ALPHA_COARSE * loss_coarse

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 train_loss += loss.item()
#                 train_hr_loss += loss_hr.item()
#                 train_coarse_loss += loss_coarse.item()

#         elif model_name == "SRResNet_HR_Aux":
#             for (
#                 Xn,
#                 X_hr_aux,
#                 yn,
#                 y_raw,
#                 mask_raw,
#                 y_coarse_n,
#                 mask_coarse_raw,
#             ) in train_dataloader:
#                 Xn = Xn.to(device)
#                 X_hr_aux = X_hr_aux.to(device)
#                 yn = yn.to(device)
#                 mask_raw = mask_raw.to(device)

#                 y_coarse_n = y_coarse_n.to(device)
#                 mask_coarse_raw = mask_coarse_raw.to(device)

#                 # normalized HR prediction
#                 y_pred_n = model(Xn, X_hr_aux)

#                 # HR land-only loss
#                 loss_hr = masked_mse(y_pred_n, yn, mask_raw)

#                 # coarse RCM consistency loss
#                 y_pred_coarse_n = F.interpolate(
#                     y_pred_n,
#                     size=y_coarse_n.shape[-2:],
#                     mode="bilinear",
#                     align_corners=False,
#                 )

#                 loss_coarse = masked_mse(
#                     y_pred_coarse_n,
#                     y_coarse_n,
#                     mask_coarse_raw,
#                 )

#                 loss = loss_hr + ALPHA_COARSE * loss_coarse

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 train_loss += loss.item()
#                 train_hr_loss += loss_hr.item()
#                 train_coarse_loss += loss_coarse.item()

#         else:
#             raise ValueError(f"Unsupported model_name: {model_name}")

#         train_loss /= len(train_dataloader)
#         train_hr_loss /= len(train_dataloader)
#         train_coarse_loss /= len(train_dataloader)

#         train_loss_list.append(train_loss)
#         train_hr_loss_list.append(train_hr_loss)
#         train_coarse_loss_list.append(train_coarse_loss)

#         # -------------------------------------------------
#         # validation in physical space
#         # prediction is hard-masked in physical space,
#         # so ocean is exactly zero.
#         # -------------------------------------------------
#         model.eval()
#         val_loss = 0.0
#         val_coarse_loss = 0.0

#         with torch.no_grad():
#             if model_name == "SRResNet":
#                 for (
#                     Xn,
#                     yn,
#                     y_raw,
#                     mask_raw,
#                     y_coarse_raw,
#                     mask_coarse_raw,
#                 ) in val_dataloader:
#                     Xn = Xn.to(device)
#                     y_raw = y_raw.to(device)
#                     mask_raw = mask_raw.to(device)

#                     y_coarse_raw = y_coarse_raw.to(device)
#                     mask_coarse_raw = mask_coarse_raw.to(device)

#                     # normalized prediction
#                     y_pred_n = model(Xn)

#                     # convert to physical space
#                     y_pred = y_pred_n * y_std_dev + y_mean_dev

#                     # enforce exact zero over ocean in physical space
#                     y_pred = y_pred * mask_raw

#                     # full-image physical MSE
#                     val_loss += loss_fn(y_pred, y_raw).item()

#                     # coarse validation loss in physical space
#                     y_pred_coarse = F.interpolate(
#                         y_pred,
#                         size=y_coarse_raw.shape[-2:],
#                         mode="bilinear",
#                         align_corners=False,
#                     )

#                     val_coarse_loss += masked_mse(
#                         y_pred_coarse,
#                         y_coarse_raw,
#                         mask_coarse_raw,
#                     ).item()

#             elif model_name == "SRResNet_HR_Aux":
#                 for (
#                     Xn,
#                     X_hr_aux,
#                     yn,
#                     y_raw,
#                     mask_raw,
#                     y_coarse_raw,
#                     mask_coarse_raw,
#                 ) in val_dataloader:
#                     Xn = Xn.to(device)
#                     X_hr_aux = X_hr_aux.to(device)
#                     y_raw = y_raw.to(device)
#                     mask_raw = mask_raw.to(device)

#                     y_coarse_raw = y_coarse_raw.to(device)
#                     mask_coarse_raw = mask_coarse_raw.to(device)

#                     # normalized prediction
#                     y_pred_n = model(Xn, X_hr_aux)

#                     # convert to physical space
#                     y_pred = y_pred_n * y_std_dev + y_mean_dev

#                     # enforce exact zero over ocean in physical space
#                     y_pred = y_pred * mask_raw

#                     # full-image physical MSE
#                     val_loss += loss_fn(y_pred, y_raw).item()

#                     # coarse validation loss in physical space
#                     y_pred_coarse = F.interpolate(
#                         y_pred,
#                         size=y_coarse_raw.shape[-2:],
#                         mode="bilinear",
#                         align_corners=False,
#                     )

#                     val_coarse_loss += masked_mse(
#                         y_pred_coarse,
#                         y_coarse_raw,
#                         mask_coarse_raw,
#                     ).item()

#             else:
#                 raise ValueError(f"Unsupported model_name: {model_name}")

#         val_loss /= len(val_dataloader)
#         val_coarse_loss /= len(val_dataloader)

#         val_loss_list.append(val_loss)
#         val_coarse_loss_list.append(val_coarse_loss)

#         scheduler.step(val_loss)

#         # report to Optuna for pruning
#         trial.report(val_loss, epoch)
#         trial.set_user_attr("best_epoch", best_epoch)
#         trial.set_user_attr("best_val_loss", best_val_loss)

#         if trial.should_prune():
#             trial.set_user_attr("status", "pruned")
#             raise optuna.TrialPruned()

#         # -------------------------------------------------
#         # save best model for this trial
#         # -------------------------------------------------
#         if val_loss < best_val_loss - MIN_DELTA:
#             best_val_loss = val_loss
#             best_epoch = epoch
#             epochs_no_improve = 0

#             trial.set_user_attr("best_epoch", best_epoch)
#             trial.set_user_attr("best_val_loss", best_val_loss)
#             trial.set_user_attr("checkpoint_path", trial_ckpt_path)

#             torch.save(
#                 {
#                     "trial_number": trial.number,
#                     "epoch": best_epoch,
#                     "model_name": model.__class__.__name__,
#                     "model_choice": model_name,
#                     "input_source": INPUT_SOURCE,
#                     "input_file": input_file,
#                     "input_mask_file": input_mask_file,
#                     "input_elevation_file": input_elev_file,
#                     "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
#                     "target_file": "high_res.pth",
#                     "target_mask_file": "high_res_mask.pth",
#                     "coarse_target_file": coarse_target_file,
#                     "coarse_target_mask_file": coarse_target_mask_file,
#                     "model_hparams": {
#                         "num_resblk": num_resblk,
#                         "num_features": num_features,
#                         "input_channels": INPUT_CHANNELS,
#                         "output_channels": 1,
#                         "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
#                         "scale": factor,
#                     },
#                     "training_hparams": {
#                         "batch_size": batch_size,
#                         "learning_rate": learning_rate,
#                         "weight_decay": weight_decay,
#                         "alpha_coarse": ALPHA_COARSE,
#                         "epochs": EPOCHS,
#                         "early_stopping_patience": EARLY_STOPPING_PATIENCE,
#                         "min_delta": MIN_DELTA,
#                         "loss": (
#                             "train_hr_masked_mse_land_only "
#                             "+ alpha_coarse * coarse_rcm_consistency"
#                         ),
#                         "validation_loss": "val_full_image_physical_mse",
#                         "validation_coarse_loss": "downsampled_prediction_vs_coarse_RCM_masked_mse",
#                         "input_channels_description": input_channels_description,
#                         "ocean_constraint": "prediction multiplied by HR mask in physical space",
#                         "factor": factor,
#                         "rcm_var": rcm_var,
#                         "gcm_name": gcm_name,
#                         "rcm_name": rcm_name,
#                         "grid": grid,
#                         "train_end_year": train_end_year,
#                         "val_start_year": val_start_year,
#                     },
#                     "model_state": copy.deepcopy(model.state_dict()),
#                     "optimizer_state": optimizer.state_dict(),
#                     "scheduler_state": scheduler.state_dict(),
#                     "best_val_loss": best_val_loss,
#                     "best_epoch": best_epoch,
#                     "train_loss": train_loss,
#                     "train_hr_loss": train_hr_loss,
#                     "train_coarse_loss": train_coarse_loss,
#                     "val_loss": val_loss,
#                     "val_coarse_loss": val_coarse_loss,
#                     "train_loss_list": train_loss_list,
#                     "train_hr_loss_list": train_hr_loss_list,
#                     "train_coarse_loss_list": train_coarse_loss_list,
#                     "val_loss_list": val_loss_list,
#                     "val_coarse_loss_list": val_coarse_loss_list,
#                     "X_mean": X_mean,
#                     "X_std": X_std,
#                     "X_max": X_max,
#                     "y_mean": y_mean,
#                     "y_std": y_std,
#                     "y_max": y_max,
#                 },
#                 trial_ckpt_path,
#             )

#         else:
#             epochs_no_improve += 1

#         # print(
#         #     f"Trial {trial.number:03d} | "
#         #     f"Epoch {epoch:03d} | "
#         #     f"Model: {model_name} | "
#         #     f"Input: {INPUT_SOURCE} | "
#         #     f"Channels: {INPUT_CHANNELS} | "
#         #     f"HR aux channels: {HR_AUX_CHANNELS if model_name == 'SRResNet_HR_Aux' else 0} | "
#         #     f"Train Total: {train_loss:.6f} | "
#         #     f"Train HR: {train_hr_loss:.6f} | "
#         #     f"Train Coarse: {train_coarse_loss:.6f} | "
#         #     f"Val(full image): {val_loss:.6f} | "
#         #     f"Val Coarse: {val_coarse_loss:.6f} | "
#         #     f"Best Val: {best_val_loss:.6f} | "
#         #     f"Best Epoch: {best_epoch:03d} | "
#         #     f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
#         #     f"lr: {optimizer.param_groups[0]['lr']:.6e}"
#         # )

#         if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
#             print(
#                 f"Early stopping triggered for trial {trial.number:03d} "
#                 f"at epoch {epoch:03d} | best epoch: {best_epoch:03d}"
#             )
#             break

#     trial.set_user_attr("status", "completed")
#     trial.set_user_attr("best_epoch", best_epoch)
#     trial.set_user_attr("best_val_loss", best_val_loss)
#     trial.set_user_attr("checkpoint_path", trial_ckpt_path)

#     return best_val_loss


# # =========================================================
# # create and run study
# # =========================================================
# study = optuna.create_study(
#     direction="minimize",
#     pruner=optuna.pruners.MedianPruner(
#         n_startup_trials=5,
#         n_warmup_steps=5,
#     ),
# )

# study.optimize(objective, n_trials=N_TRIALS)


# # =========================================================
# # save best params summary
# # =========================================================
# best_trial = study.best_trial

# best_summary = {
#     "best_trial_number": best_trial.number,
#     "best_value": best_trial.value,
#     "best_params": best_trial.params,
#     "best_epoch": best_trial.user_attrs.get("best_epoch", None),
#     "best_checkpoint_path": best_trial.user_attrs.get(
#         "checkpoint_path",
#         os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
#     ),
#     "input_source": INPUT_SOURCE,
#     "input_file": input_file,
#     "input_mask_file": input_mask_file,
#     "input_elevation_file": input_elev_file,
#     "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
#     "target_file": "high_res.pth",
#     "target_mask_file": "high_res_mask.pth",
#     "coarse_target_file": coarse_target_file,
#     "coarse_target_mask_file": coarse_target_mask_file,
#     "alpha_coarse": ALPHA_COARSE,
#     "input_channels": INPUT_CHANNELS,
#     "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
#     "input_channels_description": input_channels_description,
#     "model_name": model_name,
#     "rcm_var": rcm_var,
#     "gcm_name": gcm_name,
#     "rcm_name": rcm_name,
#     "grid": grid,
# }

# with open(study_summary_path, "w") as f:
#     json.dump(best_summary, f, indent=4)


# # =========================================================
# # save all trial summaries
# # =========================================================
# all_trials_summary = []

# for t in study.trials:
#     trial_info = {
#         "trial_number": t.number,
#         "state": str(t.state),
#         "value": t.value if t.value is not None else None,
#         "params": t.params,
#         "best_epoch": t.user_attrs.get("best_epoch", None),
#         "best_val_loss": t.user_attrs.get("best_val_loss", None),
#         "checkpoint_path": t.user_attrs.get(
#             "checkpoint_path",
#             os.path.join(save_root, f"trial_{t.number:03d}.pth"),
#         ),
#         "input_source": INPUT_SOURCE,
#         "input_file": input_file,
#         "input_mask_file": input_mask_file,
#         "input_elevation_file": input_elev_file,
#         "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
#         "coarse_target_file": coarse_target_file,
#         "coarse_target_mask_file": coarse_target_mask_file,
#         "alpha_coarse": ALPHA_COARSE,
#         "input_channels": INPUT_CHANNELS,
#         "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
#         "model_name": model_name,
#         "rcm_var": rcm_var,
#         "gcm_name": gcm_name,
#         "rcm_name": rcm_name,
#         "grid": grid,
#     }

#     all_trials_summary.append(trial_info)

# with open(all_trials_summary_path, "w") as f:
#     json.dump(all_trials_summary, f, indent=4)


# # =========================================================
# # final logs
# # =========================================================
# print("\nOptuna study finished.")
# print(f"Model name: {model_name}")
# print(f"Input source: {INPUT_SOURCE}")
# print(f"Input file: {input_file}")
# print(f"Input mask file: {input_mask_file}")
# print(f"Input elevation file: {input_elev_file}")
# print(f"HR elevation file: {HR_ELEVATION_FILE if model_name == 'SRResNet_HR_Aux' else None}")
# print(f"Coarse target file: {coarse_target_file}")
# print(f"Coarse target mask file: {coarse_target_mask_file}")
# print(f"Alpha coarse: {ALPHA_COARSE}")
# print(f"Input channels: {INPUT_CHANNELS}")
# print(f"HR aux channels: {HR_AUX_CHANNELS if model_name == 'SRResNet_HR_Aux' else None}")
# print(f"Best trial: {best_trial.number}")
# print(f"Best value: {best_trial.value:.6f}")
# print(f"Best epoch: {best_trial.user_attrs.get('best_epoch', None)}")
# print("Best params:", best_trial.params)

# print(
#     "Best checkpoint:",
#     best_trial.user_attrs.get(
#         "checkpoint_path",
#         os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
#     ),
# )

# print("Best summary saved to:", study_summary_path)
# print("All trial summaries saved to:", all_trials_summary_path)
# print("All per-trial checkpoints saved under:", save_root)








# # # =========================================================
# # # imports
# # # =========================================================
# # import os
# # import copy
# # import json
# # import argparse

# # import torch
# # import optuna
# # import numpy as np

# # from torch import nn
# # from torch.utils.data import DataLoader, TensorDataset
# # from torch.optim.lr_scheduler import ReduceLROnPlateau

# # from models import UNet, ESPCNx4, SRResNet, SRResNet_HR_Aux


# # # =========================================================
# # # argument parser
# # # =========================================================
# # parser = argparse.ArgumentParser()

# # parser.add_argument(
# #     "--rcm_var",
# #     type=str,
# #     choices=["tmean", "tmax", "tmin", "prec"],
# #     required=True,
# # )

# # parser.add_argument(
# #     "--gcm_name",
# #     type=str,
# #     required=True,
# # )

# # parser.add_argument(
# #     "--rcm_name",
# #     type=str,
# #     required=True,
# # )

# # parser.add_argument(
# #     "--grid",
# #     type=str,
# #     default="NAM-44i",
# # )

# # parser.add_argument(
# #     "--factor",
# #     type=int,
# #     default=4,
# # )

# # parser.add_argument(
# #     "--input_source",
# #     type=str,
# #     choices=["gcm", "coarse"],
# #     default="gcm",
# #     help="'gcm' uses low_res.pth; 'coarse' uses coarse_4x.pth",
# # )

# # parser.add_argument(
# #     "--input_channels",
# #     type=int,
# #     default=3,
# #     help=(
# #         "Input channels for original SRResNet. Default 3: "
# #         "LR climate input, LR land-sea mask, LR elevation. "
# #         "For SRResNet_HR_Aux, this is automatically set to 1."
# #     ),
# # )

# # parser.add_argument(
# #     "--hr_aux_channels",
# #     type=int,
# #     default=2,
# #     help="HR auxiliary channels for SRResNet_HR_Aux. Default 2: HR mask + HR elevation.",
# # )

# # parser.add_argument(
# #     "--hr_elevation_file",
# #     type=str,
# #     default="high_res_elevation.pth",
# #     help="High-resolution elevation tensor used by SRResNet_HR_Aux.",
# # )

# # parser.add_argument(
# #     "--n_trials",
# #     type=int,
# #     default=3000,
# # )

# # parser.add_argument(
# #     "--epochs",
# #     type=int,
# #     default=300,
# # )

# # parser.add_argument(
# #     "--early_stopping_patience",
# #     type=int,
# #     default=30,
# # )

# # parser.add_argument(
# #     "--min_delta",
# #     type=float,
# #     default=0.0,
# # )

# # parser.add_argument(
# #     "--random_seed",
# #     type=int,
# #     default=42,
# # )

# # parser.add_argument(
# #     "--batch_size",
# #     type=int,
# #     default=256,
# # )

# # parser.add_argument(
# #     "--model_name",
# #     type=str,
# #     choices=["SRResNet", "SRResNet_HR_Aux"],
# #     default="SRResNet",
# #     help=(
# #         "SRResNet: original model with LR climate + LR mask + LR elevation as input. "
# #         "SRResNet_HR_Aux: LR climate input, then HR mask/elevation fused after upsampling."
# #     ),
# # )

# # args = parser.parse_args()


# # # =========================================================
# # # settings from args
# # # =========================================================
# # rcm_var = args.rcm_var
# # gcm_name = args.gcm_name
# # rcm_name = args.rcm_name
# # grid = args.grid
# # factor = args.factor

# # INPUT_SOURCE = args.input_source
# # HR_AUX_CHANNELS = args.hr_aux_channels
# # HR_ELEVATION_FILE = args.hr_elevation_file

# # # Original SRResNet uses LR climate + LR mask + LR elevation.
# # # HR-aux SRResNet uses only LR climate as the main model input.
# # if args.model_name == "SRResNet_HR_Aux":
# #     INPUT_CHANNELS = 1
# # else:
# #     INPUT_CHANNELS = args.input_channels

# # N_TRIALS = args.n_trials
# # EPOCHS = args.epochs
# # EARLY_STOPPING_PATIENCE = args.early_stopping_patience
# # MIN_DELTA = args.min_delta
# # RANDOM_SEED = args.random_seed
# # BATCH_SIZE = args.batch_size

# # model_name = args.model_name

# # print("Running with:")
# # print("rcm_var =", rcm_var)
# # print("gcm_name =", gcm_name)
# # print("rcm_name =", rcm_name)
# # print("grid =", grid)
# # print("factor =", factor)
# # print("input_source =", INPUT_SOURCE)
# # print("input_channels =", INPUT_CHANNELS)
# # print("hr_aux_channels =", HR_AUX_CHANNELS)
# # print("hr_elevation_file =", HR_ELEVATION_FILE)
# # print("n_trials =", N_TRIALS)
# # print("epochs =", EPOCHS)
# # print("early_stopping_patience =", EARLY_STOPPING_PATIENCE)
# # print("batch_size =", BATCH_SIZE)
# # print("model_name =", model_name)


# # # =========================================================
# # # data folder
# # # This should match the preprocessing output folder
# # # =========================================================
# # exp_folder_name = (
# #     f"/projects/sds-lab/Shuochen/downscaling/CORDEX/"
# #     f"{rcm_var}.{gcm_name}.{rcm_name}.day.{grid}.GCM_to_HR_RCM/"
# # )


# # # =========================================================
# # # device
# # # =========================================================
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print("Using device:", device)

# # torch.manual_seed(RANDOM_SEED)
# # np.random.seed(RANDOM_SEED)

# # if torch.cuda.is_available():
# #     torch.cuda.manual_seed_all(RANDOM_SEED)


# # # =========================================================
# # # choose input file
# # # =========================================================
# # if INPUT_SOURCE == "gcm":
# #     input_file = "low_res.pth"
# #     input_mask_file = "low_res_mask.pth"
# #     input_elev_file = "low_res_elevation.pth"
# #     input_tag = "GCM_low_res"

# # elif INPUT_SOURCE == "coarse":
# #     input_file = f"coarse_{factor}x.pth"
# #     input_mask_file = f"coarse_{factor}x_mask.pth"
# #     input_elev_file = f"coarse_{factor}x_elevation.pth"
# #     input_tag = f"coarsened_RCM_{factor}x"

# # else:
# #     raise ValueError(
# #         "INPUT_SOURCE must be either 'gcm' or 'coarse'. "
# #         f"Got: {INPUT_SOURCE}"
# #     )


# # # =========================================================
# # # save paths
# # # =========================================================
# # if model_name == "SRResNet_HR_Aux":
# #     input_setup_tag = f"{input_tag}_with_HR_mask_elevation_aux"
# # else:
# #     input_setup_tag = f"{input_tag}_with_mask_elevation"

# # save_root = os.path.join(
# #     exp_folder_name,
# #     "trained_models",
# #     model_name,
# #     input_setup_tag,
# # )

# # os.makedirs(save_root, exist_ok=True)

# # study_summary_path = os.path.join(save_root, "optuna_best_params.json")
# # all_trials_summary_path = os.path.join(save_root, "optuna_all_trials_summary.json")

# # print("Input source:", INPUT_SOURCE)
# # print("Input file:", input_file)
# # print("Input mask file:", input_mask_file)
# # print("Input elevation file:", input_elev_file)
# # print("HR elevation file:", HR_ELEVATION_FILE)
# # print("Input channels:", INPUT_CHANNELS)
# # print("All trial checkpoints will be saved to:", save_root)


# # # =========================================================
# # # helper function
# # # =========================================================
# # def match_sample_dim(tensor, n_samples, name):
# #     """
# #     Allows either:
# #       [N, C, H, W] full-time tensor
# #     or:
# #       [1, C, H, W] static tensor.

# #     Returns tensor with first dimension compatible with N.
# #     """
# #     if tensor.shape[0] == n_samples:
# #         return tensor

# #     if tensor.shape[0] == 1:
# #         print(f"{name} has one sample; expanding to {n_samples} samples.")
# #         return tensor.expand(n_samples, -1, -1, -1)

# #     raise ValueError(
# #         f"{name} has incompatible sample dimension: "
# #         f"{tensor.shape[0]} vs expected {n_samples}"
# #     )


# # # =========================================================
# # # load tensors
# # # =========================================================
# # X = torch.load(os.path.join(exp_folder_name, input_file))
# # X_mask = torch.load(os.path.join(exp_folder_name, input_mask_file))

# # y = torch.load(os.path.join(exp_folder_name, "high_res.pth"))
# # mask_hr = torch.load(os.path.join(exp_folder_name, "high_res_mask.pth"))

# # # LR elevation is needed by original SRResNet.
# # # HR elevation is needed by SRResNet_HR_Aux.
# # X_elev = None
# # X_hr_elev = None

# # if model_name == "SRResNet":
# #     X_elev = torch.load(os.path.join(exp_folder_name, input_elev_file))

# # if model_name == "SRResNet_HR_Aux":
# #     hr_elev_path = os.path.join(exp_folder_name, HR_ELEVATION_FILE)
# #     if not os.path.exists(hr_elev_path):
# #         raise FileNotFoundError(
# #             f"SRResNet_HR_Aux requires HR elevation, but this file was not found: {hr_elev_path}"
# #         )
# #     X_hr_elev = torch.load(hr_elev_path)

# # print("Loaded shapes:")
# # print("X       :", X.shape)
# # print("X_mask  :", X_mask.shape)
# # if X_elev is not None:
# #     print("X_elev  :", X_elev.shape)
# # if X_hr_elev is not None:
# #     print("X_hr_elev:", X_hr_elev.shape)
# # print("y       :", y.shape)
# # print("mask_hr :", mask_hr.shape)

# # # Optional: also load coarsened RCM for later comparison/reference
# # coarse_path = os.path.join(exp_folder_name, f"coarse_{factor}x.pth")
# # coarse_mask_path = os.path.join(exp_folder_name, f"coarse_{factor}x_mask.pth")

# # if os.path.exists(coarse_path):
# #     X_coarse_reference = torch.load(coarse_path)
# #     print("X_coarse_reference:", X_coarse_reference.shape)
# # else:
# #     X_coarse_reference = None
# #     print("No coarse RCM reference file found.")

# # if os.path.exists(coarse_mask_path):
# #     X_coarse_mask_reference = torch.load(coarse_mask_path)
# #     print("X_coarse_mask_reference:", X_coarse_mask_reference.shape)
# # else:
# #     X_coarse_mask_reference = None


# # # =========================================================
# # # shape checks and static/full expansion
# # # =========================================================
# # X_mask = match_sample_dim(X_mask, X.shape[0], "X_mask")
# # mask_hr = match_sample_dim(mask_hr, y.shape[0], "mask_hr")

# # if X_elev is not None:
# #     X_elev = match_sample_dim(X_elev, X.shape[0], "X_elev")

# # if X_hr_elev is not None:
# #     X_hr_elev = match_sample_dim(X_hr_elev, y.shape[0], "X_hr_elev")

# # if y.shape != mask_hr.shape:
# #     raise ValueError(
# #         f"Shape mismatch: y shape {y.shape} does not match HR mask shape {mask_hr.shape}"
# #     )

# # if X.shape != X_mask.shape:
# #     raise ValueError(
# #         f"Shape mismatch: X shape {X.shape} does not match X_mask shape {X_mask.shape}"
# #     )

# # if X_elev is not None and X.shape != X_elev.shape:
# #     raise ValueError(
# #         f"Shape mismatch: X shape {X.shape} does not match X_elev shape {X_elev.shape}"
# #     )

# # if X_hr_elev is not None and y.shape != X_hr_elev.shape:
# #     raise ValueError(
# #         f"Shape mismatch: y shape {y.shape} does not match X_hr_elev shape {X_hr_elev.shape}"
# #     )

# # if X.shape[0] != y.shape[0]:
# #     raise ValueError(
# #         f"Sample mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples"
# #     )

# # if y.shape[-2] != factor * X.shape[-2]:
# #     raise ValueError(
# #         f"Height mismatch: HR height {y.shape[-2]} != "
# #         f"{factor} * LR height {X.shape[-2]}"
# #     )

# # if y.shape[-1] != factor * X.shape[-1]:
# #     raise ValueError(
# #         f"Width mismatch: HR width {y.shape[-1]} != "
# #         f"{factor} * LR width {X.shape[-1]}"
# #     )

# # # ensure masks are float and binary
# # mask_hr = mask_hr.float()
# # mask_hr = (mask_hr > 0.5).float()

# # X_mask = X_mask.float()
# # X_mask = (X_mask > 0.5).float()

# # # elevation should already be normalized in preprocessing
# # if X_elev is not None:
# #     X_elev = X_elev.float()
# #     X_elev = torch.nan_to_num(X_elev, nan=0.0, posinf=0.0, neginf=0.0)

# # if X_hr_elev is not None:
# #     X_hr_elev = X_hr_elev.float()
# #     X_hr_elev = torch.nan_to_num(X_hr_elev, nan=0.0, posinf=0.0, neginf=0.0)

# # # climate tensors
# # X = X.float()
# # y = y.float()

# # print("After checks:")
# # print("X       :", X.shape)
# # print("X_mask  :", X_mask.shape)
# # if X_elev is not None:
# #     print("X_elev  :", X_elev.shape)
# # if X_hr_elev is not None:
# #     print("X_hr_elev:", X_hr_elev.shape)
# # print("y       :", y.shape)
# # print("mask_hr :", mask_hr.shape)


# # # =========================================================
# # # split train/val by year
# # # train: historical period
# # # val: future period
# # # =========================================================
# # if gcm_name == "CanESM2":
# #     # CanESM2/CORDEX preprocessing currently gives 1950-2100,
# #     # assuming no leap years in the aligned dataset.
# #     start_year = 1950
# #     train_end_year = 2005
# #     val_start_year = 2081

# #     train_end_idx = (train_end_year - start_year + 1) * 365
# #     val_start_idx = (val_start_year - start_year) * 365

# # elif gcm_name == "EC-EARTH":
# #     def is_leap(year):
# #         return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

# #     def days_between_years(start_year, end_year_inclusive):
# #         total = 0
# #         for y in range(start_year, end_year_inclusive + 1):
# #             total += 366 if is_leap(y) else 365
# #         return total

# #     start_year = 1951
# #     train_end_year = 2005
# #     val_start_year = 2081

# #     train_end_idx = days_between_years(start_year, train_end_year)
# #     val_start_idx = days_between_years(start_year, val_start_year - 1)

# # else:
# #     raise ValueError(f"Year split not defined for gcm_name={gcm_name}")

# # print("train_end_idx =", train_end_idx)
# # print("val_start_idx =", val_start_idx)
# # print("Total samples  =", X.shape[0])


# # # =========================================================
# # # create train/val tensors
# # # =========================================================
# # X_train = X[:train_end_idx]
# # X_mask_train = X_mask[:train_end_idx]

# # y_train = y[:train_end_idx]
# # mask_train = mask_hr[:train_end_idx]

# # X_val = X[val_start_idx:]
# # X_mask_val = X_mask[val_start_idx:]

# # y_val = y[val_start_idx:]
# # mask_val = mask_hr[val_start_idx:]

# # if model_name == "SRResNet":
# #     X_elev_train = X_elev[:train_end_idx]
# #     X_elev_val = X_elev[val_start_idx:]
# # else:
# #     X_elev_train = None
# #     X_elev_val = None

# # if model_name == "SRResNet_HR_Aux":
# #     X_hr_elev_train = X_hr_elev[:train_end_idx]
# #     X_hr_elev_val = X_hr_elev[val_start_idx:]
# # else:
# #     X_hr_elev_train = None
# #     X_hr_elev_val = None

# # print("Train/Val shapes before model input construction:")
# # print("X_train      :", X_train.shape)
# # print("X_mask_train :", X_mask_train.shape)
# # if X_elev_train is not None:
# #     print("X_elev_train :", X_elev_train.shape)
# # if X_hr_elev_train is not None:
# #     print("X_hr_elev_train:", X_hr_elev_train.shape)
# # print("y_train      :", y_train.shape)
# # print("mask_train   :", mask_train.shape)

# # print("X_val        :", X_val.shape)
# # print("X_mask_val   :", X_mask_val.shape)
# # if X_elev_val is not None:
# #     print("X_elev_val   :", X_elev_val.shape)
# # if X_hr_elev_val is not None:
# #     print("X_hr_elev_val:", X_hr_elev_val.shape)
# # print("y_val        :", y_val.shape)
# # print("mask_val     :", mask_val.shape)


# # # =========================================================
# # # normalization stats from training set only
# # # Only normalize the climate variable channel.
# # # Do NOT normalize land-sea masks.
# # # Elevation is assumed already normalized by preprocessing.
# # # =========================================================
# # X_train_valid = X_train[X_mask_train > 0.5]

# # if X_train_valid.numel() == 0:
# #     raise ValueError("No valid points found in input mask.")

# # X_mean = X_train_valid.mean()
# # X_std = X_train_valid.std()
# # X_max = X_train_valid.max()

# # y_train_land = y_train[mask_train > 0.5]

# # if y_train_land.numel() == 0:
# #     raise ValueError("No land points found in HR training mask.")

# # y_mean = y_train_land.mean()
# # y_std = y_train_land.std()
# # y_max = y_train_land.max()

# # # avoid divide-by-zero
# # if X_std.item() == 0:
# #     X_std = torch.tensor(1.0, dtype=X_train.dtype)

# # if y_std.item() == 0:
# #     y_std = torch.tensor(1.0, dtype=y_train.dtype)

# # print("Normalization stats:")
# # print("X_mean:", X_mean.item())
# # print("X_std :", X_std.item())
# # print("X_max :", X_max.item())
# # print("y_mean:", y_mean.item())
# # print("y_std :", y_std.item())
# # print("y_max :", y_max.item())


# # # =========================================================
# # # normalize climate variable and build model inputs
# # # =========================================================
# # X_train_clim_n = (X_train - X_mean) / X_std
# # X_val_clim_n = (X_val - X_mean) / X_std

# # # force invalid LR input pixels to zero after normalization
# # X_train_clim_n = X_train_clim_n * X_mask_train
# # X_val_clim_n = X_val_clim_n * X_mask_val

# # # target normalization
# # y_train_n = (y_train - y_mean) / y_std
# # y_val_n = (y_val - y_mean) / y_std

# # if model_name == "SRResNet":
# #     # Original model input:
# #     # channel 0: normalized LR climate variable
# #     # channel 1: LR land-sea mask
# #     # channel 2: LR elevation
# #     X_train_in = torch.cat(
# #         [X_train_clim_n, X_mask_train, X_elev_train],
# #         dim=1,
# #     ).contiguous()

# #     X_val_in = torch.cat(
# #         [X_val_clim_n, X_mask_val, X_elev_val],
# #         dim=1,
# #     ).contiguous()

# #     X_train_hr_aux = None
# #     X_val_hr_aux = None

# #     input_channels_description = (
# #         "SRResNet input channels: "
# #         "channel0=normalized_LR_climate_input; "
# #         "channel1=LR_land_sea_mask; "
# #         "channel2=LR_elevation"
# #     )

# # elif model_name == "SRResNet_HR_Aux":
# #     # HR-aux model input:
# #     # main input is only the normalized LR climate variable.
# #     # HR auxiliary input is concatenated after upsampling inside the model.
# #     X_train_in = X_train_clim_n.contiguous()
# #     X_val_in = X_val_clim_n.contiguous()

# #     X_train_hr_aux = torch.cat(
# #         [mask_train, X_hr_elev_train],
# #         dim=1,
# #     ).contiguous()

# #     X_val_hr_aux = torch.cat(
# #         [mask_val, X_hr_elev_val],
# #         dim=1,
# #     ).contiguous()

# #     input_channels_description = (
# #         "SRResNet_HR_Aux main input: channel0=normalized_LR_climate_input; "
# #         "HR auxiliary channels after upsampling: "
# #         "channel0=HR_land_sea_mask; channel1=HR_elevation"
# #     )

# # else:
# #     raise ValueError(f"Unsupported model_name: {model_name}")

# # print("Train/Val shapes after model input construction:")
# # print("X_train_in:", X_train_in.shape)
# # print("X_val_in  :", X_val_in.shape)
# # if X_train_hr_aux is not None:
# #     print("X_train_hr_aux:", X_train_hr_aux.shape)
# # if X_val_hr_aux is not None:
# #     print("X_val_hr_aux  :", X_val_hr_aux.shape)

# # if X_train_in.shape[1] != INPUT_CHANNELS:
# #     raise ValueError(
# #         f"Expected {INPUT_CHANNELS} input channels, got {X_train_in.shape[1]}"
# #     )

# # if X_val_in.shape[1] != INPUT_CHANNELS:
# #     raise ValueError(
# #         f"Expected {INPUT_CHANNELS} input channels, got {X_val_in.shape[1]}"
# #     )

# # if model_name == "SRResNet_HR_Aux":
# #     if X_train_hr_aux.shape[1] != HR_AUX_CHANNELS:
# #         raise ValueError(
# #             f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
# #             f"got {X_train_hr_aux.shape[1]}"
# #         )
# #     if X_val_hr_aux.shape[1] != HR_AUX_CHANNELS:
# #         raise ValueError(
# #             f"Expected {HR_AUX_CHANNELS} HR auxiliary channels, "
# #             f"got {X_val_hr_aux.shape[1]}"
# #         )


# # # =========================================================
# # # losses
# # # =========================================================
# # def masked_mse(pred, target, mask, eps=1e-8):
# #     """
# #     pred, target, mask: [B, C, H, W]
# #     mask = 1 over land, 0 over ocean.
# #     Returns mean squared error over land pixels only.
# #     """
# #     se = (pred - target) ** 2
# #     se = se * mask
# #     return se.sum() / (mask.sum() + eps)


# # loss_fn = nn.MSELoss()


# # # =========================================================
# # # datasets
# # # =========================================================
# # if model_name == "SRResNet":
# #     training_set = TensorDataset(
# #         X_train_in,
# #         y_train_n,
# #         y_train,
# #         mask_train,
# #     )

# #     validation_set = TensorDataset(
# #         X_val_in,
# #         y_val_n,
# #         y_val,
# #         mask_val,
# #     )

# # elif model_name == "SRResNet_HR_Aux":
# #     training_set = TensorDataset(
# #         X_train_in,
# #         X_train_hr_aux,
# #         y_train_n,
# #         y_train,
# #         mask_train,
# #     )

# #     validation_set = TensorDataset(
# #         X_val_in,
# #         X_val_hr_aux,
# #         y_val_n,
# #         y_val,
# #         mask_val,
# #     )

# # else:
# #     raise ValueError(f"Unsupported model_name: {model_name}")


# # # =========================================================
# # # objective function
# # # =========================================================
# # def objective(trial):
# #     # -----------------------------------------------------
# #     # hyperparameters to search
# #     # -----------------------------------------------------
# #     batch_size = trial.suggest_categorical("batch_size", [BATCH_SIZE])
# #     learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True)
# #     num_resblk = trial.suggest_int("num_resblk", 4, 16, step=2)
# #     num_features = trial.suggest_categorical("num_features", [32, 64, 96, 128])
# #     weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

# #     # -----------------------------------------------------
# #     # paths for this trial
# #     # -----------------------------------------------------
# #     trial_ckpt_path = os.path.join(save_root, f"trial_{trial.number:03d}.pth")

# #     # -----------------------------------------------------
# #     # model
# #     # -----------------------------------------------------
# #     if model_name == "SRResNet":
# #         model = SRResNet(
# #             num_resblk=num_resblk,
# #             num_features=num_features,
# #             input_channels=INPUT_CHANNELS,
# #             output_channels=1,
# #             scale=factor,
# #         ).to(device)

# #     elif model_name == "SRResNet_HR_Aux":
# #         model = SRResNet_HR_Aux(
# #             num_resblk=num_resblk,
# #             num_features=num_features,
# #             input_channels=INPUT_CHANNELS,
# #             output_channels=1,
# #             hr_aux_channels=HR_AUX_CHANNELS,
# #             scale=factor,
# #         ).to(device)

# #     else:
# #         raise ValueError(f"Unsupported model_name: {model_name}")

# #     # -----------------------------------------------------
# #     # dataloaders
# #     # -----------------------------------------------------
# #     train_dataloader = DataLoader(
# #         training_set,
# #         batch_size=batch_size,
# #         shuffle=True,
# #     )

# #     val_dataloader = DataLoader(
# #         validation_set,
# #         batch_size=batch_size,
# #         shuffle=False,
# #     )

# #     # -----------------------------------------------------
# #     # training setup
# #     # -----------------------------------------------------
# #     optimizer = torch.optim.AdamW(
# #         model.parameters(),
# #         lr=learning_rate,
# #         weight_decay=weight_decay,
# #     )

# #     scheduler = ReduceLROnPlateau(
# #         optimizer,
# #         patience=15,
# #         factor=0.5,
# #     )

# #     train_loss_list = []
# #     val_loss_list = []

# #     best_val_loss = float("inf")
# #     best_epoch = -1
# #     epochs_no_improve = 0

# #     y_mean_dev = y_mean.to(device)
# #     y_std_dev = y_std.to(device)

# #     # -----------------------------------------------------
# #     # train loop
# #     # -----------------------------------------------------
# #     for epoch in range(EPOCHS):
# #         model.train()
# #         train_loss = 0.0

# #         if model_name == "SRResNet":
# #             for Xn, yn, y_raw, mask_raw in train_dataloader:
# #                 Xn = Xn.to(device)
# #                 yn = yn.to(device)
# #                 mask_raw = mask_raw.to(device)

# #                 # normalized prediction
# #                 y_pred_n = model(Xn)

# #                 # train on HR land only
# #                 loss = masked_mse(y_pred_n, yn, mask_raw)

# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 optimizer.step()

# #                 train_loss += loss.item()

# #         elif model_name == "SRResNet_HR_Aux":
# #             for Xn, X_hr_aux, yn, y_raw, mask_raw in train_dataloader:
# #                 Xn = Xn.to(device)
# #                 X_hr_aux = X_hr_aux.to(device)
# #                 yn = yn.to(device)
# #                 mask_raw = mask_raw.to(device)

# #                 # normalized prediction
# #                 y_pred_n = model(Xn, X_hr_aux)

# #                 # train on HR land only
# #                 loss = masked_mse(y_pred_n, yn, mask_raw)

# #                 optimizer.zero_grad()
# #                 loss.backward()
# #                 optimizer.step()

# #                 train_loss += loss.item()

# #         else:
# #             raise ValueError(f"Unsupported model_name: {model_name}")

# #         train_loss /= len(train_dataloader)
# #         train_loss_list.append(train_loss)

# #         # -------------------------------------------------
# #         # validation in physical space
# #         # prediction is hard-masked in physical space,
# #         # so ocean is exactly zero.
# #         # -------------------------------------------------
# #         model.eval()
# #         val_loss = 0.0

# #         with torch.no_grad():
# #             if model_name == "SRResNet":
# #                 for Xn, yn, y_raw, mask_raw in val_dataloader:
# #                     Xn = Xn.to(device)
# #                     y_raw = y_raw.to(device)
# #                     mask_raw = mask_raw.to(device)

# #                     # normalized prediction
# #                     y_pred_n = model(Xn)

# #                     # convert to physical space
# #                     y_pred = y_pred_n * y_std_dev + y_mean_dev

# #                     # enforce exact zero over ocean in physical space
# #                     y_pred = y_pred * mask_raw

# #                     # full-image physical MSE
# #                     val_loss += loss_fn(y_pred, y_raw).item()

# #             elif model_name == "SRResNet_HR_Aux":
# #                 for Xn, X_hr_aux, yn, y_raw, mask_raw in val_dataloader:
# #                     Xn = Xn.to(device)
# #                     X_hr_aux = X_hr_aux.to(device)
# #                     y_raw = y_raw.to(device)
# #                     mask_raw = mask_raw.to(device)

# #                     # normalized prediction
# #                     y_pred_n = model(Xn, X_hr_aux)

# #                     # convert to physical space
# #                     y_pred = y_pred_n * y_std_dev + y_mean_dev

# #                     # enforce exact zero over ocean in physical space
# #                     y_pred = y_pred * mask_raw

# #                     # full-image physical MSE
# #                     val_loss += loss_fn(y_pred, y_raw).item()

# #             else:
# #                 raise ValueError(f"Unsupported model_name: {model_name}")

# #         val_loss /= len(val_dataloader)
# #         val_loss_list.append(val_loss)

# #         scheduler.step(val_loss)

# #         # report to Optuna for pruning
# #         trial.report(val_loss, epoch)
# #         trial.set_user_attr("best_epoch", best_epoch)
# #         trial.set_user_attr("best_val_loss", best_val_loss)

# #         if trial.should_prune():
# #             trial.set_user_attr("status", "pruned")
# #             raise optuna.TrialPruned()

# #         # -------------------------------------------------
# #         # save best model for this trial
# #         # -------------------------------------------------
# #         if val_loss < best_val_loss - MIN_DELTA:
# #             best_val_loss = val_loss
# #             best_epoch = epoch
# #             epochs_no_improve = 0

# #             trial.set_user_attr("best_epoch", best_epoch)
# #             trial.set_user_attr("best_val_loss", best_val_loss)
# #             trial.set_user_attr("checkpoint_path", trial_ckpt_path)

# #             torch.save(
# #                 {
# #                     "trial_number": trial.number,
# #                     "epoch": best_epoch,
# #                     "model_name": model.__class__.__name__,
# #                     "model_choice": model_name,
# #                     "input_source": INPUT_SOURCE,
# #                     "input_file": input_file,
# #                     "input_mask_file": input_mask_file,
# #                     "input_elevation_file": input_elev_file,
# #                     "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
# #                     "target_file": "high_res.pth",
# #                     "target_mask_file": "high_res_mask.pth",
# #                     "model_hparams": {
# #                         "num_resblk": num_resblk,
# #                         "num_features": num_features,
# #                         "input_channels": INPUT_CHANNELS,
# #                         "output_channels": 1,
# #                         "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
# #                         "scale": factor,
# #                     },
# #                     "training_hparams": {
# #                         "batch_size": batch_size,
# #                         "learning_rate": learning_rate,
# #                         "weight_decay": weight_decay,
# #                         "epochs": EPOCHS,
# #                         "early_stopping_patience": EARLY_STOPPING_PATIENCE,
# #                         "min_delta": MIN_DELTA,
# #                         "loss": "train_masked_mse_land_only__val_full_image_mse",
# #                         "input_channels_description": input_channels_description,
# #                         "ocean_constraint": "prediction multiplied by HR mask in physical space",
# #                         "factor": factor,
# #                         "rcm_var": rcm_var,
# #                         "gcm_name": gcm_name,
# #                         "rcm_name": rcm_name,
# #                         "grid": grid,
# #                         "train_end_year": train_end_year,
# #                         "val_start_year": val_start_year,
# #                     },
# #                     "model_state": copy.deepcopy(model.state_dict()),
# #                     "optimizer_state": optimizer.state_dict(),
# #                     "scheduler_state": scheduler.state_dict(),
# #                     "best_val_loss": best_val_loss,
# #                     "best_epoch": best_epoch,
# #                     "train_loss": train_loss,
# #                     "val_loss": val_loss,
# #                     "train_loss_list": train_loss_list,
# #                     "val_loss_list": val_loss_list,
# #                     "X_mean": X_mean,
# #                     "X_std": X_std,
# #                     "X_max": X_max,
# #                     "y_mean": y_mean,
# #                     "y_std": y_std,
# #                     "y_max": y_max,
# #                 },
# #                 trial_ckpt_path,
# #             )

# #         else:
# #             epochs_no_improve += 1

# #         # print(
# #         #     f"Trial {trial.number:03d} | "
# #         #     f"Epoch {epoch:03d} | "
# #         #     f"Model: {model_name} | "
# #         #     f"Input: {INPUT_SOURCE} | "
# #         #     f"Channels: {INPUT_CHANNELS} | "
# #         #     f"HR aux channels: {HR_AUX_CHANNELS if model_name == 'SRResNet_HR_Aux' else 0} | "
# #         #     f"Train: {train_loss:.6f} | "
# #         #     f"Val(full image): {val_loss:.6f} | "
# #         #     f"Best Val: {best_val_loss:.6f} | "
# #         #     f"Best Epoch: {best_epoch:03d} | "
# #         #     f"No improve: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE} | "
# #         #     f"lr: {optimizer.param_groups[0]['lr']:.6e}"
# #         # )

# #         if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
# #             print(
# #                 f"Early stopping triggered for trial {trial.number:03d} "
# #                 f"at epoch {epoch:03d} | best epoch: {best_epoch:03d}"
# #             )
# #             break

# #     trial.set_user_attr("status", "completed")
# #     trial.set_user_attr("best_epoch", best_epoch)
# #     trial.set_user_attr("best_val_loss", best_val_loss)
# #     trial.set_user_attr("checkpoint_path", trial_ckpt_path)

# #     return best_val_loss


# # # =========================================================
# # # create and run study
# # # =========================================================
# # study = optuna.create_study(
# #     direction="minimize",
# #     pruner=optuna.pruners.MedianPruner(
# #         n_startup_trials=5,
# #         n_warmup_steps=5,
# #     ),
# # )

# # study.optimize(objective, n_trials=N_TRIALS)


# # # =========================================================
# # # save best params summary
# # # =========================================================
# # best_trial = study.best_trial

# # best_summary = {
# #     "best_trial_number": best_trial.number,
# #     "best_value": best_trial.value,
# #     "best_params": best_trial.params,
# #     "best_epoch": best_trial.user_attrs.get("best_epoch", None),
# #     "best_checkpoint_path": best_trial.user_attrs.get(
# #         "checkpoint_path",
# #         os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
# #     ),
# #     "input_source": INPUT_SOURCE,
# #     "input_file": input_file,
# #     "input_mask_file": input_mask_file,
# #     "input_elevation_file": input_elev_file,
# #     "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
# #     "target_file": "high_res.pth",
# #     "input_channels": INPUT_CHANNELS,
# #     "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
# #     "input_channels_description": input_channels_description,
# #     "model_name": model_name,
# #     "rcm_var": rcm_var,
# #     "gcm_name": gcm_name,
# #     "rcm_name": rcm_name,
# #     "grid": grid,
# # }

# # with open(study_summary_path, "w") as f:
# #     json.dump(best_summary, f, indent=4)


# # # =========================================================
# # # save all trial summaries
# # # =========================================================
# # all_trials_summary = []

# # for t in study.trials:
# #     trial_info = {
# #         "trial_number": t.number,
# #         "state": str(t.state),
# #         "value": t.value if t.value is not None else None,
# #         "params": t.params,
# #         "best_epoch": t.user_attrs.get("best_epoch", None),
# #         "best_val_loss": t.user_attrs.get("best_val_loss", None),
# #         "checkpoint_path": t.user_attrs.get(
# #             "checkpoint_path",
# #             os.path.join(save_root, f"trial_{t.number:03d}.pth"),
# #         ),
# #         "input_source": INPUT_SOURCE,
# #         "input_file": input_file,
# #         "input_mask_file": input_mask_file,
# #         "input_elevation_file": input_elev_file,
# #         "hr_elevation_file": HR_ELEVATION_FILE if model_name == "SRResNet_HR_Aux" else None,
# #         "input_channels": INPUT_CHANNELS,
# #         "hr_aux_channels": HR_AUX_CHANNELS if model_name == "SRResNet_HR_Aux" else None,
# #         "model_name": model_name,
# #         "rcm_var": rcm_var,
# #         "gcm_name": gcm_name,
# #         "rcm_name": rcm_name,
# #         "grid": grid,
# #     }

# #     all_trials_summary.append(trial_info)

# # with open(all_trials_summary_path, "w") as f:
# #     json.dump(all_trials_summary, f, indent=4)


# # # =========================================================
# # # final logs
# # # =========================================================
# # print("\nOptuna study finished.")
# # print(f"Model name: {model_name}")
# # print(f"Input source: {INPUT_SOURCE}")
# # print(f"Input file: {input_file}")
# # print(f"Input mask file: {input_mask_file}")
# # print(f"Input elevation file: {input_elev_file}")
# # print(f"HR elevation file: {HR_ELEVATION_FILE if model_name == 'SRResNet_HR_Aux' else None}")
# # print(f"Input channels: {INPUT_CHANNELS}")
# # print(f"HR aux channels: {HR_AUX_CHANNELS if model_name == 'SRResNet_HR_Aux' else None}")
# # print(f"Best trial: {best_trial.number}")
# # print(f"Best value: {best_trial.value:.6f}")
# # print(f"Best epoch: {best_trial.user_attrs.get('best_epoch', None)}")
# # print("Best params:", best_trial.params)

# # print(
# #     "Best checkpoint:",
# #     best_trial.user_attrs.get(
# #         "checkpoint_path",
# #         os.path.join(save_root, f"trial_{best_trial.number:03d}.pth"),
# #     ),
# # )

# # print("Best summary saved to:", study_summary_path)
# # print("All trial summaries saved to:", all_trials_summary_path)
# # print("All per-trial checkpoints saved under:", save_root)
