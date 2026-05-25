N_TRIALS = 3000
EPOCHS = 200
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 0.0
RANDOM_SEED = 42

rcm_var = "prec"  # 'tmean' 'prec'
gcm_name = "CanESM2"
gcm_rcm_dict = {"CanESM2": "CanRCM4"}
grid = "NAM-44i"
factor = 4
exp_folder_name = (
    f"/projects/sds-lab/Shuochen/downscaling/cordex_canesm2/"
    f"{rcm_var}.{gcm_name}.{gcm_rcm_dict[gcm_name]}.day.{grid}.mbcn-gridMET/"
)

import os
import copy
import json
import torch
import optuna
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from models import UNet, ESPCNx4, SRResNet

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
# save paths
# =========================================================
model_name = "SRResNet"
save_root = os.path.join(exp_folder_name, "trained_models", model_name)
os.makedirs(save_root, exist_ok=True)

study_summary_path = os.path.join(save_root, "optuna_best_params.json")
all_trials_summary_path = os.path.join(save_root, "optuna_all_trials_summary.json")

print("All trial checkpoints will be saved to:", save_root)

# =========================================================
# load tensors
# =========================================================
X = torch.load(os.path.join(exp_folder_name, f"coarse_{factor}x.pth"))
y = torch.load(os.path.join(exp_folder_name, "high_res.pth"))
mask_hr = torch.load(os.path.join(exp_folder_name, "high_res_mask.pth"))

print("Loaded shapes:")
print("X       :", X.shape)
print("y       :", y.shape)
print("mask_hr :", mask_hr.shape)

if y.shape != mask_hr.shape:
    raise ValueError(
        f"Shape mismatch: y shape {y.shape} does not match mask shape {mask_hr.shape}"
    )

# ensure mask is float and binary-like
mask_hr = mask_hr.float()
mask_hr = (mask_hr > 0.5).float()

# =========================================================
# split train/val
# keep X, y, and mask aligned
# =========================================================
X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
    X, y, mask_hr, test_size=0.3, shuffle=False
)
print("Train/Val shapes:")
print("X_train    :", X_train.shape)
print("y_train    :", y_train.shape)
print("mask_train :", mask_train.shape)
print("X_val      :", X_val.shape)
print("y_val      :", y_val.shape)
print("mask_val   :", mask_val.shape)

# =========================================================
# normalization stats from training set only
# X normalization stays global as before
# y normalization is computed on land only
# =========================================================
X_mean = X_train.mean()
X_std = X_train.std()
X_max = X_train.max()

# compute y stats on land only
y_train_land = y_train[mask_train > 0.5]
if y_train_land.numel() == 0:
    raise ValueError("No land points found in training mask.")

y_mean = y_train_land.mean()
y_std = y_train_land.std()
y_max = y_train_land.max()

# avoid divide-by-zero
if X_std.item() == 0:
    X_std = torch.tensor(1.0, dtype=X_train.dtype)
if y_std.item() == 0:
    y_std = torch.tensor(1.0, dtype=y_train.dtype)

# normalize
X_train_n = (X_train - X_mean) / X_std
X_val_n   = (X_val   - X_mean) / X_std
y_train_n = (y_train - y_mean) / y_std
y_val_n   = (y_val   - y_mean) / y_std

# =========================================================
# losses
# =========================================================
def masked_mse(pred, target, mask, eps=1e-8):
    """
    pred, target, mask: [B, C, H, W]
    mask = 1 over land, 0 over ocean
    Returns mean squared error over land pixels only.
    """
    se = (pred - target) ** 2
    se = se * mask
    return se.sum() / (mask.sum() + eps)

loss_fn = nn.MSELoss()

# =========================================================
# datasets
# include mask in both training and validation
# =========================================================
training_set = TensorDataset(X_train_n, y_train_n, X_train, y_train, mask_train)
validation_set = TensorDataset(X_val_n, y_val_n, X_val, y_val, mask_val)

# =========================================================
# objective function
# =========================================================
def objective(trial):
    # -----------------------------------------------------
    # hyperparameters to search
    # -----------------------------------------------------
    batch_size = trial.suggest_categorical("batch_size", [1024])
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
    model = SRResNet(
        num_resblk=num_resblk,
        num_features=num_features
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
    scheduler = ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

    train_loss_list = []
    val_loss_list = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    X_mean_dev = X_mean.to(device)
    X_std_dev  = X_std.to(device)
    y_mean_dev = y_mean.to(device)
    y_std_dev  = y_std.to(device)

    # -----------------------------------------------------
    # train loop
    # -----------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for Xn, yn, X_raw, y_raw, mask_raw in train_dataloader:
            Xn = Xn.to(device)
            yn = yn.to(device)
            mask_raw = mask_raw.to(device)

            # normalized prediction
            y_pred_n = model(Xn)

            # hard mask on output in normalized space
            # this guarantees exact zero over ocean after denorm below only if
            # zero physical value corresponds to normalized zero, which it does not.
            # so we must also hard-mask again in physical space for final guarantee.
            y_pred_n = y_pred_n * mask_raw

            # train on land only
            loss = masked_mse(y_pred_n, yn, mask_raw)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)
        train_loss_list.append(train_loss)

        # validation in physical space, FULL IMAGE MSE
        # prediction is hard-masked in physical space so ocean is exactly zero
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for Xn, yn, X_raw, y_raw, mask_raw in val_dataloader:
                Xn = Xn.to(device)
                y_raw = y_raw.to(device)
                mask_raw = mask_raw.to(device)

                # normalized prediction
                y_pred_n = model(Xn)

                # convert to physical space
                y_pred = y_pred_n * y_std_dev + y_mean_dev

                # enforce exact zero over ocean in physical space
                y_pred = y_pred * mask_raw

                # full-image MSE over all pixels
                val_loss += loss_fn(y_pred, y_raw).item()

        val_loss /= len(val_dataloader)
        val_loss_list.append(val_loss)

        scheduler.step(val_loss)

        # report to Optuna for pruning
        trial.report(val_loss, epoch)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("best_val_loss", best_val_loss)

        # prune if needed
        if trial.should_prune():
            trial.set_user_attr("status", "pruned")
            raise optuna.TrialPruned()

        # save best model for THIS trial only
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
                    "model_hparams": {
                        "num_resblk": num_resblk,
                        "num_features": num_features,
                    },
                    "training_hparams": {
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "epochs": EPOCHS,
                        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                        "min_delta": MIN_DELTA,
                        "loss": "train_masked_mse_land_only__val_full_image_mse",
                        "ocean_constraint": "prediction multiplied by mask in physical space",
                    },
                    "model_state": copy.deepcopy(model.state_dict()),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_loss_list": train_loss_list,
                    "val_loss_list": val_loss_list,
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
        #     f"Train: {train_loss:.6f} | "
        #     f"Val(full image): {val_loss:.6f} | "
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
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
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
        os.path.join(save_root, f"trial_{best_trial.number:03d}.pth")
    ),
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
            os.path.join(save_root, f"trial_{t.number:03d}.pth")
        ),
    }
    all_trials_summary.append(trial_info)

with open(all_trials_summary_path, "w") as f:
    json.dump(all_trials_summary, f, indent=4)

# =========================================================
# final logs
# =========================================================
print("\nOptuna study finished.")
print(f"Best trial: {best_trial.number}")
print(f"Best value: {best_trial.value:.6f}")
print(f"Best epoch: {best_trial.user_attrs.get('best_epoch', None)}")
print("Best params:", best_trial.params)
print(
    "Best checkpoint:",
    best_trial.user_attrs.get(
        "checkpoint_path",
        os.path.join(save_root, f"trial_{best_trial.number:03d}.pth")
    )
)
print("Best summary saved to:", study_summary_path)
print("All trial summaries saved to:", all_trials_summary_path)
print("All per-trial checkpoints saved under:", save_root)
