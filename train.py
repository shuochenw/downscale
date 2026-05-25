import argparse
import copy
import json
import os
import random

import matplotlib
import numpy as np
import optuna
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from models import ClimateRCAN_ShallowFusion

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna training for shallow-fusion RCAN HR-aux climate downscaling."
    )

    parser.add_argument("--rcm_var", type=str, default="tas")
    parser.add_argument("--gcm_name", type=str, default="CanESM2")
    parser.add_argument("--rcm_name", type=str, default="RCA4")
    parser.add_argument("--grid", type=str, default="NAM-44i")
    parser.add_argument("--rcm_product", type=str, default="raw")
    parser.add_argument(
        "--exp",
        type=str,
        default="GCM_RCM",
        choices=("GCM_RCM", "RCM_RCM"),
        help="Preprocessed experiment folder suffix, e.g. GCM_RCM or RCM_RCM.",
    )
    parser.add_argument("--factor", type=int, default=4)

    parser.add_argument(
        "--data_root",
        type=str,
        default="/projects/sds-lab/Shuochen/downscaling/preprocessed",
    )
    parser.add_argument("--input_file", type=str, default="low_res.pth")
    parser.add_argument("--target_file", type=str, default="high_res.pth")
    parser.add_argument("--hr_mask_file", type=str, default="high_res_mask.pth")
    parser.add_argument(
        "--hr_elevation_file",
        type=str,
        default="high_res_elevation.pth",
    )

    parser.add_argument("--train_start_year", type=int, default=1951)
    parser.add_argument("--train_end_year", type=int, default=2005)
    parser.add_argument("--val_start_year", type=int, default=2006)
    parser.add_argument("--val_end_year", type=int, default=2099)

    parser.add_argument("--n_trials", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early_stopping_patience", type=int, default=30)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--batch_size_choices",
        type=int,
        nargs="+",
        default=[16, 32, 64],
    )
    parser.add_argument(
        "--num_features_choices",
        type=int,
        nargs="+",
        default=[64, 96, 128],
    )
    parser.add_argument("--num_resblk_min", type=int, default=4)
    parser.add_argument("--num_resblk_max", type=int, default=16)
    parser.add_argument("--num_resblk_step", type=int, default=2)
    parser.add_argument(
        "--num_groups_choices",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument(
        "--reduction_choices",
        type=int,
        nargs="+",
        default=[4, 8, 16],
    )
    parser.add_argument(
        "--res_scale_choices",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
    )
    parser.add_argument("--learning_rate_min", type=float, default=1e-5)
    parser.add_argument("--learning_rate_max", type=float, default=3e-4)
    parser.add_argument("--weight_decay_min", type=float, default=1e-7)
    parser.add_argument("--weight_decay_max", type=float, default=1e-3)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)

    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optuna storage URL. If omitted, a sqlite database is created "
            "inside save_dir."
        ),
    )
    parser.add_argument(
        "--no_resume_study",
        action="store_true",
        help="Create a fresh Optuna study instead of loading an existing one.",
    )
    parser.add_argument("--no_plots", action="store_true")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or inf values.")


def safe_std(tensor):
    std = tensor.std()
    if std.item() == 0:
        return torch.tensor(1.0, dtype=tensor.dtype)
    return std


def match_sample_dim(tensor, n_samples, name):
    if tensor.shape[0] == n_samples:
        return tensor
    if tensor.shape[0] == 1:
        print(f"{name} has one sample; expanding to {n_samples} samples.")
        return tensor.expand(n_samples, -1, -1, -1).contiguous()
    raise ValueError(
        f"{name} has incompatible sample dimension: "
        f"{tensor.shape[0]} vs expected {n_samples}"
    )


def split_indices(args, n_samples):
    if args.train_end_year < args.train_start_year:
        raise ValueError("train_end_year must be >= train_start_year.")

    if args.val_end_year < args.val_start_year:
        raise ValueError("val_end_year must be >= val_start_year.")

    if args.gcm_name == "CanESM2":
        train_start_idx = 0
        train_end_idx = (
            args.train_end_year - args.train_start_year + 1
        ) * 365
        val_start_idx = (args.val_start_year - args.train_start_year) * 365
        val_end_idx = (args.val_end_year - args.train_start_year + 1) * 365

    elif args.gcm_name == "EC-EARTH":
        def is_leap(year):
            return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

        def days_between_years(start_year, end_year_inclusive):
            if end_year_inclusive < start_year:
                return 0
            total = 0
            for year in range(start_year, end_year_inclusive + 1):
                total += 366 if is_leap(year) else 365
            return total

        train_start_idx = 0
        train_end_idx = days_between_years(
            args.train_start_year,
            args.train_end_year,
        )
        val_start_idx = days_between_years(
            args.train_start_year,
            args.val_start_year - 1,
        )
        val_end_idx = days_between_years(
            args.train_start_year,
            args.val_end_year,
        )

    else:
        raise ValueError(
            f"Year split is not defined for gcm_name={args.gcm_name}"
        )

    if train_end_idx > n_samples:
        raise ValueError(
            f"Training end index {train_end_idx} exceeds total samples "
            f"{n_samples}."
        )

    if val_end_idx > n_samples:
        raise ValueError(
            f"Validation end index {val_end_idx} exceeds total samples "
            f"{n_samples}."
        )

    return train_start_idx, train_end_idx, val_start_idx, val_end_idx


def load_data(args):
    exp_folder_name = os.path.join(
        args.data_root,
        (
            f"{args.rcm_var}.{args.gcm_name}.{args.rcm_name}.day."
            f"{args.grid}.{args.rcm_product}.{args.exp}"
        ),
    )

    X = torch.load(os.path.join(exp_folder_name, args.input_file)).float()
    y = torch.load(os.path.join(exp_folder_name, args.target_file)).float()
    mask_hr = torch.load(os.path.join(exp_folder_name, args.hr_mask_file)).float()
    elev_hr = torch.load(
        os.path.join(exp_folder_name, args.hr_elevation_file)
    ).float()

    if X.ndim != 4:
        raise ValueError(f"Expected X shape [N, C, H, W], got {X.shape}")
    if y.ndim != 4:
        raise ValueError(f"Expected y shape [N, C, H, W], got {y.shape}")
    if mask_hr.ndim != 4:
        raise ValueError(
            f"Expected mask_hr shape [N, C, H, W], got {mask_hr.shape}"
        )
    if elev_hr.ndim != 4:
        raise ValueError(
            f"Expected elev_hr shape [N, C, H, W], got {elev_hr.shape}"
        )

    if X.shape[1] != 1:
        raise ValueError(f"Expected one LR input channel, got {X.shape[1]}")
    if y.shape[1] != 1:
        raise ValueError(f"Expected one HR RCM target channel, got {y.shape[1]}")
    if mask_hr.shape[1] != 1:
        raise ValueError(f"Expected one HR mask channel, got {mask_hr.shape[1]}")
    if elev_hr.shape[1] != 1:
        raise ValueError(
            f"Expected one HR elevation channel, got {elev_hr.shape[1]}"
        )

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X has {X.shape[0]}, y has {y.shape[0]}")

    mask_hr = match_sample_dim(mask_hr, y.shape[0], "mask_hr")
    elev_hr = match_sample_dim(elev_hr, y.shape[0], "elev_hr")

    if mask_hr.shape != y.shape:
        raise ValueError(f"Mask shape mismatch: {mask_hr.shape} vs {y.shape}")
    if elev_hr.shape != y.shape:
        raise ValueError(f"Elevation shape mismatch: {elev_hr.shape} vs {y.shape}")

    mask_hr = (mask_hr > 0.5).float().contiguous()
    elev_hr = torch.nan_to_num(
        elev_hr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).contiguous()
    hr_aux = torch.cat([mask_hr, elev_hr], dim=1).contiguous()

    if hr_aux.shape[1] != 2:
        raise ValueError(f"Expected 2 HR auxiliary channels, got {hr_aux.shape[1]}")
    if mask_hr.sum().item() == 0:
        raise ValueError("HR mask contains no land pixels.")

    if y.shape[-2] != args.factor * X.shape[-2]:
        raise ValueError(
            f"Height mismatch: HR height {y.shape[-2]} != "
            f"{args.factor} * LR height {X.shape[-2]}"
        )
    if y.shape[-1] != args.factor * X.shape[-1]:
        raise ValueError(
            f"Width mismatch: HR width {y.shape[-1]} != "
            f"{args.factor} * LR width {X.shape[-1]}"
        )

    assert_finite(X, "X")
    assert_finite(y, "y")
    assert_finite(mask_hr, "mask_hr")
    assert_finite(elev_hr, "elev_hr")
    assert_finite(hr_aux, "hr_aux")

    train_start_idx, train_end_idx, val_start_idx, val_end_idx = split_indices(
        args,
        X.shape[0],
    )

    X_train = X[train_start_idx:train_end_idx]
    y_train = y[train_start_idx:train_end_idx]
    aux_train = hr_aux[train_start_idx:train_end_idx]

    X_val = X[val_start_idx:val_end_idx]
    y_val = y[val_start_idx:val_end_idx]
    aux_val = hr_aux[val_start_idx:val_end_idx]
    mask_val = mask_hr[val_start_idx:val_end_idx]

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("Train or validation split is empty.")

    X_mean = X_train.mean()
    X_std = safe_std(X_train)
    y_mean = y_train.mean()
    y_std = safe_std(y_train)

    X_train_n = ((X_train - X_mean) / X_std).contiguous()
    X_val_n = ((X_val - X_mean) / X_std).contiguous()
    y_train_n = ((y_train - y_mean) / y_std).contiguous()
    y_val_n = ((y_val - y_mean) / y_std).contiguous()

    print("Experiment folder:", exp_folder_name)
    print("Experiment branch:", args.exp)
    print("Loaded shapes:")
    print("LR input X:", X.shape)
    print("HR target y:", y.shape)
    print("HR mask  :", mask_hr.shape)
    print("HR elev  :", elev_hr.shape)
    print("HR aux   :", hr_aux.shape)
    print("Year split:")
    print(f"Train years: {args.train_start_year}-{args.train_end_year}")
    print(f"Val years  : {args.val_start_year}-{args.val_end_year}")
    print("train_start_idx:", train_start_idx)
    print("train_end_idx  :", train_end_idx)
    print("val_start_idx  :", val_start_idx)
    print("val_end_idx    :", val_end_idx)
    print("Split shapes:")
    print("X_train  :", X_train.shape)
    print("y_train  :", y_train.shape)
    print("aux_train:", aux_train.shape)
    print("X_val    :", X_val.shape)
    print("y_val    :", y_val.shape)
    print("aux_val  :", aux_val.shape)
    print("mask_val :", mask_val.shape)
    print("Normalization:")
    print("X_mean:", X_mean.item())
    print("X_std :", X_std.item())
    print("y_mean:", y_mean.item())
    print("y_std :", y_std.item())

    data = {
        "exp_folder_name": exp_folder_name,
        "train_start_idx": train_start_idx,
        "train_end_idx": train_end_idx,
        "val_start_idx": val_start_idx,
        "val_end_idx": val_end_idx,
        "X_train_n": X_train_n,
        "aux_train": aux_train,
        "y_train_n": y_train_n,
        "X_val_n": X_val_n,
        "aux_val": aux_val,
        "y_val_n": y_val_n,
        "y_val": y_val,
        "mask_val": mask_val,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    return data


def make_save_root(args, exp_folder_name):
    if args.save_dir is not None:
        save_root = args.save_dir
    else:
        save_root = os.path.join(
            exp_folder_name,
            "trained_models",
            "RCAN_HR_Aux_ShallowFusion",
            "optuna_lr_gcm_with_hr_mask_elevation_to_hr_rcm_mse_shallowfusion",
        )
    os.makedirs(save_root, exist_ok=True)
    return save_root


def suggest_hparams(trial, args):
    num_resblk = trial.suggest_int(
        "num_resblk",
        args.num_resblk_min,
        args.num_resblk_max,
        step=args.num_resblk_step,
    )
    return {
        "batch_size": trial.suggest_categorical(
            "batch_size",
            args.batch_size_choices,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            args.learning_rate_min,
            args.learning_rate_max,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            args.weight_decay_min,
            args.weight_decay_max,
            log=True,
        ),
        "num_resblk": num_resblk,
        "num_features": trial.suggest_categorical(
            "num_features",
            args.num_features_choices,
        ),
        "num_groups": trial.suggest_categorical(
            "num_groups",
            args.num_groups_choices,
        ),
        "reduction": trial.suggest_categorical(
            "reduction",
            args.reduction_choices,
        ),
        "res_scale": trial.suggest_categorical(
            "res_scale",
            args.res_scale_choices,
        ),
    }


def make_model(hparams, args, input_channels=1, hr_aux_channels=2):
    return ClimateRCAN_ShallowFusion(
        num_resblk=hparams["num_resblk"],
        num_features=hparams["num_features"],
        input_channels=input_channels,
        output_channels=1,
        hr_aux_channels=hr_aux_channels,
        scale=args.factor,
        num_groups=hparams["num_groups"],
        reduction=hparams["reduction"],
        res_scale=hparams["res_scale"],
    )


def make_loaders(data, batch_size, args, include_raw_val=False):
    training_set = TensorDataset(
        data["X_train_n"],
        data["aux_train"],
        data["y_train_n"],
    )

    if include_raw_val:
        validation_set = TensorDataset(
            data["X_val_n"],
            data["aux_val"],
            data["y_val_n"],
            data["y_val"],
            data["mask_val"],
        )
    else:
        validation_set = TensorDataset(
            data["X_val_n"],
            data["aux_val"],
            data["y_val_n"],
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def evaluate(
    model,
    val_loader,
    device,
    y_mean,
    y_std,
    loss_fn,
    use_amp=False,
    include_physical=False,
):
    model.eval()
    val_loss_sum = 0.0
    val_samples = 0

    full_se_sum = 0.0
    full_pixel_count = 0
    land_se_sum = 0.0
    land_pixel_count = 0.0

    with torch.no_grad():
        for batch in val_loader:
            if include_physical:
                Xn, aux, yn, y_raw, mask_raw = batch
                y_raw = y_raw.to(device, non_blocking=True)
                mask_raw = mask_raw.to(device, non_blocking=True)
            else:
                Xn, aux, yn = batch
                y_raw = None
                mask_raw = None

            Xn = Xn.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            yn = yn.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                y_pred_n = model(Xn, aux)
                loss = loss_fn(y_pred_n, yn)

            batch_size = Xn.shape[0]
            val_loss_sum += loss.item() * batch_size
            val_samples += batch_size

            if include_physical:
                y_pred = y_pred_n * y_std + y_mean
                squared_error = (y_pred - y_raw) ** 2
                full_se_sum += squared_error.sum().item()
                full_pixel_count += squared_error.numel()
                land_se_sum += (squared_error * mask_raw).sum().item()
                land_pixel_count += mask_raw.sum().item()

    result = {
        "val_loss": val_loss_sum / val_samples,
    }

    if include_physical:
        result["val_physical_mse_full_image"] = full_se_sum / full_pixel_count
        result["val_physical_mse_land_only"] = land_se_sum / land_pixel_count

    return result


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def train_one_trial(trial, args, data, save_root, device):
    set_seed(args.random_seed + trial.number)

    hparams = suggest_hparams(trial, args)
    trial_dir = os.path.join(save_root, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)
    trial_ckpt_path = os.path.join(trial_dir, "best_model.pth")

    train_loader, val_loader = make_loaders(
        data,
        hparams["batch_size"],
        args,
        include_raw_val=True,
    )

    model = make_model(hparams, args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )
    loss_fn = nn.MSELoss()
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    y_mean_dev = data["y_mean"].to(device)
    y_std_dev = data["y_std"].to(device)

    train_loss_list = []
    val_loss_list = []
    val_physical_full_list = []
    val_physical_land_list = []

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss_sum = 0.0
            train_samples = 0

            for Xn, aux, yn in train_loader:
                Xn = Xn.to(device, non_blocking=True)
                aux = aux.to(device, non_blocking=True)
                yn = yn.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    y_pred_n = model(Xn, aux)
                    loss = loss_fn(y_pred_n, yn)

                scaler.scale(loss).backward()

                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip_norm,
                    )

                scaler.step(optimizer)
                scaler.update()

                batch_size = Xn.shape[0]
                train_loss_sum += loss.item() * batch_size
                train_samples += batch_size

            train_loss = train_loss_sum / train_samples
            train_loss_list.append(train_loss)

            val_result = evaluate(
                model,
                val_loader,
                device,
                y_mean_dev,
                y_std_dev,
                loss_fn,
                use_amp=use_amp,
                include_physical=True,
            )
            val_loss = val_result["val_loss"]
            val_physical_full = val_result["val_physical_mse_full_image"]
            val_physical_land = val_result["val_physical_mse_land_only"]

            val_loss_list.append(val_loss)
            val_physical_full_list.append(val_physical_full)
            val_physical_land_list.append(val_physical_land)
            scheduler.step(val_loss)

            trial.report(val_loss, epoch)
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_val_loss", best_val_loss)
            trial.set_user_attr("checkpoint_path", trial_ckpt_path)

            if trial.should_prune():
                trial.set_user_attr("status", "pruned")
                raise optuna.TrialPruned()

            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0

                trial.set_user_attr("best_epoch", best_epoch)
                trial.set_user_attr("best_val_loss", best_val_loss)
                trial.set_user_attr(
                    "best_val_physical_mse_full_image",
                    val_physical_full,
                )
                trial.set_user_attr(
                    "best_val_physical_mse_land_only",
                    val_physical_land,
                )

                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_name": "RCAN_HR_Aux_ShallowFusion",
                        "input_file": args.input_file,
                        "target_file": args.target_file,
                        "hr_mask_file": args.hr_mask_file,
                        "hr_elevation_file": args.hr_elevation_file,
                        "model_state_dict": copy.deepcopy(model.state_dict()),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "best_val_physical_mse_full_image": val_physical_full,
                        "best_val_physical_mse_land_only": val_physical_land,
                        "train_loss_list": train_loss_list,
                        "val_loss_list": val_loss_list,
                        "val_physical_mse_full_image_list": val_physical_full_list,
                        "val_physical_mse_land_only_list": val_physical_land_list,
                        "X_mean": data["X_mean"],
                        "X_std": data["X_std"],
                        "y_mean": data["y_mean"],
                        "y_std": data["y_std"],
                        "config": build_config(args, hparams),
                    },
                    trial_ckpt_path,
                )
            else:
                epochs_no_improve += 1

            # if epoch % max(1, args.log_every) == 0:
            #     print(
            #         f"Trial {trial.number:04d} | "
            #         f"Epoch {epoch:03d} | "
            #         f"Train MSE: {train_loss:.6f} | "
            #         f"Val MSE: {val_loss:.6f} | "
            #         f"Val physical full: {val_physical_full:.6f} | "
            #         f"Val physical land: {val_physical_land:.6f} | "
            #         f"Best Val: {best_val_loss:.6f} | "
            #         f"Best Epoch: {best_epoch:03d} | "
            #         f"No improve: {epochs_no_improve}/"
            #         f"{args.early_stopping_patience} | "
            #         f"lr: {optimizer.param_groups[0]['lr']:.6e}"
            #     )

            if epochs_no_improve >= args.early_stopping_patience:
                print(
                    f"Early stopping trial {trial.number:04d} at epoch "
                    f"{epoch:03d}; best epoch {best_epoch:03d}."
                )
                break

    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial.set_user_attr("status", "oom_pruned")
            raise optuna.TrialPruned() from exc
        raise

    trial.set_user_attr("status", "completed")
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("best_val_loss", best_val_loss)
    trial.set_user_attr("checkpoint_path", trial_ckpt_path)

    return best_val_loss


def build_config(args, hparams):
    return {
        "rcm_var": args.rcm_var,
        "gcm_name": args.gcm_name,
        "rcm_name": args.rcm_name,
        "grid": args.grid,
        "rcm_product": args.rcm_product,
        "exp": args.exp,
        "factor": args.factor,
        "train_start_year": args.train_start_year,
        "train_end_year": args.train_end_year,
        "val_start_year": args.val_start_year,
        "val_end_year": args.val_end_year,
        "model_name": "RCAN_HR_Aux_ShallowFusion",
        "input_file": args.input_file,
        "target_file": args.target_file,
        "hr_mask_file": args.hr_mask_file,
        "hr_elevation_file": args.hr_elevation_file,
        "input_channels": 1,
        "hr_aux_channels": 2,
        "hr_aux_description": "channel0=HR land-sea mask; channel1=HR elevation",
        "loss": "mse",
        "objective": "normalized full-image validation MSE",
        "final_reporting": "physical MSE for full image and land only",
        "batch_norm": False,
        **hparams,
    }


def plot_best_model(best_trial, args, data, save_root, device):
    checkpoint_path = best_trial.user_attrs.get("checkpoint_path")
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print("No best checkpoint found for plotting.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    hparams = {
        "num_resblk": config["num_resblk"],
        "num_features": config["num_features"],
        "num_groups": config["num_groups"],
        "reduction": config["reduction"],
        "res_scale": config["res_scale"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
    }
    model = make_model(hparams, args).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_mean_dev = data["y_mean"].to(device)
    y_std_dev = data["y_std"].to(device)

    validation_plot_path = os.path.join(save_root, "validation_sample_prediction.png")
    validation_summary_plot_path = os.path.join(save_root, "validation_set_summary.png")
    validation_rmse_plot_path = os.path.join(save_root, "validation_set_rmse.png")

    with torch.no_grad():
        X_sample = data["X_val_n"][0:1].to(device)
        aux_sample = data["aux_val"][0:1].to(device)
        y_true = data["y_val"][0, 0].cpu().numpy()
        y_pred_n = model(X_sample, aux_sample)
        y_pred = (y_pred_n * y_std_dev + y_mean_dev)[0, 0].cpu().numpy()

    error = y_pred - y_true
    value_pixels = np.concatenate(
        [
            y_true[np.isfinite(y_true)].ravel(),
            y_pred[np.isfinite(y_pred)].ravel(),
        ]
    )
    vmin = float(value_pixels.min())
    vmax = float(value_pixels.max())
    err_abs = float(np.nanmax(np.abs(error)))
    if err_abs == 0:
        err_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    im0 = axes[0].imshow(y_true, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("HR RCM target")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(y_pred, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("RCAN shallow-fusion prediction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(error, origin="lower", cmap="coolwarm", vmin=-err_abs, vmax=err_abs)
    axes[2].set_title("Prediction - target")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("LR GCM to HR RCM, RCAN shallow-fusion first validation sample")
    fig.savefig(validation_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    batch_size = int(hparams["batch_size"])
    _, val_loader = make_loaders(data, batch_size, args, include_raw_val=True)

    sum_true = None
    sum_pred = None
    sum_sq_error = None
    total_samples = 0

    with torch.no_grad():
        for Xn, aux, yn, y_raw, mask_raw in val_loader:
            Xn = Xn.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            y_raw = y_raw.to(device, non_blocking=True)
            y_pred_n = model(Xn, aux)
            y_pred_batch = y_pred_n * y_std_dev + y_mean_dev
            error_batch = y_pred_batch - y_raw

            batch_sum_true = y_raw.cpu().sum(dim=0)
            batch_sum_pred = y_pred_batch.cpu().sum(dim=0)
            batch_sum_sq_error = (error_batch.cpu() ** 2).sum(dim=0)

            if sum_true is None:
                sum_true = batch_sum_true
                sum_pred = batch_sum_pred
                sum_sq_error = batch_sum_sq_error
            else:
                sum_true += batch_sum_true
                sum_pred += batch_sum_pred
                sum_sq_error += batch_sum_sq_error

            total_samples += Xn.shape[0]

    mean_true = (sum_true / total_samples)[0].numpy()
    mean_pred = (sum_pred / total_samples)[0].numpy()
    bias = mean_true - mean_pred
    rmse = torch.sqrt(sum_sq_error / total_samples)[0].numpy()

    summary_values = np.concatenate(
        [
            mean_true[np.isfinite(mean_true)].ravel(),
            mean_pred[np.isfinite(mean_pred)].ravel(),
        ]
    )
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
    axes[0].set_title("Mean HR RCM target")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(mean_pred, origin="lower", cmap="viridis", vmin=summary_vmin, vmax=summary_vmax)
    axes[1].set_title("Mean RCAN shallow-fusion prediction")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(bias, origin="lower", cmap="coolwarm", vmin=-bias_abs, vmax=bias_abs)
    axes[2].set_title("Bias: target - prediction")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("LR GCM to HR RCM, RCAN shallow-fusion validation mean fields")
    fig.savefig(validation_summary_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(rmse, origin="lower", cmap="magma", vmin=0.0, vmax=rmse_vmax)
    ax.set_title("Validation set RMSE")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("LR GCM to HR RCM, RCAN shallow-fusion validation set")
    fig.savefig(validation_rmse_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("Validation sample plot saved to:", validation_plot_path)
    print("Validation summary plot saved to:", validation_summary_plot_path)
    print("Validation RMSE plot saved to:", validation_rmse_plot_path)


def main():
    args = parse_args()
    set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = load_data(args)
    save_root = make_save_root(args, data["exp_folder_name"])
    print("All trial outputs will be saved to:", save_root)

    if args.study_name is None:
        study_name = (
            f"rcan_hr_aux_shallowfusion_{args.rcm_var}_{args.gcm_name}_{args.rcm_name}_"
            f"{args.grid}_{args.rcm_product}_{args.exp}"
        )
    else:
        study_name = args.study_name

    storage = args.storage
    if storage is None:
        storage_path = os.path.join(save_root, "optuna_study.db")
        storage = f"sqlite:///{storage_path}"

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=not args.no_resume_study,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
        ),
    )

    print("Optuna study:", study_name)
    print("Optuna storage:", storage)

    def objective(trial):
        return train_one_trial(trial, args, data, save_root, device)

    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    best_summary = {
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_epoch": best_trial.user_attrs.get("best_epoch"),
        "best_val_loss": best_trial.user_attrs.get("best_val_loss"),
        "best_val_physical_mse_full_image": best_trial.user_attrs.get(
            "best_val_physical_mse_full_image"
        ),
        "best_val_physical_mse_land_only": best_trial.user_attrs.get(
            "best_val_physical_mse_land_only"
        ),
        "best_checkpoint_path": best_trial.user_attrs.get("checkpoint_path"),
        "study_name": study_name,
        "storage": storage,
        "config": build_config(args, best_trial.params),
    }
    save_json(best_summary, os.path.join(save_root, "optuna_best_params.json"))

    all_trials_summary = []
    for t in study.trials:
        all_trials_summary.append(
            {
                "trial_number": t.number,
                "state": str(t.state),
                "value": t.value if t.value is not None else None,
                "params": t.params,
                "best_epoch": t.user_attrs.get("best_epoch"),
                "best_val_loss": t.user_attrs.get("best_val_loss"),
                "best_val_physical_mse_full_image": t.user_attrs.get(
                    "best_val_physical_mse_full_image"
                ),
                "best_val_physical_mse_land_only": t.user_attrs.get(
                    "best_val_physical_mse_land_only"
                ),
                "checkpoint_path": t.user_attrs.get("checkpoint_path"),
                "status": t.user_attrs.get("status"),
            }
        )
    save_json(
        all_trials_summary,
        os.path.join(save_root, "optuna_all_trials_summary.json"),
    )

    if not args.no_plots:
        plot_best_model(best_trial, args, data, save_root, device)

    print("\nOptuna study finished.")
    print("Best trial:", best_trial.number)
    print(f"Best normalized validation MSE: {best_trial.value:.6f}")
    print("Best params:", best_trial.params)
    print("Best checkpoint:", best_trial.user_attrs.get("checkpoint_path"))
    print("Best summary saved to:", os.path.join(save_root, "optuna_best_params.json"))
    print(
        "All trial summaries saved to:",
        os.path.join(save_root, "optuna_all_trials_summary.json"),
    )


if __name__ == "__main__":
    main()
