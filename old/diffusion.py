import os
import math
import argparse
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split


# ============================================================
# Utilities
# ============================================================

def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


def normalize(x, mean, std):
    return (x - mean) / (std + 1e-8)


def denormalize(x, mean, std):
    return x * (std + 1e-8) + mean


# ============================================================
# Sinusoidal timestep embedding
# ============================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B]
        return: [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================
# UNet blocks
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, groups=8):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(min(groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)

        time_emb = self.mlp(t_emb)
        h = h + time_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Conditional UNet for climate downscaling
# ============================================================

class ConditionalUNet(nn.Module):
    """
    Input:
        noisy HR field y_t:        [B, out_channels, H, W]
        upsampled LR condition x:  [B, cond_channels, H, W]
        timestep t:                [B]

    Output:
        predicted noise:           [B, out_channels, H, W]
    """

    def __init__(
        self,
        out_channels=1,
        cond_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=256
    ):
        super().__init__()

        self.out_channels = out_channels
        self.cond_channels = cond_channels

        input_channels = out_channels + cond_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        channels = [base_channels * m for m in channel_mults]

        self.init_conv = nn.Conv2d(input_channels, channels[0], kernel_size=3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels:
            self.downs.append(nn.ModuleList([
                ResBlock(in_ch, out_ch, time_emb_dim),
                ResBlock(out_ch, out_ch, time_emb_dim),
                Downsample(out_ch)
            ]))
            in_ch = out_ch

        # Bottleneck
        self.mid1 = ResBlock(channels[-1], channels[-1], time_emb_dim)
        self.mid2 = ResBlock(channels[-1], channels[-1], time_emb_dim)

        # Decoder
        self.ups = nn.ModuleList()
        for out_ch in reversed(channels):
            self.ups.append(nn.ModuleList([
                Upsample(in_ch),
                ResBlock(in_ch + out_ch, out_ch, time_emb_dim),
                ResBlock(out_ch, out_ch, time_emb_dim)
            ]))
            in_ch = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        )

    def forward(self, y_t, x_cond, t):
        """
        y_t:    noisy high-resolution field
        x_cond: low-resolution condition already upsampled to HR shape
        t:      diffusion timestep
        """

        x = torch.cat([y_t, x_cond], dim=1)
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        skips = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t_emb)
            x = block2(x, t_emb)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for upsample, block1, block2 in self.ups:
            x = upsample(x)

            skip = skips.pop()

            # Handle odd spatial sizes
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = block1(x, t_emb)
            x = block2(x, t_emb)

        return self.final(x)


# ============================================================
# Gaussian diffusion
# ============================================================

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        image_size=None,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2
    ):
        super().__init__()

        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

    def extract(self, a, t, x_shape):
        """
        Extract timestep-dependent coefficients.
        """
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, y_start, t, noise=None):
        """
        Forward diffusion process.
        """
        noise = default(noise, torch.randn_like(y_start))

        sqrt_alpha_bar = self.extract(self.sqrt_alphas_cumprod, t, y_start.shape)
        sqrt_one_minus_alpha_bar = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, y_start.shape
        )

        return sqrt_alpha_bar * y_start + sqrt_one_minus_alpha_bar * noise

    def p_losses(self, y_start, x_cond, t, mask=None):
        """
        Training loss.
        """
        noise = torch.randn_like(y_start)
        y_noisy = self.q_sample(y_start=y_start, t=t, noise=noise)

        pred_noise = self.model(y_noisy, x_cond, t)

        if mask is not None:
            # mask shape: [1, 1, H, W] or [B, 1, H, W]
            loss = ((noise - pred_noise) ** 2) * mask
            loss = loss.sum() / (mask.sum() * y_start.shape[1] + 1e-8)
        else:
            loss = F.mse_loss(pred_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, y, x_cond, t):
        """
        One reverse diffusion step.
        """
        betas_t = self.extract(self.betas, t, y.shape)
        sqrt_one_minus_alpha_bar_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, y.shape
        )
        sqrt_recip_alpha_t = self.extract(self.sqrt_recip_alphas, t, y.shape)

        pred_noise = self.model(y, x_cond, t)

        model_mean = sqrt_recip_alpha_t * (
            y - betas_t * pred_noise / sqrt_one_minus_alpha_bar_t
        )

        posterior_variance_t = self.extract(self.posterior_variance, t, y.shape)

        noise = torch.randn_like(y)
        nonzero_mask = (t != 0).float().reshape(y.shape[0], *((1,) * (len(y.shape) - 1)))

        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, x_cond, shape):
        """
        Generate HR climate field conditioned on LR input.

        x_cond: [B, C, H, W]
        shape:  [B, C_out, H, W]
        """
        device = x_cond.device
        y = torch.randn(shape, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            y = self.p_sample(y, x_cond, t)

        return y


# ============================================================
# Training script
# ============================================================

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs(args.save_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load tensors
    # Expected shapes:
    #   x_lr: [N, 1, H_lr, W_lr]
    #   y_hr: [N, 1, H_hr, W_hr]
    # --------------------------------------------------------
    x_lr = torch.load(args.lr_path).float()
    y_hr = torch.load(args.hr_path).float()

    print("LR shape:", x_lr.shape)
    print("HR shape:", y_hr.shape)

    if x_lr.ndim == 3:
        x_lr = x_lr.unsqueeze(1)
    if y_hr.ndim == 3:
        y_hr = y_hr.unsqueeze(1)

    # --------------------------------------------------------
    # Upsample LR to HR resolution
    # --------------------------------------------------------
    x_lr_up = F.interpolate(
        x_lr,
        size=y_hr.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # --------------------------------------------------------
    # Normalize input and target
    # --------------------------------------------------------
    x_mean = x_lr_up.mean()
    x_std = x_lr_up.std()

    y_mean = y_hr.mean()
    y_std = y_hr.std()

    x_lr_up = normalize(x_lr_up, x_mean, x_std)
    y_hr = normalize(y_hr, y_mean, y_std)

    # --------------------------------------------------------
    # Optional land-sea mask
    # --------------------------------------------------------
    mask = None
    if args.mask_path is not None:
        mask = torch.load(args.mask_path).float()

        if mask.ndim == 2:
            mask = mask[None, None, :, :]
        elif mask.ndim == 3:
            # if shape is [N, H, W], keep first time only
            mask = mask[0:1, None, :, :]        
        elif mask.ndim == 4:
            # if shape is [N, 1, H, W], keep first time only
            mask = mask[0:1, :, :, :]        
        if mask.shape[-2:] != y_hr.shape[-2:]:
            mask = F.interpolate(mask, size=y_hr.shape[-2:], mode="nearest")        
        print("Mask shape:", mask.shape)

    dataset = TensorDataset(x_lr_up, y_hr)

    n_total = len(dataset)
    n_train = int(args.train_frac * n_total)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    unet = ConditionalUNet(
        out_channels=1,
        cond_channels=1,
        base_channels=args.base_channels,
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=args.time_emb_dim
    ).to(device)

    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end
    ).to(device)

    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10
    )

    best_val_loss = np.inf
    best_epoch = -1
    patience_counter = 0

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        diffusion.train()
        train_loss = 0.0

        for x_cond, y in train_loader:
            x_cond = x_cond.to(device)
            y = y.to(device)

            b = y.shape[0]
            t = torch.randint(
                0,
                args.timesteps,
                (b,),
                device=device
            ).long()

            batch_mask = None
            if mask is not None:
                batch_mask = mask.to(device)
                if batch_mask.shape[0] == 1:
                    batch_mask = batch_mask.expand(b, -1, -1, -1)

            loss = diffusion.p_losses(
                y_start=y,
                x_cond=x_cond,
                t=t,
                mask=batch_mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * b

        train_loss /= len(train_loader.dataset)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        diffusion.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_cond, y in val_loader:
                x_cond = x_cond.to(device)
                y = y.to(device)

                b = y.shape[0]
                t = torch.randint(
                    0,
                    args.timesteps,
                    (b,),
                    device=device
                ).long()

                batch_mask = None
                if mask is not None:
                    batch_mask = mask.to(device)
                    if batch_mask.shape[0] == 1:
                        batch_mask = batch_mask.expand(b, -1, -1, -1)

                loss = diffusion.p_losses(
                    y_start=y,
                    x_cond=x_cond,
                    t=t,
                    mask=batch_mask
                )

                val_loss += loss.item() * b

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:04d} | "
            f"Train loss: {train_loss:.6e} | "
            f"Val loss: {val_loss:.6e}"
        )

        # ----------------------------------------------------
        # Save best checkpoint only
        # ----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            ckpt = {
                "epoch": epoch,
                "model_state_dict": diffusion.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "args": vars(args)
            }

            torch.save(ckpt, os.path.join(args.save_dir, "best_diffusion_downscaling.pt"))
            print(f"Saved best model at epoch {epoch}")

        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print("Early stopping triggered.")
            break

    print(f"Best epoch: {best_epoch}")
    print(f"Best val loss: {best_val_loss:.6e}")


# ============================================================
# Inference function
# ============================================================

@torch.no_grad()
def generate_downscaled_sample(
    ckpt_path,
    lr_tensor,
    device="cuda"
):
    """
    lr_tensor: coarse input, shape [B, 1, H_lr, W_lr]
    return: generated HR field in physical units
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    model = ConditionalUNet(
        out_channels=1,
        cond_channels=1,
        base_channels=args["base_channels"],
        channel_mults=(1, 2, 4, 8),
        time_emb_dim=args["time_emb_dim"]
    ).to(device)

    diffusion = GaussianDiffusion(
        model=model,
        timesteps=args["timesteps"],
        beta_start=args["beta_start"],
        beta_end=args["beta_end"]
    ).to(device)

    diffusion.load_state_dict(ckpt["model_state_dict"])
    diffusion.eval()

    lr_tensor = lr_tensor.float().to(device)

    if lr_tensor.ndim == 3:
        lr_tensor = lr_tensor.unsqueeze(1)

    # You need to set this to your HR target shape.
    # Example: if LR is 4x coarser:
    h_hr = lr_tensor.shape[-2] * 4
    w_hr = lr_tensor.shape[-1] * 4

    x_cond = F.interpolate(
        lr_tensor,
        size=(h_hr, w_hr),
        mode="bilinear",
        align_corners=False
    )

    x_cond = normalize(x_cond, ckpt["x_mean"].to(device), ckpt["x_std"].to(device))

    sample_norm = diffusion.sample(
        x_cond=x_cond,
        shape=(lr_tensor.shape[0], 1, h_hr, w_hr)
    )

    sample = denormalize(
        sample_norm,
        ckpt["y_mean"].to(device),
        ckpt["y_std"].to(device)
    )

    return sample


# ============================================================
# Arguments
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--lr_path", type=str, required=True)
    parser.add_argument("--hr_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)

    parser.add_argument("--save_dir", type=str, default="./diffusion_downscaling_ckpt")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_emb_dim", type=int, default=256)

    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)

