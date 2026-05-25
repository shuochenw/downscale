import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function


# =========================================================
# Project model backbones
# =========================================================
# Kept because they are imported by current training scripts/notebooks.
# Removed legacy/unused models to keep this file focused.

# model 2: SRResNet
class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels)
        )

    def forward(self, x):
        return x + self.seq(x)

class SubPixelConvBlock(nn.Module):
    def __init__(self, input_channel, upscale=2):
        super(SubPixelConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_channel, input_channel * (upscale ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x

class SRResNet(nn.Module):
    def __init__(self,
                 num_resblk,
                 num_features,
                 input_channels=1,
                 output_channels=1,
                 scale=4):
        super(SRResNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.resblk = nn.Sequential(*[ResBlock(num_features) for _ in range(num_resblk)])

        self.seq2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features)
        )

        # Create enough SubPixel blocks for the scale factor
        num_upsample_blocks = int(np.log2(scale))
        self.subpixconvblk = nn.Sequential(*[SubPixelConvBlock(num_features, upscale=2) for _ in range(num_upsample_blocks)])

        self.seq3 = nn.Conv2d(num_features, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.seq1(x)
        res = x
        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res
        x = self.subpixconvblk(x)
        x = self.seq3(x)
        return x


class ResBlockNoBN(nn.Module):
    def __init__(self, input_channels):
        super(ResBlockNoBN, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.seq(x)


class SRResNet_NoBN(nn.Module):
    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        scale=4,
    ):
        super(SRResNet_NoBN, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.resblk = nn.Sequential(
            *[ResBlockNoBN(num_features) for _ in range(num_resblk)]
        )

        self.seq2 = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        num_upsample_blocks = int(np.log2(scale))
        self.subpixconvblk = nn.Sequential(
            *[
                SubPixelConvBlock(num_features, upscale=2)
                for _ in range(num_upsample_blocks)
            ]
        )

        self.seq3 = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.seq1(x)
        res = x
        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res
        x = self.subpixconvblk(x)
        x = self.seq3(x)
        return x


class SRResNet_NEW(nn.Module):
    def __init__(self,
                 num_resblk,
                 num_features,
                 input_channels=1,
                 output_channels=1,
                 hr_aux_channels=2,
                 scale=4):
        super(SRResNet_NEW, self).__init__()

        self.hr_aux_channels = hr_aux_channels

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.resblk = nn.Sequential(*[ResBlock(num_features) for _ in range(num_resblk)])

        self.seq2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features)
        )

        num_upsample_blocks = int(np.log2(scale))
        if 2 ** num_upsample_blocks != scale:
            raise ValueError(f"scale must be a power of 2, but got scale={scale}")

        self.subpixconvblk = nn.Sequential(
            *[SubPixelConvBlock(num_features, upscale=2) for _ in range(num_upsample_blocks)]
        )

        self.seq3 = nn.Conv2d(num_features, output_channels, kernel_size=3, stride=1, padding=1)

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(output_channels + hr_aux_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x_lr, x_hr_aux):
        x = self.seq1(x_lr)
        res = x
        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res
        x = self.subpixconvblk(x)
        x = self.seq3(x)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled prediction shape {x.shape[-2:]}"
            )

        x = torch.cat([x, x_hr_aux], dim=1)
        x = self.hr_fusion(x)
        return x


# =========================================================
# EDSR without BatchNorm, with HR land-sea mask/elevation fusion
# =========================================================
class EDSRResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super(EDSRResidualBlock, self).__init__()

        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class EDSR_HR_Aux(nn.Module):
    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        res_scale=0.1,
    ):
        super(EDSR_HR_Aux, self).__init__()

        self.hr_aux_channels = hr_aux_channels

        num_upsample_blocks = int(np.log2(scale))
        if 2 ** num_upsample_blocks != scale:
            raise ValueError(f"scale must be a power of 2, but got scale={scale}")

        self.head = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        body = [
            EDSRResidualBlock(
                num_features,
                res_scale=res_scale,
            )
            for _ in range(num_resblk)
        ]
        body.append(
            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.body = nn.Sequential(*body)

        upsample_blocks = []
        for _ in range(num_upsample_blocks):
            upsample_blocks.extend(
                [
                    nn.Conv2d(
                        num_features,
                        num_features * 4,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        self.upsampler = nn.Sequential(*upsample_blocks)

        self.prediction_head = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                output_channels + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x_lr, x_hr_aux):
        x = self.head(x_lr)
        res = x
        x = self.body(x)
        x = x + res
        x = self.upsampler(x)
        x = self.prediction_head(x)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled prediction shape {x.shape[-2:]}"
            )

        x = torch.cat([x, x_hr_aux], dim=1)
        x = self.hr_fusion(x)
        return x




# =========================================================
# SRResNet with HR land-sea mask/elevation fusion
# =========================================================
class SRResNet_HR_Aux(nn.Module):
    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
    ):
        """
        input_channels:
            Number of LR climate input channels.
            Usually 1 for tas/pr/tmean/etc.

        hr_aux_channels:
            Number of HR auxiliary channels concatenated after upsampling.
            Usually 2:
                channel 0 = HR land-sea mask
                channel 1 = HR elevation

        scale:
            Upsampling factor, e.g. 4.
        """

        super(SRResNet_HR_Aux, self).__init__()

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        # LR feature extraction
        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.resblk = nn.Sequential(
            *[ResBlock(num_features) for _ in range(num_resblk)]
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
        )

        # LR -> HR upsampling
        num_upsample_blocks = int(np.log2(scale))

        if 2 ** num_upsample_blocks != scale:
            raise ValueError(
                f"scale must be a power of 2, but got scale={scale}"
            )

        self.subpixconvblk = nn.Sequential(
            *[
                SubPixelConvBlock(num_features, upscale=2)
                for _ in range(num_upsample_blocks)
            ]
        )

        # HR fusion after upsampling
        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                num_features,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Final HR prediction
        self.seq3 = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x_lr, x_hr_aux):
        """
        x_lr:
            Low-resolution climate input.
            Shape: [B, input_channels, H_lr, W_lr]

        x_hr_aux:
            High-resolution auxiliary input.
            Shape: [B, hr_aux_channels, H_hr, W_hr]
            Example channels:
                HR land-sea mask
                HR elevation
        """

        # LR feature extraction
        x = self.seq1(x_lr)
        res = x

        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res

        # Upsample climate features to HR
        x = self.subpixconvblk(x)

        # Check HR auxiliary shape
        if x_hr_aux.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {x.shape[-2:]}"
            )

        # Concatenate HR land-sea mask/elevation after upsampling
        x = torch.cat([x, x_hr_aux], dim=1)

        # HR refinement
        x = self.hr_fusion(x)

        # Final prediction
        x = self.seq3(x)

        return x


# =========================================================
# SRResNet with deeper HR feature fusion
# =========================================================
class HRFusionResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super(HRFusionResidualBlock, self).__init__()

        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class SRResNet_HR_Aux_DeepFusion(nn.Module):
    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        num_fusion_blocks=4,
        fusion_res_scale=0.1,
    ):
        super(SRResNet_HR_Aux_DeepFusion, self).__init__()

        self.hr_aux_channels = hr_aux_channels

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.resblk = nn.Sequential(
            *[ResBlock(num_features) for _ in range(num_resblk)]
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features),
        )

        num_upsample_blocks = int(np.log2(scale))
        if 2 ** num_upsample_blocks != scale:
            raise ValueError(f"scale must be a power of 2, but got scale={scale}")

        self.subpixconvblk = nn.Sequential(
            *[
                SubPixelConvBlock(num_features, upscale=2)
                for _ in range(num_upsample_blocks)
            ]
        )

        self.fusion_stem = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.fusion_body = nn.Sequential(
            *[
                HRFusionResidualBlock(
                    num_features,
                    res_scale=fusion_res_scale,
                )
                for _ in range(num_fusion_blocks)
            ]
        )

        self.fusion_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.seq3 = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x_lr, x_hr_aux):
        x = self.seq1(x_lr)
        res = x
        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res
        x = self.subpixconvblk(x)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {x.shape[-2:]}"
            )

        x = torch.cat([x, x_hr_aux], dim=1)
        x = self.fusion_stem(x)
        fusion_res = x
        x = self.fusion_tail(self.fusion_body(x))
        x = x + fusion_res
        x = self.seq3(x)
        return x




# =========================================================
# Climate RCAN
# =========================================================
class ClimateChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ClimateChannelAttention, self).__init__()

        hidden_channels = max(channels // reduction, 4)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class ClimateRCAB(nn.Module):
    """
    Residual channel-attention block for climate super-resolution.

    This follows the EDSR/RCAN practice of removing BatchNorm so the network
    can preserve absolute regression values instead of normalizing feature
    statistics inside each mini-batch.
    """

    def __init__(self, channels, reduction=8, res_scale=0.1):
        super(ClimateRCAB, self).__init__()

        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ClimateChannelAttention(channels, reduction=reduction),
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class ClimateResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks, reduction=8, res_scale=0.1):
        super(ClimateResidualGroup, self).__init__()

        blocks = [
            ClimateRCAB(
                channels,
                reduction=reduction,
                res_scale=res_scale,
            )
            for _ in range(num_blocks)
        ]

        blocks.append(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.body(x)


class ClimateUpsampler(nn.Module):
    def __init__(self, channels, scale):
        super(ClimateUpsampler, self).__init__()

        num_upsample_blocks = int(np.log2(scale))

        if 2 ** num_upsample_blocks != scale:
            raise ValueError(
                f"scale must be a power of 2, but got scale={scale}"
            )

        blocks = []

        for _ in range(num_upsample_blocks):
            blocks.extend(
                [
                    nn.Conv2d(
                        channels,
                        channels * 4,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.PixelShuffle(2),
                    nn.ReLU(inplace=True),
                ]
            )

        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        return self.body(x)


class ClimateRCAN(nn.Module):
    """
    Residual Channel Attention Network for LR-GCM to HR-RCM downscaling.

    Compared with the current SRResNet, this model is designed to be stronger
    for climate regression:
      - no BatchNorm, avoiding batch-statistic drift in physical fields;
      - channel attention for adaptive feature weighting;
      - residual groups with residual scaling for stable deeper training;
      - HR land-sea mask/elevation fusion after upsampling;
      - a learned bilinear skip path from LR inputs to HR output.
    """

    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        num_groups=4,
        reduction=8,
        res_scale=0.1,
    ):
        super(ClimateRCAN, self).__init__()

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        num_groups = max(1, min(num_groups, num_resblk))
        base_blocks = num_resblk // num_groups
        extra_blocks = num_resblk % num_groups
        blocks_per_group = [
            base_blocks + (1 if group_idx < extra_blocks else 0)
            for group_idx in range(num_groups)
        ]

        self.head = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.body = nn.Sequential(
            *[
                ClimateResidualGroup(
                    num_features,
                    group_blocks,
                    reduction=reduction,
                    res_scale=res_scale,
                )
                for group_blocks in blocks_per_group
            ]
        )

        self.body_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsampler = ClimateUpsampler(num_features, scale=scale)

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ClimateRCAB(
                num_features,
                reduction=reduction,
                res_scale=res_scale,
            ),
            ClimateRCAB(
                num_features,
                reduction=reduction,
                res_scale=res_scale,
            ),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        with torch.no_grad():
            self.skip.weight[:, 0, 0, 0] = 1.0

    def forward(self, x_lr, x_hr_aux):
        shallow = self.head(x_lr)
        deep = self.body_tail(self.body(shallow))
        features = shallow + deep

        hr_features = self.upsampler(features)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != hr_features.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {hr_features.shape[-2:]}"
            )

        hr_features = torch.cat([hr_features, x_hr_aux], dim=1)
        residual = self.tail(self.hr_fusion(hr_features))
        baseline = self.skip(
            F.interpolate(
                x_lr,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        )

        return baseline + residual


class ClimateRCAN_ShallowFusion(nn.Module):
    """
    RCAN backbone with shallow HR auxiliary fusion.

    This keeps the same LR RCAN trunk as ClimateRCAN, but after HR upsampling it
    concatenates HR mask/elevation and applies only a shallow convolutional head
    before the final prediction.
    """

    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        num_groups=4,
        reduction=8,
        res_scale=0.1,
    ):
        super(ClimateRCAN_ShallowFusion, self).__init__()

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        num_groups = max(1, min(num_groups, num_resblk))
        base_blocks = num_resblk // num_groups
        extra_blocks = num_resblk % num_groups
        blocks_per_group = [
            base_blocks + (1 if group_idx < extra_blocks else 0)
            for group_idx in range(num_groups)
        ]

        self.head = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.body = nn.Sequential(
            *[
                ClimateResidualGroup(
                    num_features,
                    group_blocks,
                    reduction=reduction,
                    res_scale=res_scale,
                )
                for group_blocks in blocks_per_group
            ]
        )

        self.body_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsampler = ClimateUpsampler(num_features, scale=scale)

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        with torch.no_grad():
            self.skip.weight[:, 0, 0, 0] = 1.0

    def forward(self, x_lr, x_hr_aux):
        shallow = self.head(x_lr)
        deep = self.body_tail(self.body(shallow))
        features = shallow + deep
        hr_features = self.upsampler(features)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != hr_features.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {hr_features.shape[-2:]}"
            )

        hr_features = torch.cat([hr_features, x_hr_aux], dim=1)
        residual = self.tail(self.hr_fusion(hr_features))
        baseline = self.skip(
            F.interpolate(
                x_lr,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        )

        return baseline + residual


# =========================================================
# RRDB with HR land-sea mask/elevation fusion
# =========================================================
class ClimateResidualDenseBlock(nn.Module):
    def __init__(self, num_features, growth_channels=32, res_scale=0.2):
        super(ClimateResidualDenseBlock, self).__init__()

        self.res_scale = res_scale
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(num_features, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_features + growth_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_features + 2 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(num_features + 3 * growth_channels, growth_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(num_features + 4 * growth_channels, num_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + self.res_scale * x5


class ClimateRRDB(nn.Module):
    def __init__(self, num_features, growth_channels=32, res_scale=0.2):
        super(ClimateRRDB, self).__init__()

        self.res_scale = res_scale
        self.rdb1 = ClimateResidualDenseBlock(
            num_features,
            growth_channels=growth_channels,
            res_scale=res_scale,
        )
        self.rdb2 = ClimateResidualDenseBlock(
            num_features,
            growth_channels=growth_channels,
            res_scale=res_scale,
        )
        self.rdb3 = ClimateResidualDenseBlock(
            num_features,
            growth_channels=growth_channels,
            res_scale=res_scale,
        )

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + self.res_scale * out


class RRDB_HR_Aux(nn.Module):
    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        growth_channels=32,
        res_scale=0.2,
        num_fusion_blocks=2,
    ):
        super(RRDB_HR_Aux, self).__init__()

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        self.head = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.body = nn.Sequential(
            *[
                ClimateRRDB(
                    num_features,
                    growth_channels=growth_channels,
                    res_scale=res_scale,
                )
                for _ in range(num_resblk)
            ]
        )

        self.body_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsampler = ClimateUpsampler(num_features, scale=scale)

        self.hr_fusion_stem = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.hr_fusion_body = nn.Sequential(
            *[
                ClimateRRDB(
                    num_features,
                    growth_channels=growth_channels,
                    res_scale=res_scale,
                )
                for _ in range(num_fusion_blocks)
            ]
        )

        self.hr_fusion_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.tail = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        with torch.no_grad():
            self.skip.weight[:, 0, 0, 0] = 1.0

    def forward(self, x_lr, x_hr_aux):
        shallow = self.head(x_lr)
        deep = self.body_tail(self.body(shallow))
        features = shallow + deep
        hr_features = self.upsampler(features)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != hr_features.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {hr_features.shape[-2:]}"
            )

        hr_features = torch.cat([hr_features, x_hr_aux], dim=1)
        fused = self.hr_fusion_stem(hr_features)
        fused_res = fused
        fused = self.hr_fusion_tail(self.hr_fusion_body(fused))
        residual = self.tail(fused + fused_res)

        baseline = self.skip(
            F.interpolate(
                x_lr,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        )

        return baseline + residual


# =========================================================
# Climate SwinIR
# =========================================================
def climate_window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, C)


def climate_window_reverse(windows, window_size, H, W):
    windows_per_sample = (H // window_size) * (W // window_size)
    B = windows.shape[0] // windows_per_sample
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)


class ClimateSwinMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2.0):
        super(ClimateSwinMLP, self).__init__()

        hidden_dim = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ClimateWindowAttention(nn.Module):
    def __init__(self, dim, window_size=5, num_heads=4):
        super(ClimateWindowAttention, self).__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by num_heads={num_heads}"
            )

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ]
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(
                B_ // num_windows,
                num_windows,
                self.num_heads,
                N,
                N,
            )
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class ClimateSwinBlock(nn.Module):
    def __init__(
        self,
        dim,
        window_size=5,
        shift_size=0,
        num_heads=4,
        mlp_ratio=2.0,
        res_scale=0.1,
    ):
        super(ClimateSwinBlock, self).__init__()

        if shift_size >= window_size:
            raise ValueError("shift_size must be smaller than window_size")

        self.window_size = window_size
        self.shift_size = shift_size
        self.res_scale = res_scale

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ClimateWindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ClimateSwinMLP(dim, mlp_ratio=mlp_ratio)

    def calculate_mask(self, H, W, device):
        if self.shift_size == 0:
            return None

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1

        mask_windows = climate_window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size

        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        actual_shift = self.shift_size
        if Hp <= self.window_size or Wp <= self.window_size:
            actual_shift = 0

        shortcut = x
        x_hw = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_hw)

        if actual_shift > 0:
            shifted_x = torch.roll(
                x_norm,
                shifts=(-actual_shift, -actual_shift),
                dims=(1, 2),
            )
            attn_mask = self.calculate_mask(Hp, Wp, x.device)
        else:
            shifted_x = x_norm
            attn_mask = None

        x_windows = climate_window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        shifted_x = climate_window_reverse(
            attn_windows,
            self.window_size,
            Hp,
            Wp,
        )

        if actual_shift > 0:
            x_hw = torch.roll(
                shifted_x,
                shifts=(actual_shift, actual_shift),
                dims=(1, 2),
            )
        else:
            x_hw = shifted_x

        x = shortcut + self.res_scale * x_hw.permute(0, 3, 1, 2).contiguous()

        x_hw = x.permute(0, 2, 3, 1).contiguous()
        x = x + self.res_scale * self.mlp(self.norm2(x_hw)).permute(
            0, 3, 1, 2
        ).contiguous()

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W].contiguous()

        return x


class ClimateResidualSwinGroup(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        window_size=5,
        num_heads=4,
        mlp_ratio=2.0,
        res_scale=0.1,
    ):
        super(ClimateResidualSwinGroup, self).__init__()

        blocks = []
        for block_idx in range(depth):
            shift_size = 0 if block_idx % 2 == 0 else window_size // 2
            blocks.append(
                ClimateSwinBlock(
                    dim,
                    window_size=window_size,
                    shift_size=shift_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    res_scale=res_scale,
                )
            )

        self.blocks = nn.Sequential(*blocks)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.conv(self.blocks(x))


class ClimateCBAMChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ClimateCBAMChannelAttention, self).__init__()

        hidden_channels = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_attention = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_attention = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * self.sigmoid(avg_attention + max_attention)


class ClimateCBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(ClimateCBAMSpatialAttention, self).__init__()

        if kernel_size not in [3, 7]:
            raise ValueError("CBAM spatial kernel_size must be 3 or 7")

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * attention


class ClimateCBAM(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel_size=7):
        super(ClimateCBAM, self).__init__()

        self.channel_attention = ClimateCBAMChannelAttention(
            channels,
            reduction=reduction,
        )
        self.spatial_attention = ClimateCBAMSpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class ClimateCBAMBlock(nn.Module):
    def __init__(
        self,
        channels,
        reduction=8,
        spatial_kernel_size=7,
        res_scale=0.1,
    ):
        super(ClimateCBAMBlock, self).__init__()

        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            ClimateCBAM(
                channels,
                reduction=reduction,
                spatial_kernel_size=spatial_kernel_size,
            ),
        )

    def forward(self, x):
        return x + self.res_scale * self.body(x)


class ClimateCBAMFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels,
        num_features,
        num_blocks=3,
        reduction=8,
        spatial_kernel_size=7,
        res_scale=0.1,
    ):
        super(ClimateCBAMFeatureExtractor, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.refine = nn.Sequential(
            *[
                ClimateCBAMBlock(
                    num_features,
                    reduction=reduction,
                    spatial_kernel_size=spatial_kernel_size,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        shallow = self.stem(x)
        return shallow + self.refine(shallow)


class ClimateSwinIR(nn.Module):
    """
    SwinIR-style downscaler with HR auxiliary fusion.

    LR climate is first processed by a CBAM convolutional feature extractor,
    then by shifted-window transformer blocks on the LR grid. The upsampled HR
    features are concatenated with HR land-sea mask and HR elevation before the
    final regression head.
    """

    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        window_size=5,
        num_heads=4,
        num_groups=2,
        num_cbam_blocks=3,
        mlp_ratio=2.0,
        res_scale=0.1,
    ):
        super(ClimateSwinIR, self).__init__()

        if num_features % num_heads != 0:
            valid_heads = [
                heads
                for heads in [8, 4, 2, 1]
                if num_features % heads == 0
            ]
            num_heads = valid_heads[0]

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        num_groups = max(1, min(num_groups, num_resblk))
        base_blocks = num_resblk // num_groups
        extra_blocks = num_resblk % num_groups
        blocks_per_group = [
            base_blocks + (1 if group_idx < extra_blocks else 0)
            for group_idx in range(num_groups)
        ]

        self.head = ClimateCBAMFeatureExtractor(
            input_channels,
            num_features,
            num_blocks=num_cbam_blocks,
            reduction=8,
            spatial_kernel_size=7,
            res_scale=res_scale,
        )

        self.body = nn.Sequential(
            *[
                ClimateResidualSwinGroup(
                    num_features,
                    depth=group_blocks,
                    window_size=window_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    res_scale=res_scale,
                )
                for group_blocks in blocks_per_group
            ]
        )

        self.body_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsampler = ClimateUpsampler(num_features, scale=scale)

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            ClimateRCAB(num_features, reduction=8, res_scale=res_scale),
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        with torch.no_grad():
            self.skip.weight[:, 0, 0, 0] = 1.0

    def forward(self, x_lr, x_hr_aux):
        shallow = self.head(x_lr)
        deep = self.body_tail(self.body(shallow))
        features = shallow + deep
        hr_features = self.upsampler(features)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != hr_features.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {hr_features.shape[-2:]}"
            )

        hr_features = torch.cat([hr_features, x_hr_aux], dim=1)
        residual = self.tail(self.hr_fusion(hr_features))
        baseline = self.skip(
            F.interpolate(
                x_lr,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        )

        return baseline + residual




# =========================================================
# Gradient Reversal Layer
# =========================================================
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None



class DomainClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, num_domains=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ClimateRCAN_ShallowFusion_DANN(nn.Module):
    """
    Shallow-fusion RCAN with a DANN domain classifier on LR feature maps.

    The prediction path matches ClimateRCAN_ShallowFusion:
      LR input -> RCAN trunk -> pixel-shuffle upsampling -> concat HR aux
      -> shallow fusion conv -> final HR prediction.

    The domain path branches from the LR trunk features before upsampling:
      LR features -> gradient reversal -> domain classifier.
    """

    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        num_groups=4,
        reduction=8,
        res_scale=0.1,
        domain_hidden_dim=128,
        num_domains=2,
    ):
        super(ClimateRCAN_ShallowFusion_DANN, self).__init__()

        self.scale = scale
        self.hr_aux_channels = hr_aux_channels

        num_groups = max(1, min(num_groups, num_resblk))
        base_blocks = num_resblk // num_groups
        extra_blocks = num_resblk % num_groups
        blocks_per_group = [
            base_blocks + (1 if group_idx < extra_blocks else 0)
            for group_idx in range(num_groups)
        ]

        self.head = nn.Conv2d(
            input_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.body = nn.Sequential(
            *[
                ClimateResidualGroup(
                    num_features,
                    group_blocks,
                    reduction=reduction,
                    res_scale=res_scale,
                )
                for group_blocks in blocks_per_group
            ]
        )

        self.body_tail = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.upsampler = ClimateUpsampler(num_features, scale=scale)

        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                num_features + hr_aux_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.tail = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.skip = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

        with torch.no_grad():
            self.skip.weight[:, 0, 0, 0] = 1.0

        self.domain_classifier = DomainClassifier(
            in_channels=num_features,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
        )

    def extract_features(self, x_lr):
        shallow = self.head(x_lr)
        deep = self.body_tail(self.body(shallow))
        return shallow + deep

    def predict_from_features(self, features, x_lr, x_hr_aux):
        hr_features = self.upsampler(features)

        if x_hr_aux is None:
            raise ValueError("x_hr_aux is required when predict=True.")

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != hr_features.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"upsampled feature shape {hr_features.shape[-2:]}"
            )

        hr_features = torch.cat([hr_features, x_hr_aux], dim=1)
        residual = self.tail(self.hr_fusion(hr_features))
        baseline = self.skip(
            F.interpolate(
                x_lr,
                scale_factor=self.scale,
                mode="bilinear",
                align_corners=False,
            )
        )
        return baseline + residual

    def classify_from_features(self, features, lambda_grl):
        reversed_features = GradientReversalFunction.apply(features, lambda_grl)
        return self.domain_classifier(reversed_features)

    def forward(
        self,
        x_lr,
        x_hr_aux=None,
        lambda_grl=0.0,
        predict=True,
        classify_domain=True,
    ):
        features = self.extract_features(x_lr)

        y_pred = None
        if predict:
            y_pred = self.predict_from_features(features, x_lr, x_hr_aux)

        domain_logits = None
        if classify_domain:
            domain_logits = self.classify_from_features(features, lambda_grl)

        return y_pred, domain_logits


class ConditionalDomainClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        num_domains=2,
        condition_dim=2,
    ):
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels + self.condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, condition=None):
        x = self.features(x)
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError("Domain condition is required for conditional DANN.")
            if condition.shape[0] != x.shape[0] or condition.shape[1] != self.condition_dim:
                raise ValueError(
                    f"Expected domain condition shape [B, {self.condition_dim}], "
                    f"got {tuple(condition.shape)}"
                )
            x = torch.cat([x, condition], dim=1)
        return self.classifier(x)


class ConditionalVGGDomainClassifier(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim=128,
        num_domains=2,
        condition_dim=2,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.condition_dim = int(condition_dim)
        c1 = max(16, hidden_dim)
        c2 = max(32, hidden_dim)
        c3 = max(64, hidden_dim * 2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c3 + self.condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, x, condition=None):
        x = self.features(x)
        if self.condition_dim > 0:
            if condition is None:
                raise ValueError("Domain condition is required for conditional DANN.")
            if condition.shape[0] != x.shape[0] or condition.shape[1] != self.condition_dim:
                raise ValueError(
                    f"Expected domain condition shape [B, {self.condition_dim}], "
                    f"got {tuple(condition.shape)}"
                )
            condition = condition.to(device=x.device, dtype=x.dtype)
            x = torch.cat([x, condition], dim=1)
        return self.classifier(x)


class ClimateRCAN_ShallowFusion_DANN_new(ClimateRCAN_ShallowFusion_DANN):
    """
    Conditional DANN variant of ClimateRCAN_ShallowFusion_DANN.

    The prediction path is unchanged. The domain classifier is a VGG-style
    convolutional head over LR trunk features, with an optional low-dimensional
    condition concatenated before the final classifier.
    """

    def __init__(
        self,
        num_resblk,
        num_features,
        input_channels=1,
        output_channels=1,
        hr_aux_channels=2,
        scale=4,
        num_groups=4,
        reduction=8,
        res_scale=0.1,
        domain_hidden_dim=128,
        num_domains=2,
        domain_condition_dim=2,
    ):
        super().__init__(
            num_resblk=num_resblk,
            num_features=num_features,
            input_channels=input_channels,
            output_channels=output_channels,
            hr_aux_channels=hr_aux_channels,
            scale=scale,
            num_groups=num_groups,
            reduction=reduction,
            res_scale=res_scale,
            domain_hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
        )
        self.domain_condition_dim = int(domain_condition_dim)
        self.domain_classifier = ConditionalVGGDomainClassifier(
            in_channels=num_features,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains,
            condition_dim=self.domain_condition_dim,
        )

    def classify_from_features(self, features, lambda_grl, domain_condition=None):
        reversed_features = GradientReversalFunction.apply(features, lambda_grl)
        return self.domain_classifier(reversed_features, domain_condition)

    def forward(
        self,
        x_lr,
        x_hr_aux=None,
        lambda_grl=0.0,
        predict=True,
        classify_domain=True,
        domain_condition=None,
    ):
        features = self.extract_features(x_lr)

        y_pred = None
        if predict:
            y_pred = self.predict_from_features(features, x_lr, x_hr_aux)

        domain_logits = None
        if classify_domain:
            domain_logits = self.classify_from_features(
                features,
                lambda_grl,
                domain_condition=domain_condition,
            )

        return y_pred, domain_logits


class SRResNet_DANN(nn.Module):
    def __init__(self,
                 num_resblk,
                 num_features,
                 input_channels=1,
                 output_channels=1,
                 scale=4,
                 num_domains=2,
                 domain_hidden_dim=128,
                 lambda_grl=1.0):
        super(SRResNet_DANN, self).__init__()

        if scale < 1 or (scale & (scale - 1)) != 0:
            raise ValueError("scale must be a power of 2, e.g. 2, 4, 8")

        self.lambda_grl = lambda_grl

        self.seq1 = nn.Sequential(
            nn.Conv2d(input_channels, num_features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.resblk = nn.Sequential(
            *[ResBlock(num_features) for _ in range(num_resblk)]
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features)
        )

        num_upsample_blocks = int(np.log2(scale))
        self.subpixconvblk = nn.Sequential(
            *[SubPixelConvBlock(num_features, upscale=2) for _ in range(num_upsample_blocks)]
        )

        self.seq3 = nn.Conv2d(
            num_features,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.domain_classifier = DomainClassifier(
            in_channels=num_features,
            hidden_dim=domain_hidden_dim,
            num_domains=num_domains
        )

    def extract_features(self, x):
        x = self.seq1(x)
        res = x
        x = self.resblk(x)
        x = self.seq2(x)
        x = x + res
        return x

    def sr_head(self, feat):
        x = self.subpixconvblk(feat)
        x = self.seq3(x)
        return x

    def domain_head(self, feat, lambda_grl=None):
        if lambda_grl is None:
            lambda_grl = self.lambda_grl
        reversed_feat = GradientReversalFunction.apply(feat, lambda_grl)
        domain_logits = self.domain_classifier(reversed_feat)
        return domain_logits

    def forward(self, x, lambda_grl=None):
        feat = self.extract_features(x)
        sr_out = self.sr_head(feat)
        domain_logits = self.domain_head(feat, lambda_grl=lambda_grl)
        return sr_out, domain_logits






