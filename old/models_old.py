import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import functools

# model 1: ESPCN
class ESPCNx4(nn.Module):
    def __init__(self):
        super(ESPCNx4, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1 * 16, kernel_size=3, padding=1)  # upscale 4× -> 4×4 = 16
        )
        self.upsampler = nn.PixelShuffle(upscale_factor=4)  # (C * r^2, H, W) -> (C, H*r, W*r)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.upsampler(x)
        return x


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


# new model
# =========================================================
# Residual block
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, input_channels):
        super(ResBlock, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_channels),
        )

    def forward(self, x):
        return x + self.seq(x)
# class ResBlock(nn.Module):
#     def __init__(self, input_channels, res_scale=0.1):
#         super(ResBlock, self).__init__()

#         self.res_scale = res_scale

#         self.seq = nn.Sequential(
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1),
#         )

#     def forward(self, x):
#         return x + self.res_scale * self.seq(x)


# =========================================================
# Sub-pixel upsampling block
# =========================================================
class SubPixelConvBlock(nn.Module):
    def __init__(self, input_channel, upscale=2):
        super(SubPixelConvBlock, self).__init__()

        self.conv = nn.Conv2d(
            input_channel,
            input_channel * (upscale ** 2),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
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


class ClimateLRBiasCorrectionNet(nn.Module):
    """
    Low-resolution GCM-to-RCM bias-correction network.

    The input is the normalized LR GCM field. The output is a normalized
    LR RCM-like field on the same grid, supervised by the coarsened RCM target
    in the training script.
    """

    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        num_features=64,
        num_blocks=4,
        reduction=8,
        res_scale=0.1,
    ):
        super(ClimateLRBiasCorrectionNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                input_channels,
                num_features,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.body = nn.Sequential(
            *[
                ClimateCBAMBlock(
                    num_features,
                    reduction=reduction,
                    spatial_kernel_size=7,
                    res_scale=res_scale,
                )
                for _ in range(num_blocks)
            ],
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
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

    def forward(self, x_lr):
        shallow = self.head(x_lr)
        corrected_features = shallow + self.body(shallow)
        correction = self.tail(corrected_features)
        return self.skip(x_lr) + correction


class ClimateTwoStageSwinIR(nn.Module):
    """
    Two-stage climate downscaler.

    Stage 1: LR GCM -> LR RCM bias correction on the coarse grid.
    Stage 2: LR RCM -> HR RCM super-resolution with HR mask/elevation fusion.
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
        num_bias_blocks=4,
        mlp_ratio=2.0,
        res_scale=0.1,
    ):
        super(ClimateTwoStageSwinIR, self).__init__()

        self.bias_corrector = ClimateLRBiasCorrectionNet(
            input_channels=input_channels,
            output_channels=output_channels,
            num_features=num_features,
            num_blocks=num_bias_blocks,
            reduction=8,
            res_scale=res_scale,
        )

        self.upscaler = ClimateSwinIR(
            num_resblk=num_resblk,
            num_features=num_features,
            input_channels=output_channels,
            output_channels=output_channels,
            hr_aux_channels=hr_aux_channels,
            scale=scale,
            window_size=window_size,
            num_heads=num_heads,
            num_groups=num_groups,
            num_cbam_blocks=num_cbam_blocks,
            mlp_ratio=mlp_ratio,
            res_scale=res_scale,
        )

    def forward(self, x_lr, x_hr_aux, return_intermediate=False):
        lr_rcm = self.bias_corrector(x_lr)
        hr_rcm = self.upscaler(lr_rcm, x_hr_aux)

        if return_intermediate:
            return hr_rcm, lr_rcm

        return hr_rcm


# vision transformer
# =========================================================
# Vision Transformer model
# =========================================================
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]

        B, C, Hp, Wp = x.shape

        x = x.flatten(2)       # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]

        return x, Hp, Wp


class ViTDownscaler(nn.Module):
    """
    Vision Transformer downscaler.

    Workflow:
        LR input channels
            -> bilinear upsampling to HR grid
            -> patch embedding
            -> Transformer encoder
            -> CNN decoder
            -> HR prediction

    Input:
        x: [B, C_in, H_lr, W_lr]

    Output:
        y_pred: [B, 1, H_hr, W_hr]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        hr_img_size,
        patch_size=4,
        embed_dim=128,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        decoder_features=64,
    ):
        super().__init__()

        self.hr_img_size = tuple(hr_img_size)
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        H_hr, W_hr = self.hr_img_size

        if H_hr % patch_size != 0:
            raise ValueError(
                f"HR height {H_hr} must be divisible by patch_size={patch_size}"
            )

        if W_hr % patch_size != 0:
            raise ValueError(
                f"HR width {W_hr} must be divisible by patch_size={patch_size}"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.Hp = H_hr // patch_size
        self.Wp = W_hr // patch_size
        num_patches = self.Hp * self.Wp

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_features, decoder_features, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(decoder_features, out_channels, kernel_size=3, padding=1),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, C_in, H_lr, W_lr]

        # Upsample LR input channels to HR grid before ViT patch embedding.
        x = F.interpolate(
            x,
            size=self.hr_img_size,
            mode="bilinear",
            align_corners=False,
        )

        # Patch embedding
        tokens, Hp, Wp = self.patch_embed(x)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        # Transformer encoder
        tokens = self.transformer(tokens)

        # Tokens -> feature map
        x = tokens.transpose(1, 2)
        x = x.reshape(x.shape[0], self.embed_dim, Hp, Wp)

        # CNN decoder on patch-resolution feature map
        x = self.decoder(x)

        # Restore HR grid
        x = F.interpolate(
            x,
            size=self.hr_img_size,
            mode="bilinear",
            align_corners=False,
        )

        return x





import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

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

# =========================================================
# model blocks
# =========================================================
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
        self.conv = nn.Conv2d(
            input_channel,
            input_channel * (upscale ** 2),
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


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







# model 3: UNetSuperResolution from GPT
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNetSuperResolution(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)        # [B, 64, 14, 30]
        self.enc2 = ConvBlock(base_ch, base_ch * 2)        # [B, 128, 14, 30]
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)    # [B, 256, 14, 30]

        # Bottleneck (optional extra conv layer)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(base_ch * 4 + base_ch * 4, base_ch * 2, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(base_ch * 2 + base_ch * 2, base_ch, 3, padding=1),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(base_ch + base_ch, base_ch, 3, padding=1),
            nn.ReLU()
        )

        # Final upsampling to HR
        self.upsample = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),  # [B, base_ch, 28, 60]
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),  # [B, base_ch, 56, 120]
            nn.ReLU(),
            nn.Conv2d(base_ch, out_channels, 3, padding=1)
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)     # [B, 64, 14, 30]
        x2 = self.enc2(x1)    # [B, 128, 14, 30]
        x3 = self.enc3(x2)    # [B, 256, 14, 30]

        # Bottleneck
        x3b = self.bottleneck(x3)

        # Decoder (concat skip connections)
        x = self.up1(torch.cat([x3b, x3], dim=1))  # [B, 128, 14, 30]
        x = self.up2(torch.cat([x, x2], dim=1))    # [B, 64, 14, 30]
        x = self.up3(torch.cat([x, x1], dim=1))    # [B, 64, 14, 30]

        # Upsample to HR
        x = self.upsample(x)                       # [B, 1, 56, 120]
        return x



# model 4: UNet from Regional climate model emulator based on deep learning: concept and first evaluation of a novel hybrid downscaling approach
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)        # -> [B, 64, 14, 30]
        self.pool1 = nn.MaxPool2d(2)                        # -> [B, 64, 7, 15]
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # -> [B, 128, 7, 15]
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)        # -> [B, 128, 4, 8]

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # -> [B, 256, 4, 8]

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)  # -> [B, 128, 8, 16]
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # skip + up -> [B, 128, 7, 15]
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)  # -> [B, 64, 14, 30]
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # super-resolution
        self.transconv1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.transconv2 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_ch, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x1 = self.enc1(x)       # [B, 64, 14, 30]
        x2 = self.enc2(self.pool1(x1))  # [B, 128, 7, 15]
        x3 = self.bottleneck(self.pool2(x2))  # [B, 256, 4, 8]

        x = self.up2(x3)        # [B, 128, 8, 16]
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))  # [B, 128, 7, 15]

        x = self.up1(x)         # [B, 64, 14, 30]
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1))  # [B, 64, 14, 30]

        x = self.transconv1(x)
        x = self.conv1(x)
        x = self.transconv2(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return  x


class UNet_HR_Aux(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_ch=64,
        hr_aux_channels=2,
        scale=4,
    ):
        super().__init__()

        if scale != 4:
            raise ValueError(
                "UNet_HR_Aux follows the original UNet 4x upsampling design; "
                f"got scale={scale}"
            )

        self.hr_aux_channels = hr_aux_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(
            base_ch * 4,
            base_ch * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(
            base_ch * 2,
            base_ch,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # Super-resolution
        self.transconv1 = nn.ConvTranspose2d(
            base_ch,
            base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.transconv2 = nn.ConvTranspose2d(
            base_ch,
            base_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)

        # HR land-sea mask/elevation fusion before final prediction.
        self.hr_fusion = nn.Sequential(
            nn.Conv2d(
                base_ch + hr_aux_channels,
                base_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Conv2d(base_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x_lr, x_hr_aux):
        x1 = self.enc1(x_lr)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))

        x = self.up2(x3)
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1))

        x = self.transconv1(x)
        x = self.conv1(x)
        x = self.transconv2(x)
        x = self.conv2(x)

        if x_hr_aux.shape[1] != self.hr_aux_channels:
            raise ValueError(
                f"Expected {self.hr_aux_channels} HR auxiliary channels, "
                f"got {x_hr_aux.shape[1]}"
            )

        if x_hr_aux.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"HR auxiliary shape {x_hr_aux.shape[-2:]} does not match "
                f"UNet HR feature shape {x.shape[-2:]}"
            )

        x = torch.cat([x, x_hr_aux], dim=1)
        x = self.hr_fusion(x)
        return self.conv3(x)


# model 5: YNet
SCALING_FACTOR = 4
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
NUM_FEATURES = 64

class YNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=NUM_FEATURES, input_channels=INPUT_CHANNELS, output_channels=OUTPUT_CHANNELS, scale=SCALING_FACTOR):
        super(YNet30, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.scale = scale

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(self.input_channels, self.num_features, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(self.num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(self.num_features, self.num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(self.num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True),
                                               nn.Conv2d(self.num_features,self.num_features,kernel_size=3,padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(self.num_features, self.num_features, kernel_size=3, stride=1, padding=1, output_padding=0),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(self.num_features,self.input_channels,kernel_size=3,stride=1,padding=1)))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

        self.subpixel_conv_layer = nn.Sequential(nn.Conv2d(self.input_channels,self.input_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True),
                                                 nn.Upsample(scale_factor=self.scale,mode='bilinear',align_corners=False),
                                                 nn.Conv2d(self.input_channels,self.output_channels,kernel_size=3,stride=1,padding=1),
                                                 nn.ReLU(inplace=True))
    
        self.fusion_layer = nn.Sequential(nn.Conv2d(input_channels,self.num_features,kernel_size=3,stride=1,padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(self.num_features,self.output_channels,kernel_size=1,stride=1,padding=0))

    def forward(self, x):
        residual = x
        
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)
        
        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
                
        x = x+residual
        x = self.relu(x)
        x = self.subpixel_conv_layer(x)
        x = self.fusion_layer(x)

        return x

# model 6: YNetImproved from GPT
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class PixelShuffleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class YNetImproved(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_features=64, scale=4):
        super().__init__()
        self.scale = scale
        self.encoder1 = ConvBlock(input_channels, num_features, downsample=False)  # [B, 64, 14, 30]
        self.encoder2 = ConvBlock(num_features, num_features * 2, downsample=True) # [B, 128, 7, 15]
        self.encoder3 = ConvBlock(num_features * 2, num_features * 4, downsample=True) # [B, 256, 4, 8]

        self.bottleneck = ConvBlock(num_features * 4, num_features * 4, downsample=False)

        self.decoder3 = ConvBlock(num_features * 4 + num_features * 4, num_features * 2)
        self.decoder2 = ConvBlock(num_features * 2 + num_features * 2, num_features)
        self.decoder1 = ConvBlock(num_features + num_features, num_features)

        # Upsampling (4× = 2× followed by 2×)
        self.upsample = nn.Sequential(
            PixelShuffleBlock(num_features, upscale_factor=2),   # [B, 64, 28, 60]
            PixelShuffleBlock(num_features, upscale_factor=2),   # [B, 64, 56, 120]
        )

        self.final_conv = nn.Conv2d(num_features, output_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)   # [B, 64, 14, 30]
        x2 = self.encoder2(x1)  # [B, 128, 7, 15]
        x3 = self.encoder3(x2)  # [B, 256, 4, 8]

        # Bottleneck
        xb = self.bottleneck(x3)

        # Decoder with skip connections
        d3 = self.decoder3(torch.cat([xb, x3], dim=1))  # [B, 128, 4, 8]
        d3_up = nn.functional.interpolate(d3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.decoder2(torch.cat([d3_up, x2], dim=1))  # [B, 64, 7, 15]
        d2_up = nn.functional.interpolate(d2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.decoder1(torch.cat([d2_up, x1], dim=1))  # [B, 64, 14, 30]

        # Upsample to high-res
        up = self.upsample(d1)                          # [B, 64, 56, 120]
        out = self.final_conv(up)                       # [B, 1, 56, 120]
        return out

# model 7: DeepSD from GPT
class SRCNNBlock(nn.Module):
    def __init__(self, in_channels=1, mid_channels=64, out_channels=1):
        super(SRCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.block(x)

class DeepSD(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, upscale_factor=4):
        super(DeepSD, self).__init__()
        assert upscale_factor == 4, "This DeepSD version only supports 4× upscaling."

        self.upscale_factor = upscale_factor

        # Stage 1: 2× upsampling + SRCNN
        self.upsample_2x_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.srcnn1 = SRCNNBlock(in_channels=in_channels, out_channels=out_channels)

        # Stage 2: 2× upsampling + SRCNN
        self.upsample_2x_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.srcnn2 = SRCNNBlock(in_channels=out_channels, out_channels=out_channels)

    def forward(self, x):
        x = self.upsample_2x_1(x)  # [B, 1, 28, 60]
        x = self.srcnn1(x)
        x = self.upsample_2x_2(x)  # [B, 1, 56, 120]
        x = self.srcnn2(x)
        return x


# model 8: RRDB https://github.com/xinntao/ESRGAN/blob/master/RRDBNet_arch.py
def make_layer_RRDB(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB_blk(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB_blk, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, in_nc=1, out_nc=1, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB_blk, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = make_layer_RRDB(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out




# UNet encoder decoder with VGG domain classifier and identity decoder loss

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_ch=64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch)        # -> [B, 64, 14, 30]
        self.pool1 = nn.MaxPool2d(2)                        # -> [B, 64, 7, 15]
        self.enc2 = ConvBlock(base_ch, base_ch * 2)         # -> [B, 128, 7, 15]
        self.pool2 = nn.MaxPool2d(2, ceil_mode=True)        # -> [B, 128, 4, 8]

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 2, base_ch * 4)  # -> [B, 256, 4, 8]

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)  # -> [B, 128, 8, 16]
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)  # skip + up -> [B, 128, 7, 15]
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)  # -> [B, 64, 14, 30]
        self.dec1 = ConvBlock(base_ch * 2, base_ch)
        
    def forward(self, x):
        x1 = self.enc1(x)       # [B, 64, 14, 30]
        x2 = self.enc2(self.pool1(x1))  # [B, 128, 7, 15]
        x3 = self.bottleneck(self.pool2(x2))  # [B, 256, 4, 8]

        x = self.up2(x3)        # [B, 128, 8, 16]
        x = F.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))  # [B, 128, 7, 15]

        x = self.up1(x)         # [B, 64, 14, 30]
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x1], dim=1))  # [B, 64, 14, 30]
        
        return  x

class Decoder(nn.Module):
    def __init__(self,base_ch=64, out_channels=1):
        super().__init__()
        # super-resolution
        self.transconv1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.transconv2 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(base_ch, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.transconv1(x)
        x = self.conv1(x)
        x = self.transconv2(x)
        x = self.conv2(x)
        x = self.conv3(x)      
        return x
        
from grl import GradientReversal
class VGGDomainClassifier(nn.Module):
    def __init__(self, in_channels=64):
        super(VGGDomainClassifier, self).__init__()
        self.grl = GradientReversal()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [B, 64, 7, 15]

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # [B, 128, 4, 8]

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),  # [B, 256, 2, 4]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Flatten(),             # [B, 256]
            nn.Linear(256, 1)         # [B, 1] — domain prediction
        )

    def forward(self, x):
        x = self.grl(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

class Decoder_Identity(nn.Module):
    def __init__(self):
        super(Decoder_Identity, self).__init__()

        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, bias=True),            
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1, bias=True),
            nn.ReLU()
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, bias=True),
            nn.ReLU()
        )

    def forward(self, feat):
        featmap_2 = self.conv_up_2(feat)
        featmap_1 = self.conv_up_1(featmap_2)
        out = self.conv_last(featmap_1)

        return out

# Decoder_Identity_ResNet50
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Decoder_Identity_ResNet(nn.Module):
    def __init__(self, in_channels=64, out_channels=1):
        super(Decoder_Identity_ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Fewer blocks and channels
        self.layer1 = self._make_layer(32, 32, blocks=1)
        self.layer2 = self._make_layer(128, 64, blocks=1)  # in_channels = 32 * 4
        self.layer3 = self._make_layer(256, 64, blocks=1)  # reduce depth

        self.final_conv = nn.Conv2d(256, out_channels, kernel_size=1)

    def _make_layer(self, in_channels, mid_channels, blocks):
        layers = []

        downsample = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels * Bottleneck.expansion)
        )

        layers.append(Bottleneck(in_channels, mid_channels, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(Bottleneck(mid_channels * Bottleneck.expansion, mid_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))    # [B, 32, 14, 30]
        x = self.layer1(x)                        # [B, 128, 14, 30]
        x = self.layer2(x)                        # [B, 256, 14, 30]
        x = self.layer3(x)                        # [B, 256, 14, 30]
        x = self.final_conv(x)                    # [B, 1, 14, 30]
        return x


# try simple encoder decoder and perform transfer learning on the first few layers
class Encoder_Simple(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(Encoder_Simple, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder_Simple(nn.Module):
    def __init__(self, base_channels=64, out_channels=1, upscale_factor=4):
        super(Decoder_Simple, self).__init__()
        # We do two PixelShuffle layers, each with upscale_factor=2
        self.decoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x) 










# https://github.com/anse3832/USR_DA/blob/main/model/decoder.py
from torch.nn import init as init

def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class Decoder_Id_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=1, scale=4, num_feat=64, num_block=10, num_grow_ch=32):
        super(Decoder_Id_RRDB, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class Decoder_SR_RRDB(nn.Module):
    def __init__(self, num_in_ch, num_out_ch=1, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(Decoder_SR_RRDB, self).__init__()

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out    
        
