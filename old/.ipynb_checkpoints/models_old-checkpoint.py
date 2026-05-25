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
# Constants
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
SCALING_FACTOR = 4
NUM_FEATURES = 64
class ResBlock(nn.Module):
    def __init__(self, input_channels=NUM_FEATURES):
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
    def __init__(self, input_channel=NUM_FEATURES, upscale=2):
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
                 num_resblk=16,
                 input_channels=INPUT_CHANNELS,
                 num_features=NUM_FEATURES,
                 output_channels=OUTPUT_CHANNELS,
                 scale=SCALING_FACTOR):
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
        