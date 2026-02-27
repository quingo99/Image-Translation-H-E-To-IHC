import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


def _make_norm(norm_type, channels):
    """
    norm_type:
      - "batch"    -> BatchNorm2d
      - "instance" -> InstanceNorm2d (affine=True so it has learnable scale/shift)
      - "none"/"identity" -> Identity (no normalization)
    """
    norm = str(norm_type).lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


def _use_norm(norm_type):
    """Returns True if the norm type actually has learnable/running parameters
    (i.e. not Identity). Used to decide whether conv bias should be disabled.)"""
    return str(norm_type).lower() not in {"none", "identity"}


class ResidualBlock(nn.Module):
    """
    Standard ResNet-style residual block:
        out = x + F(x)

    Uses 2 conv layers (3x3) with padding via ReflectionPad2d to preserve spatial size.
    """
    def __init__(self, channels, norm_type="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=not _use_norm(norm_type)),
            _make_norm(norm_type, channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=not _use_norm(norm_type)),
            _make_norm(norm_type, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UNetDown(nn.Module):
    """
    Encoder (downsampling) block for U-Net:

        Conv2d(k=4, stride=2, padding=1) -> [Norm] -> LeakyReLU

    Stride=2 halves spatial dimensions (H,W -> H/2,W/2).
    Pass norm_type="none" to skip normalization (common for first layer).
    """

    def __init__(self, in_ch, out_ch, norm_type="instance"):
        super().__init__()
        has_norm = _use_norm(norm_type)
        layers = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not has_norm)
        ]
        if has_norm:
            layers.append(_make_norm(norm_type, out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """
    Decoder (upsampling) block for U-Net:

        Upsample(2x) -> Conv(3x3) -> Norm -> ReLU -> [Dropout]
        Then concatenate with the encoder skip feature map along channels.

    Note: concatenation doubles/expands channels depending on skip size.
    """
    def __init__(self, in_ch, out_ch, norm_type="instance", dropout=False):
        super().__init__()
        has_norm = _use_norm(norm_type)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=0, bias=not has_norm),
            _make_norm(norm_type, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
        )

    def forward(self, x, skip):
        # 1) Upsample spatial size
        x = self.up(x)
        # 2) Refine features with conv+norm+relu
        x = self.conv(x)
        # 3) Channel-wise concat with skip connection from encoder
        #    Shapes: x=(N, out_ch, H, W), skip=(N, skip_ch, H, W)
        #    Output=(N, out_ch+skip_ch, H, W)
        return torch.cat([x, skip], dim=1)




class Generator(nn.Module):
    """U-Net generator for HE -> IHC translation.

    For 256x256 inputs, 6 encoder stages produce a 4x4 bottleneck where
    residual blocks are actually meaningful. 

    Spatial sizes for 256x256 input (H x W):
        d1: 128x128   d2: 64x64   d3: 32x32
        d4: 16x16     d5: 8x8     d6: 4x4  (bottleneck)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_res_blocks: int = 6,
        norm_type: str = "instance",
    ):
        super().__init__()

        # --- Encoder ---
        # First block has no norm (raw pixel inputs shouldn't be normalised)
        self.down1 = UNetDown(in_channels, 64,  norm_type="none")
        self.down2 = UNetDown(64,          128, norm_type=norm_type)
        self.down3 = UNetDown(128,         256, norm_type=norm_type)
        self.down4 = UNetDown(256,         512, norm_type=norm_type)
        self.down5 = UNetDown(512,         512, norm_type=norm_type)
        # Last encoder block: no norm — spatial size already tiny (4x4)
        self.down6 = UNetDown(512,         512, norm_type="none")

        # --- Bottleneck residual blocks (operate on 4x4) ---
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(512, norm_type=norm_type) for _ in range(num_res_blocks)]
        )

        # --- Decoder ---
        # in_ch for each UNetUp = channels from previous up output + skip channels
        self.up1 = UNetUp(512,  512, norm_type=norm_type, dropout=True)   # 512 -> cat(512, d5=512) = 1024
        self.up2 = UNetUp(1024, 512, norm_type=norm_type, dropout=True)   # 512 -> cat(512, d4=512) = 1024
        self.up3 = UNetUp(1024, 512, norm_type=norm_type)                  # 512 -> cat(512, d3=256) = 768
        self.up4 = UNetUp(768,  256, norm_type=norm_type)                  # 256 -> cat(256, d2=128) = 384
        self.up5 = UNetUp(384,  128, norm_type=norm_type)                  # 128 -> cat(128, d1=64)  = 192

        self.final = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(192, out_channels, 3, stride=1, padding=0),
                    nn.Tanh(),
                )


        # Store for use by PatchDiscriminator factory method
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        d1 = self.down1(x)   # 128x128, 64ch
        d2 = self.down2(d1)  #  64x64, 128ch
        d3 = self.down3(d2)  #  32x32, 256ch
        d4 = self.down4(d3)  #  16x16, 512ch
        d5 = self.down5(d4)  #   8x8,  512ch
        d6 = self.down6(d5)  #   4x4,  512ch

        b  = self.bottleneck(d6)   # 4x4, 512ch

        u1 = self.up1(b,  d5)  # 8x8,   1024ch
        u2 = self.up2(u1, d4)  # 16x16, 1024ch
        u3 = self.up3(u2, d3)  # 32x32,  768ch
        u4 = self.up4(u3, d2)  # 64x64,  384ch
        u5 = self.up5(u4, d1)  # 128x128, 192ch

        return self.final(u5)  # 256x256, out_channels


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    Receives (real_HE, real/fake_IHC) concatenated along the channel axis.
    in_channels must match Generator.in_channels + Generator.out_channels.
    """

    def __init__(
        self,
        in_channels: int = 6,
        norm_type: str = "instance",
        use_spectral_norm: bool = False,
    ):
        super().__init__()

        def maybe_spectral(conv):
            return spectral_norm(conv) if use_spectral_norm else conv

        def block(in_ch, out_ch, stride=2, normalize=True):
            has_norm = normalize and _use_norm(norm_type)
            layers = [
                maybe_spectral(
                    nn.Conv2d(
                        in_ch, out_ch, 4, stride=stride, padding=1, bias=not has_norm
                    )
                )
            ]
            if normalize and _use_norm(norm_type):
                layers.append(_make_norm(norm_type, out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64,  normalize=False),
            block(64,          128),
            block(128,         256),
            block(256,         512, stride=1),
            maybe_spectral(nn.Conv2d(512, 1, 4, padding=1)),
        )

    def forward(self, he, ihc):
        return self.model(torch.cat([he, ihc], dim=1))

    @classmethod
    def from_generator(
        cls,
        generator: Generator,
        norm_type: str = "instance",
        use_spectral_norm: bool = False,
    ):
        """Convenience constructor that derives in_channels from the Generator,
        so the two are always in sync.

            disc = PatchDiscriminator.from_generator(gen)
        """
        return cls(
            in_channels=generator.in_channels + generator.out_channels,
            norm_type=norm_type,
            use_spectral_norm=use_spectral_norm,
        )
