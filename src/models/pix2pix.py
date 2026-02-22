import torch
import torch.nn as nn


def _make_norm(norm_type, channels):
    norm = str(norm_type).lower()
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if norm in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_type="instance"):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _make_norm(norm_type, channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            _make_norm(norm_type, channels),
        )

    def forward(self, x):
        return x + self.block(x)


class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="instance", normalize=True):
        super().__init__()
        use_norm = normalize and str(norm_type).lower() not in {"none", "identity"}
        layers = [
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm)
        ]
        if normalize:
            layers.append(_make_norm(norm_type, out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, norm_type="instance", dropout=False):
        super().__init__()
        use_norm = str(norm_type).lower() not in {"none", "identity"}
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_norm),
            _make_norm(norm_type, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class Generator(nn.Module):
    """U-Net generator for HE -> IHC translation."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_res_blocks=6,
        norm_type="instance",
    ):
        super().__init__()
        self.down1 = UNetDown(in_channels, 64, norm_type=norm_type, normalize=False)
        self.down2 = UNetDown(64, 128, norm_type=norm_type)
        self.down3 = UNetDown(128, 256, norm_type=norm_type)
        self.down4 = UNetDown(256, 512, norm_type=norm_type)
        self.down5 = UNetDown(512, 512, norm_type=norm_type)
        self.down6 = UNetDown(512, 512, norm_type=norm_type)
        self.down7 = UNetDown(512, 512, norm_type=norm_type, normalize=False)

        self.bottleneck = nn.Sequential(
            *[ResidualBlock(512, norm_type=norm_type) for _ in range(num_res_blocks)]
        )

        self.up1 = UNetUp(512, 512, norm_type=norm_type, dropout=True)
        self.up2 = UNetUp(1024, 512, norm_type=norm_type, dropout=True)
        self.up3 = UNetUp(1024, 512, norm_type=norm_type)
        self.up4 = UNetUp(1024, 256, norm_type=norm_type)
        self.up5 = UNetUp(512, 128, norm_type=norm_type)
        self.up6 = UNetUp(256, 64, norm_type=norm_type)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        b = self.bottleneck(d7)

        u1 = self.up1(b, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        return self.final(u6)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN discriminator."""

    def __init__(self, in_channels=6, norm_type="instance"):
        super().__init__()

        def block(in_ch, out_ch, stride=2, normalize=True):
            use_norm = normalize and str(norm_type).lower() not in {"none", "identity"}
            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    4,
                    stride=stride,
                    padding=1,
                    bias=not use_norm,
                )
            ]
            if normalize:
                layers.append(_make_norm(norm_type, out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, 64, normalize=False),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, he, ihc):
        return self.model(torch.cat([he, ihc], dim=1))
