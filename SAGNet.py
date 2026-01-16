from dataclasses import dataclass
from typing import Tuple, List, Literal

import torch
import torch.nn as nn


# ===============================
# Utils
# ===============================
class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        # (N, C, H, W) -> (N, H, W, C) 로 변경하여 정규화 적용 후 복구
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=2):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.net(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch * 2, 2, 2)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # ch → ch/2 after pixel shuffle
        self.conv = nn.Conv2d(ch, (ch // 2) * 4, 1)
        self.ps = nn.PixelShuffle(2)

    def forward(self, x):
        return self.ps(self.conv(x))



# ===============================
# Baseline Block
# ===============================
class BaselineBlock(nn.Module):
    def __init__(
        self,
        channels,
        dw_kernel=3,
        expand_ratio=1,
        ffn_ratio=1,
        ca_reduction=2,
        use_skip_init=True,
    ):
        super().__init__()
        hidden = channels * expand_ratio
        ffn_hidden = channels * ffn_ratio

        self.norm1 = LayerNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, dw_kernel, padding=dw_kernel // 2, groups=hidden)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, channels, 1)
        self.ca = ChannelAttention(channels, ca_reduction)

        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, ffn_hidden, 1)
        self.ffn2 = nn.Conv2d(ffn_hidden, channels, 1)

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_skip_init else 1.0
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_skip_init else 1.0

    def forward(self, x):
        y = self.norm1(x)
        y = self.pw1(y)
        y = self.dw(y)
        y = self.act(y)
        y = self.pw2(y)
        y = self.ca(y)
        x = x + self.beta * y

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.act(y)
        y = self.ffn2(y)
        return x + self.gamma * y


# ===============================
# Scattering-Aware Block
# ===============================
class ScatteringAwareBlock(nn.Module):
    def __init__(self, channels, lf_kernel=5, reduction=4, use_skip_init=True):
        super().__init__()

        self.lf = nn.Conv2d(
            channels, channels,
            kernel_size=lf_kernel,
            padding=lf_kernel // 2,
            groups=channels
        )

        hidden = max(channels // reduction, 8)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1, channels, 1, 1)) if use_skip_init else 1.0

    def forward(self, x):
        lf = self.lf(x)
        g = self.global_gate(x)
        s = self.spatial_gate(x)
        return x + self.alpha * lf * g * s


# ===============================
# Enhanced Block
# ===============================
class EnhancedBlock(nn.Module):
    def __init__(
        self,
        channels,
        use_sab,
        dw_kernel,
        expand_ratio,
        ffn_ratio,
        ca_reduction,
        use_skip_init,
        sab_lf_kernel,
        sab_reduction,
    ):
        super().__init__()

        self.base = BaselineBlock(
            channels,
            dw_kernel,
            expand_ratio,
            ffn_ratio,
            ca_reduction,
            use_skip_init,
        )

        self.sab = (
            ScatteringAwareBlock(
                channels,
                sab_lf_kernel,
                sab_reduction,
                use_skip_init,
            ) if use_sab else nn.Identity()
        )

    def forward(self, x):
        x = self.base(x)
        return self.sab(x)


# ===============================
# Config
# ===============================
SABPlacement = Literal["none", "middle", "enc+middle", "all"]

@dataclass
class Config:
    width: int = 24
    enc_blk_nums: Tuple[int, int, int, int] = (2, 2, 2, 2)
    mid_blk_num: int = 4
    dec_blk_nums: Tuple[int, int, int, int] = (2, 2, 2, 2)

    sab_placement: SABPlacement = "enc+middle"
    sab_lf_kernel: int = 3
    sab_reduction: int = 4


# ===============================
# Network
# ===============================
class SAGNet(nn.Module):
    def __init__(self, cfg=Config()):
        super().__init__()
        self.cfg = cfg

        self.intro = nn.Conv2d(3, cfg.width, 3, padding=1)
        self.ending = nn.Conv2d(cfg.width, 3, 3, padding=1)

        ch = cfg.width
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for n in cfg.enc_blk_nums:
            self.encoders.append(self._stage(ch, n, cfg.sab_placement in ["enc+middle", "all"]))
            self.downs.append(Downsample(ch))
            ch *= 2

        #self.middle = self._stage(ch, cfg.mid_blk_num, cfg.sab_placement != "none")
        use_sab_mid = cfg.sab_placement in ["middle", "enc+middle", "all"]
        self.middle = self._stage(ch, cfg.mid_blk_num, use_sab_mid)


        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in cfg.dec_blk_nums:
            self.ups.append(Upsample(ch))
            ch //= 2
            self.decoders.append(self._stage(ch, n, cfg.sab_placement == "all"))

    def _stage(self, ch, nblk, use_sab):
        return nn.Sequential(*[
            EnhancedBlock(
                ch, use_sab,
                dw_kernel=3,
                expand_ratio=2,
                ffn_ratio=2,
                ca_reduction=2,
                use_skip_init=True,
                sab_lf_kernel=self.cfg.sab_lf_kernel,
                sab_reduction=self.cfg.sab_reduction,
            )
            for _ in range(nblk)
        ])

    def forward(self, x):
        inp = x
        x = self.intro(x)

        skips = []
        for e, d in zip(self.encoders, self.downs):
            x = e(x)
            skips.append(x)
            x = d(x)

        x = self.middle(x)

        for u, d in zip(self.ups, self.decoders):
            x = u(x)
            x = x + skips.pop()
            x = d(x)

        return self.ending(x) + inp