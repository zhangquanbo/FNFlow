"""
ADM-style UNet for Flow Matching — 速度预测网络

基于 Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis",
NeurIPS 2021 的 UNet 架构简化版。

特点:
  - 纯卷积，无 Self-Attention → 在 512×512 pixel space 上高速推理
  - Adaptive Group Normalization (AdaGN) 注入 timestep 信息
  - Skip connections (U-Net encoder-decoder)
  - 保持与 FaceNormalFlux 相同的 forward(z_t, t, condition_img) 接口

参考:
  - ADM: https://arxiv.org/abs/2105.05233  (NeurIPS 2021)
  - Rectified Flow: https://arxiv.org/abs/2209.03003  (ICLR 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =============================================================
# 基础模块
# =============================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding (same as DDPM)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepMLP(nn.Module):
    """Timestep → embedding vector"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t):
        return self.net(t)


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization (AdaGN)
    - GroupNorm + scale/shift from timestep embedding
    - 核心时间步注入机制 (ADM 论文 Section 3)
    """
    def __init__(self, channels, t_dim, num_groups=32):
        super().__init__()
        # 确保 num_groups 能整除 channels
        while channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, channels)
        self.proj = nn.Linear(t_dim, channels * 2)

    def forward(self, x, t_emb):
        h = self.norm(x)
        scale, shift = self.proj(t_emb)[:, :, None, None].chunk(2, dim=1)
        return h * (1 + scale) + shift


# =============================================================
# ResBlock (ADM style)
# =============================================================

def _make_group_norm(channels, num_groups=32):
    """创建 GroupNorm，自动调整 groups 以兼容 channels"""
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


class ResBlock(nn.Module):
    """
    ADM ResBlock with timestep conditioning via AdaGN.
    
    结构:
        GroupNorm → SiLU → Conv3x3
        → AdaGN(t) → SiLU → Dropout → Conv3x3
        → + skip (with optional 1x1 conv if channels change)
    """
    def __init__(self, in_ch, out_ch, t_dim, dropout=0.0):
        super().__init__()
        self.norm1 = _make_group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.adagn = AdaGN(out_ch, t_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Skip projection
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # Zero init for residual (improves training stability)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.adagn(h, t_emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# =============================================================
# Downsample / Upsample
# =============================================================

class Downsample(nn.Module):
    """2x spatial downsampling with strided conv (ADM style)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling with nearest + conv (ADM style)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================
# FaceNormalUNet — ADM UNet for Flow Matching
# =============================================================

class FaceNormalUNet(nn.Module):
    """
    ADM-style UNet for face normal estimation via Flow Matching.
    
    架构 (Base config, channels=[64, 128, 256, 512]):
    
        Encoder:                              Decoder:
        512×512 [64]  ──skip──────────────► concat → [64]  512×512
             ↓ downsample                          ↑ upsample
        256×256 [128] ──skip──────────────► concat → [128] 256×256
             ↓ downsample                          ↑ upsample
        128×128 [256] ──skip──────────────► concat → [256] 128×128
             ↓ downsample                          ↑ upsample
         64×64  [512] ──skip──────────────► concat → [512]  64×64
                                    ↕
                            Bottleneck [512]
    
    接口与 FaceNormalFlux 完全一致:
        forward(z_t, t, condition_img) → velocity
    
    Args:
        in_channels:     输入/输出通道数 (RGB = 3)
        channels:        各 level 的通道数列表
        num_res_blocks:  每个 level 的 ResBlock 数量
        dropout:         dropout 概率
    """
    def __init__(
        self,
        in_channels=3,
        channels=(64, 128, 256, 512),
        num_res_blocks=2,
        dropout=0.0,
    ):
        super().__init__()
        self.channels = list(channels)
        self.num_levels = len(channels)
        t_dim = channels[0] * 4  # timestep embedding dimension

        # --- Timestep Embedding ---
        self.time_embed = TimestepMLP(channels[0])

        # --- Input conv: (noise 3ch + condition 3ch) → channels[0] ---
        self.input_conv = nn.Conv2d(in_channels * 2, channels[0], 3, padding=1)

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch_in = channels[0]
        for level in range(self.num_levels):
            ch_out = channels[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                level_blocks.append(ResBlock(ch_in, ch_out, t_dim, dropout))
                ch_in = ch_out
            self.encoder_blocks.append(level_blocks)

            # Downsample (except last level)
            if level < self.num_levels - 1:
                self.downsamples.append(Downsample(ch_out))
            # ch_in carries over to next level

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList([
            ResBlock(channels[-1], channels[-1], t_dim, dropout),
            ResBlock(channels[-1], channels[-1], t_dim, dropout),
        ])

        # --- Decoder (mirror of encoder) ---
        self.decoder_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in range(self.num_levels - 1, -1, -1):
            ch = channels[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                if i == 0:
                    # First block: skip concat doubles the channels
                    level_blocks.append(ResBlock(ch * 2, ch, t_dim, dropout))
                else:
                    level_blocks.append(ResBlock(ch, ch, t_dim, dropout))
            self.decoder_blocks.append(level_blocks)

            # Upsample + channel reduction (except the final/outermost level)
            if level > 0:
                ch_next = channels[level - 1]
                self.upsamples.append(ResBlock(ch, ch_next, t_dim, dropout))
            # Keep decoder at channels[0] for the outermost level

        # --- Output ---
        self.output_norm = _make_group_norm(channels[0])
        self.output_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, z_t, t, condition_img):
        """
        与 FaceNormalFlux 完全相同的接口。
        
        Args:
            z_t:            (B, 3, H, W)  当前状态 (noisy)
            t:              (B,)          timestep ∈ [0, 1]  
            condition_img:  (B, 3, H, W)  RGB 条件图
            
        Returns:
            velocity:       (B, 3, H, W)  预测的速度场
        """
        # --- Input concat ---
        x = torch.cat([z_t, condition_img], dim=1)  # (B, 6, H, W)
        x = self.input_conv(x)                       # (B, channels[0], H, W)

        # --- Timestep embedding ---
        t_emb = self.time_embed(t)                   # (B, t_dim)

        # --- Encoder ---
        skips = []
        for level in range(self.num_levels):
            for block in self.encoder_blocks[level]:
                x = block(x, t_emb)
            skips.append(x)
            if level < self.num_levels - 1:
                x = self.downsamples[level](x)

        # --- Bottleneck ---
        for block in self.bottleneck:
            x = block(x, t_emb)

        # --- Decoder ---
        for i, level in enumerate(range(self.num_levels - 1, -1, -1)):
            # Skip connection (concat)
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)           # channels doubled

            for block in self.decoder_blocks[i]:
                x = block(x, t_emb)

            # Upsample + channel reduction
            if level > 0:
                x = self.upsamples[i](x, t_emb)       # ResBlock: ch → ch_next
                x = F.interpolate(x, scale_factor=2, mode='nearest')

        # --- Output ---
        x = self.output_norm(x)
        x = F.silu(x)
        velocity = self.output_conv(x)
        return velocity


# =============================================================
# 预设模型配置
# =============================================================

def FaceNormalUNet_S(**kwargs):
    """Small: 适合快速实验"""
    return FaceNormalUNet(
        channels=(64, 128, 256, 256),
        num_res_blocks=2,
        **kwargs
    )

def FaceNormalUNet_B(**kwargs):
    """Base: 与当前 FaceNormalFlux (16.76M) 参数量相近"""
    return FaceNormalUNet(
        channels=(64, 128, 256, 512),
        num_res_blocks=2,
        **kwargs
    )

def FaceNormalUNet_L(**kwargs):
    """Large: 更高容量"""
    return FaceNormalUNet(
        channels=(128, 256, 512, 512),
        num_res_blocks=3,
        **kwargs
    )
