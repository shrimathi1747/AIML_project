"""
DDPM (Denoising Diffusion Probabilistic Model) on CIFAR-10
=============================================================
Paper: Ho et al., 2020 - "Denoising Diffusion Probabilistic Models"

Requirements:
    pip install torch torchvision tqdm matplotlib pytorch-fid

Run:
    python ddpm_cifar10.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE  = 32
CHANNELS    = 3
BATCH_SIZE  = 128
EPOCHS      = 100          # Increase to 200+ for better quality
LR          = 2e-4
T           = 1000         # Total diffusion timesteps
BETA_START  = 1e-4
BETA_END    = 0.02
SAVE_EVERY  = 10           # Save sample images every N epochs
OUTPUT_DIR  = "ddpm_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# NOISE SCHEDULER
# ─────────────────────────────────────────────
class NoiseScheduler:
    """
    Linear beta schedule from DDPM paper.
    Precomputes all alpha values needed for forward/reverse process.
    """
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        # β_t: noise schedule
        self.betas = torch.linspace(beta_start, beta_end, T).to(device)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = cumulative product of α's
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Useful precomputed terms
        self.sqrt_alpha_bars        = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_ab      = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas      = torch.sqrt(1.0 / self.alphas)
        self.alpha_bars_prev        = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.posterior_variance     = self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars)

    def q_sample(self, x0, t, noise=None):
        """
        Forward process: add noise to x0 at timestep t.
        q(x_t | x_0) = N(x_t; sqrt(ᾱ_t)*x_0, (1-ᾱ_t)*I)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab    = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_1mab  = self.sqrt_one_minus_ab[t].view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_1mab * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        """
        Reverse process: one denoising step.
        p_θ(x_{t-1} | x_t)
        """
        t_tensor = torch.full((x.shape[0],), t_idx, device=self.device, dtype=torch.long)
        predicted_noise = model(x, t_tensor)

        beta_t          = self.betas[t_idx]
        sqrt_recip_a    = self.sqrt_recip_alphas[t_idx]
        sqrt_1mab       = self.sqrt_one_minus_ab[t_idx]

        # Mean of p(x_{t-1} | x_t)
        mean = sqrt_recip_a * (x - beta_t / sqrt_1mab * predicted_noise)

        if t_idx == 0:
            return mean
        else:
            noise    = torch.randn_like(x)
            var      = self.posterior_variance[t_idx]
            return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        """Full reverse diffusion loop: pure noise → image."""
        model.eval()
        x = torch.randn(shape, device=self.device)
        for t in tqdm(reversed(range(self.T)), desc="Sampling", total=self.T, leave=False):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)


# ─────────────────────────────────────────────
# U-NET COMPONENTS
# ─────────────────────────────────────────────
class SinusoidalPositionEmbeddings(nn.Module):
    """Encodes timestep t as sinusoidal embeddings (like Transformer PE)."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time embedding injection."""
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.block1   = nn.Sequential(nn.GroupNorm(8, in_ch), nn.SiLU(),
                                       nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2   = nn.Sequential(nn.GroupNorm(8, out_ch), nn.SiLU(),
                                       nn.Conv2d(out_ch, out_ch, 3, padding=1))
        self.skip     = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]  # inject time
        h = self.block2(h)
        return h + self.skip(x)


class Attention(nn.Module):
    """Self-attention for spatial features."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scale = C ** -0.5
        attn  = torch.softmax((q.transpose(-1, -2) @ k) * scale, dim=-1)
        out   = (attn @ v.transpose(-1, -2)).transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    U-Net for DDPM noise prediction.
    Predicts ε (the noise added to x_0 to get x_t).
    """
    def __init__(self, image_channels=3, base_channels=128, time_emb_dim=256):
        super().__init__()
        C = base_channels

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(C),
            nn.Linear(C, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.init_conv = nn.Conv2d(image_channels, C, 3, padding=1)
        self.down1 = ResBlock(C,     C*2,  time_emb_dim)   # 32→32
        self.down2 = ResBlock(C*2,   C*2,  time_emb_dim)   # 32→16 (after pool)
        self.down3 = ResBlock(C*2,   C*4,  time_emb_dim)   # 16→8  (after pool)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1  = ResBlock(C*4, C*4, time_emb_dim)
        self.attn  = Attention(C*4)
        self.mid2  = ResBlock(C*4, C*4, time_emb_dim)

        # Decoder
        self.up1   = ResBlock(C*4 + C*4, C*2, time_emb_dim)   # skip from down3
        self.up2   = ResBlock(C*2 + C*2, C*2, time_emb_dim)   # skip from down2
        self.up3   = ResBlock(C*2 + C*2, C,   time_emb_dim)   # skip from down1

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Output head
        self.out = nn.Sequential(
            nn.GroupNorm(8, C),
            nn.SiLU(),
            nn.Conv2d(C, image_channels, 1)
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        x0 = self.init_conv(x)           # [B, C,   32, 32]
        x1 = self.down1(x0, t_emb)       # [B, C*2, 32, 32]
        x2 = self.down2(self.pool(x1), t_emb)   # [B, C*2, 16, 16]
        x3 = self.down3(self.pool(x2), t_emb)   # [B, C*4, 8,  8 ]

        # Bottleneck
        x_mid = self.mid1(self.pool(x3), t_emb)
        x_mid = self.attn(x_mid)
        x_mid = self.mid2(x_mid, t_emb)

        # Decoder (upsample + skip concat)
        d = self.up1(torch.cat([self.upsample(x_mid), x3], dim=1), t_emb)
        d = self.up2(torch.cat([self.upsample(d),    x2], dim=1), t_emb)
        d = self.up3(torch.cat([self.upsample(d),    x1], dim=1), t_emb)

        return self.out(d)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def get_cifar10_loader(batch_size=128):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),   # scale to [-1, 1]
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=4, pin_memory=True, drop_last=True)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train():
    loader    = get_cifar10_loader(BATCH_SIZE)
    scheduler = NoiseScheduler(T=T, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE)
    model     = UNet(image_channels=CHANNELS, base_channels=128).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on: {DEVICE}")

    losses = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")):
            images = images.to(DEVICE)

            # Sample random timesteps
            t = torch.randint(0, T, (images.shape[0],), device=DEVICE)

            # Forward process: add noise
            x_noisy, noise = scheduler.q_sample(images, t)

            # Predict noise with U-Net
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                predicted_noise = model(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)   # Simple MSE loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        print(f"  Epoch {epoch} | Loss: {avg_loss:.4f}")

        # Save sample images
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            samples = scheduler.p_sample_loop(model, (16, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
            # Denormalize from [-1,1] to [0,1]
            samples = (samples + 1) / 2
            save_image(make_grid(samples, nrow=4),
                       f"{OUTPUT_DIR}/samples_epoch_{epoch:03d}.png")
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/ddpm_epoch_{epoch:03d}.pt")
            print(f"  Saved samples & checkpoint at epoch {epoch}")

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="DDPM Training Loss")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("DDPM Training Loss (CIFAR-10)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curve.png")
    print(f"\nDone! Outputs saved to ./{OUTPUT_DIR}/")

    return model, scheduler


# ─────────────────────────────────────────────
# GENERATE SAMPLES (after training)
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_samples(model_path, n_samples=64):
    scheduler = NoiseScheduler(T=T, device=DEVICE)
    model     = UNet(image_channels=CHANNELS, base_channels=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    samples = scheduler.p_sample_loop(model, (n_samples, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    samples = (samples + 1) / 2
    save_image(make_grid(samples, nrow=8), f"{OUTPUT_DIR}/final_samples.png")
    print(f"Generated {n_samples} samples → {OUTPUT_DIR}/final_samples.png")


if __name__ == "__main__":
    model, scheduler = train()
