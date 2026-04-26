"""
DCGAN Baseline on CIFAR-10
============================
Paper: Radford et al., 2015 - "Unsupervised Representation Learning with DCGANs"

Requirements:
    pip install torch torchvision tqdm matplotlib

Run:
    python dcgan_cifar10.py
"""

import os
import torch
import torch.nn as nn
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
LATENT_DIM  = 100         # Dimension of generator input noise
BATCH_SIZE  = 128
EPOCHS      = 100
LR          = 2e-4
BETA1       = 0.5         # Adam beta1 (standard for GANs)
BETA2       = 0.999
SAVE_EVERY  = 10
OUTPUT_DIR  = "dcgan_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# WEIGHT INITIALIZATION (from DCGAN paper)
# ─────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ─────────────────────────────────────────────
# GENERATOR
# ─────────────────────────────────────────────
class Generator(nn.Module):
    """
    Maps random noise z (LATENT_DIM,) → image (3, 32, 32)
    Architecture: Linear → reshape → upsample with ConvTranspose2d
    
    z (100) → 4x4 → 8x8 → 16x16 → 32x32
    """
    def __init__(self, latent_dim=100, channels=3, base=128):
        super().__init__()
        self.net = nn.Sequential(
            # Input: z of shape (B, latent_dim)
            # Project and reshape to (B, base*4, 4, 4)
            nn.ConvTranspose2d(latent_dim, base * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(True),

            # 4 → 8
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(True),

            # 8 → 16
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(True),

            # 16 → 32
            nn.ConvTranspose2d(base, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z):
        # z shape: (B, latent_dim) → (B, latent_dim, 1, 1)
        return self.net(z.view(z.size(0), -1, 1, 1))


# ─────────────────────────────────────────────
# DISCRIMINATOR
# ─────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Maps image (3, 32, 32) → scalar probability [0,1]
    Architecture: Strided convolutions (no pooling, as per DCGAN paper)
    
    32x32 → 16x16 → 8x8 → 4x4 → scalar
    """
    def __init__(self, channels=3, base=128):
        super().__init__()
        self.net = nn.Sequential(
            # 32 → 16
            nn.Conv2d(channels, base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 16 → 8
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 → 4
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 → 1 (scalar)
            nn.Conv2d(base * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)


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
    loader = get_cifar10_loader(BATCH_SIZE)

    G = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    D = Discriminator(channels=CHANNELS).to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(BETA1, BETA2))

    criterion = nn.BCELoss()

    # Fixed noise to track generator progress across epochs
    fixed_noise = torch.randn(64, LATENT_DIM, device=DEVICE)

    g_losses, d_losses = [], []

    print(f"Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    print(f"Training on: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        epoch_g, epoch_d = 0.0, 0.0

        for images, _ in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images = images.to(DEVICE)
            B = images.size(0)

            real_labels = torch.ones(B, device=DEVICE)
            fake_labels = torch.zeros(B, device=DEVICE)

            # ── Train Discriminator ──────────────────
            # Real images → D should output 1
            D.zero_grad()
            loss_real = criterion(D(images), real_labels)

            # Fake images → D should output 0
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_images = G(z).detach()   # detach so G doesn't get gradients here
            loss_fake = criterion(D(fake_images), fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            optimizer_D.step()

            # ── Train Generator ──────────────────────
            # Generator wants D to think its fakes are real (label = 1)
            G.zero_grad()
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_images = G(z)
            loss_G = criterion(D(fake_images), real_labels)
            loss_G.backward()
            optimizer_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()

        avg_g = epoch_g / len(loader)
        avg_d = epoch_d / len(loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)
        print(f"  Epoch {epoch} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        # Save samples
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            G.eval()
            with torch.no_grad():
                samples = G(fixed_noise)
            samples = (samples + 1) / 2   # denormalize to [0, 1]
            save_image(make_grid(samples, nrow=8),
                       f"{OUTPUT_DIR}/samples_epoch_{epoch:03d}.png")
            torch.save(G.state_dict(), f"{OUTPUT_DIR}/generator_epoch_{epoch:03d}.pt")
            print(f"  Saved samples & checkpoint at epoch {epoch}")

    # Plot loss curves
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss")
    plt.title("DCGAN Training Losses (CIFAR-10)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/loss_curves.png")
    print(f"\nDone! Outputs saved to ./{OUTPUT_DIR}/")

    return G, D


# ─────────────────────────────────────────────
# GENERATE SAMPLES (after training)
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_samples(model_path, n_samples=64):
    G = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()
    z = torch.randn(n_samples, LATENT_DIM, device=DEVICE)
    samples = G(z)
    samples = (samples + 1) / 2
    save_image(make_grid(samples, nrow=8), f"{OUTPUT_DIR}/final_samples.png")
    print(f"Generated {n_samples} samples → {OUTPUT_DIR}/final_samples.png")


if __name__ == "__main__":
    G, D = train()
