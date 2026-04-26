"""
FID Evaluation & Visual Comparison
=====================================
Computes FID score for DDPM and DCGAN and creates comparison plots.

Requirements:
    pip install torch torchvision pytorch-fid matplotlib numpy

Usage:
    # After training both models, run:
    python evaluate_compare.py \
        --ddpm_path   ddpm_outputs/ddpm_epoch_100.pt \
        --dcgan_path  dcgan_outputs/generator_epoch_100.pt
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# Import models from training scripts
from ddpm_cifar10  import UNet, NoiseScheduler, T, BETA_START, BETA_END, CHANNELS, IMAGE_SIZE
from dcgan_cifar10 import Generator, LATENT_DIM

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "comparison_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# FID SCORE USING pytorch-fid
# ─────────────────────────────────────────────
def save_samples_for_fid(images_tensor, folder):
    """Save generated images as PNG files for FID computation."""
    from torchvision.utils import save_image
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images_tensor):
        save_image((img + 1) / 2, f"{folder}/{i:05d}.png")

def save_real_images_for_fid(n=10000):
    """Save real CIFAR-10 images for FID reference."""
    folder = "fid_real"
    if os.path.exists(folder) and len(os.listdir(folder)) >= n:
        print(f"Real images already saved at ./{folder}/")
        return folder

    from torchvision.utils import save_image
    os.makedirs(folder, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3)
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    loader  = DataLoader(dataset, batch_size=256, shuffle=True)
    count   = 0
    for imgs, _ in loader:
        for img in imgs:
            if count >= n: break
            save_image((img + 1) / 2, f"{folder}/{count:05d}.png")
            count += 1
        if count >= n: break
    print(f"Saved {count} real images to ./{folder}/")
    return folder


def compute_fid(real_folder, fake_folder):
    """
    Compute FID using pytorch-fid library.
    Lower FID = better image quality & distribution match.
    """
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", real_folder, fake_folder,
         "--device", DEVICE],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    # Parse FID value from output
    for line in output.split("\n"):
        if "FID" in line:
            try:
                fid = float(line.split(":")[-1].strip())
                return fid
            except:
                pass
    print("FID output:", output)
    return None


# ─────────────────────────────────────────────
# GENERATE IMAGES
# ─────────────────────────────────────────────
@torch.no_grad()
def generate_ddpm_samples(model_path, n=1000):
    print(f"Generating {n} DDPM samples...")
    scheduler = NoiseScheduler(T=T, beta_start=BETA_START, beta_end=BETA_END, device=DEVICE)
    model = UNet(image_channels=CHANNELS, base_channels=128).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_samples = []
    batch = 64
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        s  = scheduler.p_sample_loop(model, (bs, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
        all_samples.append(s.cpu())
    return torch.cat(all_samples, dim=0)[:n]


@torch.no_grad()
def generate_dcgan_samples(model_path, n=1000):
    print(f"Generating {n} DCGAN samples...")
    G = Generator(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()

    all_samples = []
    batch = 256
    for i in range(0, n, batch):
        bs = min(batch, n - i)
        z  = torch.randn(bs, LATENT_DIM, device=DEVICE)
        all_samples.append(G(z).cpu())
    return torch.cat(all_samples, dim=0)[:n]


# ─────────────────────────────────────────────
# VISUAL COMPARISON GRID
# ─────────────────────────────────────────────
def create_comparison_grid(ddpm_samples, dcgan_samples, n_show=32):
    """Create a side-by-side visual grid comparing DDPM vs DCGAN outputs."""
    def to_grid(samples, n):
        imgs = (samples[:n].clamp(-1, 1) + 1) / 2
        return make_grid(imgs, nrow=8, padding=2, normalize=False)

    ddpm_grid  = to_grid(ddpm_samples,  n_show).permute(1, 2, 0).numpy()
    dcgan_grid = to_grid(dcgan_samples, n_show).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(ddpm_grid);  axes[0].set_title("DDPM Generated Images",  fontsize=16, fontweight="bold")
    axes[1].imshow(dcgan_grid); axes[1].set_title("DCGAN Generated Images", fontsize=16, fontweight="bold")
    for ax in axes: ax.axis("off")
    plt.suptitle("DDPM vs DCGAN — CIFAR-10 Image Generation", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/visual_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Visual comparison saved → {OUTPUT_DIR}/visual_comparison.png")


# ─────────────────────────────────────────────
# METRICS COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_metrics_comparison(fid_ddpm, fid_dcgan):
    """Bar chart comparing FID scores."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["DDPM", "DCGAN"], [fid_ddpm, fid_dcgan],
                  color=["#4C72B0", "#DD8452"], width=0.5, edgecolor="black")
    for bar, val in zip(bars, [fid_ddpm, fid_dcgan]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.2f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("FID Score (lower is better)", fontsize=12)
    ax.set_title("FID Score Comparison: DDPM vs DCGAN\n(CIFAR-10)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(fid_ddpm, fid_dcgan) * 1.2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fid_comparison.png", dpi=150)
    plt.close()
    print(f"FID comparison chart saved → {OUTPUT_DIR}/fid_comparison.png")


# ─────────────────────────────────────────────
# QUALITATIVE COMPARISON TABLE (printed)
# ─────────────────────────────────────────────
def print_comparison_table(fid_ddpm, fid_dcgan):
    print("\n" + "="*55)
    print("  FINAL COMPARISON: DDPM vs DCGAN on CIFAR-10")
    print("="*55)
    print(f"  {'Metric':<30} {'DDPM':>8} {'DCGAN':>8}")
    print("-"*55)
    print(f"  {'FID Score (↓ better)':<30} {fid_ddpm:>8.2f} {fid_dcgan:>8.2f}")
    winner_fid = "DDPM ✓" if fid_ddpm < fid_dcgan else "DCGAN ✓"
    print(f"  {'Winner (FID)':<30} {winner_fid:>17}")
    print("-"*55)
    print("  Training Stability:")
    print("    DDPM  → Stable (single MSE loss)")
    print("    DCGAN → Can be unstable (adversarial)")
    print("  Inference Speed:")
    print("    DDPM  → Slow (1000 denoising steps)")
    print("    DCGAN → Fast (single forward pass)")
    print("  Sample Diversity:")
    print("    DDPM  → High (no mode collapse)")
    print("    DCGAN → May suffer mode collapse")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(ddpm_path, dcgan_path, n_fid=5000):
    # Generate samples
    ddpm_samples  = generate_ddpm_samples(ddpm_path,  n=n_fid)
    dcgan_samples = generate_dcgan_samples(dcgan_path, n=n_fid)

    # Visual grid
    create_comparison_grid(ddpm_samples, dcgan_samples)

    # Save for FID computation
    real_folder  = save_real_images_for_fid(n=n_fid)
    ddpm_folder  = f"{OUTPUT_DIR}/fid_ddpm"
    dcgan_folder = f"{OUTPUT_DIR}/fid_dcgan"
    save_samples_for_fid(ddpm_samples,  ddpm_folder)
    save_samples_for_fid(dcgan_samples, dcgan_folder)

    # Compute FID
    print("\nComputing FID scores (this may take a few minutes)...")
    fid_ddpm  = compute_fid(real_folder, ddpm_folder)
    fid_dcgan = compute_fid(real_folder, dcgan_folder)

    if fid_ddpm and fid_dcgan:
        plot_metrics_comparison(fid_ddpm, fid_dcgan)
        print_comparison_table(fid_ddpm, fid_dcgan)
    else:
        print("FID computation failed. Check pytorch-fid installation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddpm_path",  required=True, help="Path to saved DDPM checkpoint (.pt)")
    parser.add_argument("--dcgan_path", required=True, help="Path to saved DCGAN Generator checkpoint (.pt)")
    parser.add_argument("--n_fid",      type=int, default=5000, help="Number of samples for FID")
    args = parser.parse_args()
    main(args.ddpm_path, args.dcgan_path, args.n_fid)
