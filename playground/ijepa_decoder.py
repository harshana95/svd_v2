"""
I-JEPA Stochastic Decoder
=========================
Trains a stochastic (CVAE-style) decoder on top of frozen I-JEPA embeddings
to visualize predicted representations back in pixel space.

Architecture mirrors what Meta described in the I-JEPA paper:
  - Frozen I-JEPA ViT encoder (target encoder)
  - Stochastic decoder: embedding → latent z (reparameterize) → pixels
  - Loss: reconstruction (MSE) + KL divergence

Usage:
  # 1. Train the decoder
  python ijepa_decoder.py --mode train --data_dir /path/to/images --epochs 50

  # 2. Visualize embeddings → pixels
  python ijepa_decoder.py --mode visualize --image /path/to/image.jpg --checkpoint decoder.pth
"""

import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from transformers import AutoModel, ViTImageProcessor
from PIL import Image


# ─────────────────────────────────────────────
# 1.  Frozen I-JEPA Encoder wrapper
# ─────────────────────────────────────────────

class IJEPAEncoder(nn.Module):
    """Wraps the HuggingFace I-JEPA model; always runs in no-grad mode."""

    # ViT-H/14 trained on ImageNet-22k — best publicly available checkpoint
    HF_REPO = "facebook/ijepa_vith14_22k"

    def __init__(self, repo: str = HF_REPO, device="cpu"):
        super().__init__()
        print(f"Loading I-JEPA encoder from {repo} …")
        self.encoder = AutoModel.from_pretrained(repo).to(device)
        self.processor = ViTImageProcessor.from_pretrained(repo)
        self.embed_dim = self.encoder.config.hidden_size   # 1280 for ViT-H
        self.num_patches = (self.encoder.config.image_size //
                            self.encoder.config.patch_size) ** 2  # 256 for 224/14
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W]  (already normalized)
        Returns:
            patch_embeddings: [B, N, D]   N=num_patches, D=embed_dim
        """
        out = self.encoder(pixel_values=pixel_values)
        # last_hidden_state includes CLS token at position 0 — drop it
        return out.last_hidden_state[:, 1:, :]   # [B, N, D]

    def preprocess(self, images) -> torch.Tensor:
        """Accepts PIL Image or list of PIL Images, returns normalized tensor."""
        inputs = self.processor(images=images, return_tensors="pt")
        return inputs["pixel_values"]


# ─────────────────────────────────────────────
# 2.  Stochastic Decoder
# ─────────────────────────────────────────────

class TransformerPooler(nn.Module):
    """Condense N patch tokens → a single conditioning vector."""
    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D] → [B, out_dim]"""
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)          # [B, N+1, D]
        x = self.transformer(x)
        return self.proj(x[:, 0])               # [B, out_dim]


class StochasticDecoder(nn.Module):
    """
    CVAE-style stochastic decoder.

    Given a patch-embedding sequence from I-JEPA:
      1. Pool embeddings → conditioning vector c
      2. Sample z ~ N(mu(c), sigma(c))          (reparameterisation trick)
      3. Decode [z; c] → image pixels

    The stochasticity lets the decoder express uncertainty about pixel details
    that aren't encoded in the semantic embedding — matching the "sketch" quality
    described in the I-JEPA paper.
    """

    def __init__(
        self,
        embed_dim: int = 1280,   # I-JEPA ViT-H embed dim
        cond_dim: int = 512,     # pooled conditioning dimension
        latent_dim: int = 256,   # z dimension
        image_size: int = 224,
        base_channels: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # ── Pooler: patch tokens → conditioning vector ──────────────────────
        self.pooler = TransformerPooler(embed_dim, cond_dim)

        # ── VAE encoder (embedding → mu, logvar) ────────────────────────────
        self.fc_mu     = nn.Linear(cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(cond_dim, latent_dim)

        # ── Decoder: [z; c] → image ─────────────────────────────────────────
        # Start at 7×7 feature map, upsample to image_size
        self.num_ups = int(math.log2(image_size // 7))   # 224→5 ups, 256→5 ups
        dec_in_dim = latent_dim + cond_dim
        ch = base_channels * (2 ** self.num_ups)         # widest = base*{32,16,…}
        ch = min(ch, 512)

        self.fc_dec = nn.Linear(dec_in_dim, ch * 7 * 7)

        layers = []
        for i in range(self.num_ups):
            out_ch = max(ch // 2, base_channels)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ]
            ch = out_ch

        layers += [
            nn.Conv2d(ch, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),   # output in [0, 1]
        ]
        self.decoder_conv = nn.Sequential(*layers)

    # ── helpers ─────────────────────────────────────────────────────────────

    def pool(self, patch_emb: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B, cond_dim]"""
        return self.pooler(patch_emb)

    def encode(self, cond: torch.Tensor):
        """cond: [B, cond_dim] → mu, logvar each [B, latent_dim]"""
        return self.fc_mu(cond), self.fc_logvar(cond)

    def reparameterize(self, mu, logvar, temperature: float = 1.0):
        if self.training or temperature > 0:
            std = (0.5 * logvar).exp() * temperature
            return mu + std * torch.randn_like(std)
        return mu   # deterministic at test time with temperature=0

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim], [B, cond_dim] → [B, 3, H, W]"""
        x = torch.cat([z, cond], dim=1)
        x = self.fc_dec(x)
        x = x.view(x.size(0), -1, 7, 7)
        return self.decoder_conv(x)

    # ── main forward ─────────────────────────────────────────────────────────

    def forward(self, patch_emb: torch.Tensor, temperature: float = 1.0):
        """
        Args:
            patch_emb:  [B, N, D]  frozen I-JEPA patch embeddings
            temperature: float  >1 = more stochastic, 0 = deterministic
        Returns:
            recon:   [B, 3, H, W]
            mu:      [B, latent_dim]
            logvar:  [B, latent_dim]
        """
        cond        = self.pool(patch_emb)          # [B, cond_dim]
        mu, logvar  = self.encode(cond)
        z           = self.reparameterize(mu, logvar, temperature)
        recon       = self.decode(z, cond)
        return recon, mu, logvar


# ─────────────────────────────────────────────
# 3.  Loss
# ─────────────────────────────────────────────

def elbo_loss(recon, target, mu, logvar, beta: float = 1e-3):
    """
    β-VAE ELBO:  E[log p(x|z)] - β * KL(q(z|x) || p(z))

    β < 1  →  reconstruction dominates  (sharper but less diverse)
    β > 1  →  KL dominates              (smoother, more stochastic)
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ─────────────────────────────────────────────
# 4.  Training
# ─────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    dataset    = datasets.ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

    # ── Models ──────────────────────────────────────────────────────────────
    encoder   = IJEPAEncoder(device=device)
    embed_dim = encoder.embed_dim

    # Normalisation applied inside the loop (matches I-JEPA preprocessing)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    decoder   = StochasticDecoder(
        embed_dim    = embed_dim,
        cond_dim     = args.cond_dim,
        latent_dim   = args.latent_dim,
        image_size   = 224,
        base_channels= args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader))

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = recon_sum = kl_sum = 0.0

        for step, (images, _) in enumerate(dataloader):
            images = images.to(device)                      # [B, 3, 224, 224]

            # ── Get frozen embeddings ────────────────────────────────────────
            with torch.no_grad():
                patch_emb = encoder(normalize(images))      # [B, N, D]

            # ── Decode ──────────────────────────────────────────────────────
            recon, mu, logvar = decoder(patch_emb)

            # ── Loss ────────────────────────────────────────────────────────
            loss, r_loss, kl_loss = elbo_loss(recon, images, mu, logvar,
                                              beta=args.beta)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            recon_sum  += r_loss.item()
            kl_sum     += kl_loss.item()

            if (step + 1) % 100 == 0:
                avg = total_loss / (step + 1)
                print(f"  Epoch {epoch:3d} | step {step+1:5d} "
                      f"| loss={avg:.4f}  recon={recon_sum/(step+1):.4f}"
                      f"  kl={kl_sum/(step+1):.5f}")

        # ── Save samples ────────────────────────────────────────────────────
        decoder.eval()
        with torch.no_grad():
            sample_imgs = images[:8]
            emb         = encoder(normalize(sample_imgs))
            recon_det, _, _ = decoder(emb, temperature=0.0)   # deterministic
            recon_sto, _, _ = decoder(emb, temperature=1.0)   # stochastic

            grid = torch.cat([sample_imgs, recon_det, recon_sto], dim=0)
            save_image(grid,
                       os.path.join(args.out_dir, f"epoch_{epoch:03d}.png"),
                       nrow=8)

        # ── Checkpoint ──────────────────────────────────────────────────────
        ckpt_path = os.path.join(args.out_dir, "decoder.pth")
        torch.save({
            "epoch":      epoch,
            "state_dict": decoder.state_dict(),
            "args":       vars(args),
        }, ckpt_path)
        print(f"Epoch {epoch} done — checkpoint saved to {ckpt_path}\n")


# ─────────────────────────────────────────────
# 5.  Visualisation / Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load encoder ────────────────────────────────────────────────────────
    encoder = IJEPAEncoder(device=device)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    # ── Load decoder ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved_args = ckpt["args"]
    decoder = StochasticDecoder(
        embed_dim     = encoder.embed_dim,
        cond_dim      = saved_args["cond_dim"],
        latent_dim    = saved_args["latent_dim"],
        image_size    = 224,
        base_channels = saved_args["base_channels"],
    ).to(device)
    decoder.load_state_dict(ckpt["state_dict"])
    decoder.eval()
    print(f"Decoder loaded from {args.checkpoint} (epoch {ckpt['epoch']})")

    # ── Encode input image ──────────────────────────────────────────────────
    img   = Image.open(args.image).convert("RGB")
    pixel = encoder.preprocess(img).to(device)          # [1, 3, 224, 224]
    emb   = encoder(normalize(pixel))                   # [1, N, D]

    # ── Decode: one deterministic + N stochastic samples ───────────────────
    samples = []
    recon_det, mu, logvar = decoder(emb, temperature=0.0)
    samples.append(recon_det)

    for _ in range(args.num_samples - 1):
        recon_sto, _, _ = decoder(emb, temperature=args.temperature)
        samples.append(recon_sto)

    # ── Save ────────────────────────────────────────────────────────────────
    # Row 0: original | Row 1: deterministic | Rows 2+: stochastic samples
    to_tensor = transforms.ToTensor()
    original  = to_tensor(img.resize((224, 224))).unsqueeze(0)
    grid      = torch.cat([original.to(device)] + samples, dim=0)

    out_path  = args.out_image or "decoded_output.png"
    save_image(grid, out_path, nrow=len(grid))
    print(f"Saved {len(grid)} images to {out_path}")
    print("  Row 0 = original input")
    print("  Row 1 = deterministic decode (temperature=0)")
    print(f"  Rows 2+ = stochastic samples (temperature={args.temperature})")


# ─────────────────────────────────────────────
# 6.  Standalone embed → decode (no training)
# ─────────────────────────────────────────────

def decode_embedding(
    patch_emb:   torch.Tensor,   # [B, N, D] or [N, D]
    decoder:     StochasticDecoder,
    temperature: float = 1.0,
    n_samples:   int   = 4,
) -> torch.Tensor:
    """
    Given raw I-JEPA patch embeddings, return decoded pixel images.

    Returns:
        samples: [n_samples, B, 3, H, W]
    """
    if patch_emb.dim() == 2:
        patch_emb = patch_emb.unsqueeze(0)   # add batch dim

    results = []
    decoder.eval()
    with torch.no_grad():
        for _ in range(n_samples):
            recon, _, _ = decoder(patch_emb, temperature=temperature)
            results.append(recon)

    return torch.stack(results)   # [n_samples, B, 3, H, W]


# ─────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="I-JEPA Stochastic Decoder")
    sub = p.add_subparsers(dest="mode", required=True)

    # -- train ---
    t = sub.add_parser("train")
    t.add_argument("--data_dir",      required=True,
                   help="ImageFolder-style dataset root")
    t.add_argument("--out_dir",       default="decoder_out")
    t.add_argument("--epochs",        type=int,   default=50)
    t.add_argument("--batch_size",    type=int,   default=32)
    t.add_argument("--lr",            type=float, default=3e-4)
    t.add_argument("--beta",          type=float, default=1e-3,
                   help="KL weight in ELBO loss")
    t.add_argument("--cond_dim",      type=int,   default=512)
    t.add_argument("--latent_dim",    type=int,   default=256)
    t.add_argument("--base_channels", type=int,   default=64)

    # -- visualize ---
    v = sub.add_parser("visualize")
    v.add_argument("--image",       required=True)
    v.add_argument("--checkpoint",  required=True)
    v.add_argument("--out_image",   default="decoded_output.png")
    v.add_argument("--num_samples", type=int,   default=8)
    v.add_argument("--temperature", type=float, default=1.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "visualize":
        visualize(args)