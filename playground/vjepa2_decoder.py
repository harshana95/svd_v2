"""
V-JEPA 2 / 2.1  —  Stochastic Video Decoder
=============================================
Trains a CVAE-style decoder on top of frozen V-JEPA 2 (or 2.1) spatio-temporal
patch embeddings to reconstruct video frames in pixel space.

Encoder weights load only via Meta ``torch.hub`` (``facebookresearch/vjepa2``), not Hugging Face.

Architecture
------------
  Frozen V-JEPA 2 encoder  →  [B, T*N_spatial, D=1024]
        ↓
  Spatio-Temporal Pooler   →  [B, T, cond_dim]   (one vector per frame)
        ↓
  VAE head (mu, logvar)    →  z ~ N(mu, σ)  [B, T, latent_dim]
        ↓
  Frame Decoder (Conv)     →  [B, T, 3, H, W]  pixel frames

Loss: β-VAE ELBO  (MSE reconstruction + β * KL)

Usage
-----
  # Train
  python vjepa2_decoder.py train \\
      --data_dir /path/to/video_dataset \\
      --hub-entry vjepa2_1_vit_large_384 \\
      --epochs 30 --batch_size 4 --frames 8

  # Decode a video clip
  python vjepa2_decoder.py decode \\
      --video /path/to/clip.mp4 \\
      --checkpoint out/decoder.pth \\
      --num_samples 4 --out decoded.mp4

  # Decode from raw embeddings (library use)
  from vjepa2_decoder import build_vjepa_encoder, StochasticVideoDecoder, decode_embeddings
"""

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# Optional: torchcodec for fast video loading (falls back to opencv)
try:
    from torchcodec.decoders import VideoDecoder as TorchCodecDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constants — Meta torch.hub only (facebookresearch/vjepa2)
# ──────────────────────────────────────────────────────────────────────────────

HUB_REPO_DEFAULT = "facebookresearch/vjepa2"
HUB_ENTRY_DEFAULT = "vjepa2_vit_large"

# Meta's public checkpoints (their hub source sometimes ships with localhost — see _prepare_vjepa_hub_cdn).
VJEPA_CDN_BASE = "https://dl.fbaipublicfiles.com/vjepa2"

HUB_ENTRY_NATIVE_CROP = {
    "vjepa2_vit_large": 256,
    "vjepa2_vit_huge": 256,
    "vjepa2_vit_giant": 256,
    "vjepa2_ac_vit_giant": 256,
    "vjepa2_vit_giant_384": 384,
    "vjepa2_1_vit_base_384": 384,
    "vjepa2_1_vit_large_384": 384,
    "vjepa2_1_vit_giant_384": 384,
    "vjepa2_1_vit_gigantic_384": 384,
}


def hub_native_crop_size(hub_entry: str) -> int:
    return HUB_ENTRY_NATIVE_CROP.get(hub_entry, 256)


def _fix_vjepa_backbones_cdn_url(backbones_py: Path) -> bool:
    """
    Upstream facebookresearch/vjepa2 occasionally sets VJEPA_BASE_URL to
    http://localhost:8300 (internal testing). Rewrite to Meta's public CDN.
    """
    if not backbones_py.is_file():
        return False
    text = backbones_py.read_text()
    orig = text
    text = text.replace(
        'VJEPA_BASE_URL = "http://localhost:8300"',
        f'VJEPA_BASE_URL = "{VJEPA_CDN_BASE}"',
    )
    text = text.replace(
        "VJEPA_BASE_URL = 'http://localhost:8300'",
        f'VJEPA_BASE_URL = "{VJEPA_CDN_BASE}"',
    )
    if text != orig:
        backbones_py.write_text(text)
        return True
    return False


def _invalidate_cached_vjepa_backbones() -> None:
    """Force re-import of backbones.py after on-disk patch (avoids stale localhost URL)."""
    import sys

    for name, mod in list(sys.modules.items()):
        fn = getattr(mod, "__file__", None)
        if not fn:
            continue
        p = fn.replace("\\", "/")
        if "vjepa" in p.lower() and p.endswith("/hub/backbones.py"):
            del sys.modules[name]


def _prepare_vjepa_hub_cdn(hub_repo: str, hub_entry: str) -> None:
    """
    Ensure torch hub cache contains facebookresearch/vjepa2 and patch broken CDN URL
    before pretrained weights are downloaded.
    """
    hub_root = Path(torch.hub.get_dir())
    paths = list(hub_root.glob("facebookresearch_vjepa2*/src/hub/backbones.py"))
    if not paths:
        torch.hub.load(hub_repo, hub_entry, pretrained=False, trust_repo=True)
        paths = list(hub_root.glob("facebookresearch_vjepa2*/src/hub/backbones.py"))
    patched = False
    for p in paths:
        if _fix_vjepa_backbones_cdn_url(p):
            print(f"Patched V-JEPA weight URL in {p} (localhost → {VJEPA_CDN_BASE}).")
            patched = True
    if patched:
        _invalidate_cached_vjepa_backbones()


def _clean_hub_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def _load_hub_encoder_weights(encoder: torch.nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint is not a state-dict-like dict.")

    candidate_keys = ["ema_encoder", "target_encoder", "encoder", "model", "state_dict"]
    src = None
    for key in candidate_keys:
        if key in ckpt and isinstance(ckpt[key], dict):
            src = ckpt[key]
            break
    if src is None:
        src = ckpt

    state = _clean_hub_state_dict(src)
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    print(
        f"Loaded hub encoder checkpoint: {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_vjepa_encoder(cfg, device: torch.device | str) -> nn.Module:
    """
    Frozen V-JEPA encoder via Meta ``torch.hub`` (same pattern as ``check_jepa_output.py``).
    ``cfg`` may be ``argparse.Namespace`` or ``dict`` (e.g. checkpoint ``args``).
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    if _cfg_get(cfg, "backend") == "hf":
        raise ValueError(
            "HuggingFace encoder loading was removed. Retrain/decode using torch.hub only "
            "(see --hub-repo / --hub-entry)."
        )
    hub_repo = _cfg_get(cfg, "hub_repo", HUB_REPO_DEFAULT)
    hub_entry = _cfg_get(cfg, "hub_entry", HUB_ENTRY_DEFAULT)
    hub_checkpoint = _cfg_get(cfg, "hub_checkpoint") or ""
    no_pretrained = bool(_cfg_get(cfg, "no_pretrained", False))
    use_pretrained = (not no_pretrained) and (not hub_checkpoint)
    print(f"Loading V-JEPA via torch.hub: {hub_repo} / {hub_entry}")
    return VJEPAEncoder(
        hub_repo=hub_repo,
        hub_entry=hub_entry,
        device=device,
        pretrained=use_pretrained,
        hub_checkpoint=hub_checkpoint or None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Frozen V-JEPA 2 Encoder
# ──────────────────────────────────────────────────────────────────────────────

class VJEPAEncoder(nn.Module):
    """
    Meta ``facebookresearch/vjepa2`` backbone via ``torch.hub.load`` (encoder only).
    Forward: ``pixel_values`` [B, C, T, H, W] (ImageNet-normalized) → tokens [B, N, D].
    """

    def __init__(
        self,
        hub_repo: str = HUB_REPO_DEFAULT,
        hub_entry: str = HUB_ENTRY_DEFAULT,
        device="cpu",
        *,
        pretrained: bool = True,
        hub_checkpoint: str | None = None,
    ):
        super().__init__()
        use_pretrained = pretrained and not hub_checkpoint
        if use_pretrained:
            _prepare_vjepa_hub_cdn(hub_repo, hub_entry)
        model_obj = torch.hub.load(
            hub_repo,
            hub_entry,
            pretrained=use_pretrained,
            trust_repo=True,
        )
        self.model = model_obj[0] if isinstance(model_obj, (tuple, list)) else model_obj
        self.model = self.model.to(device).eval()

        if hub_checkpoint:
            _load_hub_encoder_weights(self.model, hub_checkpoint)
        elif use_pretrained:
            print(f"Loaded pretrained hub weights (CDN) for {hub_entry}.")

        m = self.model
        self.embed_dim  = m.embed_dim
        self.patch_size = m.patch_size
        self.crop_size  = int(m.img_height)
        self.tubelet    = m.tubelet_size
        self.n_spatial  = (self.crop_size // self.patch_size) ** 2

        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values)

    @property
    def num_frames_out(self) -> int:
        return max(1, self.model.num_frames // self.tubelet)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Spatio-Temporal Pooler
#     [B, T_tubes * N_spatial, D]  →  per-frame conditioning vectors [B, T, C]
# ──────────────────────────────────────────────────────────────────────────────

class SpatioTemporalPooler(nn.Module):
    """
    Splits the flat patch sequence back into (T_tubes, N_spatial) and
    applies cross-attention to pool each time-step into a single vector.
    """

    def __init__(self, embed_dim: int, n_spatial: int, cond_dim: int,
                 num_heads: int = 8):
        super().__init__()
        self.n_spatial = n_spatial
        self.proj_in   = nn.Linear(embed_dim, cond_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cond_dim))

        layer = nn.TransformerEncoderLayer(
            d_model=cond_dim, nhead=num_heads,
            dim_feedforward=cond_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(cond_dim)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        """
        patch_emb: [B, T_tubes * N, D]
        returns:   [B, T_tubes, cond_dim]
        """
        B, TN, D = patch_emb.shape
        T = TN // self.n_spatial

        # Reshape to [B*T, N, D]
        x = patch_emb.view(B * T, self.n_spatial, D)
        x = self.proj_in(x)                             # [B*T, N, cond_dim]

        # Prepend cls token per time-step
        cls = self.cls_token.expand(B * T, -1, -1)
        x   = torch.cat([cls, x], dim=1)                # [B*T, N+1, cond_dim]
        x   = self.attn(x)
        cond = self.norm(x[:, 0])                       # [B*T, cond_dim]

        return cond.view(B, T, -1)                      # [B, T, cond_dim]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Frame Decoder  (latent + cond → pixel frame)
# ──────────────────────────────────────────────────────────────────────────────

class FrameDecoder(nn.Module):
    """Convolutional upsampler: [z; cond] → [3, H, W]"""

    def __init__(self, latent_dim: int, cond_dim: int,
                 image_size: int = 256, base_ch: int = 128):
        super().__init__()
        in_dim = latent_dim + cond_dim

        # Number of 2× upsamples needed: 4×4 → image_size
        n_ups  = int(math.log2(image_size // 4))
        ch     = min(base_ch * (2 ** n_ups), 1024)

        self.fc = nn.Linear(in_dim, ch * 4 * 4)

        layers = []
        for _ in range(n_ups):
            out_ch = max(ch // 2, base_ch // 2)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(min(32, out_ch), out_ch),
                nn.SiLU(),
            ]
            ch = out_ch

        layers += [
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid(),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """z, cond: [B*T, *dim]  →  [B*T, 3, H, W]"""
        x = torch.cat([z, cond], dim=-1)
        x = self.fc(x).view(x.size(0), -1, 4, 4)
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Full Stochastic Video Decoder
# ──────────────────────────────────────────────────────────────────────────────

class StochasticVideoDecoder(nn.Module):
    """
    CVAE-style stochastic decoder for V-JEPA 2/2.1 spatio-temporal embeddings.

    Forward pass:
      patch_emb [B, T*N, D]
        → pool → cond [B, T, cond_dim]
        → VAE head → mu, logvar [B, T, latent_dim]
        → sample z
        → frame decoder → frames [B, T, 3, H, W]
    """

    def __init__(
        self,
        embed_dim:  int = 1024,   # V-JEPA 2 ViT-L
        n_spatial:  int = 256,    # spatial patches per time-step
        cond_dim:   int = 512,
        latent_dim: int = 256,
        image_size: int = 256,
        base_ch:    int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.pooler  = SpatioTemporalPooler(embed_dim, n_spatial, cond_dim)
        self.fc_mu   = nn.Linear(cond_dim, latent_dim)
        self.fc_lv   = nn.Linear(cond_dim, latent_dim)
        self.decoder = FrameDecoder(latent_dim, cond_dim, image_size, base_ch)

    def encode(self, patch_emb: torch.Tensor):
        """Returns cond [B,T,C], mu [B,T,L], logvar [B,T,L]"""
        cond   = self.pooler(patch_emb)
        mu     = self.fc_mu(cond)
        logvar = self.fc_lv(cond)
        return cond, mu, logvar

    def reparameterize(self, mu, logvar, temperature=1.0):
        if not self.training and temperature == 0:
            return mu
        std = (0.5 * logvar).exp() * temperature
        return mu + std * torch.randn_like(std)

    def forward(self, patch_emb: torch.Tensor, temperature: float = 1.0):
        """
        Args:
            patch_emb:   [B, T*N, D]
            temperature: 0 = deterministic, 1 = normal, >1 = creative
        Returns:
            frames:  [B, T, 3, H, W]  in [0, 1]
            mu:      [B, T, latent_dim]
            logvar:  [B, T, latent_dim]
        """
        B = patch_emb.size(0)

        cond, mu, logvar = self.encode(patch_emb)
        T = cond.size(1)

        z     = self.reparameterize(mu, logvar, temperature)
        # flatten time into batch for frame decoder
        cond_flat = cond.view(B * T, -1)
        z_flat    = z.view(B * T, -1)

        frames_flat = self.decoder(z_flat, cond_flat)   # [B*T, 3, H, W]
        frames = frames_flat.view(B, T, *frames_flat.shape[1:])

        return frames, mu, logvar


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Loss
# ──────────────────────────────────────────────────────────────────────────────

def elbo_loss(frames_pred, frames_target, mu, logvar, beta=1e-3):
    """
    frames_pred/target: [B, T, 3, H, W]
    β-VAE ELBO = recon_loss + β * KL
    """
    recon = F.mse_loss(frames_pred, frames_target)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Video Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_video_frames(path: str, n_frames: int, size: int) -> np.ndarray:
    """Load n_frames uniformly sampled from a video file → [T, H, W, C] uint8"""
    if HAS_TORCHCODEC:
        vr      = TorchCodecDecoder(path)
        total   = vr.metadata.num_frames_from_header or len(vr)
        idx     = np.linspace(0, total - 1, n_frames, dtype=int)
        frames  = vr.get_frames_at(indices=idx.tolist()).data  # [T, C, H, W]
        frames  = frames.permute(0, 2, 3, 1).numpy()          # [T, H, W, C]
    elif HAS_CV2:
        cap    = cv2.VideoCapture(path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx    = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        frames = np.stack(frames)
    else:
        raise RuntimeError("Install torchcodec or opencv-python for video loading")

    # Resize to crop_size
    resized = []
    for f in frames:
        from PIL import Image
        img = Image.fromarray(f).resize((size, size), Image.BICUBIC)
        resized.append(np.array(img))
    return np.stack(resized)   # [T, H, W, 3]


class VideoFolderDataset(Dataset):
    """
    Expects a flat folder (or nested ImageFolder-style) of video files.
    Supported extensions: .mp4 .avi .mov .mkv
    """

    EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, root: str, n_frames: int = 8, crop_size: int = 256):
        self.n_frames  = n_frames
        self.crop_size = crop_size
        self.paths     = []

        for dirpath, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.paths.append(os.path.join(dirpath, f))

        if not self.paths:
            raise RuntimeError(f"No video files found under {root}")
        print(f"Found {len(self.paths)} videos in {root}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            frames = load_video_frames(
                self.paths[idx], self.n_frames, self.crop_size
            )  # [T, H, W, 3]  uint8
        except Exception:
            # Skip corrupt videos — return next
            return self.__getitem__((idx + 1) % len(self))

        # [T, H, W, 3] → [T, 3, H, W] float32 [0,1]
        t = torch.from_numpy(frames).float() / 255.0
        return t.permute(0, 3, 1, 2)   # [T, 3, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Training
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    """Stack variable-length clips to fixed-length (pad/truncate)."""
    return torch.stack(batch)   # [B, T, 3, H, W]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("⚠️  GPU not found — training will be very slow on CPU.")

    # ── Encoder ─────────────────────────────────────────────────────────────
    encoder   = build_vjepa_encoder(args, device)
    embed_dim = encoder.embed_dim
    n_spatial = encoder.n_spatial
    crop_size = encoder.crop_size

    # ── Normalisation (matches V-JEPA 2 processor internals) ────────────────
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset    = VideoFolderDataset(args.data_dir,
                                    n_frames=args.frames,
                                    crop_size=crop_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
    )

    # ── Decoder ──────────────────────────────────────────────────────────────
    decoder = StochasticVideoDecoder(
        embed_dim  = embed_dim,
        n_spatial  = n_spatial,
        cond_dim   = args.cond_dim,
        latent_dim = args.latent_dim,
        image_size = crop_size,
        base_ch    = args.base_ch,
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"Decoder parameters: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-6
    )

    start_epoch = 1
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        decoder.load_state_dict(resume_ckpt["state_dict"])
        start_epoch = int(resume_ckpt.get("epoch", 0)) + 1
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        print(
            f"Resumed decoder from {args.resume} "
            f"(checkpoint epoch {resume_ckpt.get('epoch', 'unknown')})"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    end_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        decoder.train()
        totals = dict(loss=0.0, recon=0.0, kl=0.0)

        for step, frames_raw in enumerate(dataloader):
            # frames_raw: [B, T, 3, H, W]  float [0,1]
            frames_raw = frames_raw.to(device)
            B, T, C, H, W = frames_raw.shape

            # ── Encoder needs [B, C, T, H, W] normalized ────────────────────
            # reshape (not view): Normalize output can be non-contiguous
            frames_norm = normalize(
                frames_raw.reshape(B * T, C, H, W)
            ).reshape(B, C, T, H, W)

            with torch.no_grad():
                patch_emb = encoder(frames_norm)    # [B, T_tubes*N, D]

            # ── Decoder ─────────────────────────────────────────────────────
            frames_pred, mu, logvar = decoder(patch_emb, temperature=1.0)

            # Sub-sample T_tubes frames from ground-truth to match decoder output
            T_dec = frames_pred.size(1)
            if T_dec != T:
                idx_t = torch.linspace(0, T - 1, T_dec).long()
                frames_gt = frames_raw[:, idx_t]
            else:
                frames_gt = frames_raw

            loss, r, kl = elbo_loss(frames_pred, frames_gt,
                                    mu, logvar, beta=args.beta)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            totals["loss"]  += loss.item()
            totals["recon"] += r.item()
            totals["kl"]    += kl.item()

            if (step + 1) % args.log_every == 0:
                n = step + 1
                print(
                    f"  Epoch {epoch:3d} | step {n:5d} | "
                    f"loss={totals['loss']/n:.4f}  "
                    f"recon={totals['recon']/n:.4f}  "
                    f"kl={totals['kl']/n:.5f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

        # ── Visual samples every epoch ───────────────────────────────────────
        decoder.eval()
        with torch.no_grad():
            sample_raw = frames_raw[:min(2, B)]         # 2 clips
            B2, T2, C2, H2, W2 = sample_raw.shape
            fn = normalize(sample_raw.reshape(B2 * T2, C2, H2, W2)).reshape(
                B2, C2, T2, H2, W2
            )
            emb = encoder(fn)
            pred_det, _, _ = decoder(emb, temperature=0.0)  # [B2, T_dec, 3, H, W]
            pred_sto, _, _ = decoder(emb, temperature=1.0)

            # Save first frame of each clip: orig | det | stochastic
            T_dec = pred_det.size(1)
            idx_t = torch.linspace(0, T2-1, T_dec).long()
            gt_frame  = sample_raw[:, idx_t[0]]   # [B2, 3, H, W]
            det_frame = pred_det[:, 0]
            sto_frame = pred_sto[:, 0]

            grid = torch.cat([gt_frame, det_frame, sto_frame], dim=0)
            save_image(
                grid,
                os.path.join(args.out_dir, f"epoch_{epoch:03d}.png"),
                nrow=B2,
            )

        # ── Checkpoint ───────────────────────────────────────────────────────
        ckpt = dict(
            epoch      = epoch,
            state_dict = decoder.state_dict(),
            optimizer_state_dict = optimizer.state_dict(),
            scheduler_state_dict = scheduler.state_dict(),
            args       = vars(args),
            embed_dim  = embed_dim,
            n_spatial  = n_spatial,
            crop_size  = crop_size,
        )
        path = os.path.join(args.out_dir, "decoder.pth")
        torch.save(ckpt, path)
        print(f"Epoch {epoch} — saved checkpoint to {path}\n")


# ──────────────────────────────────────────────────────────────────────────────
# 8.  Decode a video file → reconstructed frames
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load encoder ─────────────────────────────────────────────────────────
    ckpt      = torch.load(args.checkpoint, map_location=device)
    saved     = ckpt["args"]
    encoder   = build_vjepa_encoder(saved, device)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )

    # ── Load decoder ─────────────────────────────────────────────────────────
    decoder = StochasticVideoDecoder(
        embed_dim  = ckpt["embed_dim"],
        n_spatial  = ckpt["n_spatial"],
        cond_dim   = saved["cond_dim"],
        latent_dim = saved["latent_dim"],
        image_size = ckpt["crop_size"],
        base_ch    = saved["base_ch"],
    ).to(device)
    decoder.load_state_dict(ckpt["state_dict"])
    decoder.eval()
    print(f"Loaded decoder (epoch {ckpt['epoch']}) from {args.checkpoint}")

    # ── Load & encode video ───────────────────────────────────────────────────
    frames_np = load_video_frames(
        args.video, saved["frames"], ckpt["crop_size"]
    )  # [T, H, W, 3]

    frames_t  = torch.from_numpy(frames_np).float() / 255.0
    frames_t  = frames_t.permute(0, 3, 1, 2)              # [T, 3, H, W]
    T, C, H, W = frames_t.shape

    frames_in = frames_t.unsqueeze(0).to(device)           # [1, T, 3, H, W]
    B0, T0, C0, H0, W0 = frames_in.shape
    fn = normalize(frames_in.reshape(B0 * T0, C0, H0, W0)).reshape(
        B0, C0, T0, H0, W0
    )
    patch_emb = encoder(fn)                                # [1, T_tubes*N, D]

    # ── Decode with multiple stochastic samples ───────────────────────────────
    samples = []
    for s in range(args.num_samples):
        temp = 0.0 if s == 0 else args.temperature
        pred, _, _ = decoder(patch_emb, temperature=temp)  # [1, T_dec, 3, H, W]
        samples.append(pred.squeeze(0))                    # [T_dec, 3, H, W]

    # ── Save frame grid PNG ───────────────────────────────────────────────────
    T_dec = samples[0].size(0)
    idx_t = torch.linspace(0, T - 1, T_dec).long()
    gt_row = frames_t[idx_t].to(samples[0].device)  # [T_dec, 3, H, W]  ground truth

    rows = [gt_row] + samples
    grid = torch.cat(rows, dim=0)   # [(1+S)*T_dec, 3, H, W]

    out_png = args.out.replace(".mp4", ".png") if args.out.endswith(".mp4") \
              else args.out
    save_image(grid, out_png, nrow=T_dec)
    print(f"Saved frame grid → {out_png}")
    print(f"  Row 0        : ground truth frames")
    print(f"  Row 1        : deterministic decode (temperature=0)")
    print(f"  Rows 2–{args.num_samples}: stochastic samples (temperature={args.temperature})")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Library API
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_embeddings(
    patch_emb: torch.Tensor,          # [B, T*N, D]  V-JEPA 2 embeddings
    decoder:   StochasticVideoDecoder,
    temperature: float = 1.0,
    n_samples:   int   = 4,
) -> torch.Tensor:
    """
    Decode raw V-JEPA 2/2.1 patch embeddings into pixel frames.

    Returns:
        samples: [n_samples, B, T_dec, 3, H, W]
    """
    if patch_emb.dim() == 2:
        patch_emb = patch_emb.unsqueeze(0)

    decoder.eval()
    results = []
    for i in range(n_samples):
        t = 0.0 if i == 0 else temperature
        frames, _, _ = decoder(patch_emb, temperature=t)
        results.append(frames)

    return torch.stack(results)   # [n_samples, B, T_dec, 3, H, W]


# ──────────────────────────────────────────────────────────────────────────────
# 10. CLI
# ──────────────────────────────────────────────────────────────────────────────

def make_parser():
    p = argparse.ArgumentParser(
        description="V-JEPA 2/2.1 Stochastic Video Decoder"
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    t = sub.add_parser("train", help="Train the stochastic decoder")
    t.add_argument("--data_dir",  required=True,
                   help="Root folder containing video files (recursive)")
    t.add_argument(
        "--hub-repo",
        default=HUB_REPO_DEFAULT,
        help="torch.hub GitHub repo (default: facebookresearch/vjepa2).",
    )
    t.add_argument(
        "--hub-entry",
        default=HUB_ENTRY_DEFAULT,
        help="torch.hub entry (e.g. vjepa2_vit_large, vjepa2_1_vit_large_384).",
    )
    t.add_argument(
        "--hub-checkpoint",
        default="",
        help="Optional local .pt for encoder weights (skips CDN; loads into backbone).",
    )
    t.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random-init backbone unless --hub-checkpoint is set.",
    )
    t.add_argument("--out_dir",   default="vjepa2_decoder_out")
    t.add_argument(
        "--resume",
        default="",
        help="Optional path to decoder checkpoint to continue training.",
    )
    t.add_argument("--epochs",    type=int,   default=30)
    t.add_argument("--batch_size",type=int,   default=4,
                   help="Clips per batch — keep low for GPU memory")
    t.add_argument("--frames",    type=int,   default=8,
                   help="Frames sampled per video clip")
    t.add_argument("--workers",   type=int,   default=4)
    t.add_argument("--lr",        type=float, default=3e-4)
    t.add_argument("--beta",      type=float, default=1e-3,
                   help="KL weight; lower=sharper, higher=more diverse")
    t.add_argument("--cond_dim",  type=int,   default=512)
    t.add_argument("--latent_dim",type=int,   default=256)
    t.add_argument("--base_ch",   type=int,   default=128)
    t.add_argument("--log_every", type=int,   default=50)

    # ── decode ─────────────────────────────────────────────────────────────
    d = sub.add_parser("decode", help="Decode a video clip using trained decoder")
    d.add_argument("--video",       required=True, help="Input video path")
    d.add_argument("--checkpoint",  required=True, help="Path to decoder.pth")
    d.add_argument("--out",         default="decoded_output.png")
    d.add_argument("--num_samples", type=int,   default=4)
    d.add_argument("--temperature", type=float, default=1.0)

    return p


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "decode":
        decode(args)