"""
Continuous Trajectory Modeling via Implicit Neural Representations (INRs).

Theory
------
Deep deblurring methods almost always treat the motion-blur kernel as a
free-form discrete 2D matrix (or a dense per-pixel flow field). Physically,
motion blur is the integral of light hitting the sensor along a *continuous*
camera trajectory theta(t) in SE(3) during the exposure window:

    y = (1 / dt) * int_0^dt  W(x, theta(t))  dt

where W is a warping function and x is the (unknown) sharp scene. Estimating
theta(t) from a single blurry image is extremely ill-posed if theta(t) is
allowed to be an arbitrary function of t (equivalent to allowing an arbitrary
dense blur kernel). This module instead represents theta(t) with a
conditional SIREN (Sitzmann et al., 2020) -- a coordinate-MLP with sinusoidal
activations that is continuous and differentiable in t by construction, and
has far fewer effective degrees of freedom than a dense kernel. Restricting
the search space to trajectories this network can express is the
regularizer that is meant to reduce the ill-posedness of the inverse
problem (see also `models/ContinuousTrajectory/README.md` for the write-up
and the smoothness prior in `TrajectoryINR_model.trajectory_smoothness_loss`
that further biases theta(t) towards physically plausible camera paths).

Pipeline
--------
1. `SharpImageEncoderDecoder` maps the blurry observation y -> a candidate
   sharp image x_hat and a global latent code z.
2. `TrajectorySIREN`, conditioned on z via FiLM, maps stratified time
   samples t in [0, 1] -> theta(t) = (rotvec(t), trans(t)), an SE(3) pose
   relative to the (arbitrary) reference frame of x_hat.
3. `pose_to_homography` converts each theta(t) to a per-time homography
   under a single fronto-parallel scene-plane assumption (the standard
   projective motion-blur model, e.g. Whyte et al. 2012, generalized here
   with a translation/parallax term). Setting the out-of-plane rotation and
   z-translation to zero recovers exactly the discrete in-plane kernel model
   already used elsewhere in this repo for synthetic data generation
   (psf/motionblur/trajectory2psf.py::get_transformation_matrix) -- i.e.
   that model is the theta_z-only special case of this one.
4. `warp_batch` forward-warps x_hat by each homography (matching
   cv2.warpPerspective's convention, so H here is directly comparable to the
   `Ms` returned by psf/motionblur/trajectory2psf.py::trajectory2psfs) and
   averages over t to synthesize a blurred image, closing the loop for
   analysis-by-synthesis (self-supervised) training.

This is a v1 scaffold: the encoder/decoder is a small U-Net (swap for
NAFNet/Restormer from models/archs/ once the trajectory-INR idea is
validated), and the scene-plane assumption ignores true depth parallax.
Both are called out again inline below.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


# ---------------------------------------------------------------------------
# Conditional SIREN: theta(t) = (rotvec(t), trans(t))
# ---------------------------------------------------------------------------
class FiLMSineLayer(nn.Module):
    """A SIREN layer (Sitzmann et al., 2020) modulated by a per-image latent
    code z via FiLM (scale/shift), so a single network is amortized across a
    whole dataset instead of being re-fit per image."""

    def __init__(self, in_features, out_features, latent_dim, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.film = nn.Linear(latent_dim, 2 * out_features)
        self._init_weights(in_features, out_features, is_first)

    def _init_weights(self, in_features, out_features, is_first):
        with torch.no_grad():
            bound = 1.0 / in_features if is_first else math.sqrt(6.0 / in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            # FiLM starts as the identity modulation (scale=1, shift=0), so
            # at init this behaves like a plain (unconditioned) SIREN.
            self.film.weight.zero_()
            self.film.bias.zero_()
            self.film.bias[:out_features] = 1.0

    def forward(self, x, z):
        # x: (B, N, in_features)   z: (B, latent_dim)
        scale, shift = self.film(z).chunk(2, dim=-1)  # (B, out_features) each
        h = self.linear(x) * scale.unsqueeze(1) + shift.unsqueeze(1)
        return torch.sin(self.omega_0 * h)


class TrajectorySIREN(nn.Module):
    """Coordinate network theta: [0, 1] -> SE(3) pose (axis-angle rotation +
    translation), conditioned on a per-image latent code z."""

    def __init__(self, latent_dim, hidden_dim=256, num_layers=5, omega_0=30.0,
                 max_rotation_deg=8.0, max_translation=0.15):
        super().__init__()
        self.max_rotation = math.radians(max_rotation_deg)
        self.max_translation = max_translation

        layers = [FiLMSineLayer(1, hidden_dim, latent_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(FiLMSineLayer(hidden_dim, hidden_dim, latent_dim, omega_0=omega_0))
        self.layers = nn.ModuleList(layers)

        self.head = nn.Linear(hidden_dim, 6)  # 3 rotvec + 3 translation
        # theta(t) = identity everywhere at init -> training starts from "no
        # blur" and only introduces motion where it helps reconstruction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, t, z):
        """t: (N,) or (B, N) in [0, 1].  z: (B, latent_dim).
        Returns rotvec, trans: (B, N, 3) each."""
        b = z.shape[0]
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(b, -1)
        h = t.unsqueeze(-1)  # (B, N, 1)
        for layer in self.layers:
            h = layer(h, z)
        out = self.head(h)  # (B, N, 6)
        rotvec = torch.tanh(out[..., :3]) * self.max_rotation
        trans = torch.tanh(out[..., 3:]) * self.max_translation
        return rotvec, trans


# ---------------------------------------------------------------------------
# SE(3) pose -> homography, and a differentiable batched warp
# ---------------------------------------------------------------------------
def rotvec_to_matrix(rotvec):
    """Rodrigues' formula, batched. rotvec: (..., 3) -> R: (..., 3, 3)."""
    theta = torch.norm(rotvec, dim=-1, keepdim=True).clamp_min(1e-8)  # (..., 1)
    axis = rotvec / theta
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    Kx = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)  # (..., 3, 3)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype).expand(Kx.shape)
    theta = theta[..., None]  # (..., 1, 1)
    R = eye + torch.sin(theta) * Kx + (1 - torch.cos(theta)) * torch.matmul(Kx, Kx)
    return R


def make_intrinsics(h, w, device, dtype, focal_px=None):
    f = float(focal_px) if focal_px is not None else float(max(h, w))
    K = torch.tensor([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
    Kinv = torch.inverse(K)
    return K, Kinv


def pose_to_homography(rotvec, trans, K, Kinv):
    """theta(t) -> homography under a single fronto-parallel scene plane at
    unit (normalized) depth: H(t) = K (R(t) + T(t) n^T) K^{-1}, n = [0,0,1]^T.
    rotvec, trans: (B, N, 3).  K, Kinv: (3, 3).  Returns H: (B, N, 3, 3).

    This is a simplifying assumption -- true parallax depends on per-pixel
    depth, which this v1 does not model. R_x = R_y = T_z = 0 recovers the
    discrete in-plane kernel model of
    psf/motionblur/trajectory2psf.py::get_transformation_matrix exactly.
    """
    R = rotvec_to_matrix(rotvec)  # (B, N, 3, 3)
    n = torch.zeros_like(trans)
    n[..., 2] = 1.0
    M = R + trans.unsqueeze(-1) * n.unsqueeze(-2)  # T n^T outer product, (B, N, 3, 3)
    H = K.view(1, 1, 3, 3) @ M @ Kinv.view(1, 1, 3, 3)
    return H


def warp_batch(image, H):
    """Forward-warps `image` by H(t), matching cv2.warpPerspective's
    convention (H maps *source* pixel coords to *destination* pixel coords,
    i.e. dst(x,y) = src(H^{-1}(x,y))), so H here is directly comparable to
    the `Ms` produced by psf/motionblur/trajectory2psf.py::trajectory2psfs.
    image: (B, C, H, W).  H: (B, N, 3, 3).  Returns (B, N, C, H, W)."""
    b, c, h, w = image.shape
    n = H.shape[1]

    ys, xs = torch.meshgrid(
        torch.linspace(0, h - 1, h, device=image.device, dtype=image.dtype),
        torch.linspace(0, w - 1, w, device=image.device, dtype=image.dtype),
        indexing='ij')
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=0).reshape(3, -1)  # (3, H*W), (x, y, 1)
    grid = grid.view(1, 1, 3, h * w).expand(b, n, 3, h * w)

    Hinv = torch.inverse(H)
    src = torch.matmul(Hinv, grid)  # (B, N, 3, H*W)
    src = src[:, :, :2] / src[:, :, 2:3].clamp_min(1e-8)
    src_x = src[:, :, 0].reshape(b, n, h, w)
    src_y = src[:, :, 1].reshape(b, n, h, w)

    src_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    src_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    sample_grid = torch.stack([src_x, src_y], dim=-1).reshape(b * n, h, w, 2)

    image_rep = image.unsqueeze(1).expand(b, n, c, h, w).reshape(b * n, c, h, w)
    warped = F.grid_sample(image_rep, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped.reshape(b, n, c, h, w)


# ---------------------------------------------------------------------------
# Backbone: blurry image -> (sharp image estimate, latent code)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SharpImageEncoderDecoder(nn.Module):
    """A small U-Net that produces (a) a global latent code conditioning the
    trajectory SIREN and (b) a candidate sharp image x_hat. Deliberately
    lightweight for a v1 scaffold -- swap for NAFNet/Restormer
    (models/archs/) once the trajectory-INR idea is validated."""

    def __init__(self, in_channels=3, out_channels=3, base_channels=32, latent_dim=256):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2, stride=2)
        self.enc3 = ConvBlock(c2, c3, stride=2)
        self.enc4 = ConvBlock(c3, c4, stride=2)

        self.to_latent = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c4, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim),
        )

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, 3, padding=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        z = self.to_latent(e4)

        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        x_hat = torch.tanh(self.out_conv(d1))[..., :h, :w]
        return x_hat, z


# ---------------------------------------------------------------------------
# Top-level arch (auto-discovered via models/archs/__init__.py::find_network_class)
# ---------------------------------------------------------------------------
class TrajectoryINR_config(PretrainedConfig):
    model_type = "TrajectoryINR_arch"

    def __init__(self, in_channels=3, out_channels=3, base_channels=32, latent_dim=256,
                 siren_hidden_dim=256, siren_num_layers=5, siren_omega_0=30.0,
                 max_rotation_deg=8.0, max_translation=0.15, num_time_samples=32,
                 focal_px=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.siren_hidden_dim = siren_hidden_dim
        self.siren_num_layers = siren_num_layers
        self.siren_omega_0 = siren_omega_0
        self.max_rotation_deg = max_rotation_deg
        self.max_translation = max_translation
        self.num_time_samples = num_time_samples
        self.focal_px = focal_px


class TrajectoryINR_arch(PreTrainedModel):
    """See module docstring for the full theory/pipeline write-up."""
    config_class = TrajectoryINR_config

    def __init__(self, config):
        super().__init__(config)
        self.backbone = SharpImageEncoderDecoder(
            in_channels=config.in_channels, out_channels=config.out_channels,
            base_channels=config.base_channels, latent_dim=config.latent_dim)
        self.trajectory = TrajectorySIREN(
            latent_dim=config.latent_dim, hidden_dim=config.siren_hidden_dim,
            num_layers=config.siren_num_layers, omega_0=config.siren_omega_0,
            max_rotation_deg=config.max_rotation_deg, max_translation=config.max_translation)
        self.num_time_samples = config.num_time_samples
        self.focal_px = config.focal_px

    def sample_times(self, n, device, dtype):
        """Stratified random samples in [0, 1] while training (keeps the
        network honest about being a *continuous* function of t, rather than
        memorizing a fixed discrete set); a uniform grid otherwise."""
        if self.training:
            edges = torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
            u = torch.rand(n, device=device, dtype=dtype)
            t = edges[:-1] + u * (edges[1:] - edges[:-1])
        else:
            t = torch.linspace(0, 1, n, device=device, dtype=dtype)
        return torch.sort(t).values

    def forward(self, blurry, num_time_samples=None, t=None):
        b, c, h, w = blurry.shape
        x_hat, z = self.backbone(blurry)

        n = num_time_samples or self.num_time_samples
        if t is None:
            t = self.sample_times(n, blurry.device, blurry.dtype)

        rotvec, trans = self.trajectory(t, z)  # (B, N, 3) each
        K, Kinv = make_intrinsics(h, w, blurry.device, blurry.dtype, focal_px=self.focal_px)
        H = pose_to_homography(rotvec, trans, K, Kinv)  # (B, N, 3, 3)
        warped = warp_batch(x_hat, H)  # (B, N, C, H, W)
        blur_hat = warped.mean(dim=1)

        return {
            'sharp': x_hat,
            'blur': blur_hat,
            'warped': warped,
            'rotvec': rotvec,
            'trans': trans,
            't': t,
        }
