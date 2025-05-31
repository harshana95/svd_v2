# Explore some image rendering with metasurfaces
import torch
torch.cuda.empty_cache()
import gc
gc.collect()

import matplotlib.pyplot as plt
import mat73
from scipy import interpolate
import numpy as np
import torch
from einops import rearrange

from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import hsi_to_rgb
from dflat.plot_utilities import format_plot
from dflat.render import Fronto_Planar_Renderer_Incoherent
out_distance_m= 1e-2
in_size = [513, 513]
wv = [550e-9]
settings = {
    "in_size": in_size,
    "in_dx_m": [10*300e-9, 10*300e-9],
    "wavelength_set_m": wv,
    "depth_set_m": [1e-2],
    "fshift_set_m": [[0.0, 0.0]],
    "out_distance_m": out_distance_m,
    "aperture_radius_m": None,
    "radial_symmetry": True  # will generate all values, but slice values along one radius
    }
amp, phase, aperture = focusing_lens(**settings)
print("amp phase aperture", amp.shape, phase.shape, aperture.shape)
#  Phase, and Aperture profiles of shape [Z, Lam, H, W]

# Reverse look-up to find the metasurface that implements the target profile
model_name = "Nanocylinders_TiO2_U300H600"
p_norm, p, err = reverse_lookup_optimize(
    amp[None, None],
    phase[None, None],
    wv,
    model_name,
    lr=1e-1,
    err_thresh=1e-6,
    max_iter=500,
    opt_phase_only=False)
print("p", p.shape)
# [B, H, W, D] where D = model.dim_in - 1 is the number of shape parameters . 

# Predict the broadband phase and amplitude imparted by this collection of meta-atoms
model = load_optical_model(model_name)
model = model.to("cuda")
sim_wl = np.linspace(400e-9, 700e-9, 31)
est_amp, est_phase = model(p, sim_wl, pre_normalized=False)
print("est_amp, est_phase", est_amp.shape, est_phase.shape) 
# [B, pol, Lam, H, W] where pol is 1 or 2

# Compute the point spread function given this broadband stack of field amplitude and phases
PSF = PointSpreadFunction(
    in_size=in_size,
    in_dx_m=[10*300e-9, 10*300e-9],
    out_distance_m=out_distance_m,
    out_size=[512, 512],
    out_dx_m=[5e-6,5e-6],
    out_resample_dx_m=None,
    radial_symmetry=True,
    diffraction_engine="ASM")
ps_locs = [[0.0, 0.0, 1e-2], [50e-6,0.0,1e-2]] # in m
ps_idx = np.array(ps_locs)
in_dx_m = np.array(PSF.in_dx_m)
ps_idx[:, 0] /= in_dx_m[0]
ps_idx[:, 1] /= in_dx_m[1]
ps_idx = ps_idx.astype(int)[:, :2]
print(ps_idx, in_dx_m)


psf_intensity, _ = PSF(
    est_amp.to(dtype=torch.float32, device='cuda'),
    est_phase.to(dtype=torch.float32, device='cuda'),
    sim_wl,
    ps_locs,
    aperture=None,
    normalize_to_aperture=True)
print("psf_intensity", psf_intensity.shape)
# shape [B P Z L H W] where Z = len(ps_locs)

# Display a slice of the radially symmetric point-spread function we will use
fig, ax = plt.subplots(1,1, figsize=(10,5))
ax.imshow(psf_intensity[0,0,0,:,256,:].detach().cpu().numpy())
format_plot(fig, ax, ylabel="wavelength (nm)", xlabel="sensor x", yvec=sim_wl*1e9, xvec = np.arange(0,512,1), setAspect="auto")
fig, ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(psf_intensity[0,0,0,10,:,:].detach().cpu().numpy())
ax[1].imshow(psf_intensity[0,0,1,10,:,:].detach().cpu().numpy())
format_plot(fig, ax[0], ylabel="sensor y", xlabel="sensor x", yvec=np.arange(0,512,1), xvec = np.arange(0,512,1), setAspect="auto")
format_plot(fig, ax[1], ylabel="sensor y", xlabel="sensor x", yvec=np.arange(0,512,1), xvec = np.arange(0,512,1), setAspect="auto")

# Load a hyperspectral image for rendering

def load_mat_dat(path):
    # Note: not sure what norm_factor is used for
    # Investigate or in the future add a note once identified
    # Upon inspection, bands are provided in units of nm and range from 400 to 700 nm for Arad1k
    # with a 10 nm step size
    mat_dat = mat73.loadmat(path)
    bands = mat_dat["bands"]
    cube = mat_dat["cube"]
    _ = mat_dat["norm_factor"]
    return bands, cube


def interpolate_HS_Cube(new_channels_nm, hs_cube, hs_bands):
    # Throw an error if we try to extrapolate
    if (min(new_channels_nm) < min(hs_bands) - 1) or (
        max(new_channels_nm) > max(hs_bands) + 1
    ):
        raise ValueError(
            f"In generator, extrapoaltion of the ARAD dataset outside of measurement data is not allowed: {min(hs_bands)}-{max(hs_bands)}"
        )

    interpfun = interpolate.interp1d(
        hs_bands,
        hs_cube,
        axis=-1,
        kind="linear",
        assume_sorted=True,
        fill_value="extrapolate",
        bounds_error=False,
    )
    resampled = interpfun(new_channels_nm)

    return resampled

# Load hyperspectral data cube needed for rendering
sim_wl = np.linspace(400e-9, 700e-9, 31)
bands, cube = load_mat_dat("./ARAD_1K_0001.mat")
hsi_data = interpolate_HS_Cube(sim_wl*1e9, cube, bands)
hsi_data = np.clip(hsi_data, 0, 1)
rgb_img = hsi_to_rgb(hsi_data[None], sim_wl)[0]
print("hsi_data, rgb_img", hsi_data.shape, rgb_img.shape)

fig, ax  = plt.subplots(1,1)
ax.imshow(rgb_img)
ax.axis('off')
ax.set_title("HSI Projected to RGB")
plt.show()

renderer = Fronto_Planar_Renderer_Incoherent()
# Need to pass inputs like
# psf has shape  [B P Z L H W]
# scene radiance [1 P Z L H W]
hsi_cube = rearrange(hsi_data, "H W C -> C H W")[None, None, None]
hsi_cube = torch.tensor(hsi_cube, dtype=torch.float32, device='cuda')
psf_intensity = psf_intensity.to('cuda')
print("psf_intensity, hsi_cube", psf_intensity.shape, hsi_cube.shape)

out_int = renderer(psf_intensity, hsi_cube, rfft=True, crop_to_psf_dim=False)
meas_rgb = renderer.rgb_measurement(out_int, sim_wl, gamma=True)#, bayer_mosaic=True)
meas_gs = torch.sum(out_int, -3, keepdim=True)
print("meas_rgb, meas_gs", meas_rgb.shape, meas_gs.shape)

# Displayed the RGB measurement after the metasurface
fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(rgb_img)
ax[1].imshow(np.transpose(meas_rgb[0,0,0].detach().cpu().numpy(), [1,2,0]))
ax[2].imshow(meas_gs[0,0,0,0].detach().cpu().numpy(), cmap='gray')
[axi.axis('off') for axi in ax.flatten()]
plt.show()