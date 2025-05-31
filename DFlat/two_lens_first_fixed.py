# Use dflat_v2 conda environment
import sys
import os
import matplotlib

# Use the Agg backend for non-interactive plotting
matplotlib.use('Agg')
# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
from dflat.initialize import focusing_lens
from dflat.metasurface import load_optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction  # Using the legacy version as it takes wavelength as a forward input instead of initialization input

import einops
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

from utils.PSFfromWavefront import PointSpreadFunctionNew
from utils.dataset_utils import crop_arr
from utils.dflat_utils import reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_size=[2049, 2049]
in_dx_m=[2.5e-6, 2.5e-6]
wavelength_set_m=[532e-9]
out_dx_m=[2.74e-6, 2.74e-6] # Basler dart M dmA4096-9gm (Mono) pixel size multiply 3
out_size=[2049, 2049]

depth_set_m=[99999]
fshift_set_m=[[0.0, 0.0]]
out_distance_m=10e-3
second_metasurface_depth_m=1e-3
second_metasurface_psf_aperture_r = 0.1e-3
aperture_radius_m=2e-3
radial_symmetry=False

threshold = 30 # Threshold for minimum cluster size
cluster_idx = 4 # select the cluster for second propagation

diffraction_engine_2nd="Fresnel"

ps_locs = [[50, 50, 1e3]]
sim_wl = [532e-9]

x,y,z = ps_locs[0]
z0 = out_distance_m - second_metasurface_depth_m
x0 = -x * z0 / z
y0 = -y * z0 / z
x0_px = int(x0 / in_dx_m[1]) + in_size[1]//2
y0_px = int(y0 / in_dx_m[0]) + in_size[0]//2

incident_angle = np.arctan((ps_locs[0][0]**2 + ps_locs[0][1]**2)**0.5/ps_locs[0][2])
print(f"Incident angle: {incident_angle*180/np.pi:.2f} degrees")

px1 = int(second_metasurface_psf_aperture_r / in_dx_m[0])
px2 = px1 + 1  # should be odd

ul = [y0_px -px1, x0_px - px1]
br = [y0_px +px2, x0_px + px2]

def add_colorbar(im, ax, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

# define the 2nd lens
amp, phase, aperture = focusing_lens(
    in_size=in_size,
    in_dx_m=in_dx_m,
    wavelength_set_m=wavelength_set_m,
    depth_set_m=depth_set_m,
    fshift_set_m=fshift_set_m,
    out_distance_m=second_metasurface_depth_m,
    aperture_radius_m=aperture_radius_m,
    radial_symmetry=radial_symmetry
)
# returns focusing profiles of shape [Z, Lam, H, W]
print(amp.shape, phase.shape, aperture.shape)

fig, ax = plt.subplots(1,2, figsize=(6,3))
ax[0].imshow(amp[0]*aperture[0], vmin=0, vmax=1)
ax[0].set_title("amplitude")
ax[1].imshow(phase[0])
ax[1].set_title("phase")
plt.savefig("5. second lens.png")
plt.close(fig)

# Nearest neightbor lookup
amp_exp = np.expand_dims(np.expand_dims(amp, axis=0), axis=0)
phase_exp = np.expand_dims(np.expand_dims(phase, axis=0), axis=0)
aperture_exp  = np.expand_dims(np.expand_dims(aperture, axis=0), axis=0)
reverse_lookup_f = reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm
p_norm, p, err = reverse_lookup_f(amp_exp, phase_exp, wavelength_set_m[0:1])
model_name = 'Nanocylinders_TiO2_U300H600'
optical_model = load_optical_model(model_name).cuda()
with torch.no_grad():
    est_amp, est_phase = optical_model(torch.from_numpy(p).cuda(), wavelength_set_m, pre_normalized=False)
print(f"est_amp: {est_amp.shape}, est_phase: {est_phase.shape}")

# ================================ 1. calculate the light field 1mm before the photosensor
# build the wavefront. This is common for all positions at this time
xx2, yy2 = torch.meshgrid(torch.arange(in_size[1])-in_size[1]//2, torch.arange(in_size[0])-in_size[0]//2)
xx2 = xx2[None]*in_dx_m[1]
yy2 = yy2[None]*in_dx_m[0]
xx2 -= x0
yy2 -= y0
phi = -2*np.pi/torch.tensor(wavelength_set_m)* torch.sqrt((xx2)**2 + (yy2)**2 + second_metasurface_depth_m**2)
A = ((xx2)**2 + (yy2)**2 < second_metasurface_psf_aperture_r**2).float()

torch_zero = torch.tensor([0.0], dtype=torch.float32)
incident_wavefront = torch.complex(A[None,None], torch_zero) * torch.exp(torch.complex(torch_zero, phi[None,None]))
amp_iw = torch.abs(incident_wavefront)
phase_iw = torch.angle(incident_wavefront)

fig = plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
plt.imshow(amp_iw.cpu().numpy()[0,0].transpose([1,2,0]))
plt.title("Amplitude")
plt.colorbar()

fig.add_subplot(1,2,2)
plt.imshow(phase_iw.cpu().numpy()[0,0].transpose([1,2,0]))
plt.title("Phase")
plt.colorbar()
plt.savefig("6.Wavefront before 2nd MS.png")
plt.close(fig)


# ==================== 2. Propgating for 1mm from the second metasurface to the photosensor


PSF_2ndprop = PointSpreadFunctionNew(
    in_size=[px1+px2,px1+px2],  # first prop gives in_size as output
    in_dx_m=in_dx_m,
    out_distance_m=second_metasurface_depth_m,
    out_size=[px1+px2,px1+px2],
    out_dx_m=out_dx_m,
    out_resample_dx_m=None,
    radial_symmetry=False,
    diffraction_engine=diffraction_engine_2nd).to(device)

phase_2nd_ms = est_phase[..., ul[0]:br[0], ul[1]:br[1]]
amp_2nd_ms = est_amp[..., ul[0]:br[0], ul[1]:br[1]]
aperture_2nd_ms = amp_exp[..., ul[0]:br[0], ul[1]:br[1]]
iw = incident_wavefront.to(device)[..., ul[0]:br[0], ul[1]:br[1]]
print(phase_2nd_ms.shape, amp_2nd_ms.shape, aperture_2nd_ms.shape, iw.shape)
psf, _ = PSF_2ndprop(
    amp_2nd_ms,
    phase_2nd_ms,
    sim_wl,
    iw,
    aperture=aperture_2nd_ms,  # We should use the mask as the aperture because we are simulating only that part of the wavefront
    normalize_to_aperture=True)
psf = psf.cpu().numpy()
intensity_sensor = np.zeros(out_size)
intensity_sensor[ul[0]:br[0], ul[1]:br[1]] = psf

amp_iw = amp_iw.cpu().numpy()
phase_iw = phase_iw.cpu().numpy()

fig, axes = plt.subplots(5, 2, figsize=(6,15), dpi=200)
rect = Rectangle(
    (ul[1], ul[0]),  # Bottom-left corner (x, y)
    br[1] - ul[1],   # Width
    br[0] - ul[0],   # Height
    linewidth=1, edgecolor='red', facecolor='none'
)
ax = axes[0,0]
im = ax.imshow(np.squeeze(amp_iw))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"Incident amplitude at {(out_distance_m-second_metasurface_depth_m)*1e3:.1f}mm")
ax = axes[0,1]
im = ax.imshow(np.squeeze(amp_iw)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[1,0]
im = ax.imshow(np.squeeze(phase_iw))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"Incident phase at {(out_distance_m-second_metasurface_depth_m)*1e3:.1f}mm")
ax = axes[1,1]
im = ax.imshow(np.squeeze(phase_iw)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[2,0]
im = ax.imshow(np.squeeze(est_phase.cpu().numpy()))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"2nd MS phase")
ax = axes[2,1]
im = ax.imshow(np.squeeze(est_phase.cpu().numpy())[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[3,0]
im = ax.imshow(np.squeeze(intensity_sensor))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"PSF at photosensor with 2nd MS")
ax = axes[3,1]
im = ax.imshow(np.squeeze(intensity_sensor)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

fig.tight_layout()
plt.savefig("7.PSF at photosensor.png")
plt.close(fig)


# representing the wavefront after 1st metasurface using Zernike polynomials
