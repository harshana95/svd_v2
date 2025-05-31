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
from utils.dflat_utils import reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_size=[2049, 2049]
in_dx_m=[2.5e-6, 2.5e-6]
wavelength_set_m=[532e-9]
out_dx_m=[2.74e-6, 2.74e-6] # Basler dart M dmA4096-9gm (Mono) pixel size multiply 3
# out_size=[375*8+1, 513*8+1]
out_size=[2049, 2049]

aperture_radius_m=2e-3
f_distance_m = 10.3e-3
depth_set_m=[99999]
fshift_set_m=[[0.0, 0.0]]

out_distance_m=10e-3
second_metasurface_depth_m=1e-3
radial_symmetry=False

threshold = 30 # Threshold for minimum cluster size
cluster_idx = 4 # select the cluster for second propagation

diffraction_engine_1st="ASM"
diffraction_engine_2nd="ASM"

ps_locs = [[50, 50, 1e3]]
sim_wl = [532e-9]
incident_angle = np.arctan((ps_locs[0][0]**2 + ps_locs[0][1]**2)**0.5/ps_locs[0][2])
print(f"Incident angle: {incident_angle*180/np.pi:.2f} degrees")

def add_colorbar(im, ax, fig):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

# define the lens
amp, phase, aperture = focusing_lens(
    in_size=in_size,
    in_dx_m=in_dx_m,
    wavelength_set_m=wavelength_set_m,
    depth_set_m=depth_set_m,
    fshift_set_m=fshift_set_m,
    out_distance_m=f_distance_m,
    aperture_radius_m=aperture_radius_m,
    radial_symmetry=radial_symmetry
)

amp2 = torch.ones_like(amp, dtype=torch.float32, device=device)
phase2 = torch.zeros_like(phase, dtype=torch.float32, device=device)

# returns focusing profiles of shape [Z, Lam, H, W]
print(amp.shape, phase.shape, aperture.shape)

fig, ax = plt.subplots(1,2, figsize=(6,3))
ax[0].imshow(amp[0]*aperture[0], vmin=0, vmax=1)
ax[0].set_title("amplitude")
ax[1].imshow(phase[0])
ax[1].set_title("phase")
plt.savefig("1.first lens.png")
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
# The ASM propagator naturally provides PSFs of a grid of incident angles due to the discrete sampling of the metasurface phase profile
PSF = PointSpreadFunction(
    in_size=in_size,
    in_dx_m=in_dx_m,
    out_distance_m=out_distance_m-second_metasurface_depth_m,
    out_size=in_size,
    out_dx_m=in_dx_m,
    out_resample_dx_m=None,
    manual_upsample_factor=1,
    radial_symmetry=radial_symmetry,
    diffraction_engine=diffraction_engine_1st).cuda()


intensity, phase = PSF(
    est_amp,
    est_phase,
    sim_wl,
    ps_locs,
    aperture=aperture_exp,
    normalize_to_aperture=True)
intensity = intensity.cpu().numpy()
amplitude = np.sqrt(intensity)
phase = phase.cpu().numpy()

fig = plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
plt.imshow(amplitude[0,0, 0, 0,:,:])
plt.title("Amplitude")
plt.colorbar()

fig.add_subplot(1,2,2)
plt.imshow(phase[0,0, 0, 0,:,:])
plt.title("Phase")
plt.colorbar()
plt.savefig("2.Wavefront before 2nd MS.png")
plt.close(fig)

# segment each PSF and separately propagate for 1mm
# label each cluster in the binary image
from scipy.ndimage import label
# Label connected clusters (8-connectivity)

binary_image = amplitude[0,0, 0, 0,:,:]>0.0005
labeled_array, num_features = label(binary_image, structure=np.ones((3,3)))


# Remove clusters smaller than threshold
filtered_labels = np.copy(labeled_array)
for label_idx in range(1, num_features + 1):
    cluster_size = np.sum(labeled_array == label_idx)
    if cluster_size < threshold:
        filtered_labels[filtered_labels == label_idx] = 0

# Re-label the remaining clusters
filtered_labels, new_num_features = label(filtered_labels > 0, structure=np.ones((3,3)))

print("Number of clusters after filtering:", new_num_features)

# Visualize the clusters
fig = plt.figure(figsize=(20,5))
fig.add_subplot(1,2,1)
plt.imshow(binary_image)
plt.title("Binary Mask")
plt.colorbar()
fig.add_subplot(1,2,2)
plt.imshow(filtered_labels, cmap='nipy_spectral')
plt.colorbar(label='Cluster labels')
plt.title('Segmented Clusters')
plt.savefig("3.Segmented clusters.png")
plt.close(fig)

def find_bounding_box(binary_map):
    rows = np.any(binary_map, axis=1)
    cols = np.any(binary_map, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError("The binary map contains no ones.")

    top, bottom = np.where(rows)[0][0], np.where(rows)[0][-1]
    left, right = np.where(cols)[0][0], np.where(cols)[0][-1]

    upper_left = (top, left)
    bottom_right = (bottom, right)

    return upper_left, bottom_right

# ==================== 2. Propgating for 1mm from the second metasurface to the photosensor

# select the PSF, the central PSF
if diffraction_engine_1st.lower() == 'asm':
    ul, br = find_bounding_box(filtered_labels == cluster_idx)
else:
    ul = [0, 0]
    br = filtered_labels.shape[-2:]

center = [(ul[0]+br[0])//2, (ul[1]+br[1])//2]
print(center)

xx, yy = np.meshgrid(np.arange(amplitude.shape[-1]), 
                     np.arange(amplitude.shape[-2]), indexing="xy")
    
# circular mask
mask = (xx - center[1])**2 + (yy - center[0])**2 < ((br[1] - ul[1])/2)**2
mask = mask.astype(np.float32)
# # rectangular mask
mask = np.zeros_like(filtered_labels)
mask[ul[0]:br[0], ul[1]:br[1]] = 1

mask = mask[None, None, None]
amp_2ndprop = amplitude[0] * mask
phase_2ndprop = phase[0] * mask

xx = (xx - center[1]) * out_dx_m[0]
yy = (yy - center[0]) * out_dx_m[1]
phase_analytical =  np.angle(np.exp(1j*-2*np.pi/sim_wl[0] * np.sqrt(xx**2 + yy**2 + second_metasurface_depth_m**2)))


# amp_2ndprop = np.ones_like(amp_2ndprop)
# phase_2ndprop = np.zeros_like(phase_2ndprop)

PSF_2ndprop = PointSpreadFunctionNew(
    in_size=in_size,  # first prop gives in_size as output
    in_dx_m=in_dx_m,
    out_distance_m=second_metasurface_depth_m,
    out_size=out_size,
    out_dx_m=out_dx_m,
    out_resample_dx_m=None,
    radial_symmetry=False,
    diffraction_engine=diffraction_engine_2nd).to(device)
torch_zero = torch.tensor([0.0], dtype=torch.float32, device=device)
amp_2ndprop = torch.tensor(amp_2ndprop, dtype=torch.float32, device=device)
phase_2ndprop = torch.tensor(phase_2ndprop, dtype=torch.float32, device=device)
incident_wavefront = torch.complex(amp_2ndprop, torch_zero) * torch.exp(torch.complex(torch_zero, phase_2ndprop))
intensity_sensor, _ = PSF_2ndprop(
    torch.ones_like(amp_2ndprop, dtype=torch.float32),
    torch.zeros_like(phase_2ndprop, dtype=torch.float32),
    sim_wl,
    incident_wavefront.to(device),
    aperture=mask,  # We should use the mask as the aperture because we are simulating only that part of the wavefront
    normalize_to_aperture=True)
intensity_sensor = intensity_sensor.cpu().numpy()

amp_2ndprop = amp_2ndprop.cpu().numpy()
phase_2ndprop = phase_2ndprop.cpu().numpy()

fig, axes = plt.subplots(5, 2, figsize=(6,15), dpi=200)
rect = Rectangle(
    (ul[1], ul[0]),  # Bottom-left corner (x, y)
    br[1] - ul[1],   # Width
    br[0] - ul[0],   # Height
    linewidth=1, edgecolor='red', facecolor='none'
)
ax = axes[0,0]
im = ax.imshow(np.squeeze(amp_2ndprop))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"Amplitude at {(out_distance_m-second_metasurface_depth_m)*1e3:.1f}mm")
ax = axes[0,1]
im = ax.imshow(np.squeeze(amp_2ndprop)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[1,0]
im = ax.imshow(np.squeeze(phase_2ndprop))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"Numerical phase at {(out_distance_m-second_metasurface_depth_m)*1e3:.1f}mm")
ax = axes[1,1]
im = ax.imshow(np.squeeze(phase_2ndprop)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[2,0]
im = ax.imshow(np.squeeze(phase_analytical))
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"Analytical phase at {(out_distance_m-second_metasurface_depth_m)*1e3:.1f}mm")
ax = axes[2,1]
im = ax.imshow(np.squeeze(phase_analytical)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

ax = axes[3,0]
im = ax.imshow(np.squeeze(intensity_sensor))#[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title(f"PSF at photosensor with 2nd MS")
ax = axes[3,1]
im = ax.imshow(np.squeeze(intensity_sensor)[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

PSF_no_2nd = PointSpreadFunction(
    in_size=in_size,
    in_dx_m=in_dx_m,
    out_distance_m=out_distance_m,
    out_size=out_size,
    out_dx_m=out_dx_m,
    out_resample_dx_m=None,
    radial_symmetry=False,
    diffraction_engine=diffraction_engine_1st).cuda()

intensity_sensor_no_2nd, _ = PSF_no_2nd(
    est_amp,
    est_phase,
    sim_wl,
    ps_locs,
    aperture=aperture_exp,
    normalize_to_aperture=True)

ax = axes[4,0]
im = ax.imshow(np.squeeze(intensity_sensor_no_2nd.cpu().numpy()))#[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)
ax.add_patch(copy.copy(rect))
ax.set_title("PSF at photosensor without 2nd MS")
ax = axes[4,1]
im = ax.imshow(np.squeeze(intensity_sensor_no_2nd.cpu().numpy())[ul[0]:br[0],ul[1]:br[1]])
add_colorbar(im, ax, fig)

fig.tight_layout()
plt.savefig("4.PSF at photosensor.png")
plt.close(fig)


# representing the wavefront after 1st metasurface using Zernike polynomials
