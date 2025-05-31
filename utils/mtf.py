
# function to plot 1D MTF
import einops
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import seaborn as sns
from scipy.integrate import quad
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model, optical_model
from dflat.propagation.propagators_legacy import PointSpreadFunction # Using the legacy version as it takes wavelength as a forward input instead of initialization input
from dflat.render import Fronto_Planar_Renderer_Incoherent

from utils.misc import fig_to_array



def plot_1d_mtf(mtf1d, ff, ps_locs, cutoff_mtf, mask=1, fc=None):  # fc: cutoff frequency in cycles/mm
    mtf1d = mtf1d[:, :, :int(mtf1d.shape[2]*mask)]
    n, c, l = mtf1d.shape
    ff = ff[:l]
    mtf1d = mtf1d.flatten(order='C')
    # 2/pi [arccos(v/fc) - v/fc*sqrt(1-(v/fc)^2)]
    diffractionlimit = [2/np.pi *(np.arccos(v/fc) - v/fc * np.sqrt(1-(v/fc)**2)) for v in ff]
    data = pd.DataFrame({
        'MTF': mtf1d,
        'Frequency (cycles/mm)': einops.repeat(ff, 'l -> n c l', n=n, c=c).flatten(),
        'Wavelength': einops.repeat(np.arange(c), 'c -> n c l', n=n, l=l).flatten(),
        'Source Depth': einops.repeat(np.array([round(ps_locs[i][2].item(),4) for i in range(n)]), 'n -> n c l', c=c, l=l).flatten(),
    })
    fig, axes = plt.subplots(c, 1, figsize=(6, 4*c))
    for i in range(c):
        ax = sns.lineplot(data=data[data['Wavelength'] == i], x='Frequency (cycles/mm)', y='MTF', hue='Source Depth', 
                          palette=sns.color_palette("rocket", n_colors=n), ax=axes[i], legend=i==0)
        ax.plot(ff, diffractionlimit, linestyle='--', color='k', label='Diffraction limit')
        ax.set(yscale='log')
        if i==0:sns.move_legend(ax, "upper right")#, bbox_to_anchor=(1, 1))
        ax.axhline(cutoff_mtf, color='r', linestyle='--')
        ax.set_ylim([5e-2, 1])
        if i==0:ax.legend()
    fig_np = fig_to_array(fig)
    plt.close(fig)
    return fig_np

# function to plot PSF and MTF
def plot_psf_mtf(psf,mtf, ps_locs, aperture_r):
    # Create a custom colormap
    # White Blue Cyan Green Yellow Red
    colors = ["#FFFFFF", "#0000FF", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"]  # Define custom colors
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)  # Create custom colormap

    N,C,H,W = mtf.shape
    # psf = crop_arr(psf, 256,256)
    psf = einops.rearrange(psf, 'b c h w -> b h w c')
    mtf = einops.rearrange(mtf, 'b c h w -> b h w c')
    
    fig, axes = plt.subplots(N,C*2,figsize=(12*C, N*5))
    L = aperture_r
    extent1 = np.array([-L, L, -L, L])*1e6
    extent2 = np.array([-1, 1, -1, 1])*L/(mtf.shape[1]/2)  # L replace with sensor size

    for i in range(N):
        for c in range(C):
            ax = axes[i,c]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(psf[i, :, :, c], extent=extent1, cmap=custom_cmap)#, vmin=psf.min(), vmax=psf.max())
            fig.colorbar(im, cax=cax, orientation='vertical')
            if i==0: ax.set_title(f'PSF {c}')
            if c==0: ax.set_ylabel(f'Source at {ps_locs[i][2]:.2e}m')
            if i==N-1:ax. set_xlabel('x (um)')
        
            # MTF plot
            ax = axes[i,C+c]
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            im = ax.imshow(mtf[i, :, :, c], extent=extent2, cmap='inferno')#, vmin=mtf.min(), vmax=mtf.max())
            fig.colorbar(im, cax=cax, orientation='vertical')
            if i==0: ax.set_title(f'MTF {c}')
            if i==N-1:ax. set_xlabel('fx (1/m)')
    plt.tight_layout()
    fig_np = fig_to_array(fig)
    plt.close(fig)
    return fig_np

def get_psf_cross_section(amp, phase, aperture, ps_locs, z_list, h, w, in_dx_m, out_dx_m, wavelength_set_m, diffraction_engine):
    psfs = []
    pbar = tqdm(z_list, desc="Calculating PSF cross-section")
    amp = amp.cuda()
    phase = phase.cuda()
    for z in pbar:
        fresnel_number = ((h/2)*in_dx_m[0])**2/ (z * min(wavelength_set_m)) if z!=0 else np.inf # a**2/L/lambda
        pbar.set_description_str(f"Fresnel number: {fresnel_number}")
        psf_gen = PointSpreadFunction(in_size=[h+1, w+1],
                                        in_dx_m=in_dx_m,
                                        out_distance_m=z,
                                        out_size=[h,w],
                                        out_dx_m=out_dx_m,
                                        out_resample_dx_m=None,
                                        radial_symmetry=False,
                                        diffraction_engine=diffraction_engine).cuda()
        psf_intensity, _ = psf_gen(amp, phase, wavelength_set_m, ps_locs, aperture=aperture, normalize_to_aperture=True)
        psf = einops.rearrange(psf_intensity[0,0].cpu().numpy(), 'b c h w -> b h w c')[0]
        psfs.append(psf)
    pbar.close()
    psfs = np.array(psfs)
    print(psfs.shape)
    return einops.rearrange(psfs[:,:, w//2, :], 'z h c -> h z c')

# MTF calculation helpers
def convert_mtf_2d_to_1d(mtf, simple=True):
    b, c, h, w = mtf.shape
    center = (h // 2, w // 2)
    if simple:
        mtf_1d = mtf[:, :, h//2, w//2:]
        freqs = torch.arange(w-w//2, device=mtf.device)+1
    else:
        # Create coordinate grid
        y, x = torch.meshgrid(torch.arange(h, device=mtf.device), torch.arange(w, device=mtf.device), indexing='ij')
        r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)

        # Normalize radial coordinates to maximum frequency
        r_max = (center[0]**2 + center[1]**2)**0.5
        r = r / r_max  # Normalize so that max frequency is 1

        # Bin the MTF values based on radial distance
        r_bins = torch.linspace(0, 1, h//2, device=mtf.device)
        mtf_1d = torch.zeros(b, c, len(r_bins) - 1, device=mtf.device)

        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            mtf_1d[:, :, i] = torch.mean(mtf[:,:, mask], dim=-1)# if torch.any(mask) else 0
            
        # Get frequency values (midpoints of bins)
        freqs = (r_bins[:-1] + r_bins[1:]) / 2
        freqs *= r_max # denormalize
    return freqs, mtf_1d

def get_mtf(psf_intensity, pixel_pitch):
    h,w = psf_intensity.shape[-2:]
    N = h
    psf = einops.rearrange(psf_intensity, 'b 1 z c h w -> (b z) c h w')
    
    # Normalize PSF
    psf = psf / torch.sum(psf, dim=(-2,-1), keepdim=True)

    # Compute OTF (Fourier transform of PSF)
    otf = torch.fft.fftshift(torch.fft.fft2(psf), dim=(-2,-1))
    # freq = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_pitch))
    # freq_mm = freq / 1e3 # cycles/mm
    df = 1/(N*pixel_pitch*1e3) # cycles/mm

    # MTF is the magnitude of the OTF 
    mtf = torch.abs(otf) 
        
    freqs, mtf_1d = convert_mtf_2d_to_1d(mtf)
    freqs = freqs*df # cycles/mm

    # normalize MTF
    mtf_1d = mtf_1d / torch.max(mtf_1d)
    return freqs, mtf_1d, mtf

@torch.no_grad()
def get_cutoff(psfs, cutoff_mtf, pixel_pitch):
    h, w = psfs.shape[-2:]
    freqs, mtf_1d, mtf = get_mtf(psfs, pixel_pitch)

    freqs = freqs.cpu().numpy()
    mtf_1d = mtf_1d.cpu().numpy()
    mtf = mtf.cpu().numpy()
    n=mtf.shape[0]
    c=mtf.shape[1]

    cutoff_at_z1 = [[] for _ in range(c)]
    for i in range(n):
        for j in range(c):
            cutoff = -1
            for freq, val in zip(freqs, mtf_1d[i,j]):
                if val < cutoff_mtf:
                    cutoff = freq
                    break
            cutoff_at_z1[j].append(cutoff if cutoff >= 0 else np.inf)
    return cutoff_at_z1, freqs, mtf_1d, mtf

