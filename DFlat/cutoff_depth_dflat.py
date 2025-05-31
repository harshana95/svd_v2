import numpy as np
import matplotlib.pyplot as plt
import torch


from dflat.initialize import focusing_lens
from dflat.propagation.propagators_legacy import PointSpreadFunction

# Parameters
wavelength = 0.5e-6                 # Wavelength in meters
k = 2 * np.pi / wavelength          # Wavenumber
z1_range = np.logspace(-1, 2, 11)   # Distance from point source to lens (in meters)
# z1_range = np.linspace(5, 15, 11)   # Distance from point source to lens (in meters)
z2 = 0.005                           # Distance from lens to sensor (in meters)
L = 0.001                           # Size of the lens plane (in meters)
N = 1024                             # Number of sample points across the lens plane
# f = 0.0025                            # Focal length of the lens (in meters)
plot = False

target_z = 1e0

device = 'cuda'
ps_locs = []
for z1 in z1_range:
    ps_locs.append([0., 0., z1])
ps_locs = torch.tensor(ps_locs).to(device)

def convert_mtf_2d_to_1d(mtf):
    """
    Convert a 2D MTF to a 1D radially averaged MTF.

    Args:
        mtf (np.ndarray): 2D MTF array.

    Returns:
        freqs (np.ndarray): 1D spatial frequency values.
        mtf_1d (np.ndarray): 1D MTF values.
    """
    center = (N // 2, N // 2)

    # Create coordinate grid
    y, x = np.indices(mtf.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Normalize radial coordinates to maximum frequency
    r_max = np.max(r)
    r = r / r_max  # Normalize so that max frequency is 1

    # Bin the MTF values based on radial distance
    r_bins = np.linspace(0, 1, N//2)
    mtf_1d = np.zeros(len(r_bins) - 1)

    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        mtf_1d[i] = np.mean(mtf[mask]) if np.any(mask) else 0

    # Get frequency values (midpoints of bins)
    freqs = (r_bins[:-1] + r_bins[1:]) / 2

    return freqs*r_max, mtf_1d


lenssettings = {
                "in_size": [N+1, N+1],
                "in_dx_m": [L/N, L/N],
                "wavelength_set_m": [wavelength],
                "depth_set_m": [[target_z]],
                "fshift_set_m": [[0.,0.]],
                "out_distance_m": z2,
                "aperture_radius_m": L/2,
                "radial_symmetry": False
                }
amp, phase, aperture = focusing_lens(**lenssettings) # [Lam, H, W]
amp = torch.tensor(amp, dtype=torch.float32, device=device)
phase = torch.tensor(phase, dtype=torch.float32, device=device)
aperture = torch.tensor(aperture, dtype=torch.float32, device=device)
aperture = aperture[None]
print(amp.shape, phase.shape, aperture.shape)

# Compute the point spread function given this broadband stack of field amplitude and phases
PSF = PointSpreadFunction(
    in_size=[N+1, N+1],
    in_dx_m=[L/N,L/N],
    out_distance_m=z2,
    out_size=[N,N],
    out_dx_m=[L/N,L/N],
    out_resample_dx_m=None,
    radial_symmetry=False,
    diffraction_engine="fresnel").cuda() 


cutoff_at_z1 = []
for i in range(len(z1_range)):
    
    psf_intensity, _ = PSF(
                    amp.to(dtype=torch.float32, device=device),
                    phase.to(dtype=torch.float32, device=device),
                    [wavelength],
                    ps_locs[i:i+1],
                    aperture=aperture,
                    normalize_to_aperture=True)
    z1 = z1_range[i]
    psf = psf_intensity[0,0,0].cpu().numpy()
    # Compute OTF (Fourier transform of PSF)
    otf = np.fft.fftshift(np.fft.fft2(psf))

    # MTF is the magnitude of the OTF 
    ptf = np.angle(otf)
    mtf = np.abs(otf) 
        
    # # MTF from autocorrelation of the wavefront u
    # mtf = np.abs(np.fft.ifft2(np.fft.fft2(u) * np.conj(np.fft.fft2(u))))

    if plot:  
        plt.figure(figsize=(12, 5))

        # PSF plot
        plt.subplot(1, 2, 1)
        plt.imshow(psf, extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
        plt.colorbar(label='Intensity')
        plt.title('Point Spread Function at Sensor Plane')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        # MTF plot
        plt.subplot(1, 2, 2)
        plt.imshow(mtf, extent=[-L/2, L/2, -L/2, L/2], cmap='inferno')
        plt.colorbar(label='MTF')
        plt.title('Modulation Transfer Function at Sensor Plane')
        plt.xlabel('fx (1/m)')
        plt.ylabel('fy (1/m)')
        plt.tight_layout()
        plt.savefig(f'psf_mtf_{z1}.png')

    freqs, mtf_1d = convert_mtf_2d_to_1d(mtf)

    # normalize MTF
    mtf_1d = mtf_1d / np.max(mtf_1d)

    if plot:
        # Plot the 1D MTF curve
        plt.figure(figsize=(6, 4))
        plt.plot(freqs, mtf_1d, 'b-', linewidth=2)
        plt.xlabel("Spatial Frequency")
        plt.ylabel("MTF")
        plt.title("1D Radially Averaged MTF")
        plt.grid(True)
        plt.savefig(f'mtf_{z1:.3f}.png')

    plt.close('all')

    cutoff = -1
    for freq, val in zip(freqs, mtf_1d):
        if val < 0.1:
            cutoff = freq
            print(f'Cutoff at z1={z1:.3f} is {cutoff}')
            break
    if cutoff == -1:
        print(f"No cutoff at z1={z1:.3f}")
    cutoff_at_z1.append(cutoff)

plt.figure(figsize=(6, 4))
plt.plot(z1_range, cutoff_at_z1, '.-')
plt.xscale('log')
plt.xlabel('Distance from Point Source to Lens (m)')
plt.ylabel('Cutoff Spatial Frequency (1/m)')
plt.grid(True)
plt.tight_layout()
plt.savefig('cutoff.png')