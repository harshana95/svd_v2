import einops
import numpy as np
import matplotlib.pyplot as plt
import torch


from dflat.initialize import focusing_lens
from dflat.propagation.propagators_legacy import PointSpreadFunction
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from tqdm import tqdm

# Parameters
device = 'cuda'
max_steps = 1000
wavelength = 0.5e-6                 # Wavelength in meters
# k = 2 * np.pi / wavelength          # Wavenumber
batch_size = 16
range_min = -1  # logspace
range_max = 2  # logspace
z1_range = np.logspace(range_min, range_max, 11)   # Distance from point source to lens (in meters)
# z1_range = np.linspace(5, 15, 11)   # Distance from point source to lens (in meters)
z2 = 0.005                           # Distance from lens to sensor (in meters)
L = 0.001                           # Size of the lens plane (in meters)
N = 1024                             # Number of sample points across the lens plane
# f = 0.0025                            # Focal length of the lens (in meters)
cutoff_mtf = 0.2 
model_name = "Nanocylinders_TiO2_U300H600"
plot = False

target_z = 1e0
f = target_z*z2/(target_z + z2)
print(f)


def convert_mtf_2d_to_1d(mtf):
    b, c, h, w = mtf.shape
    center = (N // 2, N // 2)

    # Create coordinate grid
    y, x = torch.meshgrid(torch.arange(h, device=mtf.device), torch.arange(w, device=mtf.device))
    r = torch.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Normalize radial coordinates to maximum frequency
    r_max = (center[0]**2 + center[1]**2)**0.5
    r = r / r_max  # Normalize so that max frequency is 1

    # Bin the MTF values based on radial distance
    r_bins = torch.linspace(0, 1, N//2, device=mtf.device)
    mtf_1d = torch.zeros(b, c, len(r_bins) - 1, device=mtf.device)

    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        mtf_1d[:, :, i] = torch.mean(mtf[:,:, mask], dim=-1)# if torch.any(mask) else 0
        
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
amp, phase, aperture = focusing_lens(**lenssettings) # [P, Lam, H, W]
amp = torch.tensor(amp, dtype=torch.float32, device=device)[None]
phase = torch.tensor(phase, dtype=torch.float32, device=device)[None]
aperture = torch.tensor(aperture, dtype=torch.float32, device=device)[None, None]

p_norm, p, err = reverse_lookup_optimize(
    amp,
    phase,
    [wavelength],
    model_name,
    lr=1e-1,
    err_thresh=1e-6,
    max_iter=10,
    opt_phase_only=False)
p = torch.nn.Parameter(torch.from_numpy(p).to(device), requires_grad=True)


# Compute the point spread function given this broadband stack of field amplitude and phases
PSF = PointSpreadFunction(
    in_size=[N+1, N+1],
    in_dx_m=[L/N,L/N],
    out_distance_m=z2,
    out_size=[N,N],
    out_dx_m=[L/N,L/N],
    out_resample_dx_m=None,
    radial_symmetry=False,
    diffraction_engine="fresnel").to(device)

optical_model = load_optical_model(model_name).to(device)

optimizer = torch.optim.AdamW([p], lr=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

def get_mtf(ps_locs):
    est_amp, est_phase = optical_model(p, [wavelength], pre_normalized=False)
    
    psf_intensity, _ = PSF(
                    est_amp.to(dtype=torch.float32, device=device),
                    est_phase.to(dtype=torch.float32, device=device),
                    [wavelength],
                    ps_locs,
                    aperture=aperture,
                    normalize_to_aperture=True)
    psf = psf_intensity[0,0]
    
    # Compute OTF (Fourier transform of PSF)
    otf = torch.fft.fftshift(torch.fft.fft2(psf), dim=(-2,-1))

    # MTF is the magnitude of the OTF 
    mtf = torch.abs(otf) 
        
    freqs, mtf_1d = convert_mtf_2d_to_1d(mtf)
    
    # normalize MTF
    mtf_1d = mtf_1d / torch.max(mtf_1d)
    return freqs, mtf_1d, mtf, psf

pbar = tqdm(range(max_steps+1))
losses = []
for step in pbar:
    if step% 100 == 0:  # visualize validation
        plt.plot(losses)
        plt.savefig("loss.png")
        plt.close()

        ps_locs = []
        for z1 in z1_range:
            ps_locs.append([0., 0., z1])
        ps_locs = torch.tensor(ps_locs).to(device)

        with torch.no_grad():
            freqs, mtf_1d, mtf, psf = get_mtf(ps_locs)
            freqs = freqs.cpu().numpy()
            mtf_1d = mtf_1d.cpu().numpy()
            mtf = mtf.cpu().numpy()
            psf = psf.cpu().numpy()
            n=mtf.shape[0]
            c=mtf.shape[1]

            cutoff_at_z1 = [[] for _ in range(c)]
            colors = plt.cm.jet(np.linspace(0,1,n))
            for i in range(n):
                for j in range(c):
                    plt.plot(freqs, mtf_1d[i, j], label=f"c={j} z={z1_range[i]:.2e}", color=colors[i])
                    cutoff = -1
                    for freq, val in zip(freqs, mtf_1d[i,j]):
                        if val < cutoff_mtf:
                            cutoff = freq
                            break
                    cutoff_at_z1[j].append(cutoff)
            plt.legend()
            plt.savefig(f"mtf_{step}.png")
            plt.close()
                
            plt.figure(figsize=(6, 4))
            for i in range(len(cutoff_at_z1)):
                plt.plot(z1_range, cutoff_at_z1[i], '.-', label=f"c={i}")
            plt.legend()
            plt.xscale('log')
            plt.xlabel('Distance from Point Source to Lens (m)')
            plt.ylabel('Cutoff Spatial Frequency (1/m)')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'cutoff_{step}.png')
            plt.close()

            psf = einops.rearrange(psf, 'b c h w -> b h w c')
            mtf = einops.rearrange(mtf, 'b c h w -> b h w c')
            
            plt.figure(figsize=(12, n*5))
            for i in range(n):
                # PSF plot
                plt.subplot(n, 2, i*2+1)
                plt.imshow(psf[i], extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
                plt.colorbar(label='Intensity')
                # plt.title('Point Spread Function at Sensor Plane')
                # plt.xlabel('x (m)')
                # plt.ylabel('y (m)')

                # MTF plot
                plt.subplot(n, 2, i*2+2)
                plt.imshow(mtf[i], extent=[-L/2, L/2, -L/2, L/2], cmap='inferno')
                plt.colorbar(label='MTF')
                # plt.title('Modulation Transfer Function at Sensor Plane')
                # plt.xlabel('fx (1/m)')
                # plt.ylabel('fy (1/m)')
            plt.tight_layout()
            plt.savefig(f'psf_mtf_{step}.png')
            plt.close()

    optimizer.zero_grad()
    ps_locs = []
    for _ in range(batch_size):
        ps_locs.append([0., 0., 10**(torch.rand(1)*(range_max - range_min) + range_min)])
    ps_locs = torch.tensor(ps_locs).to(device)

    freqs, mtf_1d, mtf, psf = get_mtf(ps_locs)    

    # compute loss to maximize MTF
    loss = torch.mean(torch.nn.functional.relu(-(mtf_1d-cutoff_mtf)))
    # loss = -torch.mean(freqs*mtf_1d)
    # loss = -torch.sum(mtf_1d)
    loss.backward()
    
    # torch.nn.utils.clip_grad_norm_([p], 0.01)
    optimizer.step()
    scheduler.step()

    pbar.set_postfix_str(f"Loss: {loss.item():.4f}")
    losses.append(loss.item())

    
pbar.close()

