import matplotlib.pyplot as plt
import numpy as np
import torch
import gc
from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize, load_optical_model
from dflat.plot_utilities import mp_format
from dflat.propagation.propagators_legacy import PointSpreadFunction 
def normalize(arr):
    return (arr - arr.min())/(arr.max()-arr.min())

with torch.no_grad():
    for depth in [1e-2, 1e-1, 1e0, 1e1]:
        
        wv = [550e-9]
        h =512
        w =512
        in_dx_m = [1000e-9, 1000e-9]
        out_dx_m = [1000e-9, 1000e-9]
        out_distance_m = 1e-2
        L = (h//2)*in_dx_m[0]
        print(f"L {L}")
        amp, phase, aperture = focusing_lens(
            in_size=[h+1,w+1],
            in_dx_m=in_dx_m,
            wavelength_set_m=wv,
            depth_set_m=[depth],
            fshift_set_m=[[0.0, 0.0]],
            out_distance_m=out_distance_m,
            aperture_radius_m=L,
            radial_symmetry=False
        )
        # returns focusing profiles of shape [Lam, H, W]
        print(amp.shape, phase.shape, aperture.shape)

        fig, ax = plt.subplots(1,1, figsize=(6,6))
        plt.imshow(phase[0])
        plt.axis('off')
        plt.savefig(f'phase_{depth}.png')
        plt.close(fig)

        # Reverse look-up to find the metasurface that implements the target profile
        model_name = "Nanocylinders_TiO2_U300H600"
        with torch.enable_grad():
            p_norm, p, err = reverse_lookup_optimize(
                amp[None,None],
                phase[None,None],
                wv,
                model_name,
                lr=1e-1,
                err_thresh=1e-6,
                max_iter=100,
                opt_phase_only=False)

        # As a check, lets call the optical model and see what this metasurface does vs what we desired
        model = load_optical_model(model_name)
        model = model.to("cuda")

        est_amp, est_phase = model(p, np.array(wv), pre_normalized=False)
        est_amp = est_amp.detach().cpu().numpy()
        est_phase = est_phase.detach().cpu().numpy()
        print(est_amp.shape, est_phase.shape)

        fig, ax = plt.subplots(3,2, figsize=(6,9))
        ax[0,0].imshow(p.squeeze())
        mp_format.format_plot(fig, ax[0,0], addcolorbar=True, setAspect="equal", cbartitle="cylinder radius")
        ax[0,0].axis('off')
        ax[0,1].plot(err)
        ax[1,0].imshow(normalize(est_amp[0,0]*aperture).transpose([1,2,0]), vmin=0, vmax=1)
        mp_format.format_plot(fig, ax[1,0], title="implemented amplitude", addcolorbar=True, setAspect="equal")
        ax[1,1].imshow(normalize(amp*aperture).transpose([1,2,0]), vmin=0, vmax=1)
        mp_format.format_plot(fig, ax[1,1], title="target amplitude",  addcolorbar=True, setAspect="equal")
        ax[2,0].imshow(normalize(est_phase)[0,0].transpose([1,2,0]))
        mp_format.format_plot(fig, ax[2,0], title="implemented phase", addcolorbar=True, setAspect="equal")
        ax[2,1].imshow(normalize(phase).transpose([1,2,0]))
        mp_format.format_plot(fig, ax[2,1], title="target phase", addcolorbar=True, setAspect="equal")
        plt.tight_layout()
        plt.savefig(f'phase_opt_{depth}.png')
        plt.close(fig)

        # Compute the point-spread function for this lens for different conditions
        PSF = PointSpreadFunction(
            in_size=[h+1, w+1],
            in_dx_m=in_dx_m,
            out_distance_m=out_distance_m,
            out_size=[h, w],
            out_dx_m=out_dx_m,
            out_resample_dx_m=None,
            radial_symmetry=False,
            diffraction_engine="fresnel")

        ps_locs = [[0e-4, 0e-4, 1e-2],[0e-4, 0e-4, 1e-1], [0e-4, 0e-4, 1e0], [0e-4, 0e-4, 1e1]]
        sim_wl = wv

        intensity, phase = PSF(
            torch.tensor(est_amp, dtype=torch.float32, device='cuda'),
            torch.tensor(est_phase, dtype=torch.float32, device='cuda'),
            sim_wl,
            ps_locs,
            aperture=torch.tensor(aperture[None, None], dtype=torch.float32, device='cuda'),
            normalize_to_aperture=True)
        intensity = intensity.cpu().numpy()
        phase = phase.cpu().numpy()
        print(intensity.shape)

        c = intensity.shape[-3]
        Z = len(ps_locs)
        fig, ax = plt.subplots(Z, 2*c, figsize=(4*c,2*Z))
        for z in range(Z):
            for l in range(c):
                ax[z,l].imshow(intensity[0,0, z, l,:,:])
                mp_format.format_plot(
                    fig, 
                    ax[z,l],
                    title= f"lam {sim_wl[l]}" if z==0 else "",
                    rmvxLabel=True,
                    rmvyLabel=True, 
                    setAspect="equal")
                
                ax[z,c+l].imshow(phase[0,0, z, l,:,:], cmap='hsv')
                mp_format.format_plot(
                    fig, 
                    ax[z,c+l],
                    title= f"lam {sim_wl[l]}" if z==0 else "",
                    rmvxLabel=True,
                    rmvyLabel=True, 
                    setAspect="equal")
            ax[z,0].set_ylabel(f"PS Depth {ps_locs[z][2]}")
        plt.savefig(f'psfs_{depth}.png')
        plt.close(fig)

# # Write this lens to a gds file for fabrication
# pfab = p.squeeze(0)
# mask = np.ones(pfab.shape[0:2])
# cell_size = [300e-9, 300e-9]
# block_size =[300e-9, 300e-9] # This can be used to repeat the cell as a larger block

# from dflat.GDSII.assemble import assemble_cylinder_gds
# assemble_cylinder_gds(pfab, mask, cell_size, block_size, savepath="./file.gds")