
from matplotlib import pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import nn
from mpl_toolkits.axes_grid1 import make_axes_locatable


def noll_to_zernike(i):
    '''Get the Zernike index from a Noll index.

    vectorized version of: https://github.com/ehpor/hcipy/blob/master/hcipy/mode_basis/zernike.py
    Parameters
    ----------
    i : int
        The Noll index.
    Returns
    -------
    n : int
        The radial Zernike order.
    m : int
        The azimuthal Zernike order.
    '''
    n = (np.sqrt(2 * i - 1) + 0.5).astype(int) - 1
    m = np.where(n % 2,
                 2 * ((2 * (i + 1) - n * (n + 1)) // 4).astype(int) - 1,
                 2 * ((2 * i + 1 - n * (n + 1)) // 4).astype(int))
    return n, m * (-1) ** (i % 2)


def zernike_poly(Nz, H, W, zH, zW):
    assert H>=zH and W>=zW
    # Vectorized version
    n, m = noll_to_zernike(np.arange(1, Nz + 1))
    sum_ul = int(np.amax(n - np.abs(m)) / 2)
    k = np.arange(0, sum_ul + 1)[:, None]
    rad_terms = (-1) ** (k) * scipy.special.binom(n[None, :] - k, k) * \
                scipy.special.binom(n[None, :] - 2 * k,
                                   (n[None, :] - np.abs(m[None, :])) / 2 - k)
    
    rad_terms[np.isnan(rad_terms)] = 0
    sum_mask = np.tile(np.arange(0, sum_ul + 1), (Nz, 1)).T
    sum_mask = np.where(sum_mask <= (n - np.abs(m)) / 2, 1, 0)

    rhox, rhoy = np.meshgrid(np.linspace(-1, 1, zW), np.linspace(-1, 1, zH))
    rho = np.sqrt(rhox ** 2 + rhoy ** 2)
    rad_poly = rho[None, None, :] ** (np.maximum(n[None, ...] - 2 * k, 0))[..., None, None]
    rad_poly = rad_poly * np.where(rad_poly < 1, 1, 0)
    psi = np.arctan2(rhoy, rhox)
    ang_funs = np.where(m[:, None, None] < 0, np.sin(m[:, None, None] * psi[None, ...]),
                                              np.cos(m[:, None, None] * psi[None, ...]))

    Rmn = np.sum((sum_mask * rad_terms)[..., None, None] * rad_poly, axis=0)
    zP = np.pad(Rmn * ang_funs, [(0,0), ((H-zH)//2, (H-zH)//2), ((W-zW)//2, (W-zW)//2)], mode='constant', constant_values=0)
    return zP*(zH/2*0.8) # scaling

def phase2psf(phase, D, pad=0):
    if pad>0:
        phase = F.pad(phase, (pad, pad, pad, pad), mode='constant', value=0)
        print('phase after pad', phase.shape)
    H, W = phase.shape[-2], phase.shape[-1]
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, W, device=phase.device),
                            torch.linspace(-1, 1, H, device=phase.device), indexing='xy')
    rho = np.sqrt(xx ** 2 + yy ** 2)
    A = (rho <= (H-2*pad)/H*D).float()
    # print(H,W,A.shape, D, (H-2*pad)/H*D)
    # fig = plt.figure()
    # plt.imshow(A.cpu().numpy())
    # plt.colorbar()
    # plt.savefig('A.png')
    # plt.close(fig)
    pupil = A * torch.exp(-1j * phase)
    field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil, dim=(-2,-1)), dim=(-2,-1)), dim=(-2, -1))
    psfs = torch.abs(field)**2
    if pad > 0:
        psfs = psfs[:, pad:-pad,pad:-pad]
    psfs = psfs / torch.sum(psfs, axis=(-1, -2), keepdims=True)
    return psfs#[:, int(H//2) - int(H//8):int(H//2)+int(H//8), int(W//2) - int(W//8):int(W//2)+int(W//8)]

if __name__ == "__main__":
    N = 10
    Z = 36
    H, W = 256, 256
    h, w = H//1, W//1
    D = h/H
    pad = 256
    zernike_coeffs = np.random.rand(N, Z)
    zernike_coeffs = zernike_coeffs/np.sum(zernike_coeffs, axis=1, keepdims=True)
    zernike_polynomial = zernike_poly(Z, H, W, h, w)  # rescaling works better
    zernike_phases = np.sum(zernike_polynomial * zernike_coeffs[:, :, None, None], axis=1)
    zernike_psfs = phase2psf(torch.from_numpy(zernike_phases), D=D, pad=pad)
    
    print(zernike_polynomial.shape, zernike_phases.shape, zernike_psfs.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    im=ax.imshow(zernike_psfs[0], cmap='viridis')
    ax.set_title('PSF')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax = axes[1]
    im=ax.imshow(zernike_phases[0], cmap='viridis')
    ax.set_title('Phase')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig('tmp.png')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    n = 32
    m = 33
    ax=axes[0,0]
    im=ax.imshow((zernike_polynomial[n:n+1])[0], cmap='viridis')
    ax.set_title('Phase 1')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax=axes[0,1]
    im=ax.imshow((zernike_polynomial[m:m+1])[0], cmap='viridis')
    ax.set_title('Phase 2')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)        
    fig.colorbar(im, cax=cax, orientation='vertical')
    
    ax=axes[1,0]
    im=ax.imshow(phase2psf(torch.from_numpy(zernike_polynomial[n:n+1]), D=D, pad=pad)[0], cmap='viridis')
    ax.set_title('PSF 1')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)        
    fig.colorbar(im, cax=cax, orientation='vertical')

    ax=axes[1,1]
    im=ax.imshow(phase2psf(torch.from_numpy(zernike_polynomial[m:m+1]), D=D,pad=pad)[0], cmap='viridis')
    ax.set_title('PSF 2')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)        
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig('tmp2.png')
    