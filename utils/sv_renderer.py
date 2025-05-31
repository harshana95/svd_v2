import einops
import torch


def render(conv_op, psfs, img, coeff, wavelength_set_m):
    meas = conv_op(psfs, einops.rearrange(img, 'b c h w -> b 1 1 c h w'), rfft=True, crop_to_psf_dim=False)
    meas = conv_op.rgb_measurement(meas, wavelength_set_m, gamma=True, process='demosaic')

    
    meas = (meas * coeff).sum(2, keepdim=True)/torch.sum(coeff, dim=-4, keepdim=True)

    return meas[:, 0, 0]