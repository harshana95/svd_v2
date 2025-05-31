import numpy as np
import einops
from dflat.metasurface import datasets

def closest_index(value, array):
    """
    Find the index of the closest value in an array.

    Parameters:
        array (np.ndarray): The array to search.
        value (float): The value to find the closest match for.

    Returns:
        int: The index of the closest value in the array.
    """
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def reverse_lookup_search_Nanocylinders_TiO2_U300nm_H600nm(amp, phase, wavelength_set_m, *args, **kwargs):
    L, H, W = amp.shape[-3::]
    assert amp.shape == phase.shape
    assert (
        len(wavelength_set_m) == L
    ), "Wavelength list should match amp,phase wavelength dim (dim3)."
    assert (
        len(wavelength_set_m) == 1
    ), "only support a single wavelength."

    model = datasets.Nanocylinders_TiO2_U300nm_H600nm()
    idx = closest_index(wavelength_set_m[0], model.params[1])

    lib_amp = model.trans[:,idx]  # 121 elements
    lib_phase = model.phase[:,idx]
    lib = lib_amp * np.exp(1j * lib_phase)
    target = np.expand_dims(amp * np.exp(1j * phase), axis=-1)
    diff = np.abs(lib - target)  # (1, 1, 1, 1025, 1025, 121)
    selection_index = np.argmin(diff, axis=-1)
    # print(model.params[0].min(), model.params[0].max(),model.params[0])
    op_param = model.params[0][selection_index]
    op_param = einops.rearrange(op_param, 'b p 1 h w -> b h w p')
    op_param_norm = (op_param - model.param_limits[0][0])/(model.param_limits[0][1] - model.param_limits[0][0])
    err = []
    return op_param_norm, op_param, err
