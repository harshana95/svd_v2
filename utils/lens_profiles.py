import numpy as np

import einops
import numpy as np
from scipy.integrate import quad

from dflat.initialize import focusing_lens
from dflat.metasurface import reverse_lookup_optimize
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

def reverse_lookup_search_by_phase(amp, phase, wavelength_set_m,*args, **kwargs):
    B, P, L, H, W = amp.shape
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

def get_lens_profile(reverse_lookup, lens_type, h, w, in_dx_m, wavelength_set_m, out_distance_m, aperture_radius_m, model_name, method='nearest', **kwargs):
    p = None
    p_norm = None
    # x,y = (1, H, W)
    x, y = np.meshgrid(np.arange(w+1), np.arange(h+1), indexing="xy")
    x = x - (x.shape[-1] - 1) / 2
    y = y - (y.shape[-2] - 1) / 2
    x = x[None] * in_dx_m[-1]
    y = y[None] * in_dx_m[-2]

    if method == 'nearest':
        reverse_lookup_f = reverse_lookup_search_by_phase
    elif method == 'optimize':
        reverse_lookup_f = reverse_lookup_optimize
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
    R = aperture_radius_m

    extras = {}

    def integrand(x, s_1, s_2, n, R):
        return x / ((x**2 + (s_1 + (s_2 - s_1) * ((x / R)**n))**2)**0.5)
    def phase_mask(xx, yy, wavelength, s_1, s_2, n, R):
        # All units are in meters
        r = np.sqrt(xx**2 + yy**2)
        # Vectorize the integration over the grid
        def compute_integral(r_val):
            if r_val <= 0:
                return 0.0
            else:
                return quad(integrand, 0, r_val, args=(s_1, s_2, n, R))[0]
        # Apply to each element in r (this is slow but correct)
        integral_result = np.vectorize(compute_integral)(r)
        return (2 * np.pi / wavelength[:,None,None]) * integral_result  # Phase mask in radians (edited) 
    
    if lens_type == 'focusing_lens':
        lens_settings = {
            "in_size": [h+1, w+1],
            "in_dx_m": in_dx_m,
            "wavelength_set_m": wavelength_set_m,
            "depth_set_m": kwargs.get("depth_set_m"),
            "fshift_set_m": [[0,0] for _ in range(len(wavelength_set_m))],
            "out_distance_m": out_distance_m,
            "aperture_radius_m": aperture_radius_m,
            "radial_symmetry": False  # if True slice values along one radius
            }
        amp, phase, aperture = focusing_lens(**lens_settings) # [Lam, H, W]
        amp = amp[None, None]
        phase = phase[None, None]
        aperture = aperture[None, None]
    elif lens_type == 'multiplexed':
        n, m = kwargs.get('grid_size')
        grid_w, grid_h = np.meshgrid(np.arange(m), np.arange(n))
        grid_w, grid_h = grid_w.flatten(), grid_h.flatten()
        print(f"Multiplex grid {n}x{m} idx: {grid_h} {grid_w}")

        for d, depth in enumerate(kwargs.get("depth_set_array_m")):
            print(f"Getting lens for depth {depth}")
            lens_settings = {
                "in_size": [h+1, w+1],
                "in_dx_m": in_dx_m,
                "wavelength_set_m": wavelength_set_m,
                "depth_set_m": [depth]*len(wavelength_set_m),
                "fshift_set_m": [[0,0] for _ in range(len(wavelength_set_m))],
                "out_distance_m": out_distance_m,
                "aperture_radius_m": aperture_radius_m,
                "radial_symmetry": False  # if True slice values along one radius
            }
            amp, phase, aperture = focusing_lens(**lens_settings)
            amp = amp[None, None]
            phase = phase[None, None]
            aperture = aperture[None, None]

            if reverse_lookup:
                _p_norm, _p, err = reverse_lookup_f(amp, phase, wavelength_set_m, model_name, lr=1e-1, err_thresh=1e-6, max_iter=100, opt_phase_only=False)
                B, H, W, D = _p.shape
                _i, _j = grid_h[d], grid_w[d]
                jj, ii = np.meshgrid(np.arange(_j, W, m), np.arange(_i, H, n))    
                # print(d, _i,_j, ii.shape, jj.shape, p.shape, ii[:3,:3], jj[:3,:3])
                if p is None:
                    p = _p
                else:
                    p[:, ii,jj, :] = _p[:,ii,jj, :]
                if p_norm is None:
                    p_norm = _p_norm
                else:
                    p_norm[:, ii, jj, :] = _p_norm[:, ii, jj, :]          
    elif lens_type == 'multiplexed_cell':
        n, m = kwargs.get('grid_size')
        cell_size = kwargs.get('cell_size', 16)
        grid_w, grid_h = np.meshgrid(np.arange(m), np.arange(n))
        grid_w, grid_h = grid_w.flatten(), grid_h.flatten()
        print(f"Multiplex grid {n}x{m} cell size {cell_size} idx: {grid_h} {grid_w}")

        all_p, all_p_norm = {c:[] for c in range(len(wavelength_set_m))}, {c:[] for c in range(len(wavelength_set_m))}
        for d, depth in enumerate(kwargs.get("depth_set_array_m")):
            print(f"Getting lens for depth {depth}")
            lens_settings = {
                "in_size": [h+1, w+1],
                "in_dx_m": in_dx_m,
                "wavelength_set_m": wavelength_set_m,
                "depth_set_m": [depth]*len(wavelength_set_m),
                "fshift_set_m": [[0,0] for _ in range(len(wavelength_set_m))],
                "out_distance_m": out_distance_m,
                "aperture_radius_m": aperture_radius_m,
                "radial_symmetry": False  # if True slice values along one radius
            }
            amp, phase, aperture = focusing_lens(**lens_settings)
            amp = amp[None, None]
            phase = phase[None, None]
            aperture = aperture[None, None]

            assert reverse_lookup
            for c in range(len(wavelength_set_m)):
                _p_norm, _p, err = reverse_lookup_f(amp[:, :, c:c+1], phase[:, :, c:c+1], wavelength_set_m[c:c+1], model_name, lr=1e-1, err_thresh=1e-6, max_iter=100, opt_phase_only=False)
                all_p[c].append(_p)
                all_p_norm[c].append(_p_norm)
        
        cc = 0
        for i in range(0, h, cell_size*n):
            for j in range(0, w, cell_size*m):
                p_list, p_norm_list = all_p[cc], all_p_norm[cc]
                cc = (cc + 1)%len(wavelength_set_m)
                for d in range(len(p_list)):
                    i2 = i + grid_h[d]*cell_size
                    j2 = j + grid_w[d]*cell_size
                    if p is None:
                        p = p_list[d]
                    else:
                        p[:, i2:i2+cell_size,j2:j2+cell_size, :] = p_list[d][:,i2:i2+cell_size,j2:j2+cell_size, :]
                    if p_norm is None:
                        p_norm = p_norm_list[d]
                    else:
                        p_norm[:, i2:i2+cell_size,j2:j2+cell_size, :] = p_norm_list[d][:, i2:i2+cell_size,j2:j2+cell_size, :]

    elif lens_type == 'metalens':
        # baselines on Arka's paper: Metalens
        f = kwargs.get('f')
        phase = 2 * np.pi/ wavelength_set_m[:,None,None] * (np.sqrt(f**2 + x**2 + y**2) - f)
        phase = np.angle(np.exp(1j *-phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]
        amp = np.ones_like(phase)
    elif lens_type == 'cubic':
        f = kwargs.get('f')
        alpha = 55*np.pi
        phase = 2 * np.pi/ wavelength_set_m[:,None,None] * (np.sqrt(f**2 + x**2 + y**2) - f) 
        phase += ((-x)**3 + (-y)**3)*alpha/(R**3)  # cubic part
        phase = np.angle(np.exp(1j *-phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]
        amp = np.ones_like(phase)
    elif lens_type == 'log_asphere':
        s_1 = kwargs.get('s_1')
        s_2 = kwargs.get('s_2')
        phase = phase_mask(x, y, wavelength_set_m, s_1=s_1, s_2=s_2, n=2, R=R)
        phase = np.angle(np.exp(1j * -phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]       
        amp = np.ones_like(phase)
    elif lens_type == 'shifted_axicon':
        s_1 = kwargs.get('s_1')
        s_2 = kwargs.get('s_2')
        phase = phase_mask(x, y, wavelength_set_m, s_1=s_1, s_2=s_2, n=1, R=R)
        phase = np.angle(np.exp(1j * -phase))[None, None]    
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]       
        amp = np.ones_like(phase)
    elif lens_type == 'squbic':
        A = 50
        NA = 0.45
        alpha = np.arcsin(NA)
        num = np.sqrt(1 - ((x/R)**2 + (y/R)**2)*(np.sin(alpha)**2)) - 1
        den = 1 - np.cos(alpha)
        phase = 2 * np.pi * A * (num/den + 0.5) ** 3
        phase = np.nan_to_num(phase, nan=0, posinf=0, neginf=0)
        phase = np.ones((len(wavelength_set_m), 1, 1)) * phase  # repeat for all wavelengths
        phase = np.angle(np.exp(1j * -phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]       
        amp = np.ones_like(phase)
    elif lens_type == 'quadratic':
        phase = np.pi/wavelength_set_m[:, None, None] * (x**2 + y**2)/out_distance_m
        phase = np.angle(np.exp(1j * -phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]
        amp = np.ones_like(phase)
    elif lens_type == 'quadratic_cubic':
        alpha = 55*np.pi
        phase = np.pi/wavelength_set_m[:, None, None] * (x**2 + y**2)/out_distance_m
        phase += ((-x)**3 + (-y)**3)*alpha/(R**3)
        phase = np.angle(np.exp(1j * -phase))[None, None]
        aperture = ((x**2 + y**2) < (R**2)).astype(np.float32)[None,None]
        amp = np.ones_like(phase)
    else:
        raise NotImplementedError()
    if p_norm is None and reverse_lookup:
        # Reverse look-up to find the metasurface that implements the target profile
        p_norm, p, err = reverse_lookup_f(
            amp,
            phase,
            wavelength_set_m,
            model_name,
            lr=10e-2,
            err_thresh=1e-6,
            max_iter=500,
            opt_phase_only=False)
        # p_norm2, p2, err2 = reverse_lookup_optimize(
        #     amp,
        #     phase,
        #     wavelength_set_m,
        #     model_name,
        #     lr=10e-2,
        #     err_thresh=1e-6,
        #     max_iter=500,
        #     opt_phase_only=False)
        # print("Reverse lookup p abs err", np.abs(p - p2).mean())
        # print("Reverse lookup p_norm abs err", np.abs(p_norm - p_norm2).mean())

    print(lens_type, amp.shape, p.shape, h, w)
    return amp, phase, aperture, p_norm, p

