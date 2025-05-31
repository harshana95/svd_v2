import numpy as np
import matplotlib.pyplot as plt

# Parameters
wavelength = 0.5e-6                 # Wavelength in meters
k = 2 * np.pi / wavelength          # Wavenumber
z1_range = np.logspace(-2, 4, 101)   # Distance from point source to lens (in meters)
# z1_range = np.linspace(5, 15, 101)   # Distance from point source to lens (in meters)
z2 = 0.005                           # Distance from lens to sensor (in meters)
L = 0.001                           # Size of the lens plane (in meters)
N = 1024                             # Number of sample points across the lens plane
f = 0.0025                            # Focal length of the lens (in meters)
plot = False

target_z = 100
f = target_z*z2/(target_z + z2)
print(f)

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

cutoff_at_z1 = []
for z1 in z1_range:
    # Create lens plane coordinates
    x = np.linspace(-L/2, L/2, N)
    y = np.linspace(-L/2, L/2, N)
    X, Y = np.meshgrid(x, y)

    # Distance from the point source to each point on the lens plane
    r = np.sqrt(X**2 + Y**2 + z1**2)

    # Wavefront before the lens (spherical wave)
    wavefront = np.exp(1j * k * r) / r

    if plot:
        # Plot the phase and amplitude of the wavefront
        plt.figure(figsize=(12, 10))

        # Amplitude plot
        plt.subplot(2, 2, 1)
        plt.imshow(np.abs(wavefront), extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
        plt.colorbar(label='Amplitude'), plt.title('Wavefront Amplitude before Lens Plane')
        plt.xlabel('x (m)'), plt.ylabel('y (m)')

        # Phase plot
        plt.subplot(2, 2, 2)
        plt.imshow(np.angle(wavefront), extent=[-L/2, L/2, -L/2, L/2], cmap='twilight')
        plt.colorbar(label='Phase (radians)'), plt.title('Wavefront Phase before Lens Plane')
        plt.xlabel('x (m)'), plt.ylabel('y (m)')

    # Wavefront after the lens
    # Create circular aperture mask
    aperture = (X**2 + Y**2 <= (L/2)**2).astype(float)
    lens_phase = np.exp(-1j * k / (2 * f) * (X**2 + Y**2))
    wavefront = wavefront * lens_phase * aperture

    if plot:
        # Amplitude plot
        plt.subplot(2, 2, 3)
        plt.imshow(np.abs(wavefront), extent=[-L/2, L/2, -L/2, L/2], cmap='gray')
        plt.colorbar(label='Amplitude'), plt.title('Wavefront Amplitude after Lens Plane')
        plt.xlabel('x (m)'), plt.ylabel('y (m)')

        # Phase plot
        plt.subplot(2, 2, 4)
        plt.imshow(np.angle(wavefront), extent=[-L/2, L/2, -L/2, L/2], cmap='twilight')
        plt.colorbar(label='Phase (radians)'), plt.title('Wavefront Phase after Lens Plane')
        plt.xlabel('x (m)'), plt.ylabel('y (m)')

        plt.tight_layout()
        plt.savefig(f'wavefront_{z1}.png')

    # free space propagation of the wavefront to the sensor plane
    
    # Create spatial frequency grid
    fx = np.fft.fftfreq(N, d=L / N)
    fy = np.fft.fftfreq(N, d=L / N)
    FX, FY = np.meshgrid(fx, fy)

    # Angular spectrum transfer function
    H = np.exp(1j * k * z2 * np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))

    # Propagate to sensor using angular spectrum method
    u = np.fft.ifft2(np.fft.fft2(wavefront) * H)

    # Compute PSF at the sensor (intensity)
    psf = np.abs(u)**2

    # Compute OTF (Fourier transform of PSF)
    otf = np.fft.fftshift(np.fft.fft2(psf))

    # MTF is the magnitude of the OTF 
    ptf = np.angle(otf)
    mtf = np.abs(otf) 
        
    # # MTF from autocorrelation of the wavefront u
    # mtf = np.abs(np.fft.ifft2(np.fft.fft2(u) * np.conj(np.fft.fft2(u))))

    
    # # normalize MTF
    mtf = mtf / np.max(mtf)

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
    # mtf_1d = mtf_1d / np.max(mtf_1d)

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
    cutoff_at_z1.append(cutoff)

plt.figure(figsize=(6, 4))
plt.plot(z1_range, cutoff_at_z1, '.-')
plt.xscale('log')
plt.xlabel('Distance from Point Source to Lens (m)')
plt.ylabel('Cutoff Spatial Frequency (1/m)')
plt.grid(True)
plt.tight_layout()
plt.savefig('cutoff.png')

        