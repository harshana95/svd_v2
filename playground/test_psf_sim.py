from psf.svpsf import PSFSimulator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    psfsim = PSFSimulator(image_shape=(256, 256), psf_size=31, normalize=True)
    psfsim.set_H_obj(horizontal_distance=50, vertical_distance=50, horizontal_translation=0, vertical_translation=0)
    psfsim.set_H_img(horizontal_distance=25, vertical_distance=25, horizontal_translation=0, vertical_translation=0)

    # =================================================================================================== elliptical PSF
    kwargs = {'psf_type': 'gaussian', 'gaussian_type': 'elliptical', 'sigma1': 0.04, 'sigma2': 0.10, 'sigma_min': 0.04}
    psf, metadata = psfsim.generate_psf(101, 110, 0, **kwargs)
    print(metadata)
    plt.imshow(psf)
    plt.show()

    generated_psf, metadata = psfsim.generate_all_psfs(10,10, outc=1, inc=1, **kwargs)
    psfsim.display_psfs(image_shape=(256,256), psfs=generated_psf, weights=None, metadata=metadata, title="Gaussian PSFs")
    plt.show()

    # ======================================================================================================= motion PSF
    kwargs = {'psf_type': 'motion'}
    psf, metadata = psfsim.generate_psf(101, 110, 0, **kwargs)
    print(metadata)
    plt.imshow(psf)
    plt.show()

    generated_psf, metadata = psfsim.generate_all_psfs(10,10, outc=1, inc=1, **kwargs)
    psfsim.display_psfs(image_shape=(256,256), psfs=generated_psf, weights=None, metadata=metadata, title="Gaussian PSFs")
    plt.show()

    # ======================================================================================================= zernike PSF
    kwargs = {'psf_type': 'zernike', 'n':10}
    psf, metadata = psfsim.generate_psf(101, 110, 0, **kwargs)
    print(metadata)
    plt.imshow(psf)
    plt.show()

    generated_psf, metadata = psfsim.generate_all_psfs(10,10, outc=1, inc=1, **kwargs)
    psfsim.display_psfs(image_shape=(256,256), psfs=generated_psf, weights=None, metadata=metadata, title="Gaussian PSFs")
    plt.show()