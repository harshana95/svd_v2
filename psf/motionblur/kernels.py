import sys

sys.path.append('.')
import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from .trajectory3d import create_trajectory
from .trajectory2psf import trajectory2psfs, get_camera_calibration
from .trajectory2psf import blur_with_trajectory, center_kernel


class KernelDataset(object):
    def __init__(self, psf_size=64, max_support=8.0):
        self.psf_size = psf_size
        self.calibration_params = get_camera_calibration(1.0, 1.0, psf_size // 2, psf_size // 2)

        # Trajectory generation parameter
        self.max_anxiety = 0.2
        self.max_length = 64
        # What fraction of the trajectory should be sampled to create the psf
        self.min_fraction = 0.01
        self.max_fraction = 0.05
        # Support of kernel in terms of pixel
        self.max_support = max_support  # In terms of pixel

    def sample(self):
        length = self.max_length * uniform()
        anxiety = self.max_anxiety * uniform()
        trajectory = create_trajectory(64, anxiety, 2000, length)

        support = (self.max_support * 0.5) * uniform() + self.max_support * 0.5
        fraction = self.min_fraction + (self.max_fraction - self.min_fraction) * uniform()

        kernel = trajectory2psfs(trajectory, support, fraction, self.calibration_params, self.psf_size)

        return kernel


def get_two_impulses(psf_size, factor=4):
    im = np.zeros([factor * psf_size, factor * psf_size])
    im[psf_size // 2, psf_size // 2] = 1.0
    im[(2 * factor - 1) * psf_size // 2, (2 * factor - 1) * psf_size // 2] = 1.0
    # im[psf_size//2, 3*psf_size//2] = 1.0
    # im[3*psf_size//2, psf_size//2] = 1.0
    return im


class InaccurateKernelDataset(object):
    def __init__(self, psf_size=64):
        self.psf_size = psf_size
        self.calibration_params = get_camera_calibration(1.0, 1.0, psf_size, psf_size)
        self.im = get_two_impulses(psf_size)

        # Trajectory generation parameter
        self.max_anxiety = 0.005

        # What fraction of the trajectory should be sampled to create the psf
        self.fractions = [1 / 256, 1 / 128, 1 / 64, 1 / 32, 1 / 16]

    # Support of kernel in terms of pixel

    def sample(self):
        anxiety = self.max_anxiety * uniform()
        trajectory = create_trajectory(64, anxiety, 4000, 60, z_axis=True)
        fraction = np.random.choice(self.fractions)

        N = trajectory.shape[0]
        M = int(fraction * trajectory.shape[0])
        start = np.random.randint(N - M - 1)
        out = blur_with_trajectory(self.im, trajectory[start:start + N, :])

        k1 = out[0:self.psf_size, 0:self.psf_size]
        k2 = out[-self.psf_size:, -self.psf_size:]
        return center_kernel(k1), center_kernel(k2)


if __name__ == "__main__":
    dataset = KernelDataset(64, 32)
    for idx in range(10):
        k1 = dataset.sample()
        plt.imshow(k1, cmap='gray')
        plt.show()

    dataset = InaccurateKernelDataset()
    for idx in range(10):
        k1, k2 = dataset.sample()
        plt.subplot(1, 2, 1)
        plt.imshow(k1, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(k2, cmap='gray')
        plt.show()
