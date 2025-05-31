import numpy as np
import cv2


def get_camera_calibration(fx, fy, x0, y0):
    K = np.zeros([3, 3], dtype=np.float32)
    K[0, 0], K[1, 1], K[2, 2] = fx, fy, 1.0
    K[0, 2], K[1, 2] = x0, y0
    return K, np.linalg.inv(K)


def get_R(Tx, Ty, theta_z=0):
    # Tx, Ty but theta_z in degrees
    R = np.zeros([3, 3], dtype=np.float32)
    R[0, 0] = np.cos(theta_z * np.pi / 180);
    R[1, 1] = np.cos(theta_z * np.pi / 180)
    R[1, 0] = np.sin(theta_z * np.pi / 180);
    R[0, 1] = -np.sin(theta_z * np.pi / 180)
    R[0, 2] = Tx;
    R[1, 2] = Ty
    R[2, 2] = 1.0

    return R


def get_transformation_matrix(Tx, Ty, theta_z=0, calibration_params=[np.eye(3), np.eye(3)]):
    K, K1 = calibration_params
    # Tx, Ty but theta_z in degrees
    R = get_R(Tx, Ty, theta_z)
    M = np.matmul(np.matmul(K, R), K1)
    return M


def get_first_moments(im):
    rows, cols = np.shape(im)
    seq1 = np.repeat(np.reshape(np.arange(rows), [rows, 1]), cols, axis=1)
    seq2 = np.repeat(np.reshape(np.arange(cols), [1, cols]), rows, axis=0)
    mx, my = np.mean(seq1 * im) / np.mean(im), np.mean(seq2 * im) / np.mean(im)
    return mx, my


def center_kernel(kernel):
    N, _ = np.shape(kernel)
    # Center the image
    mx, my = get_first_moments(kernel)
    shift_x, shift_y = N // 2 - np.int32(mx), N // 2 - np.int32(my)
    kernel = np.roll(kernel, (shift_x, shift_y), axis=[0, 1])
    return kernel


def get_impulse(psf_size):
    # Make an image consisting of grid points
    grid = np.zeros([psf_size, psf_size])
    grid[psf_size // 2, psf_size // 2] = 1.0
    return grid


def blur_with_trajectory(im, trajectory):
    H, W = np.shape(im)
    K, K1 = get_camera_calibration(1, 1, H // 2, W // 2)
    out = im.copy() * 0

    # Make sure the trajcetory is centered
    for idx in range(trajectory.shape[1]):
        trajectory[:, idx] -= np.mean(trajectory[:, idx])

    N_samples = trajectory.shape[0]
    z_rotation = True if trajectory.shape[1] > 2 else False

    for idx in range(N_samples):
        tx, ty = trajectory[idx, 0], trajectory[idx, 1]
        theta_z = 0 if not z_rotation else trajectory[idx, 2]

        M = get_transformation_matrix(tx, -ty, theta_z, [K, K1])
        out += cv2.warpPerspective(im, M, (H, W))

    out = out / N_samples
    return out


def trajectory2psfs(trajectory, pixel_support, fraction, calibration_params, psf_size=64, center=True):
    impulse = get_impulse(psf_size)
    psf = impulse.copy() * 0

    N_samples = trajectory.shape[0]
    length = int(fraction * N_samples)
    start = np.random.randint(0, N_samples - length - 1)
    sample = trajectory[start:start + length, :]
    sample = sample - np.min(sample)

    # Determine the scale to be used by maximum pixel support allowed
    max_shift = np.max(np.abs(sample))
    scale = pixel_support / max_shift

    for idx in range(length):
        tx, ty = sample[idx, 0], sample[idx, 1]

        M = get_transformation_matrix(tx * scale, -ty * scale, calibration_params=calibration_params)
        psf += cv2.warpPerspective(impulse, M, (psf_size, psf_size))

    psf = np.clip(psf, 0, np.inf)
    if np.sum(psf) > 0:
        psf /= np.sum(psf)
    else:
        psf = np.asarray(impulse, dtype=np.float32)
    if center:
        psf = center_kernel(psf)

    return psf


if __name__ == "__main__":
    sys.path.append('.')
    import sys
    from blurkernel.trajectory import create_trajectory
    import matplotlib.pyplot as plt

    np.random.seed(54)
    psf_size = 64
    trajectory = create_trajectory(64, 0.01, 2000, 64)

    calibration_params = get_camera_calibration(1.0, 1.0, psf_size // 2, psf_size // 2)

    T_list = [0.01, 0.05, 0.1, 0.5]
    plt.figure(figsize=(8, 3))
    # 'Sampling at fraction 0.01, 0.05, 0.1, and 0.5'
    for idx in range(4):
        plt.subplot(1, 4, idx + 1)
        kernel = trajectory2psfs(trajectory, 16.0, T_list[idx], calibration_params)
        plt.imshow(kernel, cmap='gray');
        plt.axis('off')

    plt.show()

    # Sample different sections of the same trajectory
    T_frac = 0.5
    for idx in range(4):
        plt.subplot(1, 4, idx + 1)
        kernel = trajectory2psfs(trajectory, 16.0, T_frac, calibration_params)
        plt.imshow(kernel, cmap='gray');
    plt.show()
