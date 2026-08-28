"""Image I/O helpers (OpenCV backend) supporting 8/16-bit RGB.

PIL/imageio cannot write 16-bit RGB PNG; we use OpenCV, which round-trips 16-bit
PNG/TIFF cleanly.  All functions work in RGB float [0, 1].
"""

from __future__ import annotations

import numpy as np


def imread01(path: str) -> np.ndarray:
    """Read an image as float32 HxWx3 in [0, 1] (handles 8/16-bit, gray, npy)."""
    p = str(path)
    if p.endswith(".npy"):
        a = np.load(p).astype(np.float32)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        return np.clip(a[:, :, :3], 0.0, 1.0)

    import cv2
    a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"failed to read image: {p}")
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    a = a[:, :, :3][:, :, ::-1]  # BGR -> RGB
    maxv = 65535.0 if a.dtype == np.uint16 else 255.0
    return np.clip(a.astype(np.float32) / maxv, 0.0, 1.0)


def imwrite16(path: str, rgb01: np.ndarray) -> None:
    import cv2
    a = (np.clip(rgb01, 0.0, 1.0) * 65535.0).astype(np.uint16)[:, :, ::-1]
    cv2.imwrite(str(path), a)


def imwrite8(path: str, rgb01: np.ndarray) -> None:
    import cv2
    a = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(str(path), a)
