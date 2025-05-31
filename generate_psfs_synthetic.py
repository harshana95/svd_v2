import matplotlib

matplotlib.use("TkAgg")
import argparse
import numpy as np
from matplotlib import pyplot as plt

from psf.svpsf import PSFSimulator
from utils import generate_folder

# from PySide2.QtCore import QLibraryInfo
# from PyQt5.QtCore import QLibraryInfo
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)


if __name__ == "__main__":
    DEBUG = False
    kwargs = {'psf_type': 'gaussian', 'gaussian_type': 'elliptical', 'sigma1': 0.08, 'sigma2': 0.20, 'sigma_min': 0.08, 'chromatic': [0.0,0.5,0.9], 'chromatic_trans': [0.7,0.9,0.7]}
    # kwargs = {'psf_type': 'motion'}
    """
    R: 620-750 nm  685
    G: 495-570 nm  532.8
    B: 450-495 nm  472.5
    """
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--sensor_size', default=(4,4), action='store', type=float, nargs=2)
    parser.add_argument('--sensor_res', default=(1024,1024), action='store', type=int, nargs=2)
    parser.add_argument('--max_psf_size', type=int, default=128)
    parser.add_argument('--save_dir', default="")
    parser.add_argument('--ndata', type=int, default=256 * 256)
    parser.add_argument('--wv', action='store', type=int, nargs=3, default=[472.5, 532.8, 685])
    parser.add_argument('--coordinates', default='cartesian', choices=['random', 'polar', 'cartesian'])
    parser.add_argument('--n_sectors', default=50, help='# of sectors when using polar coordinates')
    args = parser.parse_args()

    if args.save_dir == "":
        loc = f"/scratch/gilbreth/wweligam/dataset/{kwargs['psf_type']}_synthetic"
    else:
        loc = args.save_dir

    # initialize
    generate_folder(loc)
    metadata = vars(args)
    psfsim = PSFSimulator(image_shape=args.sensor_res, psf_size=args.max_psf_size, normalize=True)
    if args.coordinates == 'cartesian':
        n, m = int(args.ndata ** 0.5), int(args.ndata ** 0.5)
        for i in range(1, int(args.ndata ** 0.5)+1):
            if args.ndata/(i*i) <= args.sensor_res[1]/args.sensor_res[0]:
                n,m = i, int(args.ndata/i)
                break
        print(f"Using cartesian coordinates. {n}x{m} PSFs")
        metadata['grid_size'] = [n, m]

        psfsim.set_H_obj(horizontal_distance=1, vertical_distance=1,
                         horizontal_translation=0, vertical_translation=0)
        psfsim.set_H_img(horizontal_distance=args.sensor_res[1] / m, vertical_distance=args.sensor_res[0] / n,
                         horizontal_translation=0, vertical_translation=0)

    # generate PSFs
    generated_psf, psf_metadata = psfsim.generate_all_psfs(metadata['grid_size'][0], metadata['grid_size'][1], outc=1, inc=args.n_channels, **kwargs)
    metadata['psf_metadata'] = psf_metadata
    PSFSimulator.display_psfs(image_shape=args.sensor_res, psfs=generated_psf, weights=None, metadata=psf_metadata, title="All PSFs", skip=1)
    plt.savefig(f"{loc}/psfs.png")
    PSFSimulator.display_psfs(image_shape=args.sensor_res, psfs=generated_psf, weights=None, metadata=psf_metadata, title="10x10 PSFs", skip=generated_psf.shape[2] // 10)
    plt.savefig(f"{loc}/psfs_10x10.png")
    

    # saving PSFs
    psfsim.save_psfs(loc, generated_psf, metadata=metadata, H_img=psfsim.H_img, H_obj=psfsim.H_obj)

    # check loading
    print("Loading PSFs")
    psfs, metadata = psfsim.load_psfs(loc)
    for name in psfs:
        print(f"{name} {psfs[name].shape}")
