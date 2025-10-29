from matplotlib import pyplot as plt
import numpy as np

from deeplens.hybridlens import HybridLens
from deeplens.diffraclens import DiffractiveLens
from deeplens.optics.diffractive_surface.pixel2d import Pixel2D 
lens = HybridLens(
    filename='/depot/chan129/users/harshana/svd_v2/checkpoints/MetaZoom/metazoom2.json',
    # filename='/depot/chan129/users/harshana/svd_v2/checkpoints/DepthEstimation/a489_doe_ours.json',
    device='cuda'
    )
# lens.refocus(foc_dist=-1000)
lens.draw_layout()# draw the layout using the lens

ks = None
psf1 = lens.psf(points=[0.0, -0.05, -1000.0], ks=ks, wvln=0.550)
psf2 = lens.psf(points=[0.0, 0.0, -1000.0], ks=ks, wvln=0.550)
psf3 = lens.psf(points=[0.0, 0.05, -1000.0], ks=ks, wvln=0.550)
plt.imsave('tmp.png', np.vstack([psf1.cpu(), psf2.cpu(), psf3.cpu()]))

# meas = lens.render(torch.rand((1, 1, 500, 500), device='cuda').to(torch.float32), depth=-1000, method="psf_patch", psf_center=[0.0, 0.0], psf_ks=101)
# print(meas.shape)
