import os
# from PyQt5.QtCore import QLibraryInfo
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
import cv2

from matplotlib import pyplot as plt
import numpy as np
import torch
from deeplens.geolens import GeoLens
from deeplens.hybridlens import HybridLens
from deeplens.diffraclens import DiffractiveLens
from deeplens.optics.diffractive_surface.pixel2d import Pixel2D 
# lens = HybridLens(
#     filename='F:\\Research\\MetaZoom\\a489_doe.json',
#     device='cuda'
#     )
lens = GeoLens(
    filename='/depot/chan129/users/harshana/svd_v2/lenses/camera/rf50mm_f1.8.png',
    device='cuda'
    )
foc_dist=-100
d_sensor_new = lens.calc_sensor_plane(depth=foc_dist)
lens.refocus(foc_dist=foc_dist)
lens.draw_layout(filename="layout.png", depth=foc_dist)# draw the layout using the lens
# breakpoint()
lens.write_lens_json("tmp_lens.json")

ks = 64
psf1 = lens.psf(points=[0.0, -0.05, -1000.0], ks=ks, wvln=0.550)
psf2 = lens.psf(points=[0.0, 0.0, -1000.0], ks=ks, wvln=0.550)
psf3 = lens.psf(points=[0.0, 0.05, -1000.0], ks=ks, wvln=0.550)
plt.imsave('psfs.png', np.vstack([psf1.cpu(), psf2.cpu(), psf3.cpu()]))

img = plt.imread("/depot/chan129/users/harshana/svd_v2/941898.jpg")/255.0
img = cv2.resize(np.array(img) , lens.sensor_res)
img = torch.from_numpy(img).to(torch.float32).cuda()
# breakpoint()
render = lens.render(img[None].to(lens.device).permute(0,3,1,2), method='psf_patch', depth=-1000)[0].cpu()
plt.imsave('render.png', render.cpu().permute(1,2,0).numpy())
# meas = lens.render(torch.rand((1, 1, 500, 500), device='cuda').to(torch.float32), depth=-1000, method="psf_patch", psf_center=[0.0, 0.0], psf_ks=101)
# print(meas.shape)
