import os
import matplotlib.pyplot as plt
import numpy as np
import einops

from psf.parse_int import extract_psf_from_content 

XX = np.linspace(0, 3, 5)
YY = np.linspace(3, 0, 5)
loc = "/depot/chan129/users/harshana/Datasets/Metazoom_PSF_INTs_new"

all_arr = []
for i in range(len(YY)):
    all_arr.append([])
    for j in range(len(XX)):
        filename = f"{loc}/my_lens_psf_yim_X_{XX[j]:.2f}_Y_{YY[i]:.2f}.int"
        with open(filename) as f:
            psf = extract_psf_from_content(f.read())
        all_arr[i].append(psf)

all_arr = np.array(all_arr)

stacked = einops.rearrange(all_arr, "H W h w -> (H h) (W w)")
plt.imsave("stacked.png", stacked)
plt.imshow(stacked)
plt.show()

