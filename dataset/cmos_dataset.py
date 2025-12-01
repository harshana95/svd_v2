import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import os
import glob
import einops

try:
    from utils.dataset_utils import crop_arr
except:
    # add parent to path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_utils import crop_arr
from utils.cmos_utils import unprocess, process

# This script defines a PyTorch Dataset for CMOS images, a GainCalculator_CMOS class,
# and a torch_forward_model_CMOS function for simulating CMOS sensor output.
# It also includes a utility function `cmos_unprocess` for preprocessing CMOS raw images.
# The main block demonstrates the usage of these functions to simulate and compare
# CMOS and QIS sensor outputs from a given input image.

class CMOS_image_dataset(torch.utils.data.Dataset):
    def __init__(self, opt): # data_path, sensor_params, patch_size = 128):
        self.opt = opt
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key
        self.patch_size = opt.patch_size
        self.split_type = opt.split_type

        self.img_idx = 0

        self.sensor_params = opt.sensor_params
        self.QE = self.sensor_params.QE
        self.theta_dark = self.sensor_params.theta_dark
        self.sigma_read = self.sensor_params.sigma_read
        self.clicks_per_frame = self.sensor_params.clicks_per_frame
        self.Nbits = self.sensor_params.Nbits
        self.fwc = self.sensor_params.fwc

        pixel_size = self.sensor_params.pixel_size
        lux = self.sensor_params.lux
        self.exp_time = self.sensor_params.exp_time
        self.PPP = 65 * pixel_size * lux * 60 * self.exp_time

        ref_exp_time = min(1/30, self.exp_time)
        # assert exp_time >= ref_exp_time, "Exposure time should be greater than or equal to reference exposure time of 1/24 sec."
        self.num_imgs = int(self.exp_time / ref_exp_time)
        
        self.ref_base_folder = opt.hq_folder_path
        
        self.folder_path = sorted(glob.glob(os.path.join(opt.data_path, '*')))
        L = len(self.folder_path)
        self.folder_path = self.folder_path[int(opt.split[0]*L):int(opt.split[1]*L)]

        # finding total number of images
        self.total_images = 0
        self.image_list = []
        for folder in self.folder_path:
            img_list = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')])
            self.image_list.append(img_list)
            self.total_images += len(img_list)
        print(f"Total number of images: {self.total_images} in {len(self.folder_path)} folders")

    def __len__(self):
        return self.total_images
    
    #random cropping can be added here for training
    @staticmethod
    def random_crop(img, crop_size):
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            img = crop_arr(img.transpose([2,0,1]), max(crop_size, h), max(crop_size, w)).transpose([1,2,0])
            h, w = img.shape[:2]
        top = np.random.randint(0, h - crop_size+1)
        left = np.random.randint(0, w - crop_size+1)
        top = max(0, top - top%2)
        left = max(0, left - left%2)
        return img[top:top + crop_size, left:left + crop_size]
    
    @staticmethod
    def center_crop(img, crop_size):
        h, w = img.shape[:2]
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        top = max(0, top - top%2)
        left = max(0, left - left%2)
        return img[top:top + crop_size, left:left + crop_size]
    
    @staticmethod
    def cmos_unprocess(img_list):
        images = []
        for img_path in img_list:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.tensor(img / 255.0, dtype=torch.float32)
            images.append(img)

        avg_img = None
        #unprocess to raw
        raw_images, metadata = unprocess(images)
        for raw in raw_images:
            if avg_img is None:
                avg_img = raw
            else:
                avg_img += raw
        avg_img = avg_img / len(raw_images)

        return avg_img, metadata

    def __getitem__(self, idx):
        # find the folder that contains the image at index idx
        i = 0
        while idx >= len(self.image_list[i]):
            idx -= len(self.image_list[i])
            i += 1

        folder = self.folder_path[i]
        ref_img_basename = os.path.basename(self.image_list[i][idx])
        if idx+self.num_imgs > len(self.image_list[i]):  # we overshoot total images in the folder
            img_list = self.image_list[i][idx-self.num_imgs+1:idx+1][::-1] 
        else:
            img_list = self.image_list[i][idx:idx+self.num_imgs]
        raw, metadata  = self.cmos_unprocess(img_list)
        processed_raw = torch_forward_model_CMOS(
                        self.PPP, 
                        raw, 
                        self.QE,
                        self.theta_dark * self.exp_time, 
                        self.sigma_read,
                        self.clicks_per_frame,
                        self.Nbits, 
                        self.fwc, 
                        normalize=True
                    ) # (1, H/2, W/2, 4)
        # raw to rgb
        red_gains = metadata['red_gain'].unsqueeze(0)    # (scalar) -> (1,)
        blue_gains = metadata['blue_gain'].unsqueeze(0)   # (scalar) -> (1,)
        cam2rgbs = metadata['cam2rgb'].unsqueeze(0)   # (3, 3) -> (1, 3, 3)
        sim_cmos_img = process(processed_raw[None], red_gains, blue_gains, cam2rgbs)[0]
        # gamma correction
        sim_cmos_img = sim_cmos_img.pow(1/2.2)

        ref_img = cv2.imread(os.path.join(self.ref_base_folder, os.path.basename(folder), ref_img_basename))
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_img = torch.from_numpy(ref_img.astype(np.float32)) / 255.0  # normalize to [0,1]
        
        if self.patch_size>0:
            # if self.split_type == 'train':
            #     ref_img = self.random_crop(ref_img, self.patch_size)
            # else:
            ref_img = self.center_crop(ref_img, self.patch_size)
            sim_cmos_img = self.center_crop(sim_cmos_img, self.patch_size)

        sim_cmos_img = sim_cmos_img.permute(2, 0, 1)  # HWC to CHW
        ref_img = ref_img.permute(2, 0, 1)  # HWC to CHW
        assert torch.isnan(sim_cmos_img).sum()==0, f"Simulated CMOS image has NaN values: {sim_cmos_img}"
        assert torch.isnan(ref_img).sum()==0, f"Reference image has NaN values"
        
        data = {self.lq_key: sim_cmos_img*2-1,  # 0,1 to -1,1
                self.gt_key: ref_img*2-1}  # 0,1 to -1,1
        return data
    
class GainCalculator_CMOS:
    '''
    Gain class is mapping avg_PPP to gain value.
    avg_PPP can be converted to lux assuming exposure time = 1/2000 sec. Ex: 3.25PPP -> 1 lux and 9.75 PPP -> 3 lux.
    This mapping function works for QIS sensors with pixel size 1.1um and full well capacity 200e-.
    Sensor Parameters assumed:
    'QE': 0.56,
    'theta_dark': 1.7,  # e-/pix/frame
    'sigma_read': 2.2,  # e- RMS
    'Nbits': 14,
    'N': 1,
    'fwc': 31402  # e-
    '''
    def __init__(self):
        # Define the (x, y) pairs
        self.data = np.array([
        [2, 150], [5, 93.75], [8, 82.5], [10, 75], [15, 71.25], [20, 67.5], [30, 56.25], [40, 48.75], [50, 45],
        [60, 38.657142857], [75, 30.026785714], [90, 23.018571429], [100, 18], [120, 16.144090909], [150, 14.020833333],
        [180, 12.172467532], [200, 10.108225108], [225, 9.309063853], [250, 8.767045455], [300, 7.448051948],
        [325, 6.673295455], [350, 6.397294372], [375, 6.123511905], [400, 5.486201298], [450, 5.194669911],
        [500, 4.787878788], [550, 4.387743505], [600, 4.111742424], [650, 3.88672619], [700, 3.688311688],
        [750, 3.493003253], [800, 3.346017317], [850, 3.022159091], [900, 2.770562771], [950, 2.457575758],
        [1000, 2.172077922], [1050, 2.063896104], [1100, 1.957489179], [1150, 1.852857143], [1200, 1.75],
        [1250, 1.666666667], [1300, 1.583333333], [1350, 1.5], [1400, 1.416666667], [1450, 1.354166667],
        [1500, 1.291666667], [1550, 1.229166667], [1600, 1.166666667], [1650, 1.104166667], [1700, 1.0625],
        [1750, 1.020833333], [1800, 0.979166667], [1850, 0.9375], [1900, 0.895833333], [1950, 0.854166667],
        [2000, 0.8125], [2050, 0.770833333], [2100, 0.729166667], [2150, 0.708333333], [2200, 0.6875],
        [2250, 0.666666667], [2300, 0.645833333], [2350, 0.625], [2400, 0.604166667], [2450, 0.583333333],
        [2500, 0.5625], [2550, 0.541666667], [2600, 0.520833333], [2650, 0.5], [2700, 0.479166667],
        [2750, 0.46875], [2800, 0.458333333], [2850, 0.447916667], [2900, 0.4375], [2950, 0.427083333],
        [3000, 0.416666667], [4000, 0.354166667], [5000, 0.302083333], [6000, 0.270833333], [7000, 0.245833333],
        [8000, 0.229166667], [9000, 0.21875], [10000, 0.208333333]
        ])

        x = self.data[:, 0]
        y = self.data[:, 1]

        # Fit an RBF interpolator
        self.rbf = Rbf(x, y, function='linear')

    def get_gain(self, avg_PPP, N=1):
        # Evaluate the polynomial at avg_PPP
        return self.rbf(avg_PPP) * N


@torch.no_grad()
def torch_forward_model_CMOS(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc, normalize = True):
    min_val = 0
    max_val = 2 ** Nbits - 1
    gain_func = GainCalculator_CMOS()
    gain = gain_func.get_gain(avg_PPP)
    
    # Convert all inputs to torch tensors if they are not already
    avg_PPP = torch.tensor(avg_PPP, dtype=torch.float32)
    #photon_flux = torch.tensor(photon_flux, dtype=torch.float32)
    QE = torch.tensor(QE, dtype=torch.float32)
    theta_dark = torch.tensor(theta_dark, dtype=torch.float32)
    sigma_read = torch.tensor(sigma_read, dtype=torch.float32)
    gain = torch.tensor(gain, dtype=torch.float32)
    fwc = torch.tensor(fwc, dtype=torch.float32)

    # Calculate theta
    theta = photon_flux * (avg_PPP / (torch.mean(photon_flux) + 0.0001))

    # Calculate lam
    lam = ((QE * theta) + theta_dark) / N

    #c, m, n = theta.shape
    img_out = torch.zeros_like(theta)

    for i in range(N):
        # Poisson sampling
        # print(lam.min(), lam.max(), lam.shape, lam.type())
        tmp = torch.poisson(lam)

        # Clipping to full well capacity (fwc)
        tmp = torch.clamp(tmp, 0, fwc.item())

        # Adding read noise
        tmp = tmp + torch.normal(mean=0, std=sigma_read, size=img_out.shape, device=theta.device)

        # Amplifying, quantizing, and clipping
        tmp = torch.round(tmp * gain * max_val / fwc)
        tmp = torch.clamp(tmp, min_val, max_val)

        # Summing up the images
        img_out = img_out + tmp

    # Averaging over N frames
    img_out = img_out / N
    if normalize:
        img_out = img_out/max_val

    return img_out




if __name__ == "__main__":
    from dataset.qis_dataset import torch_forward_model
    # Lux to avg PPP mapping.
    lux = 0.1  # input lux level
    cmos_pixel_size = 2
    cmos_exp_time = 1/8
    qis_pixel_size = 1.2
    qis_exp_time = 1/120

    # samsung nx500: reference: https://www.photonstophotos.net/Charts/Sensor_Characteristics.htm
    # pixel pitch assumed to be 2um
    cmos_sensor_params = {
        'QE': 0.6,
        'theta_dark': 15,  # e-/pix/s
        'sigma_read': 3,  # e- RMS
        'Nbits': 10,
        'N': 1,
        'fwc': 8000  # e-
    }

    # qis_sensor_params = {
    #     'QE': 0.8,
    #     'theta_dark': 1.6,  # e-/pix/s
    #     'sigma_read': 0.2,  # e- RMS
    #     'Nbits': 3,
    #     'N': 1,
    #     'fwc': 200  # e-
    # }

    cmos_sensor_params['theta_dark'] *= cmos_exp_time
    print('cmos sensor_params:', cmos_sensor_params)

    # qis_sensor_params['theta_dark'] *= qis_exp_time
    # print('qis sensor_params:', qis_sensor_params)

    CMOS_PPP = 65 * cmos_pixel_size * lux * 60 * cmos_exp_time
    # QIS_PPP = 65 * qis_pixel_size * lux * 60 * qis_exp_time

    if CMOS_PPP > cmos_sensor_params['fwc']:
        CMOS_PPP = cmos_sensor_params['fwc'] - 10
        print("Clipping CMOS_PPP to fwc value.")
        cmos_exp_time = cmos_sensor_params['fwc'] / (65 * cmos_pixel_size * lux * 60)
        print(f"Adjusted CMOS exposure time to {cmos_exp_time} sec.")

    # if QIS_PPP > qis_sensor_params['fwc']:
    #     QIS_PPP = qis_sensor_params['fwc'] - 10
    #     print("Clipping QIS_PPP to fwc value.")
    #     qis_exp_time = qis_sensor_params['fwc'] / (65 * qis_pixel_size * lux * 60)
    #     print(f"Adjusted QIS exposure time to {qis_exp_time} sec.")

    print(f"Simulating for CMOS_avg_PPP: {CMOS_PPP}")
    # print(f"Simulating for QIS_avg_PPP: {QIS_PPP}")

    cmos_base_folder = '/depot/chan129/users/pchennur/datasets/test_lolblur/low_blur/'
    folder = '0123'
    raw, metadata, ref_img_basename = CMOS_image_dataset.cmos_unprocess(os.path.join(cmos_base_folder, folder), exp_time = cmos_exp_time)
    sim_raw = torch_forward_model_CMOS(photon_flux = raw, avg_PPP = CMOS_PPP, **cmos_sensor_params, normalize = True)

    ref_base_folder = '/depot/chan129/users/pchennur/datasets/test_lolblur/high_sharp_original/'
    ref_img = cv2.imread(os.path.join(ref_base_folder, folder, ref_img_basename))
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    # qis_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
    # qis_img = torch.from_numpy(qis_img.astype(np.float32)).unsqueeze(0)
    # qis_img = torch_forward_model(QIS_PPP, qis_img, **qis_sensor_params, normalize = True)
    

    batched_raw = sim_raw.unsqueeze(0)  # (H/2, W/2, 4) -> (1, H/2, W/2, 4)
    red_gains = metadata['red_gain'].unsqueeze(0)    # (scalar) -> (1,)
    blue_gains = metadata['blue_gain'].unsqueeze(0)   # (scalar) -> (1,)
    cam2rgbs = metadata['cam2rgb'].unsqueeze(0)   # (3, 3) -> (1, 3, 3)

    # process back to rgb
    sim_cmos_img = process(batched_raw, red_gains, blue_gains, cam2rgbs)

    sim_cmos_img = sim_cmos_img.squeeze(0).numpy()
    sim_cmos_img = (sim_cmos_img * 255).astype(np.uint8)
    # qis_img = qis_img.squeeze(0).numpy()
    # qis_img = (qis_img * 255).astype(np.uint8)
    # print('qis min,max:', qis_img.min(), qis_img.max())

    V_MIN = 0.0  # Set the absolute minimum value for the data type
    V_MAX = 255.0  # Set the absolute maximum value for the data type

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(ref_img, vmin=V_MIN, vmax=V_MAX)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Simulated CMOS')
    plt.imshow(sim_cmos_img, vmin=V_MIN, vmax=V_MAX)
    plt.axis('off')

    # plt.subplot(1, 3, 3)
    # plt.title('Simulated QIS')
    # plt.imshow(qis_img, cmap='gray', vmin=V_MIN, vmax=V_MAX)
    # plt.axis('off')

    plt.show()