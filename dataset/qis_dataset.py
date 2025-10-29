import cv2
import random
import einops
import numpy as np
import torch
import os
import glob
from torch.utils.data import Dataset
from scipy.interpolate import Rbf
from torchvision import transforms as T
import torch.nn as nn
try:
    from utils.dataset_utils import crop_arr
except:
    # add parent to path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.dataset_utils import crop_arr


# -----------------------
# 6) Training dataset placeholder (user should replace with real DataLoader)
#     Here we create a minimal dummy loader to illustrate training loop structure.
# -----------------------
# NOTE: Replace this Dataset with your paired QIS(1ch) -> RGB dataset.

class QIS_image_dataset(torch.utils.data.Dataset):
    def __init__(self, opt): # data_path, sensor_params, patch_size = 128):
        self.gt_key = opt.gt_key
        self.lq_key = opt.lq_key
        self.patch_size = opt.patch_size
        self.split_type = opt.split_type

        self.sensor_params = opt.sensor_params
        self.images_path = sorted(glob.glob(os.path.join(opt.data_path, '*.png')))
        L = len(self.images_path)
        self.images_path = self.images_path[int(opt.split[0]*L):int(opt.split[1]*L)]

    def __len__(self):
        return len(self.images_path)
    
    #random cropping can be added here for training
    def random_crop(self, img, crop_size):
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            img = crop_arr(img.transpose([2,0,1]), max(crop_size, h), max(crop_size, w)).transpose([1,2,0])
            h, w = img.shape[:2]
        top = np.random.randint(0, h - crop_size+1)
        left = np.random.randint(0, w - crop_size+1)
        return img[top:top + crop_size, left:left + crop_size]
    def center_crop(self, img, crop_size):
        h, w = img.shape[:2]
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        return img[top:top + crop_size, left:left + crop_size]

    def __getitem__(self, idx):
        # random QIS 1-channel (simulate 3-bit by integer levels)
        rgb = self.images_path[idx]
        rgb = cv2.imread(rgb)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if self.patch_size>0:
            if self.split_type == 'train':
                rgb = self.random_crop(rgb, self.patch_size)
            else:
                rgb = self.center_crop(rgb, self.patch_size)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = torch.from_numpy(gray.astype(np.float32)) / 255.0  # normalize to [0,1]
        gray = gray.unsqueeze(0)  # add channel dimension

        qis = torch_forward_model(self.sensor_params.get('avg_PPP', 3.25), 
                                      gray, 
                                      self.sensor_params.get('QE', 0.8),
                                      self.sensor_params.get('theta_dark', 1.6), 
                                      self.sensor_params.get('sigma_read', 0.2),
                                      self.sensor_params.get('clicks_per_frame', 1),
                                      self.sensor_params.get('Nbits', 3), 
                                      self.sensor_params.get('fwc', 200), 
                                      normalize = True)
        
        rgb = torch.from_numpy(rgb.astype(np.float32)) / 255.0  # normalize to [0,1]
        rgb = rgb.permute(2, 0, 1)  # HWC to CHW

        # print(qis.min(), qis.max(), rgb.min(), rgb.max(), qis.shape, rgb.shape)
        qis = einops.repeat(qis, '1 h w -> 3 h w')
        data = {self.lq_key: qis*2-1,  # 0,1 to -1,1
                self.gt_key: rgb*2-1}  # 0,1 to -1,1
        return data

# =================================================================== Video Dataset

class I2_2000FPS_Train_Dataset(Dataset):
    def __init__(self, 
                 gtdata_dir, 
                 image_size, 
                 num_frames, 
                 start_frame, 
                 downsample, 
                 transforms, 
                 augment_flip, 
                 patch_size,
                 avg_PPP_low, 
                 avg_PPP_high,
                 PPP,  
                 QE, 
                 theta_dark, 
                 sigma_read, 
                 clicks_per_frame, 
                 Nbits, 
                 fwc):
        
        super(I2_2000FPS_Train_Dataset, self).__init__()
        self.video_paths = sorted(glob.glob(os.path.join(gtdata_dir, '*.mp4')))
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.downsample = downsample
        self.transforms = transforms
        self.patch_size = patch_size
        self.avg_PPP_low = avg_PPP_low
        self.avg_PPP_high = avg_PPP_high
        self.PPP = PPP
        self.QE = QE
        self.theta_dark = theta_dark
        self.sigma_read = sigma_read
        self.clicks_per_frame = clicks_per_frame
        self.Nbits = Nbits
        self.fwc = fwc
        self.image_size = image_size

        self.transforms = T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip() if augment_flip else nn.Identity(),
            T.RandomVerticalFlip() if augment_flip else nn.Identity(),
            T.RandomCrop(self.image_size)
        ])
    

    def PPP_sampling(self, low=3.25, high=32.5, size=1, bias_strength=2):
        """
        Generate samples biased towards the lower end of the range using an inverse power function.
        
        :param low: Minimum value of the range
        :param high: Maximum value of the range
        :param size: Number of samples
        :param bias_strength: Higher values increase bias towards lower values
        :return: NumPy array of biased samples
        """
        samples = np.random.power(bias_strength, size)  # Generates values skewed towards 0
        scaled_samples = low + (high - low) * (1 - samples)  # Flip the distribution
        return scaled_samples
    

    def __getitem__(self, idx):
        vid_path = self.video_paths[idx]
        gt_seq = frames_extraction(vid_path, self.num_frames, start_frame = self.start_frame, downsample=self.downsample)
        gt_seq = np.float32(gt_seq)
        
        if self.num_frames > 1:
            gt_seq = gt_seq[:,None,:,:]
        
        gt_seq = torch.from_numpy(gt_seq)
        
        if self.transforms:
            gt_seq = self.transforms(gt_seq)

        if self.PPP is None:
            self.PPP = self.PPP_sampling(low=self.avg_PPP_low, high=self.avg_PPP_high, size=1, bias_strength=2)

        qis_seq = torch_forward_model(self.PPP, gt_seq, self.QE,
                                    self.theta_dark, self.sigma_read,
                                    self.clicks_per_frame, self.Nbits, self.fwc, normalize = True)

        gt_seq = gt_seq / 255.

        return {'qis': qis_seq, 
                'gt': gt_seq,
                'PPP': self.PPP,
                'basename': os.path.basename(vid_path).split('.')[0]}

    def __len__(self):
        return len(self.video_paths)



# ================================================================================
# ===================================================================== Data Utils
# ================================================================================

def frames_extraction(video_path, frames_no, start_frame = None, downsample = None):
    vid = cv2.VideoCapture(video_path)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames<15:
        print('*************************')
        print(video_path)
        print('*************************')
    if start_frame == None and total_frames>15:
        start_frame = random.randint(0, total_frames - 1 - frames_no)
    else:
        start_frame = start_frame
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    seq_list = []
    for _ in range(frames_no):
        _, img = vid.read()
        seq_list.append(img)
    
    for f in range(len(seq_list)):
        img = seq_list[f]    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if downsample != None:
            c = 1024
            r = 512
            output_shape = (c + (c % 4), r + (r % 4))
        else:
            output_shape = (img.shape[1] + (img.shape[1] % 4), img.shape[0] + (img.shape[0] % 4))
        img = cv2.resize(img, output_shape, interpolation=cv2.INTER_CUBIC)
        seq_list[f] = img
    
    seq = np.stack(seq_list, axis=0)
    
    return seq


def sensor_image_simulation(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, gain, fwc):
    min_val = 0
    max_val = 2 ** Nbits - 1

    theta = photon_flux * (avg_PPP / (np.mean(photon_flux) + 0.0001))

    lam = ((QE * theta) + theta_dark) / N

    m, n, c = theta.shape
    img_out = np.zeros((m, n, c))
    
    for i in range(N):
        tmp = np.random.poisson(lam=lam, size=(m, n, c))
        tmp = np.clip(tmp, 0, fwc)
        tmp = tmp + np.random.normal(loc=0, scale=sigma_read, size=(m, n, c))
        tmp = np.round(tmp * gain * max_val / fwc)
        tmp = np.clip(tmp, min_val, max_val)
        img_out = img_out + tmp

    img_out = img_out / N
    return img_out


class GainCalculator:
    def __init__(self):
        # Define the (x, y) pairs
        self.data = np.array([
            [3.25, 30], [6.50, 15], [9.75, 7.5], [13, 4.5], [20, 3.2], [26, 2.8],
            [36, 2.4], [45, 2.2], [54, 1.8], [67, 1.5], [80, 1.3], [90, 1.1],
            [110, 1.05], [130, 0.90], [145, 0.65], [155, 0.56], [160, 0.51]
        ])
        x = self.data[:, 0]
        y = self.data[:, 1]

        # Fit an RBF interpolator
        self.rbf = Rbf(x, y, function='multiquadric')

    def get_gain(self, avg_PPP, N=1):
        # Evaluate the polynomial at avg_PPP
        return self.rbf(avg_PPP) * N


@torch.no_grad()
def torch_forward_model(avg_PPP, photon_flux, QE, theta_dark, sigma_read, N, Nbits, fwc, normalize = True):
    min_val = 0
    max_val = 2 ** Nbits - 1
    gain_func = GainCalculator()
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


def normalize(data, max_value=255.):
    r"""Normalizes a unit8 image to a float32 image in the range [0, 1]
	Args:
		data: a unint8 numpy array to normalize from [0, 255] to [0, 1]
	"""
    return np.float32(data / max_value)

if __name__ == '__main__':
    def random_crop(img, crop_size):
        h, w = img.shape[:2]
        if h < crop_size or w < crop_size:
            img = crop_arr(img.transpose([2,0,1]), max(crop_size, h), max(crop_size, w)).transpose([1,2,0])
            h, w = img.shape[:2]
        top = np.random.randint(0, h - crop_size+1)
        left = np.random.randint(0, w - crop_size+1)
        return img[top:top + crop_size, left:left + crop_size]
    
    x = torch.randn(size=(2, 500, 3)).numpy()
    print(random_crop(x, 256).shape)