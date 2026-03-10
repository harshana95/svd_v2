import torch
import numpy as np
import cv2
import einops
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np
import torch
from torchvision import transforms 
from datasets import load_dataset
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from dataset.hf_dataset import HuggingFaceDataset
from psf.motionblur.kernels import KernelDataset
from utils import get_dataset_util
from utils.dataset_utils import DictWrapper

class SyntheticMotionBlurDataset(HuggingFaceDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.device = 'cuda'
        # setup motionblur kernel generator
        self.max_frames = 100
        self.skip_frames = getattr(opt, "skip_frames", 1)
        self.global_blur = opt.global_blur
        self.return_meta = getattr(opt, "return_meta", False)
        self.kernel_size = getattr(opt, "kernel_size", 64)  # downsampled kernel size for meta vector (meta_dim = meta_size**2)
        self.meta_key = getattr(opt, "meta_key", "meta")
        self.meta_include_kernel = getattr(opt, "meta_include_kernel", False)

        self.kernel_generator = KernelDataset(self.kernel_size, 32)
        self.is_train = opt.is_train
        self.kernel_pool = []

        if opt.foreground_dataset is not None:
            self.fg_dataset = load_dataset(opt.foreground_dataset.name, split=opt.foreground_dataset.split, trust_remote_code=opt.get('trust_remote_code', None))
            if opt.foreground_dataset.split_range is not None:
                self.fg_dataset = self.fg_dataset.select(range(int(opt.foreground_dataset.split_range[0]/100*len(self.fg_dataset)), int(opt.foreground_dataset.split_range[1]/100*len(self.fg_dataset))))
            self.fg_dataset = self.setup_dataset(self.fg_dataset, opt.foreground_dataset)
        else:
            self.fg_dataset = None
            # setup SAM
            model_id = "facebook/sam-vit-base"
            self.sam = SamModel.from_pretrained(model_id).to(self.device)
            self.processor = SamProcessor.from_pretrained(model_id)
            self.sam.eval()

    def get_fg_from_sam(self, foreground):
        # 1. Generate Mask using SAM
        # Prompting with a center point [W/2, H/2]
        w, h = foreground.size
        input_points = [[[np.random.randint(0, w), np.random.randint(0, h)]]] 
        with torch.no_grad():
            inputs = self.processor(foreground, input_points=input_points, return_tensors="pt").to(self.device)
            outputs = self.sam(**inputs)
        
        # Post-process to get binary mask
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )
        # Take the first mask (SAM usually returns 3 options, we pick the most confident)
        mask = masks[0][0][0].numpy().astype(np.uint8) 
        return mask 

    def __getitem__(self, idx):
        b_idx = int(idx)
        background_item = self.dataset[b_idx]
        background = background_item[self.gt_key]
        h, w = background.shape[-2:]
        
        if self.fg_dataset is None:
            f_idx = np.random.randint(0, len(self.dataset)) if self.is_train else idx%len(self.dataset)
            foreground_item = self.dataset[f_idx]
            foreground = foreground_item[self.gt_key]
        else:
            f_idx = np.random.randint(0, len(self.fg_dataset)) if self.is_train else idx%len(self.fg_dataset)
            foreground_item = self.fg_dataset[f_idx]
            foreground = foreground_item[self.gt_key][:3]
            mask = foreground_item[self.gt_key][3].numpy().astype(np.uint8) 
                
        # assert background.max() < 2.0 and background.min() > -0.2, f"Make sure the images are 0-1 normalized {background.min()} {background.max()}"

        background = einops.rearrange(background, 'c h w -> h w c').clip(0, 1.0)*255
        foreground = einops.rearrange(foreground, 'c h w -> h w c').clip(0, 1.0)*255
        bg_arr = np.array(background.numpy().astype(np.uint8))
        fg_arr = np.array(foreground.numpy().astype(np.uint8))

        if self.is_train: # randomly roll the image in h,w axes
            roll = int(h*0.1)
            fg_arr = np.roll(fg_arr, np.random.randint(-roll, roll), axis=0)
            fg_arr = np.roll(fg_arr, np.random.randint(-roll, roll), axis=1)

        if self.fg_dataset is None:
            mask = self.get_fg_from_sam(Image.fromarray(fg_arr))
        
        fg_only = cv2.bitwise_and(fg_arr, fg_arr, mask=mask)
        
        # 2. Apply Synthetic Motion Blur              
        
        # generate Blur kernels
        if self.is_train:
            k_bg, fv_bg, M_bg = self.kernel_generator.sample()
            k_fg, fv_fg, M_fg = self.kernel_generator.sample()
        else:
            while len(self.kernel_pool) <= f_idx:
                k_bg, fv_bg, M_bg = self.kernel_generator.sample()
                k_fg, fv_fg, M_fg = self.kernel_generator.sample()
                self.kernel_pool.append((k_bg, fv_bg, M_bg, k_fg, fv_fg, M_fg))
            print(f"Kernel pool length: {len(self.kernel_pool)} Using index: {f_idx}")
            k_bg, fv_bg, M_bg, k_fg, fv_fg, M_fg = self.kernel_pool[f_idx]
        
        # blurred_bg = cv2.filter2D(bg_arr, -1, k_bg)
        
        # Blur foreground object
        blurred_fg = cv2.filter2D(fg_only, -1, k_fg)
        
        # Blur mask to soften edges
        blurred_mask = cv2.filter2D(mask.astype(float), -1, k_fg)[:, :, None]
        mask = mask.astype(float)[:, :, None]
        
        # Alpha Composite
        composite_blur = (blurred_fg * blurred_mask + bg_arr * (1 - blurred_mask)).astype(np.float32)

        # add global blur
        if self.global_blur:
            composite_blur = cv2.filter2D(composite_blur, -1, k_bg)/127.5-1
        else:
            composite_blur = composite_blur/127.5-1

        # calculate gt
        composite_gt = (fg_arr * mask + bg_arr * (1 - mask)).astype(np.float32)/127.5-1
        
        data_dict = {
            self.lq_key: composite_blur.transpose([2,0,1]),
            self.gt_key: composite_gt.transpose([2,0,1]),
            'fg': fg_arr.transpose([2,0,1])/127.5-1,
            'bg': bg_arr.transpose([2,0,1])/127.5-1,
            'mask': mask.transpose([2,0,1]),
        }

        # Motion blur metadata for Blur_model: downsampled global (and optionally fg) kernel(s) as vector
        if self.return_meta:
            if self.meta_include_kernel:
                data_dict[self.meta_key+"_kernel_fg"] = torch.from_numpy(k_fg).to(torch.float32)
                if self.global_blur:
                    data_dict[self.meta_key+"_kernel_bg"] = torch.from_numpy(k_bg).to(torch.float32)
                
            # calculate flow 
            # pad to max_frames (2,n) (2,max_frames)
            fv_bg = np.pad(fv_bg, [(0,0),(0,self.max_frames-fv_bg.shape[1])])
            fv_fg = np.pad(fv_fg, [(0,0),(0,self.max_frames-fv_fg.shape[1])])
            
            M_bg = np.concat([M_bg, [np.eye(3)]*(self.max_frames-len(M_bg))], axis=0)
            M_fg = np.concat([M_fg, [np.eye(3)]*(self.max_frames-len(M_fg))], axis=0)
            
            # Calculate warped masks using the M_fg transformation matrices
            # mask: (h, w, 1) float, values 0 or 1
            flow = []
            mask_np = mask.squeeze(-1).astype(np.float32)  # (h, w)

            for i, M in enumerate(M_fg):
                if i % self.skip_frames == 0:
                    fv_bg_i = fv_bg[:, i, None, None]
                    fv_fg_i = fv_fg[:, i, None, None]
                else:
                    # _M = _M @ M
                    fv_bg_i += fv_bg[:, i, None, None]
                    fv_fg_i += fv_fg[:, i, None, None]
                    
                if i % self.skip_frames == self.skip_frames - 1:
                    # warping should be in the negative direction of the flow
                    M = np.linalg.inv(M)
                    warped = cv2.warpPerspective(mask_np, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    flow.append(warped[None]*fv_fg_i + (fv_bg_i if self.global_blur else 0))
            flow = np.stack(flow, axis=0)  # (num_frames, 2, h, w)
            data_dict[self.meta_key] = einops.rearrange(flow, 'f n h w -> (f n) h w')
        
        return data_dict
