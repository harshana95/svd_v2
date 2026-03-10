import einops
import torch

from models.DeblurDeform.DeblurDeform_model import DeblurDeform_model
from utils.dataset_utils import crop_arr

class AbrDeform_model(DeblurDeform_model):
    def __init__(self, opt, logger):
        super(AbrDeform_model, self).__init__(opt, logger)

    def feed_data(self, data, is_train=True):
        self.sample = data
        gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
        lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
        meta_key1 = self.opt.meta_key1
        meta_key2 = self.opt.meta_key2
        if self.opt.train.patched if is_train else self.opt.val.patched:
            self.grids(keys=[lq_key, gt_key, meta_key1], opt=self.opt.train if is_train else self.opt.val)

    def get_meta_data(self):
        basis_coef = self.sample[self.opt.meta_key1]
        basis_psfs = self.sample[self.opt.meta_key2]
        if self.opt.train.patched if is_train else self.opt.val.patched:
            basis_psfs = einops.repeat(basis_psfs, 'b q h w -> (b mb) q h w', mb=basis_coef.shape[0]//basis_psfs.shape[0])
            basis_psfs = crop_arr(basis_psfs, basis_coef.shape[-2], basis_coef.shape[-1])
        
        meta = torch.cat([basis_coef, basis_psfs], dim=1)
        return meta
    
    
        