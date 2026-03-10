import einops
import torch

from models.DeblurDeform.DeblurDeform_model import DeblurDeform_model
from utils.dataset_utils import crop_arr

class MotionDeform_model(DeblurDeform_model):
    def __init__(self, opt, logger):
        super(MotionDeform_model, self).__init__(opt, logger)

    def feed_data(self, data, is_train=True):
        self.sample = data

        if self.opt.train.patched if is_train else self.opt.val.patched:
            gt_key = self.dataloader.dataset.gt_key if is_train else self.test_dataloader.dataset.gt_key
            lq_key = self.dataloader.dataset.lq_key if is_train else self.test_dataloader.dataset.lq_key
            meta_key1 = self.opt.meta_key1
            self.grids(keys=[lq_key, gt_key, meta_key1], opt=self.opt.train if is_train else self.opt.val)

    def get_meta_data(self):
        meta = self.sample[self.opt.meta_key1]
        return meta
    
    
        