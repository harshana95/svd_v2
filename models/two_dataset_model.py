
from models.base_model import BaseModel

from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from dataset import create_dataset
from dataset.ZipDatasetWrapper import ZipDatasetWrapper

class TwoDatasetBasemodel(BaseModel):

    def __init__(self, opt, logger):
        super(TwoDatasetBasemodel, self).__init__(opt, logger)

    def setup_dataloaders(self):
        # create train and validation dataloaders
        
        train_set1 = create_dataset(self.opt.datasets.train1)
        train_set2 = create_dataset(self.opt.datasets.train2)
        self.dataloader = DataLoader(
            ZipDatasetWrapper({'1': train_set1, '2': train_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.train1.use_shuffle,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.datasets.train1.get('num_worker_per_gpu', 1),
        )
        self.dataloader.dataset.gt_key = train_set1.gt_key
        self.dataloader.dataset.lq_key = train_set1.lq_key
        self.dataloader.dataset.rf_key = train_set1.opt.get('rf_key', None)
                
        val_set1 = create_dataset(self.opt.datasets.val1)
        val_set2 = create_dataset(self.opt.datasets.val2)
        self.test_dataloader = DataLoader(
            ZipDatasetWrapper({'1': val_set1, '2': val_set2}, transforms=transforms.Compose([]), random=False),
            shuffle=self.opt.datasets.val1.use_shuffle,
            batch_size=self.opt.val.batch_size,
            num_workers=self.opt.datasets.val1.get('num_worker_per_gpu', 1),
        )
        self.test_dataloader.dataset.gt_key = val_set1.gt_key
        self.test_dataloader.dataset.lq_key = val_set1.lq_key
        self.test_dataloader.dataset.rf_key = val_set1.opt.get('rf_key', None)

