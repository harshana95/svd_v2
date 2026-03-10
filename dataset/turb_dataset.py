import os
from PIL import Image
import numpy as np
from torch.utils import data as data

class TurbDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.lq_key = 'turb'
        self.gt_key = 'gt'
        self.dirs = []
        self.data_dir = opt.data_dir
        self.cache_dir = opt.cache_dir
        self.frames = opt.frames
        self.use_shuffle = opt.use_shuffle
        self.names = []
        os.makedirs(self.cache_dir, exist_ok=True)

        for name in sorted(os.listdir(self.data_dir)): 
            self.names.append(name)
            folder = os.path.join(self.data_dir, name)
            if not os.path.isdir(folder):
                continue
            self.dirs.append(folder)
            
        L = len(self.dirs)
        self.dirs = self.dirs[int(opt.split[0]*L):int(opt.split[1]*L)]
        print(f"Initialized dataset with {len(self.dirs)}")

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        folder = self.dirs[idx]
        np_path = os.path.join(self.cache_dir, self.names[idx]+'.npz')
        try:
            if os.path.exists(np_path):
                data = np.load(np_path)
                idx = np.random.randint(0, (data['turb_imgs'].shape[0]//3)-self.frames) if self.use_shuffle else 0
                return {
                    self.gt_key: data['gt'],
                    self.lq_key: data['turb_imgs'][idx*3:(idx+self.frames)*3],
                    'name': self.names[idx]
                }
        except:
            print(f"Failed to read {np_path}")
            
        gt_path = os.path.join(folder, 'gt.jpg')
        turb_path = os.path.join(folder, 'turb')
        gt_img = np.array(Image.open(gt_path).convert('RGB'), dtype=np.float32).transpose([2,0,1])/127.5 - 1
        
        turb_imgs = []
        for file in sorted(sorted(os.listdir(turb_path))):
            file_path = os.path.join(turb_path, file)
            turb_img = np.array(Image.open(file_path).convert('RGB'), dtype=np.float32).transpose([2,0,1])/127.5 - 1
            turb_imgs.append(turb_img)
        turb_imgs = np.concatenate(turb_imgs, axis=0)
        
        np.savez_compressed(np_path, gt=gt_img, turb_imgs=turb_imgs)
        # data = np.load(np_path)
        # return {
        #         self.gt_key: data['gt'],
        #         self.lq_key: data['turb_imgs'],
        #         'name': self.names[idx]
        #     }
        idx = np.random.randint(0, (turb_imgs.shape[0]//3)-self.frames) if self.use_shuffle else 0
        return {
            self.gt_key: gt_img,
            self.lq_key: turb_imgs[idx*3:(idx+self.frames)*3],
            'name': self.names[idx]
        }