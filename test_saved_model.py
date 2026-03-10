from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch
import yaml
from dataset import create_dataset
from utils.misc import DictAsMember
from models.archs import find_network_class
from torch.utils.data import DataLoader


model_name = "DeMoE_arch" #"NAFNet_arch"
model_path = "/depot/chan129/users/harshana/svd_v2/weights/DeMoE/" #"/scratch/gilbreth/wweligam/experiments/SVD_Real/NAFNetRealColor 2025 09 11 23.28.33/checkpoint-6000/NAFNet_arch_1/"

with open("/depot/chan129/users/harshana/svd_v2/checkpoints/ECCV26/motionNAFNet.yml", mode='r') as f:
    data = OmegaConf.load(f)
    data = OmegaConf.to_yaml(data, resolve=True)
    data = yaml.safe_load(data)
    if isinstance(data, dict):
        opt = DictAsMember(**data)
        opt.pretty_print()
    else:
        opt = data
    
dataset = create_dataset(opt.datasets.val)

dataloader = DataLoader(
    dataset,
    shuffle=False,
    batch_size=1,
    num_workers=4,
)

print(f"Loading {model_name} model from {model_path}")
c, _ = find_network_class(model_name)
# m = c.from_config(model_path)
# sd = load_file(model_path+"diffusion_pytorch_model.safetensors")
print(c)
model = c.from_pretrained(model_path)

model.eval()
model = model.cuda()  # always run the model in GPU

with torch.no_grad():
    idx = 0
    for batch in dataloader:
        img = batch['blur'].cuda()*0.5+0.5
        # img = torch.flip(img, dims=[1])  # rgb to bgr
        out = model(img)
        print(img.min(), img.max())
        print(out.min(), out.max())

        plt.imsave(f"in_{idx}.png", np.clip(img[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        plt.imsave(f"out_{idx}.png", np.clip(out[0].cpu().numpy().transpose([1,2,0]), 0, 1))
        idx += 1

        if idx > 5:
            break