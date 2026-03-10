import numpy as np
import cv2
from transformers import SamModel, SamProcessor
import numpy as np
import torch
from psf.motionblur.kernels import KernelDataset

kernel_generator = KernelDataset(64, 32)

device = 'cuda'
model_id = "facebook/sam-vit-base"
sam = SamModel.from_pretrained(model_id).to(device)
processor = SamProcessor.from_pretrained(model_id)
sam.eval()

foreground = cv2.imread("000004.png", cv2.IMREAD_COLOR_RGB)
background = cv2.imread("941898.jpg", cv2.IMREAD_COLOR_RGB)

h, w = background.shape[-2:]
input_points = [[[w // 2, h // 2]]] 

inputs = processor(foreground, input_points=input_points, return_tensors="pt").to(device)
outputs = sam(**inputs)

# Post-process to get binary mask
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), 
    inputs["original_sizes"].cpu(), 
    inputs["reshaped_input_sizes"].cpu()
)
# Take the first mask (SAM usually returns 3 options, we pick the most confident)
mask = masks[0][0][0].numpy().astype(np.uint8) 
breakpoint()