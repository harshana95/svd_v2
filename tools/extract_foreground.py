import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image
from torchvision.ops import box_convert
# -----------------------------
# GroundingDINO imports
# -----------------------------
from groundingdino.util.inference import load_model, load_image, predict, annotate

# -----------------------------
# SAM imports
# -----------------------------
from transformers import SamModel, SamProcessor

folder_path = "/depot/chan129/users/harshana/Datasets/Deconvolution/hybrid_Flickr2k_gt_v2/"
output_path = "/scratch/gilbreth/wweligam/dataset/foreground_images/"
os.makedirs(os.path.join(output_path, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_path, "foreground"), exist_ok=True)
os.makedirs(os.path.join(output_path, "boxes"), exist_ok=True)

# -----------------------------
# Load GroundingDINO
# -----------------------------
dino_model = load_model("/home/wweligam/home_env/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/depot/chan129/users/harshana/svd_v2/weights/groundingdino_swint_ogc.pth")
TEXT_PROMPT = "subject, foreground, person, dog, car"   # or "person", "dog", "car", etc.
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25

# -----------------------------
# Load SAM
# -----------------------------
sam_checkpoint = "facebook/sam-vit-base"
sam = SamModel.from_pretrained(sam_checkpoint)
sam_processor = SamProcessor.from_pretrained(sam_checkpoint)
sam.eval()


for image_name in tqdm(os.listdir(folder_path)):    
    image_path = os.path.join(folder_path, image_name)

    # -----------------------------
    # Load image
    # -----------------------------
    image_source, image = load_image(image_path)

    # -----------------------------
    # GroundingDINO detection
    # -----------------------------
    boxes, logits, phrases = predict(
        model=dino_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Draw the detected boxes on the source image and save/show the result
    image_with_boxes = annotate(image_source.copy(), boxes, logits, phrases)
    cv2.imwrite(os.path.join(output_path, "boxes", image_name), image_with_boxes)

    # Convert boxes to pixel coordinates
    H, W = image_source.shape[:2]
    boxes = boxes * torch.tensor([W, H, W, H])
    # Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
    if len(boxes) == 0:
        continue

    # -----------------------------
    # Run SAM on each detected box
    # -----------------------------
    image_source_pil = Image.fromarray((image_source).astype(np.uint8))
    masks_out = []
    for box in boxes:
        box = box.cpu().numpy().astype(float)
        x1, y1, x2, y2 = box

        inputs = sam_processor(image_source_pil,  input_boxes=[[[x1, y1, x2, y2]]], return_tensors="pt")
        outputs = sam(**inputs)
            
        # Post-process to get binary mask
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"].cpu(), 
            inputs["reshaped_input_sizes"].cpu()
        )

        # Take the first mask (SAM usually returns 3 options, we pick the most confident)
        mask = masks[0][0][0].numpy().astype(np.uint8) 

        masks_out.append(mask)
        break # no need of multiple masks

    # -----------------------------
    # Combine masks (if multiple)
    # -----------------------------
    final_mask = np.zeros_like(masks_out[0], dtype=np.uint8)
    for m in masks_out:
        final_mask = np.logical_or(final_mask, m)

    final_mask = final_mask.astype(np.uint8)

    # -----------------------------
    # Extract subject
    # -----------------------------
    image_bgr = cv2.imread(image_path)
    subject = image_bgr.copy()
    subject[final_mask == 0] = 0

    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = final_mask * 255
    cv2.imwrite(os.path.join(output_path, "foreground", image_name), rgba)
    # cv2.imwrite(os.path.join(output_path, "foreground", image_name), subject)
    cv2.imwrite(os.path.join(output_path, "masks", image_name), final_mask*255)


print("Done.")
