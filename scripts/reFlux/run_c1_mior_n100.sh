#!/bin/bash
# C1 (blur invariance at dataset scale) -- MIOR arm at n=100.
# The MIOR arm of Table 1 was n=11 and labelled "directional"; val is now 100 paired
# images. The analysis scripts read image directories directly rather than through the
# HuggingFace loader, so they see the expanded split immediately -- the 1 TB stale cache
# that still blocks training does not affect this.
set -e
cd /home/wweligam/svd_v2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/wweligam/home_env
export TORCH_HOME=/depot/chan129/users/harshana/weights/torch_hub
python playground/analyze_jepa_svd.py \
  --gt-dir /depot/chan129/data/Deblur_MIOR/val/gt \
  --blur-dir /depot/chan129/data/Deblur_MIOR/val/blur \
  --n-images 100 --encoders vjepa dino clip --save-vk \
  --out-dir playground/jepa_svd_mior_n100
