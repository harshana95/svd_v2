#!/bin/bash
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
for m in frozen random; do /home/wweligam/home_env/bin/python localize.py --root /scratch/gilbreth/wweligam/vjepa_blur_layers/full2 --model $m --n 60 --nreal 40 > /scratch/gilbreth/wweligam/vjepa_blur_layers/logs/k012_loc_$m.log 2>&1; done
echo DONE > /scratch/gilbreth/wweligam/vjepa_blur_layers/logs/k012_chain3.done
