#!/bin/bash
#SBATCH -A stanchan
#SBATCH --qos=standby
#SBATCH -p a30
#SBATCH --gpus-per-node=1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o /scratch/gilbreth/wweligam/vjepa_blur_layers/logs/%x_%j.out
PY=/home/wweligam/home_env/bin/python
cd /home/wweligam/svd_v2/playground/vjepa_blur_layers
$PY extract.py --split "$1" --n "$2" --out "$3"
