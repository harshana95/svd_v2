#!/bin/bash
#SBATCH -A stanchan
#SBATCH -p a100-80gb
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH -t 1:00:00
#SBATCH -J smoke_blurmap
#SBATCH -o /home/wweligam/svd_v2/playground/smoke_context_flux_blurmap_%j.out
#SBATCH -e /home/wweligam/svd_v2/playground/smoke_context_flux_blurmap_%j.err

set -euo pipefail
cd /home/wweligam/svd_v2
source /home/wweligam/home_env/etc/profile.d/conda.sh 2>/dev/null || true
export PATH=/home/wweligam/home_env/bin:$PATH
export PYTHONPATH=$PWD
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

echo "host=$(hostname) cuda=$(nvidia-smi -L | head -1)"
/home/wweligam/home_env/bin/python playground/smoke_context_flux_blurmap.py --full --crop 256
