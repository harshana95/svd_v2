#!/bin/bash
#SBATCH -A stanchan
#SBATCH -p a100-80gb
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J lumaflux
#SBATCH -o logs/lumaflux_%j.out
#SBATCH -e logs/lumaflux_%j.err
#
# Real LumaFlux training on one A100-80GB (Gilbreth).
# Submit:  sbatch scripts/train_a100.slurm [--key value ...]
# The paper uses an effective batch of 16; on a single GPU reach it with
# `--grad_accum 16` (per-step batch 1).  For multi-GPU use `accelerate launch`.

set -euo pipefail
cd /home/wweligam/lumaflux
mkdir -p logs

export HF_HOME=/scratch/gilbreth/wweligam/.cache/hub/
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/wweligam/lumaflux

PY=/home/wweligam/home_env/bin/python
echo "host=$(hostname)  job=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$PY train.py --config configs/lumaflux.yaml "$@"
