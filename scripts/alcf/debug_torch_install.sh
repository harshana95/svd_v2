#!/bin/bash
#PBS -N torchdbg
#PBS -A lighthouse-purdue
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2/scripts/alcf/torchdbg.log

set -x
cd /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2
source scripts/alcf/load_conda.sh
conda activate /eagle/lighthouse-purdue/harshana/my_env
echo "python=$(command -v python) CONDA_PREFIX=${CONDA_PREFIX}"
nvidia-smi

PIP=/eagle/lighthouse-purdue/harshana/my_env/bin/pip
PY=/eagle/lighthouse-purdue/harshana/my_env/bin/python

echo "=== installing torch/torchvision ==="
"$PIP" install torch==2.8.0 torchvision==0.23.0

echo "=== torch import / CUDA check ==="
"$PY" -c "
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
print('cuda version', torch.version.cuda)
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    print('matmul ok, sum=', y.sum().item())
"

echo "=== DONE marker ==="
