#!/bin/bash
#PBS -N depsinstall
#PBS -A lighthouse-purdue
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:59:00
#PBS -l filesystems=home:eagle
#PBS -j oe
#PBS -o /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2/scripts/alcf/depsinstall.log

set -x
cd /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2
source scripts/alcf/load_conda.sh
conda activate /eagle/lighthouse-purdue/harshana/my_env
echo "python=$(command -v python) CONDA_PREFIX=${CONDA_PREFIX}"

PIP=/eagle/lighthouse-purdue/harshana/my_env/bin/pip

echo "=== installing remaining environment.yml pip deps (flash-attn excluded) ==="
"$PIP" install -r scripts/alcf/requirements_no_flashattn.txt
rc=$?

echo "=== EXIT_CODE=${rc} ==="
echo "=== DONE marker ==="
