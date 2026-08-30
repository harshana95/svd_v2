#!/bin/bash
# Load ALCF Polaris conda into the CURRENT shell.
#
#   source /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2/scripts/alcf/load_conda.sh
#
# Must be sourced. Running it with ./load_conda.sh starts a subshell, so
# `conda` disappears as soon as the script exits.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this file (do not ./ it):" >&2
  echo "  source ${0}" >&2
  exit 1
fi

_alcf_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_conda_root="/soft/applications/conda/2025-09-25/mconda3"
_cuda_home="/soft/compilers/cudatoolkit/cuda-13.0.3"

module use "${_alcf_dir}/modulefiles"
module use /soft/modulefiles

# Default env is PrgEnv-nvidia; GNU is what the conda stack was built against.
if module is-loaded PrgEnv-nvidia 2>/dev/null; then
  module swap PrgEnv-nvidia PrgEnv-gnu
else
  module load PrgEnv-gnu
fi

# Skip `module load conda` — that modulefile pins gcc-native/14.2 and
# cray-hdf5-parallel/1.14.3.5, which this CPE no longer provides.
module load cray-hdf5-parallel/1.14.3.9 2>/dev/null || true
module load cudnn/9.13.0 2>/dev/null || true

# Shell hook (defines the `conda` function) + put conda on PATH
# so the command works even if the hook is skipped.
source "${_conda_root}/etc/profile.d/conda.sh"
export PATH="${_conda_root}/bin:${PATH}"
conda activate base

# Driver on Polaris compute nodes reports CUDA 13.0; the conda modulefile
# still points at 12.9.1, which breaks `import torch` (libcudart.so.13).
export CUDA_HOME="${_cuda_home}"
export CUDA_PATH="${_cuda_home}"
export CUDA_TOOLKIT_BASE="${_cuda_home}"
export PATH="${_cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${_cuda_home}/lib64:${_cuda_home}/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"

export NCCL_ROOT="/soft/libraries/nccl/nccl_2.28.3-1+cuda12.9_x86_64"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH}"
export MPICH_GPU_SUPPORT_ENABLED=1
export TORCH_CUDA_ARCH_LIST=8.0
export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="${HTTP_PROXY}"
export https_proxy="${HTTPS_PROXY}"

unset _alcf_dir _conda_root _cuda_home
echo "python=$(command -v python)  conda=$(command -v conda)  CONDA_PREFIX=${CONDA_PREFIX:-unset}"
echo "CUDA_HOME=${CUDA_HOME}"
