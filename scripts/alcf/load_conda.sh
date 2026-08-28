#!/bin/bash
# Load ALCF Polaris conda (source this file; do not execute).
#
#   source /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2/scripts/alcf/load_conda.sh
#
# conda/2025-09-25 pins gcc-native/14.2 and cray-hdf5-parallel/1.14.3.5,
# which this CPE no longer ships (only gcc-native/14 and hdf5 1.14.3.9).
# scripts/alcf/modulefiles provides version shims so the module can load.

_alcf_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module use "${_alcf_dir}/modulefiles"
module use /soft/modulefiles

# Default login env is PrgEnv-nvidia; conda requires GNU.
if module is-loaded PrgEnv-nvidia 2>/dev/null; then
  module swap PrgEnv-nvidia PrgEnv-gnu
else
  module load PrgEnv-gnu
fi

module load conda
conda activate base

unset _alcf_dir
echo "python=$(command -v python)  CONDA_PREFIX=${CONDA_PREFIX:-unset}"
