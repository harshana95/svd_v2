#!/bin/bash
# Per-node launcher for multi-node accelerate training on Polaris (PBS).
#
# Invoked once per NODE by `mpiexec -n <NNODES> --ppn 1 ...` from
# scripts/alcf/sjob_ctx_alignA_2node_debug (or any other multi-node sjob that
# sources the same convention). `accelerate launch` itself spawns the
# NRANKS_PER_NODE local GPU processes on whichever node runs it, so this
# script must run exactly once per node, not once per GPU.
#
# Expects these exported from the parent job script:
#   REPO, PROJECT_ENV, PBS_NODEFILE, NNODES, NTOTRANKS, NRANKS_PER_NODE,
#   MASTER_ADDR, MASTER_PORT, OPT

set -euo pipefail

REPO="${REPO:-/lus/eagle/projects/lighthouse-purdue/harshana/svd_v2}"
cd "${REPO}"

source "${REPO}/scripts/alcf/load_conda.sh"

# The real project env (installed by scripts/alcf/debug_deps_install.sh from
# requirements_no_flashattn.txt -- accelerate, diffusers, transformers,
# comet-ml, xformers, etc., and the datasets==3.3.0 this repo's custom
# HuggingFaceDataset loading scripts require) lives here, not in a
# .venv/alcf -- that path was never created.
conda activate "${PROJECT_ENV:-/eagle/lighthouse-purdue/harshana/my_env}"
echo "python=$(command -v python)  CONDA_PREFIX=${CONDA_PREFIX:-unset}"

# With `--ppn 1`, Polaris' Cray-PALS mpiexec gives one rank per node, in
# PBS_NODEFILE order, so PMI_RANK is this node's 0-based index into the job.
# Fall back to locating our own hostname in PBS_NODEFILE if PMI_RANK is ever
# unset (e.g. a non-PALS mpiexec build).
if [[ -n "${PMI_RANK:-}" ]]; then
  NODE_RANK="${PMI_RANK}"
else
  short_host="$(hostname -s)"
  NODE_RANK="$(grep -n -m1 "^${short_host}\b" "${PBS_NODEFILE}" | cut -d: -f1)"
  NODE_RANK=$(( NODE_RANK - 1 ))
fi

echo "[node ${NODE_RANK}] host=$(hostname)  nnodes=${NNODES}  ntotranks=${NTOTRANKS}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv

accelerate launch \
  --multi_gpu \
  --num_machines "${NNODES}" \
  --num_processes "${NTOTRANKS}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  trainer.py -opt "${OPT}"
