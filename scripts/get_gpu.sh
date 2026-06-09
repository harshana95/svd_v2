#!/usr/bin/env bash
set -euo pipefail

# Helper for interactive GPU allocations on Gilbreth.
# Defaults match the user's standard A100 request pattern.
#
# Examples:
#   ./scripts/get_gpu.sh
#   ./scripts/get_gpu.sh a100-80gb 4-0:00:00 128G
#   ./scripts/get_gpu.sh a30 4:00:00 128G standby

PARTITION="${1:-a100-80gb}"
TIME_LIMIT="${2:-4-0:00:00}"
MEMORY="${3:-128G}"
QOS="${4:-}"

CMD=(
  sinteractive
  -A stanchan
  -N1
  -n32
  --gpus-per-node=1
  --mem="${MEMORY}"
  -p "${PARTITION}"
  -t "${TIME_LIMIT}"
)

if [[ -n "${QOS}" ]]; then
  CMD=(sinteractive -A stanchan --qos="${QOS}" -N1 -n32 --gpus-per-node=1 --mem="${MEMORY}" -p "${PARTITION}" -t "${TIME_LIMIT}")
fi

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
