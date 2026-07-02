#!/usr/bin/env bash
# End-to-end V-JEPA analysis on 2 Deblur_MIOR GT/blur pairs.
#
# Usage:
#   bash playground/vjepa_pipeline.sh
#   PAIRS=("04641" "06390") bash playground/vjepa_pipeline.sh
#   VJEPA_HUB_CHECKPOINT=/path/to/vjepa.pt bash playground/vjepa_pipeline.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/wweligam/home_env
fi

# ── Config (override via env) ────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-/depot/chan129/data/Deblur_MIOR/val}"
OUT_DIR="${OUT_DIR:-playground/vjepa_pipeline_out}"
HUB_ENTRY="${HUB_ENTRY:-vjepa2_1_vit_large_384}"
HUB_CHECKPOINT="${VJEPA_HUB_CHECKPOINT:-}"

if [[ -n "${PAIRS:-}" ]]; then
  # shellcheck disable=SC2206
  PAIRS=($PAIRS)
else
  PAIRS=("05714" "05805")
fi

echo "=== V-JEPA Deblur_MIOR pipeline ==="
echo "DATA_ROOT:  $DATA_ROOT"
echo "OUT_DIR:    $OUT_DIR"
echo "PAIRS:      ${PAIRS[*]}"
echo "HUB_ENTRY:  $HUB_ENTRY"
echo

# ── Stage 1: staging dirs (symlink exactly the chosen pairs) ─────────────────
echo "[1/5] Staging ${#PAIRS[@]} GT/blur pairs..."
rm -rf "$OUT_DIR/staging"
mkdir -p "$OUT_DIR/staging/gt" "$OUT_DIR/staging/blur"

for stem in "${PAIRS[@]}"; do
  gt="$DATA_ROOT/gt/${stem}.png"
  blur="$DATA_ROOT/blur/${stem}.png"
  if [[ ! -f "$gt" ]]; then
    echo "ERROR: GT image not found: $gt" >&2
    exit 1
  fi
  if [[ ! -f "$blur" ]]; then
    echo "ERROR: blur image not found: $blur" >&2
    exit 1
  fi
  ln -sf "$gt" "$OUT_DIR/staging/gt/${stem}.png"
  ln -sf "$blur" "$OUT_DIR/staging/blur/${stem}.png"
done

# ── Stage 2: SVD analysis ───────────────────────────────────────────────────
echo "[2/5] Running SVD analysis (analyze_jepa_svd.py)..."
python playground/analyze_jepa_svd.py \
  --gt-dir "$OUT_DIR/staging/gt" \
  --blur-dir "$OUT_DIR/staging/blur" \
  --n-images "${#PAIRS[@]}" \
  --encoders vjepa \
  --out-dir "$OUT_DIR/svd"

# ── Stage 3: per-image token extraction ─────────────────────────────────────
echo "[3/5] Extracting V-JEPA tokens (check_jepa_output.py)..."
PRETRAINED_ARGS=()
if [[ -z "$HUB_CHECKPOINT" ]]; then
  PRETRAINED_ARGS=(--pretrained)
fi

for stem in "${PAIRS[@]}"; do
  for kind in gt blur; do
    save_dir="$OUT_DIR/tokens/${stem}/${kind}"
    mkdir -p "$save_dir"
    checkpoint_args=()
    if [[ -n "$HUB_CHECKPOINT" ]]; then
      checkpoint_args=(--hub-checkpoint "$HUB_CHECKPOINT")
    fi
    python playground/check_jepa_output.py \
      --image "$DATA_ROOT/${kind}/${stem}.png" \
      --backend hub \
      --hub-entry "$HUB_ENTRY" \
      "${PRETRAINED_ARGS[@]}" \
      "${checkpoint_args[@]}" \
      --save-dir "$save_dir"
  done
done

# ── Stage 4: GT vs blur token comparison ──────────────────────────────────────
echo "[4/5] Comparing GT vs blur tokens (compare_jepa_features.py)..."
for stem in "${PAIRS[@]}"; do
  echo "--- Pair: $stem ---"
  python playground/compare_jepa_features.py \
    --gt-dir "$OUT_DIR/tokens/${stem}/gt" \
    --blur-dir "$OUT_DIR/tokens/${stem}/blur"
done

# ── Stage 5: resolution check on first GT ───────────────────────────────────
echo "[5/5] Resolution check on GT ${PAIRS[0]} (check_jepa_resolutions.py)..."
python playground/check_jepa_resolutions.py \
  --image "$DATA_ROOT/gt/${PAIRS[0]}.png" \
  --out-dir "$OUT_DIR/resolution"

echo
echo "=== Pipeline complete ==="
echo "Outputs:"
echo "  SVD report:     $OUT_DIR/svd/svd_analysis.png"
echo "  Token dumps:    $OUT_DIR/tokens/<stem>/{gt,blur}/"
echo "  Resolution:     $OUT_DIR/resolution/"
