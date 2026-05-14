#!/bin/bash
set -a && source .env 2>/dev/null; set +a

DATA_ROOT="${DATA_ROOT:-/mnt/apple/k66/minhdd/data/brats2021}"
SPLIT_FILE="${SPLIT_FILE:-$DATA_ROOT/preprocessed_split_old_val251.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./anomaly_results_unet}"
SPLIT="${SPLIT:-test}"
CKPT_DIR="${CKPT_DIR:-./output_unet}"
CHECKPOINT="${CHECKPOINT:-$(ls -t "$CKPT_DIR"/checkpoint_best.pth 2>/dev/null | head -1)}"
BASE_CH="${BASE_CH:-64}"
THRESHOLD="${THRESHOLD:-0.5}"
THRESHOLD_STEPS="${THRESHOLD_STEPS:-200}"
SAVE_PNG="${SAVE_PNG:-0}"

if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: no checkpoint found. Set CHECKPOINT= or train first."
    exit 1
fi

echo "Checkpoint: $CHECKPOINT"
echo "Split:      $SPLIT"
echo "Output:     $OUTPUT_DIR"

EXTRA=()
[ "$SAVE_PNG" -eq 1 ] && EXTRA+=(--save_png)

python infer_anomaly.py \
    --checkpoint      "$CHECKPOINT" \
    --data_root       "$DATA_ROOT" \
    --split_file      "$SPLIT_FILE" \
    --output_dir      "$OUTPUT_DIR" \
    --split           "$SPLIT" \
    --base_ch         "$BASE_CH" \
    --threshold       "$THRESHOLD" \
    --threshold_steps "$THRESHOLD_STEPS" \
    --device          cuda \
    "${EXTRA[@]}" \
    2>&1 | tee "$OUTPUT_DIR/infer.log"

echo "Inference complete. Results at $OUTPUT_DIR/"
