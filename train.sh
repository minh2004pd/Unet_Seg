#!/bin/bash
set -a && source .env 2>/dev/null; set +a

DATA_ROOT="${DATA_ROOT:-/mnt/apple/k66/minhdd/data/brats2021}"
SPLIT_FILE="${SPLIT_FILE:-$DATA_ROOT/preprocessed_split_old_val251.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_unet}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
BASE_CH="${BASE_CH:-64}"
SAVE_EVERY="${SAVE_EVERY:-5}"
RESUME="${RESUME:-}"

mkdir -p "$OUTPUT_DIR"

echo "Data:        $DATA_ROOT"
echo "Split:       $SPLIT_FILE"
echo "Output:      $OUTPUT_DIR"
echo "Epochs:      $EPOCHS  BS=$BATCH_SIZE  LR=$LR  base_ch=$BASE_CH"
echo "Save every:  $SAVE_EVERY epochs  (+ last + best)"
echo "Grad ckpt:   ON"

EXTRA=(--gradient_checkpointing --save_every "$SAVE_EVERY")
[ -n "$RESUME" ] && EXTRA+=(--resume "$RESUME")

python train.py \
    --data_root   "$DATA_ROOT" \
    --split_file  "$SPLIT_FILE" \
    --output_dir  "$OUTPUT_DIR" \
    --epochs      "$EPOCHS" \
    --batch_size  "$BATCH_SIZE" \
    --lr          "$LR" \
    --base_ch     "$BASE_CH" \
    --device      cuda \
    "${EXTRA[@]}" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training complete. Checkpoint at $OUTPUT_DIR/"
