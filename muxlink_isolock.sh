#!/bin/bash

# === Configuration ===
CIRCUIT="c499"
KEY_SIZE=32
HOP=3
THRESHOLD=0.01

TRAIN_LOG="Log_train_${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO.txt"
PRED_LOG_POS="Log_pos_predict_${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO.txt"
PRED_LOG_NEG="Log_neg_predict_${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO.txt"

cd ./MuxLink

echo "[MuxLink] Starting pipeline for ${CIRCUIT} with key size ${KEY_SIZE}"

# === Step 1: Train MuxLink ===
echo "[Step 1] Training MuxLink..."
python Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO" \
    --train-name links_train.txt \
    --test-name links_test.txt \
    --testneg-name link_test_n.txt \
    --hop "$HOP" \
    --save-model > "$TRAIN_LOG"

# === Step 2: Get the Predictions ===
echo "[Step 2] Getting positive predictions..."
python Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO" \
    --train-name links_train.txt \
    --test-name links_test.txt \
    --hop "$HOP" \
    --only-predict > "$PRED_LOG_POS"

echo "[Step 2] Getting negative predictions..."
python Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO" \
    --train-name links_train.txt \
    --test-name link_test_n.txt \
    --hop "$HOP" \
    --only-predict > "$PRED_LOG_NEG"

# === Step 3: Parse Predictions ===
echo "[Step 3] Parsing predictions with threshold=$THRESHOLD..."
perl break_Iso.pl "${CIRCUIT}_K${KEY_SIZE}_H${HOP}_ISO" "$THRESHOLD" "$HOP"

echo "[Done] Pipeline completed for ${CIRCUIT} with key size ${KEY_SIZE}"