#!/bin/bash

# === Configuration ===
CIRCUIT="c499"
KEY_SIZE=32
HOP=3
THRESHOLD=0.01

LOCK_SCRIPT="./MuxLink/DMUX_Locking/convert_DMUX.py"
LOCKED_DIR="./data/${CIRCUIT}_K${KEY_SIZE}_DMUX"
BENCH_LOCKED="${LOCKED_DIR}/${CIRCUIT}_K${KEY_SIZE}.bench"

TRAIN_LOG="Log_train_${CIRCUIT}_K${KEY_SIZE}_DMUX.txt"
PRED_LOG_POS="Log_pos_predict_${CIRCUIT}_K${KEY_SIZE}_DMUX.txt"
PRED_LOG_NEG="Log_neg_predict_${CIRCUIT}_K${KEY_SIZE}_DMUX.txt"

# === Step 1: Lock the Design ===
echo "[Step 1] Locking the design..."
cd ./MuxLink/DMUX_Locking || exit 1
python3 convert_DMUX.py "$CIRCUIT" "$KEY_SIZE" "../data/${CIRCUIT}_K${KEY_SIZE}_DMUX"
cd ../

# === Step 2: Train MuxLink ===
echo "[Step 2] Training MuxLink..."
python3 Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_DMUX" \
    --train-name links_train.txt \
    --test-name links_test.txt \
    --testneg-name link_test_n.txt \
    --hop "$HOP" \
    --save-model > "$TRAIN_LOG"

# === Step 3: Get the Predictions ===
echo "[Step 3] Getting positive predictions..."
python3 Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_DMUX" \
    --train-name links_train.txt \
    --test-name links_test.txt \
    --hop "$HOP" \
    --only-predict > "$PRED_LOG_POS"

echo "[Step 3] Getting negative predictions..."
python3 Main.py \
    --file-name "${CIRCUIT}_K${KEY_SIZE}_DMUX" \
    --train-name links_train.txt \
    --test-name link_test_n.txt \
    --hop "$HOP" \
    --only-predict > "$PRED_LOG_NEG"

# === Step 4: Parse Predictions ===
echo "[Step 4] Parsing predictions with threshold=$THRESHOLD..."
perl break_DMUX.pl "${CIRCUIT}_K${KEY_SIZE}_DMUX" "$THRESHOLD" "$HOP"

echo "[Done] Pipeline completed for ${CIRCUIT} with key size ${KEY_SIZE}"