#!/bin/bash
# Simple script to train GAT model on LCoT2Tree graphs

# Default configuration
DATA_PATH="${1:-deepseek/final.json}"
OUTPUT_DIR="${2:-outputs/models}"
EPOCHS="${3:-100}"
BATCH_SIZE="${4:-32}"
HIDDEN_DIM="${5:-64}"
LEARNING_RATE="${6:-0.001}"
NUM_RUNS="${7:-5}"
SEED="${8:-42}"

echo "Training GAT Model"
echo "=================="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Hidden dim: $HIDDEN_DIM"
echo "Learning rate: $LEARNING_RATE"
echo "Number of runs: $NUM_RUNS"
echo "Seed: $SEED"
echo "=================="
echo ""

poetry run python scripts/02_train_model.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --hidden_dim "$HIDDEN_DIM" \
    --lr "$LEARNING_RATE" \
    --num_runs "$NUM_RUNS" \
    --seed "$SEED" \
    --selected_features 0 2 5 7 8 \
    --save_model \
    --early_stopping \
    --patience 15 \
    --print_every 10

echo ""
echo "Training complete! Check $OUTPUT_DIR for results."

