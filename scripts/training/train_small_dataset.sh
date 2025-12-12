#!/bin/bash
# Script optimized for training on small datasets (< 50 examples)

DATA_PATH="${1:-deepseek/final.json}"
OUTPUT_DIR="${2:-outputs/models_small}"

echo "Training GAT Model on Small Dataset"
echo "===================================="
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Using small-dataset optimized settings:"
echo "  - Batch size: 4"
echo "  - Hidden dim: 32"
echo "  - Epochs: 50"
echo "  - Number of runs: 3"
echo "===================================="
echo ""

poetry run python scripts/02_train_model.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 50 \
    --batch_size 4 \
    --hidden_dim 32 \
    --lr 0.001 \
    --num_runs 3 \
    --seed 42 \
    --selected_features 0 2 5 7 8 \
    --save_model \
    --early_stopping \
    --patience 15 \
    --print_every 5

echo ""
echo "Training complete! Check $OUTPUT_DIR for results."

