#!/bin/bash

# List of dataset folders
FOLDERS=(
  "combined_easy"
  "combined_hard"
)

BASE_DATA="./data/processed/deepseekNEW"
BASE_OUT="./outputsNEW/encoder64"
MODEL_TYPE="hetero_gin"

for FOLDER in "${FOLDERS[@]}"; do
  echo "==============================="
  echo " Running experiments for $FOLDER"
  echo "==============================="

  # ---------- UNDIRECTED RUN ----------
  UND_PT="${BASE_DATA}/${FOLDER}/undirected/final_regraded_with_rev.pt"
  UND_OUT="${BASE_OUT}/${FOLDER}/undirected/models/${MODEL_TYPE}"
  UND_LOG="${UND_OUT}/train.log"

  # Create output directory (including parents) if not exists
  mkdir -p "$UND_OUT"

  echo "Running undirected model for $FOLDER"
  poetry run python scripts/02_train_model.py \
    --pt-file "$UND_PT" \
    --output-dir "$UND_OUT" \
    --model-type "$MODEL_TYPE" \
    --log-interval 1 \
    --encoder-type robust \
    > "$UND_LOG" 2>&1

  # ---------- DIRECTED RUN ----------
  DIR_PT="${BASE_DATA}/${FOLDER}/directed/final_regraded.pt"
  DIR_OUT="${BASE_OUT}/${FOLDER}/directed/models/${MODEL_TYPE}"
  DIR_LOG="${DIR_OUT}/train.log"

  mkdir -p "$DIR_OUT"

  echo "Running directed model for $FOLDER"
  poetry run python scripts/02_train_model.py \
    --pt-file "$DIR_PT" \
    --output-dir "$DIR_OUT" \
    --model-type "$MODEL_TYPE" \
    --log-interval 1 \
    --encoder-type robust \
    > "$DIR_LOG" 2>&1

done

echo "All training runs complete."
