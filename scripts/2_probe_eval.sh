#!/bin/bash

# Root path for models
BASE_MODEL_DIR="/home/amyrouillard/project-files/models"

# List of (subdir, epoch) pairs
JOBS=(
#   "2025_06_02_13_07 25"
#   "2025_06_02_14_32 32"
  "2025_06_02_14_44 55 '499 499 499 176 230 50'"  
  "2025_06_02_15_24 83 '499 499 499 304 101 70'"
  "2025_06_02_15_40 81 '499 499 499 499 499 106'"
  # Add more here as needed
)

# Path to your Python script
SCRIPT="/home/amyrouillard/project-files/binGPT/scripts/2_probe_eval.py"  # change this if neede

# Loop through jobs and run each
for job in "${JOBS[@]}"; do
  read SUBDIR EPOCH PROBE<<< "$job"
  MODEL_DIR="${BASE_MODEL_DIR}/${SUBDIR}"

  echo "Running: python $SCRIPT --model_dir $MODEL_DIR --transformer_load_epoch $EPOCH" 
  python "$SCRIPT" --model_dir "$MODEL_DIR" --transformer_load_epoch "$EPOCH" --probe_load_epoch "$PROBE"
done