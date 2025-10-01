#!/bin/bash

# Configuration - Modify these variables as needed
DATASET_SRC="huggingface"
LEVEL=3
PROBLEM_ID=9
KERNEL_FILE="KernelBench/src/prompts/tilelang_model_new_lvl3_9.py"
LANGUAGE="tilelang"  # cuda or tilelang
GPU="H100"  # H100, A100, L40S, L4, T4, A10G
LOG="true"
VERBOSE="false"

# Check if kernel file exists
if [ ! -f "$KERNEL_FILE" ]; then
    echo "Error: Kernel file not found: $KERNEL_FILE"
    exit 1
fi

echo "Running kernel evaluation with the following configuration:"
echo "  Level: $LEVEL"
echo "  Problem ID: $PROBLEM_ID"
echo "  Kernel file: $KERNEL_FILE"
echo "  Language: $LANGUAGE"
echo "  GPU: $GPU"
echo "  Logging: $LOG"
echo "  Verbose: $VERBOSE"
echo ""

# Run the evaluation script
python KernelBench/scripts/eval_kernel_from_file.py \
  --dataset_src "$DATASET_SRC" \
  --level "$LEVEL" \
  --problem_id "$PROBLEM_ID" \
  --kernel_file "$KERNEL_FILE" \
  --language "$LANGUAGE" \
  --gpu "$GPU" \
  --log "$LOG" \
  --verbose "$VERBOSE" 