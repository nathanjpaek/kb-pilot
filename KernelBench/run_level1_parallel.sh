#!/usr/bin/env bash
set -euo pipefail

# Config
MAX_PROCS=10           # how many problems to run in parallel
START_ID=1             # first problem_id (adjust if your IDs start at 1)
END_ID=49              # last problem_id (first 50 problems: 0–49 or 1–50)
DATASET_SRC="huggingface"
LANGUAGE="cute"   # change to cute / tilelang / etc.
LEVEL=1

run_task() {
  local pid="$1"
  echo "=== Starting problem_id=${pid} ==="
  python scripts/generate_and_eval_rag_modal.py \
    dataset_src="${DATASET_SRC}" \
    language="${LANGUAGE}" \
    level="${LEVEL}" \
    problem_id="${pid}"
  echo "=== Finished problem_id=${pid} ==="
}

for ((i=START_ID; i<=END_ID; i++)); do
  run_task "$i" &

  # Limit to MAX_PROCS concurrent jobs
  while (( $(jobs -r | wc -l) >= MAX_PROCS )); do
    sleep 1
  done
done

wait
echo "All problems completed."