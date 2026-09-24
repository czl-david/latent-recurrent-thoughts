#!/usr/bin/env bash
# Train one stage on one or more GPUs (DDP over the small trainable module), detached, logging to logs/.
# Run from the activated training environment, after `source scripts/env.sh`:
#   GPUS=0,1 bash scripts/train.sh configs/experiments/<task>/<name>.yaml 1 [--set key=value ...]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LRT_DECODER_PATH:?source scripts/env.sh first}"
CONFIG=$1; STAGE=$2; shift 2
GPUS="${GPUS:-0}"
N=$(echo "$GPUS" | tr ',' '\n' | wc -l)
PORT="${MASTER_PORT:-$((29500 + RANDOM % 400))}"
NAME="$(basename "$(dirname "$CONFIG")")_$(basename "$CONFIG" .yaml)"
mkdir -p logs
LOG="logs/${NAME}_stage${STAGE}.log"
CUDA_VISIBLE_DEVICES="$GPUS" setsid nohup torchrun --standalone --nproc_per_node="$N" --master_port="$PORT" \
  -m lrt.run train --config "$CONFIG" --stage "$STAGE" "$@" > "$LOG" 2>&1 < /dev/null &
echo "training $CONFIG stage $STAGE on GPUs $GPUS (pid $!); log: $LOG"
