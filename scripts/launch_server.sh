#!/usr/bin/env bash
# Launch an SGLang server for the frozen decoder (evaluation, anchors and lrt.distill), detached.
# Run from the ACTIVATED serving environment, after `source scripts/env.sh`:
#   GPU=0 bash scripts/launch_server.sh
# --disable-radix-cache is required: evaluation sends input embeddings.
# --chunked-prefill-size -1 keeps an input_embeds prefill in one chunk. Do not send text requests
# (lrt.distill) and input_embeds requests (evaluation) to the same server at the same time.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LRT_DECODER_PATH:?source scripts/env.sh first}"
mkdir -p logs
GPU="${GPU:-0}"
PORT="${SGLANG_PORT:-30000}"
MEM="${SGLANG_MEM_FRACTION:-0.8}"          # fraction of the GPU memory that is free at launch
LOG="logs/sglang_gpu${GPU}_port${PORT}.log"
CUDA_VISIBLE_DEVICES="$GPU" setsid nohup python -m sglang.launch_server \
  --model-path "$LRT_DECODER_PATH" --port "$PORT" --dtype bfloat16 \
  --disable-radix-cache --chunked-prefill-size -1 --mem-fraction-static="$MEM" > "$LOG" 2>&1 < /dev/null &
echo "sglang starting on GPU $GPU, port $PORT (pid $!), log $LOG"
for i in $(seq 1 300); do
  if curl -s --noproxy '*' "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "sglang healthy after ${i}s"; exit 0; fi
  sleep 1
done
echo "sglang not healthy after 300s; see $LOG"
exit 1
