# Environment for every command in this repository. Copy to scripts/env.sh, edit the paths, then:
#   source scripts/env.sh
export LRT_DATA_ROOT="${LRT_DATA_ROOT:-/path/to/datasets}"          # contains sudoku_ye2024/, cd4/, coding/, strategyqa/
export LRT_DECODER_PATH="${LRT_DECODER_PATH:-/path/to/Qwen3-8B}"    # Hugging Face checkpoint directory of the frozen decoder
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LRT_OUT_ROOT="${LRT_OUT_ROOT:-$REPO/outputs}"               # runs, evaluations, generated data
export SGLANG_PORT="${SGLANG_PORT:-30000}"
export LRT_EVAL_BACKEND="${LRT_EVAL_BACKEND:-sglang}"              # default decoding backend: sglang | hf
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export NO_PROXY="localhost,127.0.0.1${NO_PROXY:+,$NO_PROXY}"       # never route localhost traffic through a proxy
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
