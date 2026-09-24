# Latent Recurrent Thoughts

Training and evaluation code for a latent-reasoning pipeline on four benchmarks: Sudoku, Countdown-4 (cd4), MBPP (with HumanEval as a zero-shot transfer evaluation), and StrategyQA.

## Quick Start

**1. Environments.** Training (Python 3.12):

```bash
conda create -n lrt python=3.12 -y && conda activate lrt
pip install torch --index-url https://download.pytorch.org/whl/cu128    # the wheel for your CUDA version
pip install -r requirements.txt
```

For fast evaluation, a second environment with [SGLang](https://github.com/sgl-project/sglang) (0.5.x).

**2. Paths.** Copy `scripts/env.example.sh` to `scripts/env.sh`, set the dataset root and the Qwen3-8B checkpoint directory in it, then `source scripts/env.sh`. The dataset root is expected to contain:

```
sudoku/{train,test}.jsonl
cd4/cd4_{train,test}.jsonl
coding/mbpp/{train,test}.jsonl  coding/human-eval/test.jsonl  coding/verify.py
strategyqa/{train,test}.jsonl  strategyqa/train_distilled_nothink.jsonl
```

**3. Check the data** (per task):

```bash
python -m lrt.run check-data --config configs/experiments/sqa/default.yaml
```

**4. Train** both stages:

```bash
python -m lrt.run train --config configs/experiments/sqa/default.yaml --stage 1
python -m lrt.run train --config configs/experiments/sqa/default.yaml --stage 2
```

Multi-GPU: `GPUS=0,1,2,3 bash scripts/train.sh configs/experiments/sqa/default.yaml 1`. Checkpoints go to `outputs/<task>/<config name>/`.

**5. Evaluate.** Start an SGLang server from the SGLang environment (`GPU=0 bash scripts/launch_server.sh`), then:

```bash
python -m lrt.run eval --config configs/experiments/sqa/default.yaml --split test
```

(`--backend hf` decodes in-process without a server; for `mbpp`, also `--split humaneval`.)

## Configs

The default version of each task is `configs/experiments/<task>/default.yaml`. The same directory also has other tuned versions (`tuned_*.yaml`) and variants of the default. Every setting lives in the config files (`configs/base.yaml` lists them all) and can be overridden on the command line with `--set key=value`. The data preprocessing pipeline is chosen with `data.pipeline`; `none` uses the data as provided.

Tests: `python -m pytest tests/`.


## Acknowledgements

This codebase builds on prior open-source work:

- **SoftCoT** ([Xu et al., 2025](https://arxiv.org/abs/2502.12134), [code](https://github.com/xuyige/SoftCoT)) — the soft chain-of-thought formulation this pipeline follows: a small trainable module produces continuous thought vectors that are linearly projected into a frozen decoder's input embedding space (the proposer in `lrt/modules.py`, assembled in `lrt/latents.py`).
- **TRM — Tiny Recursive Model** ([Jolicoeur-Martineau, 2025](https://arxiv.org/abs/2510.04871), [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)) — the refiner in `lrt/modules.py` adapts its recursive latent update.
- **HRM — Hierarchical Reasoning Model** ([Wang et al., 2025](https://arxiv.org/abs/2506.21734), [code](https://github.com/sapientinc/HRM)) — the architectural ancestor of TRM.
- **sglang** ([code](https://github.com/sgl-project/sglang)) — serves the frozen decoder for evaluation, including the embedding-input path used to feed the latent thoughts (`lrt/sglang_client.py`).

Benchmarks and datasets:

- **Sudoku-Extreme** — the sudoku split released with HRM ([Wang et al., 2025](https://arxiv.org/abs/2506.21734)).
- **Countdown** — the arithmetic puzzle task popularized by [Stream of Search](https://arxiv.org/abs/2404.03683) (Gandhi et al., 2024) and TinyZero.
- **MBPP** ([Austin et al., 2021](https://arxiv.org/abs/2108.07732)) and **HumanEval** ([Chen et al., 2021](https://arxiv.org/abs/2107.03374)).
- **StrategyQA** ([Geva et al., 2021](https://arxiv.org/abs/2101.02235)), via the `ChilleD/StrategyQA` HuggingFace release.
