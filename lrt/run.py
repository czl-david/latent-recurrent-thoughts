"""Command line.

    python -m lrt.run list                                              experiment configs
    python -m lrt.run describe   --config C [--set k=v ...]             the effective configuration
    python -m lrt.run params     --config C [--d-dec 4096]              trainable-parameter breakdown
    python -m lrt.run check-data --config C                             expected dataset files and row counts
    python -m lrt.run train      --config C --stage {1,2} [--set k=v ...] [--resume] [--bench-steps N]
                                 [--stage1-ckpt PATH]
    python -m lrt.run eval       --config C --split S [--ckpt best|last] [--infer-S N] [--stage1-only]
                                 [--backend sglang|hf] [--port P] [--limit N]      (S: an eval split or holdout)
    python -m lrt.run anchor     --config C --split S [--backend sglang|hf] [--port P] [--limit N]

Multi-GPU training: torchrun --nproc_per_node=N -m lrt.run train ... (or scripts/train.sh).
Runs live in $LRT_OUT_ROOT/<task>/<run_name>/.
"""

import argparse
import json
import os
import sys

from lrt import config as C
from lrt.paths import configs_dir, data_root


def check_data(cfg) -> bool:
    d = cfg["data"]
    ok = True
    for split, want in (d.get("expect_rows") or {}).items():
        path = os.path.join(data_root(), d["files"][split])
        if not os.path.exists(path):
            print(f"FAIL {cfg.task}/{split}: missing {path}")
            ok = False
            continue
        with open(path) as f:
            n = sum(1 for line in f if line.strip())
        print(f"{'PASS' if n == want else 'FAIL'} {cfg.task}/{split}: {n} rows (expected {want}) {path}")
        ok &= n == want
    for rel in d.get("required_files") or []:
        path = os.path.join(data_root(), rel)
        print(f"{'PASS' if os.path.exists(path) else 'FAIL'} {cfg.task}: {path}")
        ok &= os.path.exists(path)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(prog="lrt.run")
    ap.add_argument("cmd", choices=["list", "describe", "params", "check-data", "train", "eval", "anchor"])
    ap.add_argument("--config", default=None, help="an experiment file under configs/experiments/")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="override one key, e.g. stage1.lr=1e-4")
    ap.add_argument("--stage", type=int, choices=[1, 2])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench-steps", type=int, default=0, help="train only N steps without evaluation (throughput)")
    ap.add_argument("--stage1-ckpt", default=None, help="stage 2: an explicit stage-1 checkpoint path")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ckpt", default="best", choices=["best", "last"])
    ap.add_argument("--infer-S", dest="infer_S", type=int, default=0,
                    help="refiner outer iterations at inference (0 = as trained)")
    ap.add_argument("--stage1-only", action="store_true", help="evaluate the stage-1 checkpoint without the refiner")
    ap.add_argument("--backend", default=os.environ.get("LRT_EVAL_BACKEND", "sglang"), choices=["sglang", "hf"])
    ap.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", "30000")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--d-dec", type=int, default=4096, help="params: the decoder's hidden size")
    args = ap.parse_args()

    if args.cmd == "list":
        for path in C.experiment_files():
            print(os.path.relpath(path, os.path.dirname(configs_dir())))
        return
    if not args.config:
        ap.error(f"--config is required for {args.cmd}")
    cfg = C.load(args.config, args.sets)
    if args.cmd == "describe":
        print(cfg.describe())
        print(json.dumps(cfg.to_dict(), indent=2))
        return
    if args.cmd == "params":
        from lrt.modules import Proposer, Refiner, format_breakdown, param_breakdown
        print(cfg.describe())
        print(format_breakdown(param_breakdown(Proposer(cfg["model"], args.d_dec, 1.0),
                                               Refiner(cfg["model"], args.d_dec, 1.0))))
        return
    if args.cmd == "check-data":
        sys.exit(0 if check_data(cfg) else 1)
    if args.cmd == "train":
        if not args.stage:
            ap.error("--stage is required")
        from lrt.train import StageTrainer
        StageTrainer(cfg, args.stage, stage1_ckpt=args.stage1_ckpt).run(bench_steps=args.bench_steps,
                                                                        resume=args.resume)
        return
    from lrt.evaluate import run_eval
    run_eval(cfg, args.split, backend=args.backend, port=args.port, anchor=(args.cmd == "anchor"),
             stage1_only=args.stage1_only, limit=args.limit, infer_S=args.infer_S, ckpt=args.ckpt)


if __name__ == "__main__":
    main()
