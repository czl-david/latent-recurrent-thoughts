"""End-to-end runs on CPU with the tiny decoder: both stages, checkpoints, holdout evaluation, test-split
evaluation (hf backend), under the base settings and under the tuning-run settings."""

import json
import os

import pytest
import torch

from lrt import config as C
from lrt.decoder import FrozenDecoder
from lrt.evaluate import run_eval
from lrt.train import BatchStream, StageTrainer, lr_at
from tests.conftest import SMALL, exp, make_tiny_model, needs_data


def _records(path):
    return [json.loads(line) for line in open(path)]


@needs_data
@pytest.mark.parametrize("name", ["smoke", "tuned_k32_reveal"])
def test_two_stage_end_to_end(tmp_path, monkeypatch, tok, name):
    monkeypatch.setenv("LRT_OUT_ROOT", str(tmp_path))
    cfg = C.load(exp("sudoku", name), SMALL)
    dec = FrozenDecoder(make_tiny_model(), tok, grad_ckpt=bool(cfg["train"]["grad_ckpt"]))
    StageTrainer(cfg, 1, decoder=dec).run()
    for f in ("best.pt", "last.pt", "metrics.jsonl", "run_config.json"):
        assert os.path.exists(os.path.join(cfg.run_dir(), "stage1", f)), f
    StageTrainer(cfg, 2, decoder=dec).run()
    ck2 = torch.load(os.path.join(cfg.run_dir(), "stage2", "best.pt"), weights_only=False)
    assert ck2["stage1_ckpt"].endswith(os.path.join("stage1", "best.pt"))
    ev = [r for r in _records(os.path.join(cfg.run_dir(), "stage2", "metrics.jsonl")) if r["kind"] == "eval"][-1]
    for k in ("tf_em_train", "tf_em_deploy", "shuffle_gap_deploy", "gen_acc_deploy", "norm_delta_deploy"):
        assert k in ev, k
    assert ev["norm_delta_deploy"] > 0, "stage 2 must move Delta off its zero initialization"
    out = str(tmp_path / "eval")
    s = run_eval(cfg, "test", backend="hf", limit=3, decoder=dec, out_dir=out)
    assert s["n"] == 3 and s["mode"] == "lrt" and os.path.exists(os.path.join(out, "test_lrt_k4_hf_n3.jsonl"))
    s6 = run_eval(cfg, "holdout", backend="hf", limit=2, decoder=dec, out_dir=out, infer_S=6)
    assert s6["infer_S"] == 6 and os.path.exists(os.path.join(out, "holdout_lrt_k4_S6_hf_n2.jsonl"))
    assert run_eval(cfg, "test", backend="hf", limit=2, decoder=dec, out_dir=out, stage1_only=True)["mode"] == "stage1_only"
    assert run_eval(cfg, "test", backend="hf", anchor=True, limit=2, decoder=dec, out_dir=out)["checkpoints"] == {}


@needs_data
def test_aux_readout_and_stage1_reuse(tmp_path, monkeypatch, tok):
    monkeypatch.setenv("LRT_OUT_ROOT", str(tmp_path))
    base = C.load(exp("sudoku", "tuned_k81_s6_aux"), SMALL + ["model.K_train=8", "model.K_infer=8",
                                                               "train.batch_size=4", "stage1.batch_size=4"])
    dec = FrozenDecoder(make_tiny_model(), tok)
    StageTrainer(base, 1, decoder=dec).run()
    tr = [r for r in _records(os.path.join(base.run_dir(), "stage1", "metrics.jsonl")) if r["kind"] == "train"]
    assert "aux" in tr[-1] and tr[-1]["aux"] > 0
    reuse = C.load(exp("sudoku", "tuned_k81_s6_aux10_ce1"), SMALL + ["model.K_train=8", "model.K_infer=8",
                                                                      "train.batch_size=4", "stage1.batch_size=4"])
    StageTrainer(reuse, 2, decoder=dec).run()
    ck2 = torch.load(os.path.join(reuse.run_dir(), "stage2", "best.pt"), weights_only=False)
    assert ck2["stage1_ckpt"] == os.path.join(base.run_dir(), "stage1", "best.pt")
    ev = [r for r in _records(os.path.join(reuse.run_dir(), "stage2", "metrics.jsonl")) if r["kind"] == "eval"][-1]
    assert "aux_acc_deploy" in ev


def test_schedule_and_batches():
    sc = C.load(exp("sudoku", "default")).stage(1)
    total = 1000
    assert lr_at(1, total, sc) == pytest.approx(3e-4 / 50)                      # 5% linear warmup
    assert lr_at(50, total, sc) == pytest.approx(3e-4)
    assert lr_at(total, total, sc) == pytest.approx(0.0, abs=1e-12)             # cosine to zero
    a = BatchStream(100, 4, world=2, rank=0, seed=0)
    b = BatchStream(100, 4, world=2, rank=1, seed=0)
    assert not set(a.indices(3)) & set(b.indices(3)) and a.indices(3) == BatchStream(100, 4, 2, 0, 0).indices(3)
