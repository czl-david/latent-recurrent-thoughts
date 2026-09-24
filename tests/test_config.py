import os

import pytest

from lrt import config as C
from tests.conftest import exp


def test_every_experiment_config_loads_and_validates():
    files = C.experiment_files()
    assert len(files) >= 40
    names = {}
    for path in files:
        cfg = C.load(path)
        names.setdefault(cfg.task, set()).add(os.path.splitext(os.path.basename(path))[0])
        assert cfg.task in C.available_tasks()
        assert cfg.pipeline_name(1) in cfg["data"]["pipelines"] and cfg.pipeline_name(2) in cfg["data"]["pipelines"]
    for path in files:                       # stage-2 reuse points at an existing experiment of the same task
        cfg = C.load(path)
        ref = cfg.stage(2)["stage1_from"]
        assert ref is None or ref in names[cfg.task], (path, ref)


def test_every_task_offers_the_unprocessed_pipeline_and_a_default_config():
    for task in C.available_tasks():
        cfg = C.load(exp(task, "default"))
        assert cfg["data"]["pipelines"]["none"] == []
        assert os.path.exists(exp(task, "smoke"))


def test_base_values_are_the_method_hyperparameters():
    cfg = C.load(exp("sudoku", "default"))
    m, r = cfg["model"], cfg["model"]["refiner"]
    assert (m["d_prime"], m["K_train"], m["K_infer"], m["lambda_delta"]) == (256, 32, 4, 0.01)
    assert (r["S"], r["H"], r["T"]) == (3, 3, 4)
    assert m["block_norm"] == "pre" and m["proposer"]["blocks"] == 2 and r["init_state"] == "learned"
    for s, lr in ((1, 3e-4), (2, 2e-4)):
        sc = cfg.stage(s)
        assert sc["lr"] == lr and sc["betas"] == [0.9, 0.999] and sc["eps"] == 1e-8
        assert sc["weight_decay"] == 0.01 and sc["schedule"] == "cosine" and sc["warmup_frac"] == 0.05
        assert sc["grad_clip"] == 1.0 and sc["epochs"] == 30 and cfg.batch_size(s) == 64
        assert sc["aux_readout"]["weight"] == 0.0
    tr = cfg["train"]
    assert tr["precision"] == "bf16" and tr["grad_ckpt"] is True and tr["prefix_cache"] is False
    assert cfg["data"]["pipeline"] == "none"


def test_merge_is_strict_and_set_works(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("task: sudoku\nmodel: {K_trian: 81}\n")
    with pytest.raises(KeyError, match="K_trian"):
        C.load(str(bad))
    ok = tmp_path / "ok.yaml"
    ok.write_text("task: sudoku\ndata:\n  pipelines:\n    mine: [{step: symmetry}]\n  pipeline: mine\n")
    cfg = C.load(str(ok), ["stage1.lr=1e-4", "model.K_infer=8", "stage2.pipeline=none"])
    assert cfg.stage(1)["lr"] == 1e-4 and cfg["model"]["K_infer"] == 8
    assert cfg.pipeline_name(1) == "mine" and cfg.pipeline_name(2) == "none"
    with pytest.raises(KeyError):
        C.load(str(ok), ["stage1.lrr=1"])
    with pytest.raises(ValueError, match="not defined"):
        C.load(str(ok), ["data.pipeline=missing"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        C.load(str(ok), ["train.prefix_cache=true"])
    with pytest.raises(ValueError, match="K_infer"):
        C.load(str(ok), ["model.K_infer=64"])


def test_presets_and_run_names():
    cfg = C.load(exp("sudoku", "tuned_k81_s6_aux10_ce1"))
    assert cfg["model"]["block_norm"] == "post" and cfg["model"]["refiner"]["init_state"] == "fixed"
    assert cfg.stage(2)["stage1_from"] == "tuned_k81_s6_aux" and cfg.stage(2)["aux_readout"]["ce_every"] == 1
    k32 = C.load(exp("sqa", "tuned_distilled_rationales_h250_kinfer32"))
    assert k32.name == "tuned_distilled_rationales_h250" and k32["model"]["K_infer"] == 32
    assert os.path.dirname(k32.run_dir()) == os.path.dirname(C.load(exp("sqa", "default")).run_dir())

