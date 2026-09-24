"""Two-process DDP (CPU, gloo) through both stages: the ranks stay identical, and every trainable parameter
takes part in the loss (DDP fails on parameters that never receive a gradient)."""

import os
import subprocess
import sys

import pytest
import torch

from tests.conftest import REPO, needs_data, needs_decoder


@needs_data
@needs_decoder
@pytest.mark.parametrize("name", ["default", "tuned_k32_reveal"])
def test_two_rank_ddp_both_stages(tmp_path, name):
    env = {**os.environ, "LRT_OUT_ROOT": str(tmp_path), "PYTHONPATH": REPO, "CUDA_VISIBLE_DEVICES": ""}
    for stage in (1, 2):
        cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               os.path.join(REPO, "tests", "ddp_worker.py"), "sudoku", name, str(stage), str(tmp_path)]
        res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
        assert res.returncode == 0, f"stage {stage}:\n{res.stdout[-3000:]}\n{res.stderr[-3000:]}"
        r0 = torch.load(tmp_path / f"stage{stage}_rank0.pt")
        r1 = torch.load(tmp_path / f"stage{stage}_rank1.pt")
        assert r0.keys() == r1.keys() and all(torch.equal(r0[k], r1[k]) for k in r0), "DDP ranks diverged"
