"""Run under torchrun on CPU (gloo) by tests/test_ddp.py: one training stage of an experiment with the tiny
decoder; every rank saves its final trainable parameters so the test can check they agree."""

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from transformers import AutoTokenizer

from lrt import config as C
from lrt.decoder import FrozenDecoder
from lrt.train import StageTrainer
from tests.conftest import SMALL, exp, make_tiny_model

task, name, stage, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cfg = C.load(exp(task, name), SMALL + ["train.batch_size=2", "stage1.batch_size=null", "stage2.batch_size=null",
                                       "train.gen_eval_n=0", "stage1.steps=3", "stage2.steps=3",
                                       "train.eval_every=3"])
dec = FrozenDecoder(make_tiny_model(seed=0), AutoTokenizer.from_pretrained(os.environ["LRT_DECODER_PATH"]))
tr = StageTrainer(cfg, stage, decoder=dec)
tr.run()
torch.save({k: v.detach().clone() for k, v in tr.trainable.state_dict().items()},
           os.path.join(out, f"stage{stage}_rank{os.environ['RANK']}.pt"))
