"""CPU-only fixtures. A tiny random Qwen3 (the real tokenizer and vocabulary size, tiny width) stands in for
the frozen decoder, so every code path runs without a GPU. Tests that need the tokenizer ($LRT_DECODER_PATH)
or the datasets ($LRT_DATA_ROOT) are skipped when those are not configured."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _has_dir(var: str) -> bool:
    return bool(os.environ.get(var)) and os.path.isdir(os.environ[var])


needs_decoder = pytest.mark.skipif(not _has_dir("LRT_DECODER_PATH"), reason="LRT_DECODER_PATH is not set")
needs_data = pytest.mark.skipif(not _has_dir("LRT_DATA_ROOT"), reason="LRT_DATA_ROOT is not set")


# shrinks any experiment to a CPU-sized run with the tiny decoder
SMALL = ["model.d_prime=64", "model.K_train=8", "model.K_infer=4", "model.proposer.mlp_hidden=128",
         "model.refiner.mlp_hidden=128", "data.train_limit=8", "train.batch_size=4", "train.eval_holdout_n=4",
         "train.eval_every=2", "train.log_every=1", "train.gen_eval_n=2", "train.gen_max_new_tokens=8",
         "stage1.steps=2", "stage2.steps=2", "stage1.warmup_steps=1", "stage2.warmup_steps=1",
         "eval.max_new_tokens=8", "eval.batch_size=4"]


def exp(task: str, name: str) -> str:
    return os.path.join(REPO, "configs", "experiments", task, f"{name}.yaml")


@pytest.fixture(scope="session")
def tok():
    if not _has_dir("LRT_DECODER_PATH"):
        pytest.skip("LRT_DECODER_PATH is not set")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.environ["LRT_DECODER_PATH"])


def make_tiny_model(seed: int = 0, dtype=torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, rope_theta=1_000_000.0, tie_word_embeddings=False)
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).to(dtype)
    with torch.no_grad():                    # embedding rows of a realistic (small) norm
        model.get_input_embeddings().weight.normal_(0.0, 0.17)
    return model


@pytest.fixture()
def tiny_decoder(tok):
    from lrt.decoder import FrozenDecoder
    return FrozenDecoder(make_tiny_model(), tok, latent_sep=" ")
