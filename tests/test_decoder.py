import pytest
import torch

from lrt.decoder import FrozenDecoder
from tests.conftest import make_tiny_model

PRE = ["Solve this.\n\n5300700", "A much longer instruction text here.\n\nNumbers: 97,2,59,5,14", "Q?"]
ANS = ["534678912", "97-59=38,38/2=19,19-5=14", "Answer: yes"]


def _latents(b=3, k=4, d=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, k, d, generator=g) * 0.17).requires_grad_(True)


def test_prefix_cache_matches_full_sequence(tiny_decoder):
    lat_a = _latents()
    out_full = tiny_decoder.ce(PRE, lat_a, ANS, use_prefix_cache=False)
    out_full["loss"].backward()
    lat_b = lat_a.detach().clone().requires_grad_(True)
    out_pc = tiny_decoder.ce(PRE, lat_b, ANS, use_prefix_cache=True)
    out_pc["loss"].backward()
    assert torch.allclose(out_full["loss"], out_pc["loss"], atol=1e-5)
    assert torch.allclose(lat_a.grad, lat_b.grad, atol=1e-6, rtol=1e-4)
    assert torch.equal(out_full["em_per_example"], out_pc["em_per_example"])


def test_gradient_checkpointing_matches_and_refuses_the_prefix_cache(tok):
    dec_plain = FrozenDecoder(make_tiny_model(seed=3), tok)
    dec_ckpt = FrozenDecoder(make_tiny_model(seed=3), tok, grad_ckpt=True)
    lat_a, lat_b = _latents(), _latents()
    a = dec_plain.ce(PRE, lat_a, ANS, use_prefix_cache=True)
    b = dec_ckpt.ce(PRE, lat_b, ANS, use_prefix_cache=False)
    a["loss"].backward()
    b["loss"].backward()
    assert torch.allclose(a["loss"], b["loss"], atol=1e-5)
    assert torch.allclose(lat_a.grad, lat_b.grad, atol=1e-6, rtol=1e-4)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dec_ckpt.ce(PRE, _latents(), ANS, use_prefix_cache=True)


def test_labels_cover_only_the_answer(tiny_decoder, tok):
    out = tiny_decoder.ce(PRE, None, ANS, use_prefix_cache=True)
    assert int(out["n_tokens"]) == sum(len(tok(a + "<|im_end|>", add_special_tokens=False).input_ids) for a in ANS)
    assert torch.isfinite(out["loss"])


def test_decoder_weights_get_no_gradient(tiny_decoder):
    lat = _latents()
    tiny_decoder.ce(PRE, lat, ANS)["loss"].backward()
    assert all(p.grad is None for p in tiny_decoder.model.parameters())
    assert lat.grad is not None and lat.grad.abs().sum() > 0


def test_greedy_generation_starts_with_the_argmax_token(tiny_decoder):
    prefixes = tiny_decoder.prefix_embeds(PRE, _latents().detach())
    texts = tiny_decoder.generate_greedy(prefixes, max_new_tokens=4)
    assert len(texts) == 3
    for i, p in enumerate(prefixes):
        first = int(tiny_decoder.model(inputs_embeds=p.unsqueeze(0)).logits[0, -1].argmax())
        if first != tiny_decoder.eos_id:
            assert texts[i].startswith(tiny_decoder.tok.decode([first]))


def test_input_layout(tiny_decoder, tok):
    assert tok.decode(tiny_decoder.pre_ids("INSTR\n\nX")) == "<|im_start|>user\nINSTR\n\nX "
    assert tok.decode(tiny_decoder.post_ids) == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
