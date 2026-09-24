import copy

import pytest
import torch

from lrt import config as C
from lrt.latents import compute_latents, mean_pairwise_cos
from lrt.modules import Proposer, Refiner, count_params, param_breakdown
from tests.conftest import exp

SHELL = 1.3758


def _model(name="default", task="sudoku", **refiner):
    m = copy.deepcopy(C.load(exp(task, name))["model"])
    m["refiner"].update(refiner)
    return m


def test_parameter_breakdown_of_the_base_config():
    rows = dict(param_breakdown(Proposer(_model(), 4096, SHELL), Refiner(_model(), 4096, SHELL)))
    m = lambda v: round(v / 1e6, 1)
    assert m(rows["proposer: input projection P_down"]) == 1.0
    assert rows["proposer: query embeddings"] == 32 * 256
    assert m(rows["proposer: encoder blocks"]) == 2.2
    assert m(rows["proposer: output projection P_up"]) == 1.1
    assert m(rows["refiner: input projection P'_down"]) == 1.0
    assert m(rows["refiner: transition block f"]) == 4.8
    assert m(rows["refiner: output projection P'_up"]) == 1.1
    assert rows["refiner: initial states z_L^0, z_H^0"] == 2 * 32 * 256
    assert round(sum(rows.values()) / 1e6, 1) == 11.2


@pytest.mark.parametrize("name", ["default", "tuned_k32_reveal"])
def test_proposer_shapes_and_padding_invariance(name):
    torch.manual_seed(0)
    prop = Proposer(_model(name), 64, SHELL).eval()
    short, long = torch.randn(1, 5, 64) * 0.2, torch.randn(1, 9, 64) * 0.2
    junk = torch.randn(1, 4, 64) * 5                              # left padding of the shorter question
    x = torch.cat([torch.cat([junk, short], dim=1), long], dim=0)
    mask = torch.tensor([[False] * 4 + [True] * 5, [True] * 9])
    with torch.no_grad():
        l0_batch = prop(x, mask)
        l0_alone = prop(short, torch.ones(1, 5, dtype=torch.bool))
    assert l0_batch.shape == (2, 32, 64)
    assert torch.allclose(l0_batch[0], l0_alone[0], atol=1e-5), "the proposer output depends on padding"
    if prop.shell_rescale:
        assert torch.allclose(l0_batch.norm(dim=-1), torch.full((2, 32), SHELL), atol=1e-4)


@pytest.mark.parametrize("name", ["default", "tuned_k32_reveal"])
def test_refiner_starts_at_identity_and_differentiates_only_the_last_cycle(name):
    m = _model(name)
    ref = Refiner(m, 64, SHELL)
    calls = []
    orig = ref.f.forward

    def spy(*a, **k):
        calls.append(torch.is_grad_enabled())
        return orig(*a, **k)

    ref.f.forward = spy
    l0 = torch.randn(3, 32, 64)
    l_star, delta = ref(l0)
    s, h, t = (m["refiner"][k] for k in ("S", "H", "T"))
    assert len(calls) == s * h * (t + 1) == 45
    assert sum(calls) == t + 1 and all(calls[-(t + 1):]), "only the final cycle may build a graph"
    assert torch.equal(delta, torch.zeros_like(delta)) and torch.equal(l_star, l0)      # P'_up is zero-initialized
    l_star.pow(2).sum().backward()
    assert ref.up.weight.grad is not None and ref.up.weight.grad.abs().sum() > 0
    if isinstance(ref.zL0, torch.nn.Parameter):                   # learned, but only reachable when S x H == 1
        assert ref.zL0.grad is None
    calls.clear()
    with torch.no_grad():
        ref(l0[:, :4], S=6)                                        # inference: first K_infer slots, more outer iterations
    assert len(calls) == 6 * h * (t + 1)


def test_initial_state_options():
    learned = Refiner(_model(), 64, SHELL)
    fixed = Refiner(_model(init_state="fixed", init_state_shape="shared"), 64, SHELL)
    assert isinstance(learned.zL0, torch.nn.Parameter) and learned.zL0.shape == (32, 256)
    assert not isinstance(fixed.zL0, torch.nn.Parameter) and fixed.zL0.shape == (256,)
    assert count_params(learned) - count_params(fixed) == 2 * 32 * 256
    one_cycle = Refiner(_model(S=1, H=1), 64, SHELL)
    with torch.no_grad():
        one_cycle.up.weight.normal_()
    one_cycle(torch.randn(2, 8, 64))[0].sum().backward()
    assert one_cycle.zL0.grad is not None, "with a single cycle the learned initial states receive gradient"


def test_pre_norm_recursion_stays_bounded():
    ref = Refiner(_model(S=6), 64, SHELL)
    norms = []
    orig = ref.f.forward

    def spy(*a, **k):
        out = orig(*a, **k)
        norms.append(float(out.norm(dim=-1).mean()))
        return out

    ref.f.forward = spy
    with torch.no_grad():
        ref(torch.randn(2, 32, 64) * SHELL)
    assert max(norms) < 50 * norms[0], "the refiner state grows without bound"


def test_latent_conditions_and_collapse_metric():
    m = _model()
    prop, ref = Proposer(m, 64, SHELL), Refiner(m, 64, SHELL)
    x, mask = torch.randn(4, 6, 64), torch.ones(4, 6, dtype=torch.bool)
    train = compute_latents(prop, ref, x, mask, "train", 4)
    deploy = compute_latents(prop, ref, x, mask, "deploy", 4)
    assert train["latents"].shape == (4, 32, 64) and deploy["latents"].shape == (4, 4, 64)
    same = torch.randn(1, 3, 8).expand(5, -1, -1)
    assert mean_pairwise_cos(same) == pytest.approx(1.0, abs=1e-5)
