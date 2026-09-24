import json
import random

import pytest

from lrt import config as C
from lrt.data import get_task
from lrt.data.countdown import HINT_PREFIX, dfs_first_chain
from lrt.data.countdown_verify import self_test as countdown_verify_self_test
from lrt.data.sudoku import deduction_rounds, is_valid_solution
from tests.conftest import exp, needs_data

pytestmark = needs_data


def _task(task: str, name: str = "default", sets=()):
    return get_task(C.load(exp(task, name), sets))


def test_the_none_pipeline_is_the_raw_data():
    for task in C.available_tasks():
        t = _task(task, sets=["data.pipeline=none"])
        pipe = t.pipeline("none")
        rows = t.load("train")[:20]
        assert pipe.prepare(rows) == rows and not pipe.per_example
        assert all(pipe.sample(e, random.Random(0), 0.5) is e for e in rows)


def test_holdout_is_raw_and_independent_of_the_pipeline():
    for task in C.available_tasks():
        raw = _task(task, sets=["data.pipeline=none"])
        _, hold, _ = raw.train_pool_and_holdout()
        for name in raw.d["pipelines"]:
            if name == "none":
                continue
            t = _task(task, sets=[f"data.pipeline={name}"])
            _, hold2, _ = t.train_pool_and_holdout()
            assert [(e.uid, e.answer) for e in hold2] == [(e.uid, e.answer) for e in hold]


# -- Sudoku --------------------------------------------------------------------------------

def _sudoku_step(name, **params):
    t = _task("sudoku")
    t.d["pipelines"]["probe"] = [{"step": name, **params}]
    return t, t.pipeline("probe")


def test_sudoku_symmetry_and_reveal_keep_valid_unique_puzzles():
    t, pipe = _sudoku_step("symmetry")
    _, reveal = _sudoku_step("reveal", max_start=0.9, max_end=0.9, ramp_frac=0.5, p_none=0.0)
    rng = random.Random(0)
    for e in t.load("train")[:40]:
        a = reveal.sample(pipe.sample(e, rng, 0.0), rng, 0.0)
        assert is_valid_solution(a.answer) and a.answer == a.meta["solution"]
        assert all(p in ("0", s) for p, s in zip(a.x, a.answer)) and a.display == a.x
        assert deduction_rounds(a.x)[1] == a.answer and a.x.count("0") <= e.x.count("0")


def test_sudoku_given_drop_and_autoencode_change_only_what_the_decoder_sees():
    t, drop = _sudoku_step("given_drop", p_apply=1.0, q_min=0.5, q_max=0.5)
    _, ae = _sudoku_step("autoencode", p_apply=1.0)
    rng = random.Random(3)
    for e in t.load("train")[:30]:
        a = drop.sample(e, rng, 0.0)
        givens = [i for i, ch in enumerate(e.x) if ch != "0"]
        assert a.x == e.x and a.answer == e.answer
        assert sum(a.display[i] == "0" for i in givens) == round(0.5 * len(givens))
        b = ae.sample(e, rng, 0.0)
        assert b.x == e.x and b.display == "0" * 81 and b.answer == e.x and b.meta["solution"] == e.answer
        assert t.score(e.answer, b)["correct"]                       # scoring still uses the true solution


# -- Countdown -----------------------------------------------------------------------------

def _cd4_step(name, **params):
    t = _task("cd4")
    t.d["pipelines"]["probe"] = [{"step": name, **params}]
    return t, t.pipeline("probe")


def test_countdown_search_reproduces_the_dataset_chains():
    t = _task("cd4")
    for e in t.load("test") + t.load("train")[:2000]:
        parts = [int(v) for v in e.x.split(",")]
        assert dfs_first_chain(parts[:4], parts[4]) == e.answer, e.uid
    countdown_verify_self_test()


def test_countdown_steps():
    t, shuffle = _cd4_step("shuffle")
    _, hints = _cd4_step("hints", prob_start=1.0, prob_end=1.0, ramp_frac=0.5, max_steps=2)
    _, drop = _cd4_step("num_drop", p_apply=1.0, q=0.5)
    _, ae = _cd4_step("autoencode", p_apply=1.0)
    rng = random.Random(0)
    for e in t.load("train")[:200]:
        s = shuffle.sample(e, rng, 0.0)
        nums = [int(v) for v in s.meta["input"].split(",")]
        assert s.answer == dfs_first_chain(nums[:4], nums[4]) and t.score(s.answer, s)["correct"]
        h = hints.sample(e, rng, 0.0)
        assert h.meta["hint"].startswith(HINT_PREFIX) and e.answer.startswith(h.meta["hint"][len(HINT_PREFIX):])
        assert h.x == f"{e.x}\n{h.meta['hint']}" and h.display == f"Numbers: {e.x}\n{h.meta['hint']}"
        d = drop.sample(e, rng, 0.0)
        assert d.x == e.x and d.answer == e.answer
        assert all(v in ("?", g) for v, g in zip(d.display[len("Numbers: "):].split(","), e.x.split(",")))
        a = ae.sample(e, rng, 0.0)
        assert a.x == e.x and a.answer == e.x and a.display == "Numbers: ?,?,?,?,?"
    assert all("\n" not in e.x for e in t.load("test"))


# -- StrategyQA ----------------------------------------------------------------------------

def test_strategyqa_raw_targets_and_rationale_pipeline():
    raw = _task("sqa", sets=["data.pipeline=none"])
    rows = raw.load("train")
    assert len(rows) == 2061 and {e.answer for e in rows} == {"Answer: yes", "Answer: no"}
    t = _task("sqa")                                                  # default: reference rationales
    pool, hold, _ = t.train_pool_and_holdout()
    prepared = t.pipeline(t.cfg.pipeline_name(1)).prepare(pool)
    assert 0 < len(prepared) < len(pool) and not {e.x for e in prepared} & {e.x for e in hold}
    assert all(t.score(e.answer, e)["correct"] and e.answer != "Answer: yes" for e in prepared)


def test_strategyqa_generated_rationales_are_used(tmp_path, monkeypatch):
    monkeypatch.setenv("LRT_OUT_ROOT", str(tmp_path))
    t = _task("sqa")
    pool, _, _ = t.train_pool_and_holdout()
    ref = t.pipeline("reference_rationales").prepare(pool)
    have = {e.x for e in ref}
    missing = [e for e in pool if e.x not in have][:3]
    gen = tmp_path / "data_gen" / "sqa_rationales.jsonl"
    gen.parent.mkdir(parents=True)
    rows = [{"uid": e.uid, "x": e.x, "target": f"Because reasons.\n\n{e.answer}", "source": "sample"} for e in missing[:2]]
    rows.append({"uid": missing[2].uid, "x": missing[2].x, "source": "sample",           # inconsistent: ignored
                 "target": "Nope.\n\nAnswer: " + ("no" if missing[2].meta["answer"] else "yes")})
    gen.write_text("\n".join(json.dumps(r) for r in rows))
    assert len(t.pipeline("distilled_rationales").prepare(pool)) == len(ref) + 2
    assert t.gold_hint(missing[0]) and t.hint_leaked("Since the correct answer is yes, ...")


# -- coding --------------------------------------------------------------------------------

def test_coding_target_files_step(tmp_path, monkeypatch):
    monkeypatch.setenv("LRT_OUT_ROOT", str(tmp_path))
    t = _task("mbpp")
    pool, _, _ = t.train_pool_and_holdout()
    (tmp_path / "data_gen").mkdir()
    (tmp_path / "data_gen" / "mbpp_selfdistill.jsonl").write_text(json.dumps({"uid": pool[0].uid, "target": "```python\nX\n```"}))
    out = t.pipeline("self_distilled").prepare(pool)
    assert out[0].answer == "```python\nX\n```" and out[1].answer == pool[1].answer
    he = t.load("humaneval")
    assert len(he) == 164 and t.pre_text(he[0]).startswith(t.he_instruction)
    assert t.pre_text(t.load("test")[0]) == t.load("test")[0].display          # MBPP: no instruction
