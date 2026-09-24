"""The data layer. Everything task-specific lives behind this interface: loading, prompting, the
data-preparation pipelines and scoring. Model, training and evaluation code only call these methods
and never branch on the task.

An Example carries
    x        the bare question the proposer encodes (no instruction)
    display  how the question is shown to the decoder after the instruction
    answer   the cross-entropy target
    meta     whatever the steps and the scorer need

The decoder's user text is pre_text(ex) = "{instruction}\\n\\n{display}" (just the display when the
task has no instruction), followed by the latents.

Data-preparation pipelines
--------------------------
A task module defines STEPS with the decorators below, next to its raw loader:

    @load_step("name")     def f(self, rows, **params) -> rows              applied once to the training pool
    @sample_step("name")   def f(self, ex, rng, progress, **params) -> ex   applied to every sampled training
                                                                            example (progress in [0, 1] is the
                                                                            fraction of the stage completed)

A PIPELINE is an ordered list of steps with parameters, named in the config:

    data:
      pipeline: none                   # the knob: which named pipeline prepares the training data
      pipelines:
        none: []                       # the dataset as-is
        foo:  [{step: a}, {step: b, p: 0.5}]

(stage1.pipeline / stage2.pipeline override the knob per stage.) Pipelines touch TRAINING data only:
the held-out rows used for checkpoint selection and every evaluation split are always the raw data.
"""

import json
import os
import random
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

from lrt.paths import data_root, generated_data_dir


@dataclass
class Example:
    uid: str
    x: str
    display: str
    answer: str
    meta: Dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def ramp(progress: float, start: float, end: float, ramp_frac: float) -> float:
    """Linear schedule: `start` at progress 0, `end` from progress `ramp_frac` on."""
    if ramp_frac <= 0:
        return end
    a = min(1.0, max(0.0, progress / ramp_frac))
    return start + (end - start) * a


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def load_step(name: str):
    def mark(fn):
        fn._lrt_step = ("load", name)
        return fn
    return mark


def sample_step(name: str):
    def mark(fn):
        fn._lrt_step = ("sample", name)
        return fn
    return mark


class Pipeline:
    """A named, ordered list of steps bound to a task."""

    def __init__(self, name: str, load_steps: List[Tuple[str, Callable, dict]],
                 sample_steps: List[Tuple[str, Callable, dict]]):
        self.name, self.load_steps, self.sample_steps = name, load_steps, sample_steps

    def prepare(self, rows: List[Example]) -> List[Example]:
        for _, fn, params in self.load_steps:
            rows = fn(rows, **params)
        return rows

    def sample(self, ex: Example, rng: random.Random, progress: float) -> Example:
        for _, fn, params in self.sample_steps:
            ex = fn(ex, rng, progress, **params)
        return ex

    @property
    def per_example(self) -> bool:
        return bool(self.sample_steps)

    def __repr__(self) -> str:
        steps = [f"{n}({', '.join(f'{k}={v}' for k, v in p.items())})" for n, _, p in self.load_steps + self.sample_steps]
        return f"{self.name}: [{', '.join(steps)}]"


class Task:
    def __init__(self, cfg):
        self.cfg = cfg
        self.d = cfg["data"]
        self.prompt = cfg["prompt"]
        self.instruction = self.prompt.get("instruction", "") or ""

    # -- steps and pipelines ------------------------------------------------------------

    @classmethod
    def steps(cls) -> Dict[str, Tuple[str, str]]:
        """{step name: (kind, method name)} of every step this task defines."""
        out = {}
        for klass in reversed(cls.__mro__):
            for attr, fn in vars(klass).items():
                if callable(fn) and hasattr(fn, "_lrt_step"):
                    kind, name = fn._lrt_step
                    out[name] = (kind, attr)
        return out

    def pipeline(self, name: str) -> Pipeline:
        specs = self.d["pipelines"]
        if name not in specs:
            raise KeyError(f"pipeline {name!r} is not defined under data.pipelines ({sorted(specs)})")
        available = self.steps()
        load, sample = [], []
        for item in specs[name] or []:
            item = dict(item)
            step = item.pop("step")
            if step not in available:
                raise KeyError(f"pipeline {name!r}: unknown step {step!r} (this task defines {sorted(available)})")
            kind, attr = available[step]
            (load if kind == "load" else sample).append((step, getattr(self, attr), item))
        return Pipeline(name, load, sample)

    # -- data ---------------------------------------------------------------------------

    def data_path(self, rel: str) -> str:
        """Paths are relative to $LRT_DATA_ROOT; 'gen:<file>' is a file written by lrt.distill."""
        if rel.startswith("gen:"):
            return os.path.join(generated_data_dir(), rel[4:])
        return os.path.join(data_root(), rel)

    def load(self, split: str) -> List[Example]:
        """The raw examples of a split, exactly as the dataset provides them."""
        raise NotImplementedError

    def eval_splits(self) -> List[str]:
        return list(self.cfg["eval"]["splits"])

    def dedup_against_eval(self, rows: List[Example]) -> List[Example]:
        """Drop training rows whose question also appears in an evaluation split."""
        if not self.d.get("dedup_against_eval", False):
            return rows
        seen = {norm_text(e.x) for s in self.eval_splits() for e in self.load(s)}
        return [e for e in rows if norm_text(e.x) not in seen]

    def train_pool_and_holdout(self) -> Tuple[List[Example], List[Example], bool]:
        """(pool, holdout, is_fit_probe), both raw. data.train_limit > 0: the pool is the first N train
        rows and the holdout is the SAME rows (a fit probe). Otherwise the last data.holdout rows."""
        rows = self.dedup_against_eval(self.load("train"))
        limit = int(self.d.get("train_limit", 0) or 0)
        if limit > 0:
            return rows[:limit], rows[:limit], True
        h = int(self.d["holdout"])
        assert 0 < h < len(rows), f"holdout {h} vs {len(rows)} train rows"
        return rows[:-h], rows[-h:], False

    # -- prompting ----------------------------------------------------------------------

    def pre_text(self, ex: Example) -> str:
        return f"{self.instruction}\n\n{ex.display}" if self.instruction else ex.display

    # -- scoring ------------------------------------------------------------------------

    def score(self, text: str, ex: Example) -> Dict[str, Any]:
        """Must return at least {"correct": bool}."""
        raise NotImplementedError

    def score_many(self, texts: List[str], examples: List[Example]) -> List[Dict[str, Any]]:
        return [self.score(t, e) for t, e in zip(texts, examples)]

    def summarize(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        k = sum(bool(r["correct"]) for r in records)
        return {"n": n, "correct": k, "accuracy": k / max(n, 1)}

    # -- hooks of the self-distillation tool (lrt.distill) ------------------------------

    def target_from_response(self, text: str, ex: Example) -> str:
        """Turn a correct sampled decoder response into a training target."""
        return text.strip()

    def gold_hint(self, ex: Example) -> Optional[str]:
        """Text appended to the question to elicit a response for a known answer; None = unsupported."""
        return None

    def hint_leaked(self, text: str) -> bool:
        """True if a hinted response betrays that it was given the answer."""
        return False


def with_changes(ex: Example, **changes) -> Example:
    meta = changes.pop("meta", None)
    return replace(ex, **changes, meta={**ex.meta, **(meta or {})})
