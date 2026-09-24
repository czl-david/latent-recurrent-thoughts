"""Countdown-4: four numbers and a target; the answer is a 3-step chain such as '97-59=38,38/2=19,19-5=14'.

Raw example: x = "a,b,c,d,target"; display = "Numbers: a,b,c,d,target"; answer = the dataset's chain.
Scorer: STRICT chain validity (countdown_verify.py): any chain in the output that uses each given number
exactly once with exact arithmetic and ends at the target counts; the lenient final-value metric is
reported alongside.

Data fact used by `shuffle`: every dataset chain is the FIRST integer, non-negative solution of a fixed
depth-first search over the numbers in their given order (`dfs_first_chain`), so targets are deterministic
but depend on the order of the numbers.

Steps (training data only):
  shuffle     shuffle the four numbers and recompute the target with the same search
  hints       with probability ramping from prob_start to prob_end over the first `ramp_frac` of the stage,
              append the first k <= max_steps gold steps as a "Start with: ..." line (proposer input and
              display); the target stays the full chain
  num_drop    with probability p_apply, show each of the five numbers to the DECODER as '?' with
              probability q; the proposer's x keeps all numbers
  autoencode  with probability p_apply, hide all numbers from the decoder and make the target the number
              string itself
Apply `shuffle` before `hints`.
"""

import random
from itertools import combinations
from typing import Any, Dict, List, Optional

from lrt.data.base import Example, Task, load_jsonl, ramp, sample_step, with_changes
from lrt.data.countdown_verify import cd4_strict_correct

HINT_PREFIX = "Start with: "


def dfs_first_chain(nums: List[int], target: int) -> Optional[str]:
    """The data generator's search: pairs in combinations order over the current value list (the new
    value appended at the end), ops tried as +, *, a-b, b-a, a/b, b/a, integer non-negative
    intermediates only. Returns the first chain reaching target."""

    def rec(vals: List[int], steps: List[str]) -> Optional[str]:
        if len(vals) == 1:
            return ",".join(steps) if vals[0] == target else None
        for i, j in combinations(range(len(vals)), 2):
            a, b = vals[i], vals[j]
            rest = [v for k, v in enumerate(vals) if k not in (i, j)]
            cands = [(a, "+", b, a + b), (a, "*", b, a * b)]
            if a >= b:
                cands.append((a, "-", b, a - b))
            if b >= a:
                cands.append((b, "-", a, b - a))
            if b != 0 and a % b == 0:
                cands.append((a, "/", b, a // b))
            if a != 0 and b % a == 0:
                cands.append((b, "/", a, b // a))
            for x, op, y, r in cands:
                found = rec(rest + [r], steps + [f"{x}{op}{y}={r}"])
                if found is not None:
                    return found
        return None

    return rec(list(nums), [])


def _render(ex: Example, **meta) -> Example:
    """Rebuild x / display from meta: input (ordered numbers), shown (what the decoder sees of them),
    hint (optional 'Start with: ...' line) and hint_shown."""
    m = {**ex.meta, **meta}
    hint = f"\n{m['hint']}" if m["hint"] else ""
    display = f"Numbers: {m['shown']}" + (hint if m["hint_shown"] else "")
    return with_changes(ex, x=m["input"] + hint, display=display, meta=meta)


class CountdownTask(Task):
    def load(self, split: str) -> List[Example]:
        rows = load_jsonl(self.data_path(self.d["files"][split]))
        return [Example(uid=f"{split}-{i}", x=r["input"], display=f"Numbers: {r['input']}", answer=r["output"],
                        meta={"input": r["input"], "shown": r["input"], "hint": "", "hint_shown": True})
                for i, r in enumerate(rows)]

    # -- steps --------------------------------------------------------------------------

    @sample_step("shuffle")
    def shuffle(self, ex: Example, rng: random.Random, progress: float) -> Example:
        parts = ex.meta["input"].split(",")
        nums = parts[:4]
        rng.shuffle(nums)
        inp = ",".join(nums + parts[4:])
        answer = dfs_first_chain([int(n) for n in nums], int(parts[4]))
        assert answer is not None, f"no solution after reordering {ex.meta['input']}"
        return _render(with_changes(ex, answer=answer), input=inp, shown=inp, hint="")

    @sample_step("hints")
    def hints(self, ex: Example, rng: random.Random, progress: float, prob_start: float, prob_end: float,
              ramp_frac: float, max_steps: int) -> Example:
        if rng.random() >= ramp(progress, prob_start, prob_end, ramp_frac):
            return ex
        steps = ex.answer.split(",")
        k = rng.randint(1, max(1, min(int(max_steps), len(steps) - 1)))
        return _render(ex, hint=HINT_PREFIX + ",".join(steps[:k]))

    @sample_step("num_drop")
    def num_drop(self, ex: Example, rng: random.Random, progress: float, p_apply: float, q: float) -> Example:
        if rng.random() >= p_apply:
            return ex
        shown = ",".join("?" if rng.random() < q else v for v in ex.meta["input"].split(","))
        return _render(ex, shown=shown)

    @sample_step("autoencode")
    def autoencode(self, ex: Example, rng: random.Random, progress: float, p_apply: float) -> Example:
        if rng.random() >= p_apply:
            return ex
        hidden = ",".join("?" for _ in ex.meta["input"].split(","))
        return _render(with_changes(ex, answer=ex.meta["input"]), shown=hidden, hint_shown=False)

    # -- scoring ------------------------------------------------------------------------

    def score(self, text: str, ex: Example) -> Dict[str, Any]:
        res = cd4_strict_correct(text, {"input": ex.meta["input"]})
        return {"correct": bool(res["strict"]), **res}

    def summarize(self, records):
        out = super().summarize(records)
        n = max(len(records), 1)
        out.update(legacy=sum(r["legacy"] for r in records) / n, parsed=sum(r["parsed"] for r in records) / n)
        return out
