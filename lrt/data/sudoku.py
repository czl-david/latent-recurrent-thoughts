"""Sudoku: 81-character puzzle strings (0 = blank) and their unique solutions.

Raw example: x = display = the puzzle; answer = the 81-digit solution.
Scorer: the FIRST maximal run of exactly 81 digits in the output must equal the solution.

Steps (training data only; see lrt/data/base.py for how pipelines are built from them):
  symmetry    answer-preserving grid symmetries: digit relabelling, band/stack permutations, row/column
              permutations inside them, transposition (applied to x, display, answer and the solution alike)
  reveal      fill a random fraction r of the blanks from the solution (x and display): a valid puzzle
              with the same unique answer, only easier. With probability 1 - p_none, r ~ U(lo, hi) where
              lo/hi ramp from min_start/max_start to min_end/max_end over the first `ramp` of the stage
  given_drop  with probability p_apply, hide a fraction q ~ U(q_min, q_max) of the givens from the DISPLAY
              only; the proposer's x keeps the full puzzle and the target is unchanged
  autoencode  with probability p_apply, show a blank grid and make the target the puzzle x itself
"""

import random
import re
from typing import Any, Dict, List

from lrt.data.base import Example, Task, load_jsonl, ramp, sample_step, with_changes


def extract_81_runs(text: str) -> List[str]:
    """All maximal digit runs of length exactly 81."""
    return [run for run in re.findall(r"\d+", text) if len(run) == 81]


def is_valid_solution(sol: str) -> bool:
    if len(sol) != 81 or not all(c in "123456789" for c in sol):
        return False
    g = [sol[r * 9:(r + 1) * 9] for r in range(9)]
    units = [set(row) for row in g]
    units += [set(g[r][c] for r in range(9)) for c in range(9)]
    units += [set(g[br * 3 + r][bc * 3 + c] for r in range(3) for c in range(3)) for br in range(3) for bc in range(3)]
    return all(len(u) == 9 for u in units)


_PEERS = []
for _i in range(81):
    _r, _c = divmod(_i, 9)
    _p = {_r * 9 + k for k in range(9)} | {k * 9 + _c for k in range(9)}
    _p |= {(3 * (_r // 3) + a) * 9 + 3 * (_c // 3) + b for a in range(3) for b in range(3)}
    _p.discard(_i)
    _PEERS.append(_p)
_UNITS = ([[r * 9 + c for c in range(9)] for r in range(9)] + [[r * 9 + c for r in range(9)] for c in range(9)]
          + [[(br + a) * 9 + bc + b for a in range(3) for b in range(3)] for br in (0, 3, 6) for bc in (0, 3, 6)])


def deduction_rounds(puzzle: str):
    """Solve by rounds of simultaneous naked + hidden singles (no search).
    Returns (rounds, solved_grid), or (None, None) if propagation stalls."""
    g = [int(ch) for ch in puzzle]
    rounds = 0
    while 0 in g:
        cand = {i: set(range(1, 10)) - {g[p] for p in _PEERS[i]} for i in range(81) if g[i] == 0}
        fills = {i: next(iter(c)) for i, c in cand.items() if len(c) == 1}
        for u in _UNITS:
            for d in range(1, 10):
                spots = [i for i in u if g[i] == 0 and d in cand[i]]
                if len(spots) == 1:
                    fills.setdefault(spots[0], d)
        if not fills:
            return None, None
        for i, d in fills.items():
            g[i] = d
        rounds += 1
    return rounds, "".join(map(str, g))


def grid_symmetry(rng: random.Random):
    """A random answer-preserving symmetry, returned as a function on 81-character grid strings."""
    digits = list("123456789")
    rng.shuffle(digits)
    relabel = {str(i + 1): digits[i] for i in range(9)}
    relabel["0"] = "0"
    bands, stacks = [0, 1, 2], [0, 1, 2]
    rng.shuffle(bands)
    rng.shuffle(stacks)
    rows = [b * 3 + r for b in bands for r in rng.sample(range(3), 3)]
    cols = [s * 3 + c for s in stacks for c in rng.sample(range(3), 3)]
    transpose = rng.random() < 0.5

    def tf(g: str) -> str:
        cells = [[g[r * 9 + c] for c in cols] for r in rows]
        if transpose:
            cells = [list(col) for col in zip(*cells)]
        return "".join(relabel[ch] for row in cells for ch in row)

    return tf


class SudokuTask(Task):
    def load(self, split: str) -> List[Example]:
        rows = load_jsonl(self.data_path(self.d["files"][split]))
        out = []
        for i, r in enumerate(rows):
            p, s = r["puzzle"], r["solution"]
            assert len(p) == 81 and len(s) == 81, f"{split} row {i}: bad length"
            out.append(Example(uid=f"{split}-{i}", x=p, display=p, answer=s, meta={"puzzle": p, "solution": s}))
        return out

    # -- steps --------------------------------------------------------------------------

    @sample_step("symmetry")
    def symmetry(self, ex: Example, rng: random.Random, progress: float) -> Example:
        tf = grid_symmetry(rng)
        return with_changes(ex, x=tf(ex.x), display=tf(ex.display), answer=tf(ex.answer),
                            meta={"puzzle": tf(ex.meta["puzzle"]), "solution": tf(ex.meta["solution"])})

    @sample_step("reveal")
    def reveal(self, ex: Example, rng: random.Random, progress: float, max_start: float, max_end: float,
               ramp_frac: float, p_none: float, min_start: float = 0.0, min_end: float = 0.0) -> Example:
        if rng.random() < p_none:
            return ex
        hi = ramp(progress, max_start, max_end, ramp_frac)
        lo = ramp(progress, min_start, min_end, ramp_frac)
        r = rng.uniform(min(lo, hi), hi)
        blanks = [i for i, ch in enumerate(ex.x) if ch == "0"]
        k = int(round(r * len(blanks)))
        if not k:
            return ex
        cells = list(ex.x)
        for i in rng.sample(blanks, k):
            cells[i] = ex.meta["solution"][i]
        p = "".join(cells)
        return with_changes(ex, x=p, display=p, meta={"puzzle": p})

    @sample_step("given_drop")
    def given_drop(self, ex: Example, rng: random.Random, progress: float, p_apply: float, q_min: float,
                   q_max: float) -> Example:
        if rng.random() >= p_apply:
            return ex
        givens = [i for i, ch in enumerate(ex.x) if ch != "0"]
        k = int(round(rng.uniform(q_min, q_max) * len(givens)))
        if not k:
            return ex
        cells = list(ex.display)
        for i in rng.sample(givens, k):
            cells[i] = "0"
        return with_changes(ex, display="".join(cells))

    @sample_step("autoencode")
    def autoencode(self, ex: Example, rng: random.Random, progress: float, p_apply: float) -> Example:
        if rng.random() >= p_apply:
            return ex
        return with_changes(ex, display="0" * 81, answer=ex.x)

    # -- scoring ------------------------------------------------------------------------

    def score(self, text: str, ex: Example) -> Dict[str, Any]:
        sol, puz = ex.meta["solution"], ex.meta["puzzle"]
        runs = extract_81_runs(text)
        first = runs[0] if runs else None
        cell_acc = sum(a == b for a, b in zip(first, sol)) / 81.0 if first else 0.0
        return {"correct": first == sol, "pred": first, "cell_acc": cell_acc,
                "extraction_fail": first is None, "puzzle_echo": first == puz}

    def summarize(self, records):
        out = super().summarize(records)
        n = max(len(records), 1)
        out.update(cell_acc=sum(r["cell_acc"] for r in records) / n,
                   extraction_fail=sum(r["extraction_fail"] for r in records),
                   puzzle_echo=sum(bool(r["puzzle_echo"]) for r in records))
        return out
