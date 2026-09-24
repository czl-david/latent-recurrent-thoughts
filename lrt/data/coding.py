"""Python coding: MBPP (train and a local test split) and HumanEval, an evaluation-only split scored on the
same MBPP-trained stack.

MBPP raw example:      x = display = the task text + its test assertions (they pin the function name);
                       no instruction; answer = "```python\n{reference code}\n```".
HumanEval example:     x = the function prompt (signature + docstring); display = the prompt fenced in
                       ```python; instruction = prompt.humaneval_instruction.
Scoring:               strip any thinking -> extract the code block defining the target function ->
                       the sandboxed verifier ($LRT_DATA_ROOT/<data.verifier>) -> retry with the expected
                       name aliased to each top-level def.

Steps (training data only):
  target_files   replace training targets by uid from jsonl files of {"uid", "target"} (for example
                 self-distilled solutions written by lrt.distill); rows without an entry keep the raw target
"""

import importlib.util
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

from lrt.data.base import Example, Task, load_jsonl, load_step

MBPP_TESTS_HEADER = "Your code should pass these tests:"
VERIFY_TIMEOUT = 10.0
SANDBOX_IMPORT_MARK = "not allowed in sandbox"
_FENCE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)


def strip_thinking(text: str) -> str:
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def target_func_name_mbpp(row: dict) -> Optional[str]:
    m = re.search(r"def\s+(\w+)", row.get("code", "") or "")
    return m.group(1) if m else None


def _strip_toplevel_usage(code: str, fname: Optional[str]) -> str:
    out = []
    for line in code.splitlines():
        stripped = line.strip()
        if bool(line) and line[0] not in (" ", "\t"):
            if stripped.startswith("print(") or stripped.startswith("if __name__"):
                continue
            if fname and re.match(r"^" + re.escape(fname) + r"\s*\(", stripped):
                continue
        out.append(line)
    return "\n".join(out)


def _toplevel_defs(code: str) -> List[str]:
    return re.findall(r"(?m)^def\s+(\w+)\s*\(", code)


def hardened_extract(clean: str, fname: Optional[str]) -> str:
    blocks = _FENCE.findall(clean)
    if not blocks:
        return clean.strip()
    def_blocks = []
    if fname:
        pat = re.compile(r"(?m)^\s*def\s+" + re.escape(fname) + r"\b")
        def_blocks = [b for b in blocks if pat.search(b)]
    if def_blocks:
        code = "\n\n".join(def_blocks)
    else:
        code_blocks = [b for b in blocks if re.search(r"(?m)^\s*(def|class|import|from|@)\s", b)]
        code = "\n\n".join(code_blocks) if code_blocks else blocks[-1]
    return _strip_toplevel_usage(code, fname).strip()


_VERIFIER: Optional[ModuleType] = None


def load_code_verifier(path: str) -> ModuleType:
    global _VERIFIER
    if _VERIFIER is None:
        spec = importlib.util.spec_from_file_location("code_verify", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _VERIFIER = mod
    return _VERIFIER


def _verify(verifier: ModuleType, split: str, code: str, row: dict) -> Tuple[bool, str]:
    if split == "humaneval":
        ok, err = verifier._verify_human_eval_single(code, row["prompt"], row["entry_point"], row["test"],
                                                     VERIFY_TIMEOUT)
    else:
        ok, err = verifier._verify_mbpp_single(code, row.get("test_setup_code", ""), row["test_list"],
                                               VERIFY_TIMEOUT)
    return bool(ok), err


def score_code(verifier: ModuleType, split: str, code: str, expected: Optional[str], row: dict):
    """(correct, direct_correct, alias_used, direct_error)."""
    ok, err = _verify(verifier, split, code, row)
    if ok:
        return True, True, None, ""
    if expected:
        for d in _toplevel_defs(code):
            if d != expected and _verify(verifier, split, code + f"\n{expected} = {d}\n", row)[0]:
                return True, False, d, err
    return False, False, None, err


# verify.py forks a sandbox process per check (default mp.Process). Forking a
# multi-threaded CUDA/NCCL training process can deadlock, so every check runs
# inside a SPAWNED worker (clean, single-threaded, no CUDA) that does the forking.
_POOL: Optional[ProcessPoolExecutor] = None


def _init_worker() -> None:
    # A spawned child inherits "spawn" as its default start method, but verify.py hands a
    # local closure to mp.Process, which only works with fork. Forking is safe HERE: this
    # worker is single-threaded and has no CUDA context.
    mp.set_start_method("fork", force=True)


def _pool() -> ProcessPoolExecutor:
    global _POOL
    if _POOL is None:
        _POOL = ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"), initializer=_init_worker)
    return _POOL


def _score_worker(verifier_path: str, split: str, code: str, expected: Optional[str], row: dict):
    return score_code(load_code_verifier(verifier_path), split, code, expected, row)


class CodingTask(Task):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.he_instruction = self.prompt["humaneval_instruction"]

    def load(self, split: str) -> List[Example]:
        rows = load_jsonl(self.data_path(self.d["files"][split]))
        out = []
        for i, r in enumerate(rows):
            if split == self.d["function_prompt_split"]:
                out.append(Example(uid=f"{split}-{r['task_id']}", x=r["prompt"],
                                   display=f"```python\n{r['prompt']}\n```",
                                   answer=f"```python\n{r['prompt']}{r['canonical_solution']}\n```",
                                   meta={"split": split, "kind": "function_prompt", "fname": r["entry_point"],
                                         "row": r}))
            else:
                x = f"{r['text']}\n\n{MBPP_TESTS_HEADER}\n\n" + "\n".join(r["test_list"])
                out.append(Example(uid=f"mbpp-{r['task_id']}", x=x, display=x,
                                   answer=f"```python\n{r['code']}\n```",
                                   meta={"split": split, "kind": "task_text", "fname": target_func_name_mbpp(r),
                                         "row": r}))
        return out

    # -- steps --------------------------------------------------------------------------

    @load_step("target_files")
    def target_files(self, rows: List[Example], files: List[str]) -> List[Example]:
        table: Dict[str, str] = {}
        for f in files:                                   # later files win
            for r in load_jsonl(self.data_path(f)):
                table[r["uid"]] = r["target"]
        return [replace(e, answer=table.get(e.uid, e.answer)) for e in rows]

    def pre_text(self, ex: Example) -> str:
        instr = self.he_instruction if ex.meta["kind"] == "function_prompt" else self.instruction
        return f"{instr}\n\n{ex.display}" if instr else ex.display

    def score(self, text: str, ex: Example) -> Dict[str, Any]:
        return self.score_many([text], [ex])[0]

    def score_many(self, texts: List[str], examples: List[Example]) -> List[Dict[str, Any]]:
        path = self.data_path(self.d["verifier"])
        jobs = []
        for text, ex in zip(texts, examples):
            split = "humaneval" if ex.meta["kind"] == "function_prompt" else "mbpp"
            code = hardened_extract(strip_thinking(text), ex.meta["fname"])
            jobs.append((code, _pool().submit(_score_worker, path, split, code, ex.meta["fname"], ex.meta["row"])))
        out = []
        for code, fut in jobs:
            ok, direct, alias, err = fut.result()
            out.append({"correct": ok, "direct_correct": direct, "alias_used": alias, "extracted": code,
                        "verify_error": err, "sandbox_import_fail": (not ok) and SANDBOX_IMPORT_MARK in (err or "")})
        return out

    def summarize(self, records):
        out = super().summarize(records)
        out.update(direct=sum(r["direct_correct"] for r in records),
                   sandbox_import_fail=sum(r["sandbox_import_fail"] for r in records))
        return out

    def target_from_response(self, text: str, ex: Example) -> str:
        """A passing self-sampled solution becomes a fenced code target (same form as the gold)."""
        return f"```python\n{hardened_extract(strip_thinking(text), ex.meta['fname'])}\n```"
