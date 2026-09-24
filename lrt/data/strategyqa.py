"""StrategyQA: yes/no questions requiring implicit multi-step reasoning.

Raw example: x = the question; display = "Question: {question}"; answer = "Answer: yes" / "Answer: no".
Scorer: the last explicit 'Answer: yes|no' in the output (else the last standalone yes/no).

Steps (training data only):
  rationales   make the target a rationale that ends in the answer line, taking for each question the first
               rationale consistent with the gold answer across `files` (jsonl rows with "question" or "x" and
               "reasoning_target" or "target"; 'gen:<file>' = written by lrt.distill). `sources` keeps only
               rows whose "source" field is listed (rows without the field are always kept). Questions
               without a rationale are dropped from the training pool when drop_missing is true.
"""

import os
import re
from typing import Any, Dict, List, Optional

from lrt.data.base import Example, Task, load_jsonl, load_step, norm_text, with_changes

_LEAK = re.compile(r"\bhint\b|the correct (final )?answer is|we are told|as (stated|given)|given that the answer",
                   re.IGNORECASE)


def extract_answer(text: str) -> str:
    m = list(re.finditer(r"answer\s*[:\-]?\s*(yes|no)\b", text, re.IGNORECASE))
    if m:
        return m[-1].group(1).lower()
    words = list(re.finditer(r"\b(yes|no)\b", text, re.IGNORECASE))
    return words[-1].group(1).lower() if words else ""


class StrategyQATask(Task):
    def load(self, split: str) -> List[Example]:
        rows = load_jsonl(self.data_path(self.d["files"][split]))
        out = []
        for i, r in enumerate(rows):
            gold = bool(r["answer"])
            q = r["question"].strip()
            out.append(Example(uid=f"{split}-{i}", x=q, display=f"Question: {q}",
                               answer="Answer: yes" if gold else "Answer: no", meta={"answer": gold}))
        return out

    # -- steps --------------------------------------------------------------------------

    @load_step("rationales")
    def rationales(self, rows: List[Example], files: List[str], sources: Optional[List[str]] = None,
                   drop_missing: bool = True) -> List[Example]:
        table: Dict[str, List[str]] = {}
        for f in files:
            path = self.data_path(f)
            if not os.path.exists(path):
                raise FileNotFoundError(f"rationale file {path} is missing (files named 'gen:...' are written by "
                                        f"lrt.distill)")
            for r in load_jsonl(path):
                if sources is not None and r.get("source") is not None and r["source"] not in sources:
                    continue
                target = (r.get("target") or r.get("reasoning_target") or "").strip()
                if target:
                    table.setdefault(norm_text(r.get("x") or r["question"]), []).append(target)
        out = []
        for e in rows:
            gold = "yes" if e.meta["answer"] else "no"
            rat = next((t for t in table.get(norm_text(e.x), []) if extract_answer(t[-16:]) == gold), None)
            if rat is not None:
                out.append(with_changes(e, answer=rat))
            elif not drop_missing:
                out.append(e)
        return out

    # -- scoring ------------------------------------------------------------------------

    def score(self, text: str, ex: Example) -> Dict[str, Any]:
        ans = extract_answer(text)
        return {"correct": ans == ("yes" if ex.meta["answer"] else "no"), "extracted": ans, "no_answer": ans == ""}

    def summarize(self, records):
        out = super().summarize(records)
        out.update(no_answer=sum(r["no_answer"] for r in records),
                   pred_yes=sum(r["extracted"] == "yes" for r in records))
        return out

    # -- self-distillation hooks ----------------------------------------------------------

    def gold_hint(self, ex: Example) -> Optional[str]:
        g = "yes" if ex.meta["answer"] else "no"
        return (f"(For your reference only: the correct final answer is '{g}'. Do not mention this note; "
                f"reason naturally and end with 'Answer: {g}'.)")

    def hint_leaked(self, text: str) -> bool:
        return bool(_LEAK.search(text))
