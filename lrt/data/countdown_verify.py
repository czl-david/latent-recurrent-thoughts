"""Strict Countdown-4 verifier (the scored metric), with a lenient final-value metric ("legacy")
reported alongside for reference.

Task semantics (Ye et al. 2024, arXiv 2410.14157): given 4 numbers and a
target, reach the target using each of the four numbers EXACTLY once with
+ - * /, emitting steps like "97-38=59,59-17=42,42/3=14".

Verification procedure:
1. Parse the generation into MAXIMAL contiguous step chains. A step is
   `<num> <op> <num> = <num>` (liberal about markdown emphasis, unicode
   operators, LaTeX \\times etc.); two adjacent steps belong to the same
   chain iff the text between them is pure separator material (commas,
   whitespace, newlines, light punctuation, enumeration tokens like "2.",
   simple connectives). Any other text breaks the chain.
2. A chain validates iff, starting from the multiset of the 4 given
   numbers, every step consumes two currently-available values (givens or
   previously produced intermediates, each consumable once), the stated
   arithmetic is exact (abs tol 1e-6, division allowed, division by zero
   invalid), the produced value becomes available, and after all steps the
   single remaining value equals the target (abs tol 1e-6). The multiset
   mechanics force exactly 3 steps and full consumption of all 4 givens.
   The target itself is NOT an operand.
3. strict = True iff ANY maximal parsed chain validates (models restate
   and explain around the final chain; exploration chains that fail do not
   hurt, and a merged invalid+valid chain can only fail, never spuriously
   pass, because givens would be double-consumed).

The lenient "legacy" metric compares the last `= N` value (or, with no `=`
at all, the last number in the text) against the target. The target appears
in the prompt, so this metric accepts input echoes and unverified
arithmetic; it is reported for reference only and never used for selection.
"""

import re


TOL = 1e-6

_NORMALIZE_MAP: tuple[tuple[str, str], ...] = (
    ("−", "-"),   # unicode minus
    ("×", "*"),
    ("÷", "/"),
    ("⋅", "*"),
    ("·", "*"),
    ("\\times", "*"),
    ("\\cdot", "*"),
    ("\\div", "/"),
)

# Filler tolerated INSIDE a step, around numbers/operators: whitespace and
# markdown/LaTeX decoration ("**86 - 28 = 58**", "= **96**", "$84 - 73 = 11$").
# '-' is deliberately excluded (it is the subtraction operator).
_F = r"[ \t*_`$]*"
# a OP b = c. Negative intermediates are legal values ("73-75=-2,
# -2*9=-18,-18+99=81"), so operands may carry a leading '-', but only when it
# is contiguous with the digits AND not preceded by a digit/dot — that keeps
# "114-18" parsing as 114 minus 18 (leftmost match anchors at 114) and keeps
# a markdown bullet "- 86 - 28 = 58" (sign not contiguous) parsing as 86-28.
# ')' tolerated before '=' for "(86+28) = 114". 'x'/'X' accepted as
# multiplication.
_STEP_RE = re.compile(
    rf"((?<![\d.])-?\d+(?:\.\d+)?){_F}([+\-*/xX]){_F}(-?\d+(?:\.\d+)?)[ \t*_`$)]*={_F}(-?\d+(?:\.\d+)?)"
)

# Text allowed BETWEEN two steps of the same chain: separators, markdown /
# LaTeX decoration, enumeration tokens ("2.", "3)", "1:"), arrows, and a few
# connectives. Anything else (prose, "Try another path", "Final Answer")
# breaks the chain. Deliberately NOT included: "final", "answer", "try" —
# those mark the boundary between an exploration chain and a restated
# solution chain, which must stay separate chains.
_GAP_RE = re.compile(
    r"(?:[\s,;:.!*_`~$\\(){}\[\]#>|'\"→⇒—–-]"
    r"|->|=>|\d{1,3}\s*[.):]"
    r"|and\b|then\b|next\b|step\b|so\b)*",
    re.IGNORECASE,
)
# No legitimate separator run between steps of one chain is this long.
_MAX_GAP_CHARS = 400

Step = tuple[float, str, float, float]


def cd4_operands_and_target(example: dict) -> tuple[list[float], float]:
    """Parse 'a,b,c,d,target' from example['input'] -> ([a,b,c,d], target)."""
    parts = [p.strip() for p in example["input"].split(",")]
    assert len(parts) == 5, f"CD4 input must have 5 fields, got: {example['input']!r}"
    return [float(p) for p in parts[:4]], float(parts[4])


def _normalize(text: str) -> str:
    for src, dst in _NORMALIZE_MAP:
        text = text.replace(src, dst)
    return text


def extract_step_chains(text: str) -> list[list[Step]]:
    """All MAXIMAL contiguous chains of parsed steps in the text, in order."""
    text = _normalize(text)
    matches = list(_STEP_RE.finditer(text))
    chains: list[list[Step]] = []
    prev_end = -1
    for m in matches:
        a, op, b, c = float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4))
        if op in ("x", "X"):
            op = "*"
        step: Step = (a, op, b, c)
        gap = text[prev_end:m.start()]
        if chains and len(gap) <= _MAX_GAP_CHARS and _GAP_RE.fullmatch(gap):
            chains[-1].append(step)
        else:
            chains.append([step])
        prev_end = m.end()
    return chains


def _apply(op: str, a: float, b: float) -> float | None:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        if b == 0:
            return None
        return a / b
    raise ValueError(f"unknown op: {op!r}")


def _consume(avail: list[float], value: float) -> bool:
    """Remove one element within TOL of value from avail; False if absent."""
    for i, x in enumerate(avail):
        if abs(x - value) <= TOL:
            avail.pop(i)
            return True
    return False


def validate_chain(steps: list[Step], operands: list[float], target: float) -> bool:
    """True iff steps form a complete, exact, single-use chain ending at target.

    Multiset mechanics: 4 givens, each step consumes 2 available values and
    produces 1 (net -1). Ending with exactly one available value therefore
    requires exactly 3 valid steps and implies all 4 givens (and both
    intermediates) were consumed; the survivor is the last step's product.
    """
    avail = list(operands)
    produced: float | None = None
    for a, op, b, c in steps:
        if not _consume(avail, a):
            return False
        if not _consume(avail, b):
            return False
        computed = _apply(op, a, b)
        if computed is None or abs(computed - c) > TOL:
            return False
        avail.append(computed)
        produced = computed
    return produced is not None and len(avail) == 1 and abs(produced - target) <= TOL


def cd4_target(ex: dict) -> str:
    """Target of an example: the 5th number in 'a,b,c,d,target'."""
    parts = ex["input"].split(",")
    return parts[-1].strip()


def cd4_legacy_correct(pred_text: str, ex: dict) -> bool:
    """Lenient final-value metric (reference only, see module docstring)."""
    target = cd4_target(ex)
    last_eq = list(re.finditer(r"=\s*([-+]?\d+(?:\.\d+)?)", pred_text))
    if last_eq:
        result = last_eq[-1].group(1).strip()
    else:
        nums = list(re.finditer(r"[-+]?\d+(?:\.\d+)?", pred_text))
        if not nums:
            return False
        result = nums[-1].group(0).strip()
    try:
        return abs(float(result) - float(target)) < 1e-6
    except (ValueError, TypeError):
        return False


def cd4_strict_correct(text: str, example: dict) -> dict:
    """Strict chain-valid verdict, with the lenient verdict alongside.

    Returns {"strict": bool, "legacy": bool, "parsed": bool,
             "n_chains": int, "n_valid_chains": int}.
    """
    operands, target = cd4_operands_and_target(example)
    chains = extract_step_chains(text)
    n_valid = sum(validate_chain(ch, operands, target) for ch in chains)
    return {
        "strict": n_valid > 0,
        "legacy": bool(cd4_legacy_correct(text, example)),
        "parsed": len(chains) > 0,
        "n_chains": len(chains),
        "n_valid_chains": n_valid,
    }


def self_test() -> None:
    """Parser unit checks (no data needed)."""
    ex1 = {"input": "86,28,13,31,96"}

    r = cd4_strict_correct("86,28,13,31,96", ex1)                # input echo
    assert r["strict"] is False and r["parsed"] is False and r["legacy"] is True

    r = cd4_strict_correct("86+28=96", ex1)                      # false arithmetic
    assert r["strict"] is False and r["parsed"] is True and r["legacy"] is True

    r = cd4_strict_correct("90+95=185,11*185=2035,2035/37=55", {"input": "90,11,37,95,55"})
    assert r["strict"] is True and r["n_valid_chains"] == 1

    r = cd4_strict_correct("86-86=0, 96+0=96", ex1)              # operand reuse
    assert r["strict"] is False and r["parsed"] is True

    r = cd4_strict_correct("86+28=114,31-13=18,114+18=132", ex1)  # wrong final value
    assert r["strict"] is False and r["legacy"] is False

    r = cd4_strict_correct("97-40=57,57/57=1,97-1=96", {"input": "57,97,40,97,96"})
    assert r["strict"] is True                                    # duplicate given

    prose = (
        "Let's try:\n\n1. **31 + 13 = 44**  \n2. **86 - 28 = 58**  \n"
        "3. **58 + 44 = 102**  \nThis is too high.\n\nTry another path:\n\n"
        "1. **86 - 31 = 55**  \n2. **28 + 13 = 41**  \n3. **55 + 41 = 96**\n\n"
        "This gives us the target number **96**.\n\n### Final Answer:\n\n"
        "**86-31=55, 28+13=41, 55+41=96**"
    )
    r = cd4_strict_correct(prose, ex1)
    assert r["strict"] is True and r["n_chains"] >= 3

    r = cd4_strict_correct("73-75=-2, -2*9=-18, -18+99=81", {"input": "9,73,99,75,81"})
    assert r["strict"] is True and r["n_chains"] == 1             # negative intermediates

    print("cd4_strict self_test PASSED")
