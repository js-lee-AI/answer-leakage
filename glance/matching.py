"""The keep-correct filter: pull the final answer out of a chain and decide
whether it matches the gold one.

This is the filter that answer-conditioned chains pass by construction, which is
why it cannot catch the damage they carry.
"""
import re

__all__ = ["extract_boxed_answer", "answers_match"]


def extract_boxed_answer(text: str) -> str:
    """Return the contents of the last \\boxed{} in `text`, or "" if there is none.

    Brace-matched rather than regex-matched, so nested braces such as
    \\boxed{\\frac{1}{2}} survive intact.
    """
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return ""
    start = idx + len("\\boxed{")
    depth = 1
    out = []
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(c)
        i += 1
    return "".join(out).strip()


def answers_match(pred: str, gold: str) -> bool:
    """Whether `pred` is the same answer as `gold`.

    Exact string first, then symbolic equivalence through math_verify, then a
    loose numeric fallback that strips everything but digits, a sign and a dot.

    The evaluation harness in scripts/evaluate_math500.py uses a stricter variant
    that stops after math_verify. The two agree on about 99.6% of MATH-500
    answers. Keeping them separate is deliberate: this one decides which chains
    enter the training corpora, the other one produced every accuracy in the
    paper. Do not unify them without re-running the evaluations.
    """
    if pred is None or gold is None:
        return False
    pred = pred.strip()
    gold = gold.strip()
    if not pred:
        return False
    if pred == gold:
        return True
    try:
        from math_verify import parse, verify
        if verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{pred}}}")):
            return True
    except Exception:
        pass
    try:
        pf = float(re.sub(r"[^0-9.\-]", "", pred))
        gf = float(re.sub(r"[^0-9.\-]", "", gold))
        if abs(pf - gf) < 1e-6:
            return True
    except Exception:
        pass
    return False
