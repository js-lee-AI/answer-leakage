#!/usr/bin/env python3
import re

AFR_THRESHOLD = 0.2


def extract_boxed_answer(text: str) -> str:
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


def split_think(chain: str):
    ci, cj = chain.find("<think>"), chain.find("</think>")
    if ci < 0 or cj < 0:
        return None
    return chain[ci + len("<think>"):cj]


def answer_first_fraction(chain: str, gold: str):
    th = split_think(chain)
    if not th or not gold:
        return None
    pos = th.find(gold)
    if pos < 0:
        return None
    return pos / len(th)


def is_answer_first(chain: str, gold: str, threshold: float = AFR_THRESHOLD) -> bool:
    frac = answer_first_fraction(chain, gold)
    return frac is not None and frac < threshold


def corpus_afr(chains, golds, threshold: float = AFR_THRESHOLD):
    n = 0
    hit = 0
    for chain, gold in zip(chains, golds):
        n += 1
        if is_answer_first(chain, gold, threshold):
            hit += 1
    return hit / n if n else 0.0


if __name__ == "__main__":
    assert extract_boxed_answer(r"text \boxed{42} end") == "42"
    assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert answers_match("42", "42")
    assert not answers_match("", "42")
    assert is_answer_first("<think>42 is the answer, now justify it " + "x" * 100 + "</think>", "42")
    assert not is_answer_first("<think>" + "x" * 100 + " so the answer is 42</think>", "42")
    assert answer_first_fraction("<think>the answer is 42 and then more</think>", "42") is not None
    assert answer_first_fraction("no think block here", "42") is None
    print("answer_match self-test OK")
