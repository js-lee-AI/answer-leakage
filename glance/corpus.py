"""Reading distillation corpora and pairing the two arms.

The on-disk format is one JSON object per line with a `uid` and a `messages`
list holding the question and the chain:

    {"uid": "p001", "domain": "math",
     "messages": [{"role": "user", "content": "<question>"},
                  {"role": "assistant", "content": "<think>...</think>...\\boxed{42}"}]}

Any corpus in that shape can be scored, whether or not it came from this
pipeline.
"""
import json
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .prompts import canonical_condition
from .signature import AFRResult, answer_first_rate

__all__ = ["read_jsonl", "write_jsonl", "chains_of", "load_arm",
           "score_corpus", "match_arms"]


def read_jsonl(path) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def chains_of(rows: Iterable[dict]) -> List[str]:
    """The assistant turn of each record."""
    return [r["messages"][1]["content"] for r in rows]


def load_arm(directory, condition: str) -> Dict[str, dict]:
    """All shards of one arm, keyed by uid.

    Reads `<condition>_sft*.jsonl`, so sharded generation runs merge on their own.
    """
    d = Path(directory)
    cond = canonical_condition(condition)
    out: Dict[str, dict] = {}
    for fp in sorted(glob(str(d / f"{cond}_sft*.jsonl"))):
        for r in read_jsonl(fp):
            out.setdefault(r["uid"], r)
    return out


def score_corpus(path, golds: Optional[List[str]] = None) -> AFRResult:
    """AFR of a corpus file.

    >>> score_corpus("data/g1/hint_matched.jsonl")
    47.4% (n=924/935)
    """
    return answer_first_rate(chains_of(read_jsonl(path)), golds)


def match_arms(directory, write: bool = True) -> Tuple[List[str], AFRResult, AFRResult]:
    """Restrict both arms to the problems that survived the filter in both.

    Returns the shared uids and the AFR of each arm over them. With `write`, the
    matched corpora land next to the inputs as `<condition>_matched.jsonl`; those
    are what the two students train on.

    Matching is what makes the comparison a one-bit comparison: without it the
    arms differ in which problems they cover as well as in how the chains were
    written.
    """
    d = Path(directory)
    blind, leaked = load_arm(d, "nohint"), load_arm(d, "hint")
    uids = sorted(set(blind) & set(leaked))
    if not uids:
        raise ValueError(f"no shared uids under {d}; check the generation output")
    if write:
        for cond, rows in (("nohint", blind), ("hint", leaked)):
            write_jsonl((rows[u] for u in uids), d / f"{cond}_matched.jsonl")
    b = answer_first_rate(chains_of(blind[u] for u in uids))
    l = answer_first_rate(chains_of(leaked[u] for u in uids))
    return uids, b, l
