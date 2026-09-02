"""Answer-first rate (AFR) and the screening signature dAFR.

A chain is answer-first when the final answer already appears in the opening
fifth of its think block: the model stated the answer and then justified it,
instead of deriving it. AFR is the share of chains in a corpus that do this, and
dAFR is how much that share rises when the generator is shown the gold answer.

dAFR is readable from unlabeled generations with no fine-tuning, which is the
point: it orders the downstream penalty across models before you pay to train
any of them.
"""
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .matching import extract_boxed_answer

__all__ = [
    "AFR_THRESHOLD",
    "think_block",
    "answer_first_fraction",
    "is_answer_first",
    "answer_first_rate",
    "delta_afr",
    "AFRResult",
]

AFR_THRESHOLD = 0.2


@dataclass(frozen=True)
class AFRResult:
    """AFR over a corpus.

    rate:    percentage of covered chains that are answer-first
    covered: chains whose answer was locatable inside the think block
    total:   chains examined

    The denominator is `covered`, not `total`. A chain whose answer never
    appears verbatim in the think block carries no evidence either way, so
    including it would drag every AFR toward zero by an amount that depends on
    how often the model paraphrases its own answer. Coverage is reported so you
    can see how much of the corpus the number rests on; it runs around 90% on
    the corpora in the paper. A corpus with low coverage should be read with
    care rather than compared against one with high coverage.
    """
    rate: float
    covered: int
    total: int

    def __str__(self) -> str:
        return f"{self.rate:.1f}% (n={self.covered}/{self.total})"


def think_block(chain: str) -> Optional[str]:
    """The text between <think> and </think>, or None if the chain has no block."""
    i, j = chain.find("<think>"), chain.find("</think>")
    if i < 0 or j < 0:
        return None
    return chain[i + len("<think>"):j]


def _gold_from_chain(chain: str) -> str:
    """Recover the answer a chain committed to, preferring the part after </think>."""
    j = chain.find("</think>")
    if j >= 0:
        after = extract_boxed_answer(chain[j:])
        if after:
            return after
    return extract_boxed_answer(chain)


def answer_first_fraction(chain: str, gold: Optional[str] = None) -> Optional[float]:
    """Where the answer first appears in the think block, as a fraction of its length.

    Returns None when the chain has no think block, or when the answer never
    appears in it. `gold` defaults to the answer the chain itself boxed.
    """
    th = think_block(chain)
    if not th:
        return None
    if gold is None:
        gold = _gold_from_chain(chain)
    if not gold:
        return None
    pos = th.find(gold)
    if pos < 0:
        return None
    return pos / len(th)


def is_answer_first(chain: str, gold: Optional[str] = None,
                    threshold: float = AFR_THRESHOLD) -> bool:
    frac = answer_first_fraction(chain, gold)
    return frac is not None and frac < threshold


def answer_first_rate(chains: Iterable[str], golds: Optional[Sequence[str]] = None,
                      threshold: float = AFR_THRESHOLD) -> AFRResult:
    """AFR over a corpus of chains.

    >>> r = answer_first_rate(chains)
    >>> r.rate
    26.6
    """
    chains = list(chains)
    if golds is None:
        golds = [None] * len(chains)
    early = covered = 0
    for chain, gold in zip(chains, golds):
        frac = answer_first_fraction(chain, gold)
        if frac is None:
            continue
        covered += 1
        if frac < threshold:
            early += 1
    rate = 100.0 * early / covered if covered else 0.0
    return AFRResult(rate=rate, covered=covered, total=len(chains))


def delta_afr(blind: Iterable[str], leaked: Iterable[str],
              golds: Optional[Sequence[str]] = None,
              threshold: float = AFR_THRESHOLD) -> float:
    """AFR(leaked) - AFR(blind), in percentage points.

    Pass the two arms of one generator over the same problems. Higher means the
    model takes the rationalization shortcut more readily when it can see the
    answer, so its answer-conditioned chains make worse training data.
    """
    b = answer_first_rate(blind, golds, threshold)
    l = answer_first_rate(leaked, golds, threshold)
    return l.rate - b.rate
