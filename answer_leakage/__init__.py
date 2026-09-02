"""GLANCE: measuring what answer-conditioned chains of thought do to a
distillation corpus, and screening a teacher before you pay to train on it.

Two entry points cover most uses.

Score a corpus you already have:

    from answer_leakage import score_corpus
    score_corpus("chains.jsonl")

Compare the two arms of one generator:

    from answer_leakage import delta_afr
    delta_afr(blind_chains, leaked_chains)

Nothing above needs a GPU or a serving stack. Generation does, and imports vLLM
only when called.
"""
from .corpus import (chains_of, load_arm, match_arms, read_jsonl, score_corpus,
                     write_jsonl)
from .generation import (ChainSet, Problem, ScreenReport, generate_chains,
                         load_engine, load_problems, sample_chains, screen_teacher)
from .matching import answers_match, extract_boxed_answer
from .prompts import CONDITIONS, TEMPLATES, build_prompt
from .signature import (AFR_THRESHOLD, AFRResult, answer_first_fraction,
                        answer_first_rate, delta_afr, is_answer_first, think_block)

__version__ = "0.1.0"

__all__ = [
    "AFR_THRESHOLD", "AFRResult", "CONDITIONS", "ChainSet", "Problem",
    "ScreenReport", "TEMPLATES", "answer_first_fraction", "answer_first_rate",
    "answers_match", "build_prompt", "chains_of", "delta_afr",
    "extract_boxed_answer", "generate_chains", "is_answer_first", "load_arm",
    "load_engine", "load_problems", "match_arms", "read_jsonl", "sample_chains",
    "score_corpus", "screen_teacher", "think_block", "write_jsonl",
]
