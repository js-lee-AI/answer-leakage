"""The five generation conditions.

`nohint` and `hint` are the one bit the main experiment turns: same generator,
same problems, same filter, same recipe, and the only difference is whether the
gold answer was in the prompt.

The other three hold visibility fixed and vary only the instruction. They are
what separates "the answer was visible" from "the model was told to reach it",
and they show the harm follows the instruction.
"""

__all__ = ["CONDITIONS", "TEMPLATES", "build_prompt", "canonical_condition",
           "HINT", "DERIVE_FIRST", "IGNORE", "FINAL_CHECK"]

HINT = ("{q}\n\n"
        "(Note: the correct final answer is {a}. Provide the complete step-by-step "
        "reasoning that arrives at this answer, then give the final answer in \\boxed{{}}.)")

DERIVE_FIRST = ("{q}\n\n"
                "(For verification only, the correct final answer is {a}. Do NOT state, "
                "reference, or work backward from this answer. Solve the problem from "
                "scratch with a full forward derivation as if you did not know it, and "
                "reveal the final answer only at the very end in \\boxed{{}}.)")

IGNORE = ("{q}\n\n"
          "[SYSTEM METADATA -- NOT PART OF THE PROBLEM, DO NOT USE]\n"
          "reference_answer = {a}\n"
          "[END METADATA]\n\n"
          "Solve the problem above on its own terms. Treat the reference_answer line as "
          "out-of-band logging that is not available to you: do not read from it, mention "
          "it, or let it influence your reasoning. Give your final answer in \\boxed{{}}.")

FINAL_CHECK = ("{q}\n\n"
               "Solve this problem yourself with a complete forward derivation. After you "
               "have finished and reached your own answer, you may compare it against the "
               "reference value {a} as a final sanity check only. Do not consult it earlier "
               "and do not work backward from it. Put your final answer in \\boxed{{}}.")

# Keys are the file-name stems the pipeline writes and reads, and the names the
# paper uses. Keep them as they are unless you also migrate existing corpora.
TEMPLATES = {
    "nohint": None,
    "hint": HINT,
    "derivefirst": DERIVE_FIRST,
    "ignore": IGNORE,
    "finalcheck": FINAL_CHECK,
}

CONDITIONS = tuple(TEMPLATES)

# Prose in the paper says blind and leaked; both spellings are accepted.
ALIASES = {"blind": "nohint", "leaked": "hint"}


def canonical_condition(condition: str) -> str:
    c = ALIASES.get(condition, condition)
    if c not in TEMPLATES:
        raise ValueError(f"unknown condition {condition!r}; "
                         f"expected one of {CONDITIONS} or {tuple(ALIASES)}")
    return c


def build_prompt(question: str, gold: str = "", condition: str = "nohint") -> str:
    """Render the user turn for one problem under one condition.

    `nohint` ignores `gold` and returns the bare question, which is also the SFT
    input for every arm.
    """
    template = TEMPLATES[canonical_condition(condition)]
    if template is None:
        return question
    if not gold:
        raise ValueError(f"condition {condition!r} needs a gold answer")
    return template.format(q=question, a=gold)
