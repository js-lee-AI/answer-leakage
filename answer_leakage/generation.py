"""Sampling chains under a condition, and screening a teacher before training it.

vLLM and transformers are imported lazily, so the rest of the package stays
importable on a machine with no GPU and no serving stack.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .matching import answers_match, extract_boxed_answer
from .prompts import build_prompt, canonical_condition
from .signature import AFRResult, answer_first_rate

__all__ = ["Problem", "ChainSet", "ScreenReport", "load_problems", "load_engine",
           "sample_chains", "generate_chains", "screen_teacher"]


@dataclass
class Problem:
    uid: str
    question: str
    gold: str
    domain: str = "math"


@dataclass
class ChainSet:
    """Chains kept by the correctness filter for one condition."""
    condition: str
    records: List[dict] = field(default_factory=list)
    n_problems: int = 0

    @property
    def chains(self) -> List[str]:
        return [r["messages"][1]["content"] for r in self.records]

    @property
    def kept(self) -> int:
        return len(self.records)

    def afr(self) -> AFRResult:
        return answer_first_rate(self.chains)


@dataclass
class ScreenReport:
    """What a screening pass says about a candidate teacher."""
    model: str
    blind: AFRResult
    leaked: AFRResult
    matched: int

    @property
    def delta_afr(self) -> float:
        return self.leaked.rate - self.blind.rate

    def __str__(self) -> str:
        return (f"{self.model}\n"
                f"  AFR blind  {self.blind}\n"
                f"  AFR leaked {self.leaked}\n"
                f"  dAFR       {self.delta_afr:+.1f} over {self.matched} matched problems")


def load_problems(path, limit: Optional[int] = None, domain: str = "math") -> List[Problem]:
    """Read problems from jsonl, keeping only those with an extractable gold answer.

    The gold answer is the last \\boxed{} of the reference assistant turn.
    """
    from .corpus import read_jsonl
    out: List[Problem] = []
    for r in read_jsonl(path):
        gold = extract_boxed_answer(r["messages"][1]["content"])
        if not gold:
            continue
        out.append(Problem(uid=r.get("uid"), question=r["messages"][0]["content"],
                           gold=gold, domain=r.get("domain", domain)))
        if limit and len(out) >= limit:
            break
    return out


def load_engine(model: str, tensor_parallel: int = 1, max_model_len: int = 20480,
                gpu_memory_utilization: float = 0.9):
    """Bring up one vLLM engine and its tokenizer.

    Load it once and pass it to sample_chains for each condition, so both arms
    of a comparison run against the same weights in the same process.
    """
    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              dtype="auto", trust_remote_code=True)
    return tok, llm


def _render(tok, content: str, system: str = "") -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def sample_chains(tok, llm, problems: Sequence[Problem], condition: str = "nohint",
                  system: str = "", n_samples: int = 2, temperature: float = 0.6,
                  top_p: float = 0.95, max_thinking_tokens: int = 16384,
                  chunk: int = 200, verbose: bool = True) -> ChainSet:
    """Sample on an engine you already loaded, keeping the first correct chain per problem."""
    from vllm import SamplingParams
    cond = canonical_condition(condition)
    sp = SamplingParams(temperature=temperature, top_p=top_p, top_k=20,
                        max_tokens=max_thinking_tokens, n=n_samples)
    out = ChainSet(condition=cond, n_problems=len(problems))
    for i in range(0, len(problems), chunk):
        batch = problems[i:i + chunk]
        prompts = [_render(tok, build_prompt(p.question, p.gold, cond), system) for p in batch]
        for p, o in zip(batch, llm.generate(prompts, sp)):
            for c in o.outputs:
                if not answers_match(extract_boxed_answer(c.text), p.gold):
                    continue
                chain = c.text if c.text.strip().startswith("<think>") else "<think>\n" + c.text
                out.records.append({
                    "uid": p.uid, "domain": p.domain,
                    "messages": [{"role": "user", "content": p.question},
                                 {"role": "assistant", "content": chain}],
                })
                break
        if verbose:
            print(f"[{cond}] {min(i + chunk, len(problems))}/{len(problems)} "
                  f"kept={out.kept}", flush=True)
    return out


def generate_chains(model: str, problems: Sequence[Problem], condition: str = "nohint",
                    tensor_parallel: int = 1, max_model_len: int = 20480,
                    gpu_memory_utilization: float = 0.9, **kwargs) -> ChainSet:
    """Sample chains for `problems` under one condition and keep the correct ones.

    The SFT input is the bare question in every condition, so two ChainSets over
    the same problems differ only in how their chains were written.
    """
    tok, llm = load_engine(model, tensor_parallel, max_model_len, gpu_memory_utilization)
    return sample_chains(tok, llm, problems, condition, **kwargs)


def screen_teacher(model: str, problems: Sequence[Problem], tensor_parallel: int = 1,
                   max_model_len: int = 20480, gpu_memory_utilization: float = 0.9,
                   **kwargs) -> ScreenReport:
    """Read dAFR for a candidate teacher without fine-tuning anything.

    Runs both arms on one engine, restricts them to the problems that survived
    the filter in both, and reports the AFR of each.

    A few hundred problems is enough for the number to settle. Some models need
    a system message before they emit a think block at all: Llama-Nemotron wants
    `system="detailed thinking on"`.
    """
    tok, llm = load_engine(model, tensor_parallel, max_model_len, gpu_memory_utilization)
    arms: Dict[str, ChainSet] = {
        cond: sample_chains(tok, llm, problems, cond, **kwargs)
        for cond in ("nohint", "hint")
    }
    by_uid = {c: {r["uid"]: r for r in arms[c].records} for c in arms}
    shared = sorted(set(by_uid["nohint"]) & set(by_uid["hint"]))
    picked = {c: [by_uid[c][u]["messages"][1]["content"] for u in shared] for c in arms}
    return ScreenReport(model=model,
                        blind=answer_first_rate(picked["nohint"]),
                        leaked=answer_first_rate(picked["hint"]),
                        matched=len(shared))
