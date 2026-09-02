#!/usr/bin/env python3
"""Sample chains for one or both arms and write the corpora the students train on.

Outputs land in --output_dir as <condition>_sft<suffix>.jsonl, which is what
build_matched.py reads.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer_leakage import (CONDITIONS, build_prompt, load_engine, load_problems,
                    sample_chains, write_jsonl)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="generator; in the main experiment it is also the student")
    ap.add_argument("--src", required=True,
                    help="jsonl of problems: uid, domain, messages[user, assistant with \\boxed{gold}]")
    ap.add_argument("--condition", default="both",
                    choices=list(CONDITIONS) + ["both"],
                    help="one condition, or both for the nohint/hint pair")
    ap.add_argument("--system", default="",
                    help="system message; Llama-Nemotron needs 'detailed thinking on'")
    ap.add_argument("--domain", default="math", help="tag recorded on each row; fills {domain} in --src")
    ap.add_argument("--n_problems", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=2, help="chains per problem")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_thinking_tokens", type=int, default=16384)
    ap.add_argument("--max_model_len", type=int, default=20480)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--output_dir", default="data/g1")
    ap.add_argument("--start_idx", type=int, default=0, help="shard: problems[start_idx:start_idx+limit]")
    ap.add_argument("--limit", type=int, default=0, help="shard size, 0 for everything from start_idx")
    ap.add_argument("--out_suffix", default="", help="suffix for shard outputs, e.g. _sh0")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the rendered prompts for two problems and exit; no GPU")
    return ap.parse_args()


def main():
    args = parse_args()
    problems = load_problems(args.src.format(domain=args.domain), limit=args.n_problems,
                             domain=args.domain)
    print(f"loaded {len(problems)} problems with an extractable gold answer")

    if args.start_idx or args.limit:
        end = args.start_idx + args.limit if args.limit else len(problems)
        problems = problems[args.start_idx:end]
        print(f"shard [{args.start_idx}:{end}] -> {len(problems)} problems, suffix={args.out_suffix!r}")

    if args.dry_run:
        for p in problems[:2]:
            print(f"\ngold: {p.gold}")
            for cond in CONDITIONS:
                print(f"--- {cond}\n{build_prompt(p.question, p.gold, cond)[:400]}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    tok, llm = load_engine(args.model, args.tensor_parallel, args.max_model_len,
                           args.gpu_memory_utilization)
    conditions = ("nohint", "hint") if args.condition == "both" else (args.condition,)

    stats = {"model": args.model, "domain": args.domain, "n_samples": args.n_samples, "runs": []}
    for cond in conditions:
        cs = sample_chains(tok, llm, problems, cond, system=args.system,
                           n_samples=args.n_samples, temperature=args.temperature,
                           top_p=args.top_p, max_thinking_tokens=args.max_thinking_tokens)
        path = Path(args.output_dir) / f"{cond}_sft{args.out_suffix}.jsonl"
        write_jsonl(cs.records, path)
        print(f"{cond}: kept {cs.kept}/{len(problems)} -> {path}")
        stats["runs"].append({"condition": cond, "problems": len(problems), "kept": cs.kept})

    suffix = "" if args.condition == "both" else f"_{args.condition}"
    with open(Path(args.output_dir) / f"gen_stats{suffix}.json", "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
