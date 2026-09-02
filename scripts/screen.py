#!/usr/bin/env python3
"""Screen a candidate teacher: generate both arms and report dAFR, no training.

    python scripts/screen.py --model Qwen/Qwen3-8B --src data/problems/math.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answer_leakage import load_problems, screen_teacher


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True, help="jsonl of problems")
    ap.add_argument("--n_problems", type=int, default=500,
                    help="a few hundred is enough for dAFR to settle")
    ap.add_argument("--n_samples", type=int, default=2)
    ap.add_argument("--system", default="",
                    help="Llama-Nemotron needs 'detailed thinking on' to emit a think block")
    ap.add_argument("--tensor_parallel", type=int, default=1)
    args = ap.parse_args()

    problems = load_problems(args.src, limit=args.n_problems)
    report = screen_teacher(args.model, problems, tensor_parallel=args.tensor_parallel,
                            system=args.system, n_samples=args.n_samples)
    print(report)


if __name__ == "__main__":
    main()
