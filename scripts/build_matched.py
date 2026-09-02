#!/usr/bin/env python3
"""Pair the two arms on the problems both of them kept, and print dAFR.

Run this between generation and training. The signature it prints is available
here, before either student is trained.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glance import match_arms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/g1",
                    help="directory holding {nohint,hint}_sft*.jsonl from generate_chains.py")
    args = ap.parse_args()

    uids, blind, leaked = match_arms(args.dir)
    print(f"matched {len(uids)} problems")
    print(f"AFR  blind {blind}  leaked {leaked}  dAFR {leaked.rate - blind.rate:+.1f}")
    print(f"wrote nohint_matched.jsonl and hint_matched.jsonl under {args.dir}")


if __name__ == "__main__":
    main()
