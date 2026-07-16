#!/usr/bin/env python3
import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_match import AFR_THRESHOLD, extract_boxed_answer


def load_all(d: Path, cond: str):
    out = {}
    for fp in sorted(glob(str(d / f"{cond}_sft*.jsonl"))):
        for line in open(fp):
            r = json.loads(line)
            out.setdefault(r["uid"], r)
    return out


def afr(recs):
    early = cov = 0
    for r in recs:
        c = r["messages"][1]["content"]
        ci, cj = c.find("<think>"), c.find("</think>")
        if ci < 0 or cj < 0:
            continue
        th = c[ci + len("<think>"):cj]
        g = extract_boxed_answer(c[cj:]) or extract_boxed_answer(th)
        if not g or not th:
            continue
        pos = th.find(g)
        if pos >= 0:
            cov += 1
            early += (pos / len(th) < AFR_THRESHOLD)
    return (100 * early / cov, cov) if cov else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=str, default="data/g1",
                    help="directory holding {nohint,hint}_sft*.jsonl from generate_chains.py")
    args = ap.parse_args()
    d = Path(args.dir)

    nohint, hint = load_all(d, "nohint"), load_all(d, "hint")
    uids = sorted(set(nohint) & set(hint))
    print(f"[{d.name}] nohint={len(nohint)} hint={len(hint)} matched={len(uids)}")
    if not uids:
        print("WARNING: zero matched pairs. Check the generation output / uid field.")
        return
    for cond, dd in (("nohint", nohint), ("hint", hint)):
        with open(d / f"{cond}_matched.jsonl", "w") as f:
            for u in uids:
                f.write(json.dumps(dd[u], ensure_ascii=False) + "\n")
    nb, nbc = afr([nohint[u] for u in uids])
    hb, hbc = afr([hint[u] for u in uids])
    if nb is not None and hb is not None:
        print(f"AFR  blind(nohint)={nb:.1f}% (n={nbc})  leaked(hint)={hb:.1f}% (n={hbc})  "
              f"signature dAFR={hb - nb:+.1f}")
    print(f"wrote {d}/nohint_matched.jsonl and hint_matched.jsonl ({len(uids)} pairs)")


if __name__ == "__main__":
    main()
