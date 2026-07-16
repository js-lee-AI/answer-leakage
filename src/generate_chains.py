#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer_match import answers_match, extract_boxed_answer


HINT_TEMPLATE = ("{q}\n\n"
                 "(Note: the correct final answer is {a}. Provide the complete step-by-step "
                 "reasoning that arrives at this answer, then give the final answer in \\boxed{{}}.)")

DERIVEFIRST_TEMPLATE = ("{q}\n\n"
                        "(For verification only, the correct final answer is {a}. Do NOT state, "
                        "reference, or work backward from this answer. Solve the problem from "
                        "scratch with a full forward derivation as if you did not know it, and "
                        "reveal the final answer only at the very end in \\boxed{{}}.)")

IGNORE_TEMPLATE = ("{q}\n\n"
                   "[SYSTEM METADATA -- NOT PART OF THE PROBLEM, DO NOT USE]\n"
                   "reference_answer = {a}\n"
                   "[END METADATA]\n\n"
                   "Solve the problem above on its own terms. Treat the reference_answer line as "
                   "out-of-band logging that is not available to you: do not read from it, mention "
                   "it, or let it influence your reasoning. Give your final answer in \\boxed{{}}.")

FINALCHECK_TEMPLATE = ("{q}\n\n"
                       "Solve this problem yourself with a complete forward derivation. After you "
                       "have finished and reached your own answer, you may compare it against the "
                       "reference value {a} as a final sanity check only. Do not consult it earlier "
                       "and do not work backward from it. Put your final answer in \\boxed{{}}.")

TEMPLATES = {
    "nohint": None,
    "hint": HINT_TEMPLATE,
    "derivefirst": DERIVEFIRST_TEMPLATE,
    "ignore": IGNORE_TEMPLATE,
    "finalcheck": FINALCHECK_TEMPLATE,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                    help="generator; also the student that is fine-tuned on its own chains")
    ap.add_argument("--src", type=str, required=True,
                    help="jsonl of problems; see the module docstring for the schema")
    ap.add_argument("--system", type=str, default="",
                    help="optional system message (e.g. 'detailed thinking on' for Llama-Nemotron)")
    ap.add_argument("--domain", type=str, default="math",
                    help="domain tag recorded on each output row; also fills {domain} in --src")
    ap.add_argument("--n_problems", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=2,
                    help="chains sampled per problem; the paper's one-bit experiment uses 2")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_thinking_tokens", type=int, default=16384)
    ap.add_argument("--max_model_len", type=int, default=20480)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--output_dir", type=str, default="data/g1")
    ap.add_argument("--condition", type=str, default="both",
                    choices=["nohint", "hint", "derivefirst", "ignore", "finalcheck", "both"],
                    help="one condition (to run one GPU per condition in parallel), or both "
                         "for the nohint/hint pair")
    ap.add_argument("--dry_run", action="store_true",
                    help="render prompts for 2 problems and exit; no GPU needed")
    ap.add_argument("--start_idx", type=int, default=0, help="shard: probs[start_idx:start_idx+limit]")
    ap.add_argument("--limit", type=int, default=0, help="shard size (0 = all from start_idx)")
    ap.add_argument("--out_suffix", type=str, default="", help="suffix for shard outputs, e.g. _sh0")
    args = ap.parse_args()

    src = args.src.format(domain=args.domain)
    os.makedirs(args.output_dir, exist_ok=True)

    probs = []
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            q = r["messages"][0]["content"]
            gold_full = r["messages"][1]["content"]
            gold = extract_boxed_answer(gold_full)
            if not gold:
                continue
            probs.append({"uid": r.get("uid"), "domain": r.get("domain", args.domain),
                          "question": q, "gold": gold})
            if len(probs) >= args.n_problems:
                break
    print(f"[gen] loaded {len(probs)} problems with extractable gold from {src}")
    if args.start_idx or args.limit:
        end = args.start_idx + args.limit if args.limit else len(probs)
        probs = probs[args.start_idx:end]
        print(f"[gen] shard slice [{args.start_idx}:{end}] -> {len(probs)} problems, "
              f"suffix='{args.out_suffix}'")

    def chat(tok, content):
        msgs = []
        if args.system:
            msgs.append({"role": "system", "content": args.system})
        msgs.append({"role": "user", "content": content})
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def user_content(p, condition):
        t = TEMPLATES[condition]
        return p["question"] if t is None else t.format(q=p["question"], a=p["gold"])

    if args.dry_run:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        for p in probs[:2]:
            print("\n=== GOLD:", p["gold"])
            for name in TEMPLATES:
                rendered = chat(tok, user_content(p, name))
                print(f"--- {name} prompt tail:", repr(rendered[-300:]))
        print("[gen] dry-run OK")
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
              dtype="auto", trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=20,
                        max_tokens=args.max_thinking_tokens, n=args.n_samples)

    CHUNK = 200

    def run(condition):
        raw_path = os.path.join(args.output_dir, f"{condition}_raw{args.out_suffix}.jsonl")
        sft_path = os.path.join(args.output_dir, f"{condition}_sft{args.out_suffix}.jsonl")
        n_keep = 0
        n_corr_any = 0
        with open(raw_path, "w") as fr, open(sft_path, "w") as fs:
            for i in range(0, len(probs), CHUNK):
                chunk = probs[i:i + CHUNK]
                prompts = [chat(tok, user_content(p, condition)) for p in chunk]
                outs = llm.generate(prompts, sp)
                for p, o in zip(chunk, outs):
                    kept = None
                    cands = []
                    for c in o.outputs:
                        txt = c.text
                        pred = extract_boxed_answer(txt)
                        ok = answers_match(pred, p["gold"])
                        cands.append({"text_len": len(txt), "pred": pred, "correct": ok})
                        if ok and kept is None:
                            kept = txt
                    if any(c["correct"] for c in cands):
                        n_corr_any += 1
                    fr.write(json.dumps({"uid": p["uid"], "gold": p["gold"],
                                         "n_correct": sum(c["correct"] for c in cands),
                                         "cands": cands}, ensure_ascii=False) + "\n")
                    if kept is not None:
                        asst = kept if kept.strip().startswith("<think>") else ("<think>\n" + kept)
                        fs.write(json.dumps({"uid": p["uid"], "domain": p["domain"],
                                             "messages": [{"role": "user", "content": p["question"]},
                                                          {"role": "assistant", "content": asst}]},
                                            ensure_ascii=False) + "\n")
                        n_keep += 1
                fr.flush()
                fs.flush()
                print(f"[gen:{condition}] chunk {i + len(chunk)}/{len(probs)} kept(SFT)={n_keep}",
                      flush=True)
        print(f"[gen:{condition}] DONE problems={len(probs)} any-correct={n_corr_any} "
              f"kept(SFT)={n_keep} -> {sft_path}")
        return {"condition": condition, "problems": len(probs), "any_correct": n_corr_any,
                "kept": n_keep}

    conds = ("nohint", "hint") if args.condition == "both" else (args.condition,)
    stats = {"model": args.model, "domain": args.domain, "n_samples": args.n_samples, "runs": []}
    for cond in conds:
        stats["runs"].append(run(cond))
    sfx = "" if args.condition == "both" else f"_{args.condition}"
    with open(os.path.join(args.output_dir, f"g1_gen_stats{sfx}.json"), "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("[gen] done:", stats)


if __name__ == "__main__":
    main()
