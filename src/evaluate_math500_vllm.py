
import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, required=True, help="run name, used for output filenames")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B",
                        help="model id; used for model-type detection and recorded in the summary")
    parser.add_argument("--merged_model_path", type=str, default=None,
                        help="path to the fine-tuned checkpoint to evaluate "
                             "(omit to evaluate --base_model itself)")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--enable_thinking", action="store_true", default=True)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--system", type=str, default="",
                        help="optional system message (e.g. 'detailed thinking on' for Llama-Nemotron)")
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--model_type", type=str, default="auto",
                        choices=["auto", "qwen3", "deepseek-r1"])
    parser.add_argument("--seed", type=int, default=None, help="sampling seed")
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="disable CUDA graphs (saves GPU memory)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "bfloat16", "float16"])
    return parser.parse_args()


def detect_model_type(model_name: str, tokenizer) -> str:
    name_lower = model_name.lower()
    if "deepseek" in name_lower or "r1-distill" in name_lower:
        return "deepseek-r1"
    if "qwen3" in name_lower:
        return "qwen3"
    added_vocab = tokenizer.get_added_vocab()
    if "<｜User｜>" in added_vocab:
        return "deepseek-r1"
    if "<|im_start|>" in added_vocab:
        return "qwen3"
    return "qwen3"


def extract_boxed_answer(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return ""


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("\\,", "").replace("\\;", "").replace("\\!", "")
    answer = answer.replace("\\left(", "(").replace("\\right)", ")")
    answer = answer.replace("\\left[", "[").replace("\\right]", "]")
    answer = answer.replace("\\dfrac", "\\frac")
    return answer.strip()


def answers_match(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False

    try:
        from math_verify import parse, verify
        parsed_pred = parse(f"\\boxed{{{pred}}}")
        parsed_gold = parse(f"\\boxed{{{gold}}}")
        if parsed_pred and parsed_gold:
            return bool(verify(parsed_pred, parsed_gold))
    except Exception:
        pass

    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True

    try:
        if abs(float(pred_norm) - float(gold_norm)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return False


def main():
    args = parse_args()
    if args.no_thinking:
        args.enable_thinking = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.merged_model_path or args.base_model

    print(f"\n{'=' * 60}")
    print("  MATH-500 evaluation (vLLM)")
    print(f"  experiment : {args.experiment}")
    print(f"  model      : {model_path}")
    print(f"  thinking   : {args.enable_thinking}")
    print(f"{'=' * 60}\n")

    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
    print(f"Loaded {len(dataset)} problems")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model_type = args.model_type
    if model_type == "auto":
        model_type = detect_model_type(args.base_model or model_path, tokenizer)
    print(f"model type: {model_type}")

    prompts = []
    for example in dataset:
        messages = ([{"role": "system", "content": args.system}] if args.system else []) \
                   + [{"role": "user", "content": example["problem"]}]
        if model_type == "qwen3":
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)

    print(f"loading vLLM engine: {model_path}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )

    if model_type == "deepseek-r1":
        stop_token_ids = [tokenizer.eos_token_id]
    else:
        stop_token_ids = [
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        stop_token_ids = [t for t in stop_token_ids if t is not None]
    print(f"stop token ids: {stop_token_ids}")

    sampling_kwargs = dict(
        max_tokens=args.max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        top_k=20 if model_type == "qwen3" else -1,
        min_p=0.0,
        stop_token_ids=stop_token_ids,
        repetition_penalty=1.3 if model_type == "deepseek-r1" else 1.0,
        presence_penalty=0.4 if model_type == "qwen3" else 0.0,
    )
    if args.seed is not None:
        sampling_kwargs["seed"] = args.seed
    sampling_params = SamplingParams(**sampling_kwargs)

    print("generating")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    correct = 0
    total = 0
    for i, (output, example) in enumerate(zip(outputs, dataset)):
        response = output.outputs[0].text
        gold_answer = example["answer"]

        thinking = ""
        answer_text = response
        if "</think>" in response:
            parts = response.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            answer_text = parts[1].strip()

        pred_answer = extract_boxed_answer(answer_text)
        if not pred_answer:
            pred_answer = extract_boxed_answer(response)

        is_correct = answers_match(pred_answer, gold_answer)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": i,
            "problem": example["problem"][:200],
            "gold": gold_answer,
            "pred": pred_answer,
            "correct": is_correct,
            "subject": example.get("subject", ""),
            "level": example.get("level", ""),
            "thinking_len": len(thinking),
            "response_len": len(response),
            "response": response,
        })

    accuracy = correct / total * 100
    print(f"\n{'=' * 60}")
    print(f"  MATH-500: {accuracy:.2f}% ({correct}/{total})")
    print(f"{'=' * 60}")

    subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        subject_stats[r["subject"]]["total"] += 1
        level_stats[r["level"]]["total"] += 1
        if r["correct"]:
            subject_stats[r["subject"]]["correct"] += 1
            level_stats[r["level"]]["correct"] += 1

    print("\nby subject:")
    for subj, stats in sorted(subject_stats.items()):
        print(f"  {subj}: {stats['correct'] / stats['total'] * 100:.1f}% "
              f"({stats['correct']}/{stats['total']})")

    print("\nby level:")
    for level, stats in sorted(level_stats.items()):
        print(f"  Level {level}: {stats['correct'] / stats['total'] * 100:.1f}% "
              f"({stats['correct']}/{stats['total']})")

    avg_thinking = sum(r["thinking_len"] for r in results) / max(len(results), 1)
    print(f"\nmean thinking length: {avg_thinking:.0f} chars")

    summary = {
        "experiment": args.experiment,
        "base_model": args.base_model,
        "model_path": model_path,
        "enable_thinking": args.enable_thinking,
        "benchmark": "MATH-500",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_thinking_len": avg_thinking,
        "subject_accuracy": {k: v["correct"] / v["total"] * 100 for k, v in subject_stats.items()},
        "level_accuracy": {k: v["correct"] / v["total"] * 100 for k, v in level_stats.items()},
    }

    summary_path = output_dir / f"math500_{args.experiment}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    details_path = output_dir / f"math500_{args.experiment}_details.jsonl"
    with open(details_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsaved: {summary_path}")
    return accuracy


if __name__ == "__main__":
    main()
