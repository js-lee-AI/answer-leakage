
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=str, required=True,
                        help="run name; outputs land in <output_root>/<experiment>")
    parser.add_argument("--data_path", type=str, required=True,
                        help="training jsonl with a `messages` field (question -> chain)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_root", type=str, default="outputs",
                        help="parent directory for run outputs (repo-relative by default)")
    parser.add_argument("--max_seq_length", type=int, default=10240)
    parser.add_argument("--epochs", type=int, default=1,
                        help="the paper uses 1; more epochs overfit these small corpora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=16,
                        help="set to 48 // NUM_GPUS to hold the effective batch at 48")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--use_liger", action="store_true",
                        help="Liger fused linear cross-entropy: numerically equivalent, "
                             "lower memory")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="added automatically by DeepSpeed")
    parser.add_argument("--ds_config", type=str, default="configs/ds_zero3.json",
                        help="DeepSpeed config; the paper uses ZeRO-3")
    parser.add_argument("--no_deepspeed", action="store_true",
                        help="disable DeepSpeed (single-GPU debugging only)")
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 instead of bf16 (for GPUs without bf16)")
    parser.add_argument("--save_steps", type=int, default=None,
                        help="checkpoint every N steps (default: once per epoch)")
    parser.add_argument("--save_total_limit", type=int, default=6)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42,
                        help="the paper reports seeds 42, 123 and 7777")
    parser.add_argument("--no_final_save", action="store_true",
                        help="skip the final/ save. ZeRO-3 parameter gather on save can "
                             "hang on short runs; evaluate the last checkpoint-N instead")
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="answer-leakage")
    return parser.parse_args()


def load_data(data_path: str):
    print(f"Loading data from: {data_path}")
    ds = load_dataset("json", data_files=data_path, split="train")
    print(f"Loaded {len(ds)} records")
    return ds


DEEPSEEK_R1_TRAIN_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{bos_token}}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{message['content']}}{%- endif %}"
    "{%- if message['role'] == 'user' %}{{'<｜User｜>' + message['content']}}{%- endif %}"
    "{%- if message['role'] == 'assistant' %}{{'<｜Assistant｜>' + message['content'] + eos_token}}{%- endif %}"
    "{%- endfor -%}"
    "{% if add_generation_prompt %}{{'<｜Assistant｜><think>\n'}}{% endif %}"
)

GLM_TRAIN_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
    "{{ '[gMASK]<sop>' }}"
    "{%- for message in messages %}"
    "{%- if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] }}{%- endif %}"
    "{%- if message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] }}{%- endif %}"
    "{%- if message['role'] == 'assistant' %}{{ '<|assistant|>\n' + message['content'] + eos_token }}{%- endif %}"
    "{%- endfor -%}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
)


def detect_model_type(model_name: str) -> str:
    name_lower = model_name.lower()
    if "deepseek" in name_lower or "r1-distill" in name_lower:
        return "deepseek-r1"
    if "glm" in name_lower:
        return "glm"
    return "qwen3"


def main():
    args = parse_args()

    run_name = args.experiment
    output_dir = Path(args.output_root) / run_name

    print(f"\n{'=' * 60}")
    print(f"  experiment : {args.experiment}")
    print(f"  data       : {args.data_path}")
    print(f"  model      : {args.base_model}")
    print(f"  max_seq    : {args.max_seq_length}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch      : {args.batch_size} x {args.gradient_accumulation} per GPU "
          f"= {args.batch_size * args.gradient_accumulation} (x NUM_GPUS)")
    print(f"  seed       : {args.seed}")
    print(f"  output     : {output_dir}")
    print(f"{'=' * 60}\n")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.wandb and local_rank == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    dataset = load_data(args.data_path)

    model_type = detect_model_type(args.base_model)
    print(f"model type: {model_type}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_type == "deepseek-r1":
        print("DeepSeek-R1 detected: applying the thinking-preserving chat template")
        tokenizer.chat_template = DEEPSEEK_R1_TRAIN_TEMPLATE
    if model_type == "glm":
        print("GLM detected: applying the thinking-preserving chat template")
        tokenizer.chat_template = GLM_TRAIN_TEMPLATE

    print(f"loading model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        dtype=torch.float16 if args.fp16 else torch.bfloat16,
        attn_implementation="sdpa",
    )


    ds_config = None if args.no_deepspeed else args.ds_config

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_seq_length,
        bf16=not args.fp16,
        fp16=args.fp16,
        logging_steps=10,
        save_strategy="steps" if args.save_steps else "epoch",
        save_steps=args.save_steps if args.save_steps else None,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=args.use_liger,
        deepspeed=ds_config,
        report_to="wandb" if args.wandb else "none",
        run_name=run_name,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=True,
    )

    class BestModelCallback(TrainerCallback):

        def __init__(self, output_dir):
            self.best_loss = float("inf")
            self.best_dir = os.path.join(output_dir, "best")

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                current_loss = logs["loss"]
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.best_step = state.global_step

        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            if hasattr(self, "best_step") and state.global_step == self.best_step:
                latest_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                if os.path.exists(latest_ckpt):
                    if os.path.exists(self.best_dir):
                        shutil.rmtree(self.best_dir)
                    shutil.copytree(latest_ckpt, self.best_dir)
                    print(f"best model saved: step {state.global_step}, loss {self.best_loss:.4f}")

    best_callback = BestModelCallback(str(output_dir))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[best_callback],
    )

    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt:
        print(f"\nresuming from checkpoint: {resume_ckpt}")
    else:
        print("\ntraining")
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)

    if args.no_final_save:
        print("\nskipping final/ save (--no_final_save); evaluate the last checkpoint-N")
        try:
            trainer.save_state()
        except Exception as e:
            print(f"  save_state skipped: {e}")
    else:
        print("\nsaving model")
        trainer.save_model(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))

    metrics = train_result.metrics
    metrics["experiment"] = args.experiment
    metrics["data_path"] = args.data_path
    metrics["seed"] = args.seed
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\ndone. metrics: {metrics_path}")
    print(f"  train_loss: {metrics.get('train_loss', 'N/A')}")

    if args.wandb and local_rank == 0:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
