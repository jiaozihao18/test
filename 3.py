import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
import os

# ===== SID utils =====
import re
from typing import List

SID_PATTERN = re.compile(r"<\|sid_(\d+)\|>")

def extract_sids(text: str) -> List[str]:
    if text is None:
        return []
    return SID_PATTERN.findall(text)

def sid_prefix_reward(completions, ground_truth, **kwargs):
    rewards = []
    prefix_lens = []

    for comp, gt in zip(completions, ground_truth):
        if isinstance(comp, str):
            comp_text = comp
        else:
            comp_text = comp[0]["content"] if len(comp) > 0 else ""

        pred = extract_sids(comp_text)
        gold = extract_sids(gt)

        k = 0
        for i in range(min(4, len(pred), len(gold))):
            if pred[i] == gold[i]:
                k += 1
            else:
                break

        rewards.append(k / 4.0)
        prefix_lens.append(k)

    if "metrics" in kwargs:
        kwargs["metrics"]["prefix_len"] = sum(prefix_lens) / len(prefix_lens)
        kwargs["metrics"]["reward"] = sum(rewards) / len(rewards)

    return rewards


# ===== Dataset =====
from datasets import load_dataset

def build_dataset(path: Path):
    ds = load_dataset("parquet", data_files=str(path), split="train")
    return ds.map(
        lambda x: {
            "prompt": x["conversations"][:-1],
            "ground_truth": x["conversations"][-1]["content"],
        },
        remove_columns=ds.column_names,
    )


# ===== Config =====
@dataclass
class CFG:
    model_name: Path = Path("/home/j00960957/j00960957/llm4rec_add_general/checkpoints_zh/Qwen3-1.7b-sft_all/final")
    train_path: Path = Path("/home/j00960957/j00960957/llm4rec_add_general/data/output_zh/sft_data/sft_train_only_seq.parquet")
    output_dir: Path = Path("/home/j00960957/j00960957/llm4rec_add_general/checkpoints_zh/qwen3-1.7b-sft_all-grpo")

    bf16: bool = True
    seed: int = 42

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-6
    num_train_epochs: float = 1.0

    num_generations: int = 8
    max_prompt_length: int = 2048
    max_completion_length: int = 32

    temperature: float = 1.0
    top_p: float = 0.95
    beta: float = 0.02

    logging_steps: int = 10
    save_strategy: str = "epoch"

    deepspeed: str = "./dp_zero2.json"


# ===== Train =====
def main():
    cfg = CFG()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(cfg.train_path)

    args = GRPOConfig(
        output_dir=str(cfg.output_dir),
        seed=cfg.seed,
        bf16=cfg.bf16,

        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,

        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,

        temperature=cfg.temperature,
        top_p=cfg.top_p,
        beta=cfg.beta,

        mask_truncated_completions=True,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,

        deepspeed=cfg.deepspeed,
        report_to=None,
    )

    trainer = GRPOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        reward_funcs=sid_prefix_reward,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(cfg.output_dir / "final"))
    tokenizer.save_pretrained(str(cfg.output_dir / "final"))


if __name__ == "__main__":
    main()
