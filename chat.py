from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import polars as pl
from typing import List
import re
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt

def parse_semantic_id(semantic_id: str) -> List[str]:
    """
    Parse a semantic ID string into its component levels.

    Example input: '<|sid_start|><|sid_127|><|sid_45|><|sid_89|><|sid_12|><|sid_end|>'
    Returns: ['<|sid_127|>', '<|sid_45|>', '<|sid_89|>', '<|sid_12|>']
    """
    # Remove start and end tokens
    sid = semantic_id.replace("<|sid_start|>", "").replace("<|sid_end|>", "")

    # Extract all sid tokens
    pattern = r"<\|sid_\d+\|>"
    levels = re.findall(pattern, sid)

    return levels


def map_semantic_id_to_titles(semantic_id_str: str, mapping_df: pd.DataFrame) -> dict:
    """
    Map a semantic ID to titles with 4-token exact match and 3-token fallback.

    Returns:
        dict with 'match_level', 'titles', and 'count' keys
    """
    # Parse the input semantic ID
    levels = parse_semantic_id(semantic_id_str)

    if not levels:
        return {"match_level": 0, "titles": [], "count": 0}

    # First try exact match (all 4 tokens)
    exact_matches = mapping_df[mapping_df["semantic_id"] == semantic_id_str]
    if len(exact_matches) > 0:
        titles = exact_matches["标题"].tolist()
        return {"match_level": 4, "titles": titles, "count": len(titles), "match_type": "exact"}

    # Fallback to prefix matching (3 tokens, then 2, then 1)
    for depth in range(min(3, len(levels)), 0, -1):
        # Build the prefix for this depth
        prefix = "<|sid_start|>" + "".join(levels[:depth])

        # Find matches
        matches = mapping_df[mapping_df["semantic_id"].str.startswith(prefix)]

        if len(matches) > 0:
            # Found matches at this level
            titles = matches["标题"].tolist()
            return {
                "match_level": depth,
                "titles": titles[:5],  # Limit to 5 for display
                "count": len(titles),
                "match_type": "prefix",
                "prefix_used": prefix,
            }

    # No matches found at any level
    return {"match_level": 0, "titles": [], "count": 0, "match_type": "none"}

def extract_semantic_ids_from_text(text: str) -> List[str]:
    """
    Extract all semantic IDs from a text string.

    Returns list of full semantic IDs found in the text.
    """
    # Pattern to match complete semantic IDs
    pattern = r"<\|sid_start\|>(?:<\|sid_\d+\|>)+<\|sid_end\|>"
    semantic_ids = re.findall(pattern, text)
    return semantic_ids


def replace_semantic_ids_with_titles(text: str, mapping_df: pd.DataFrame = None, show_match_level: bool = True) -> str:
    """
    Replace all semantic IDs in text with their corresponding titles.

    Args:
        text: Input text containing semantic IDs
        mapping_df: DataFrame with semantic_id to title mapping
        show_match_level: Whether to append match info after the title

    Returns:
        Text with semantic IDs replaced by titles
    """
    if mapping_df is None:
        mapping_df = globals().get("mapping_df")
        if mapping_df is None:
            raise ValueError("No mapping_df provided or found in global scope")

    # Find all semantic IDs in the text
    semantic_ids = extract_semantic_ids_from_text(text)

    # Create a copy of the text to modify
    result_text = text

    # Replace each semantic ID with its title(s)
    for sid in semantic_ids:
        # Get matching titles
        match_result = map_semantic_id_to_titles(sid, mapping_df)

        if match_result["count"] > 0:
            # Use the first title if multiple matches
            title = match_result["titles"][0]

            # Add match level if requested
            if show_match_level:
                if match_result["match_type"] == "exact":
                    replacement = f'"{title}"'
                else:
                    replacement = f'"{title}" (L{match_result["match_level"]} match)'
            else:
                replacement = f'"{title}"'

            # If multiple matches, indicate this
            if match_result["count"] > 1:
                replacement += f" [+{match_result['count'] - 1} similar]"
        else:
            # No match found
            replacement = "[Unknown Item]"

        # Replace the semantic ID with the title
        result_text = result_text.replace(sid, replacement)

    return result_text
GLOBAL_MESSAGES = []
SYSTEM_PROMPT = """你是一个专业的电影商品推荐助手。"""

def clean_output(text: str) -> str:
    """Remove known special tokens from output."""
    special_tokens = ["<think>", "</think>", "<|im_end|>", "<|im_start|>", "<|endoftext|>"]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


def chat(
    model,
    tokenizer,
    text_input: str,
    new_convo: bool = True,
    max_new_tokens: int = 2048,
    stream: bool = True,
    mapping_df: pd.DataFrame = None
) -> str:
    """
    Chat with the model, maintaining conversation history in GLOBAL_MESSAGES.

    Args:
        text_input: User input text
        new_convo: If True, clear global message history for a fresh start
        temperature: Generation temperature
        max_new_tokens: Maximum tokens to generate
        stream: Whether to stream output
        mapping_df: DataFrame for semantic ID mapping

    Returns:
        Generated response text
    """
    global GLOBAL_MESSAGES

    if mapping_df is None:
        mapping_df = globals().get("mapping_df")

    # Handle conversation history
    if new_convo:
        GLOBAL_MESSAGES = []
    else:
        # Display previous conversation turns if continuing
        if GLOBAL_MESSAGES:
            print(f"{'=' * 41} Conversation History {'=' * 41}")
            for i, msg in enumerate(GLOBAL_MESSAGES, 1):
                role = msg["role"].upper()
                content = replace_semantic_ids_with_titles(msg["content"], mapping_df)
                print(f"[Turn {(i+1)//2}] {role}: {content}")
            print(f"{'=' * 35} Current Turn {'=' * 35}")

    GLOBAL_MESSAGES.append({"role": "user", "content": text_input})

    # Build messages with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + GLOBAL_MESSAGES

    # Log input with readable semantic IDs
    readable_input = replace_semantic_ids_with_titles(text_input, mapping_df)
    print(f"USER: {readable_input}")
    print(f"{'=' * 20} START RAW MODEL OUTPUT (WITH SEMANTIC IDS) {'=' * 20}")

    # Prepare input for model
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate response
    streamer = TextStreamer(tokenizer, skip_prompt=True) if stream else None

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[:, input_length:]
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    # Clean output
    generated_text = clean_output(generated_text)

    # Add to conversation history
    GLOBAL_MESSAGES.append({"role": "assistant", "content": generated_text})

    # Log output with readable semantic IDs
    print(f"{'=' * 21} END RAW MODEL OUTPUT (WITH SEMANTIC IDS) {'=' * 21}")
    readable_output = replace_semantic_ids_with_titles(generated_text, mapping_df)
    print(f"ASSISTANT: {readable_output}")

    return generated_text

MODEL_REPO = "/home/j00960957/j00960957/llm4rec_add_general/checkpoints_zh/Qwen3-1.7b-sft_all/final"
device = torch.device(f"npu:0")

model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
    padding_side="left",  # Important for generation
    use_fast=False
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

mapping_df = pl.read_parquet("data/output_zh/Movies_and_TV_semantic_ids.parquet").to_pandas()
INPUT = """已知以下电影商品：<|sid_start|><|sid_112|><|sid_267|><|sid_645|><|sid_768|><|sid_end|>，请提供其详细的信息。""".strip()
response = chat(model, tokenizer, INPUT, mapping_df=mapping_df)

