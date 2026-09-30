"""
generate.py -- Local GPU code generation pipeline using Qwen2.5-Coder.

Generates N completions per BigCodeBench task with per-token logprobs and
offset mappings. Designed for local RTX 4090 (24GB VRAM) but also supports
4-bit quantization for smaller GPUs.

Features:
  - Checkpoint/resume: saves after each task, crash-safe
  - Offset mapping capture for token-to-source alignment (Phase 6)
  - Top-K logprobs per generated token (entropy signal input)
  - Pilot mode (--pilot) for quick validation on 10 tasks
  - Full mode for the complete filtered dataset

Usage:
    python src/generate.py --pilot              # 10 pilot tasks, 1.5B model
    python src/generate.py --pilot --model 7B   # 10 pilot tasks, 7B model
    python src/generate.py                      # full dataset, 1.5B model
    python src/generate.py --model 7B           # full dataset, 7B model
    python src/generate.py --resume             # resume from checkpoint
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Model configurations ──────────────────────────────────────────────
MODELS = {
    "1.5B": {
        "name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "dtype": torch.float16,       # fits in 3-4 GB, fp16 is fine
        "quantize": False,
    },
    "7B": {
        "name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "dtype": torch.float16,       # fits in ~14 GB on RTX 4090
        "quantize": False,            # set True for GPUs with <16GB VRAM
    },
}

# ── Generation defaults ───────────────────────────────────────────────
N_SAMPLES_PER_TASK = 10
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 1024      # Increased token budget to prevent mid-function syntax truncation
TOP_K_LOGPROBS = 10        # store top-10 logprobs per token

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PILOT_TASKS = PROJECT_ROOT / "data" / "pilot" / "pilot_tasks.json"
FULL_TASKS = PROJECT_ROOT / "data" / "raw_tasks" / "filtered_tasks.json"
GENERATIONS_DIR = PROJECT_ROOT / "data" / "generations"


def load_model(model_key: str = "1.5B"):
    """Load the model and tokenizer onto GPU."""
    config = MODELS[model_key]
    model_name = config["name"]

    print(f"\n  Loading model: {model_name}")
    print(f"  dtype: {config['dtype']}, quantize: {config['quantize']}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = {
        "dtype": config["dtype"],
        "device_map": "auto",
    }

    if config["quantize"]:
        if not torch.cuda.is_available():
            print("  [WARN] 4-bit quantization (BitsAndBytes) requires an NVIDIA CUDA GPU.")
            print("         Running in standard float16 on Apple Silicon / CPU instead.\n")
        else:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    # Check memory usage
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1024**3
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {mem_used:.1f} / {mem_total:.1f} GB used\n")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  Using Apple Silicon GPU (Metal Performance Shaders - MPS)\n")

    return tokenizer, model


def clean_generated_code(text: str) -> str:
    """
    Strips markdown code fences and conversational preambles/fillers from generated code:
      - Leading/trailing ```python or ``` code fences
      - Conversational preamble before the first line of actual code
    """
    if not text:
        return ""

    # 1. Strip markdown fences if present
    if "```python" in text:
        text = text.split("```python", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
        elif "```" in text:
            text = text.replace("```", "")

    # 2. Strip conversational preamble before first line of actual code
    lines = text.splitlines()
    code_start_idx = 0
    code_starters = (
        "def ", "class ", "import ", "from ", "try:", "if ", "for ", "while ",
        "with ", "return ", "@", "#", "print(", "raise "
    )
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(code_starters) or "=" in stripped:
            code_start_idx = i
            break
        if any(stripped.lower().startswith(p) for p in ("sure", "here is", "here's", "below is", "certainly", "the function")):
            continue

    cleaned = "\n".join(lines[code_start_idx:]).rstrip()
    return cleaned


def generate_with_logprobs(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_k_logprobs: int = TOP_K_LOGPROBS,
) -> dict:
    """
    Generate one completion with per-token logprobs and offset mapping.

    Returns:
        dict with keys:
          - generated_code: str (cleaned of fences & filler)
          - topk_logprobs: list of {tokens, probs, token_ids}
          - prompt_token_count: int
          - generated_token_count: int
          - offset_mapping: list of [start, end] char offsets in generated_code
    """
    # Tokenize with offset mapping for later token-to-source alignment
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # offset_mapping is for the prompt; we'll build one for generated text below
    prompt_offsets = inputs.pop("offset_mapping")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Extract generated tokens (excluding prompt)
    generated_ids = out.sequences[0][prompt_len:].cpu()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    cleaned_code = clean_generated_code(generated_text)

    # Build offset mapping for the cleaned generated text
    target_text = cleaned_code if cleaned_code else generated_text
    enc = tokenizer(target_text, return_offsets_mapping=True, add_special_tokens=False)
    gen_offset_mapping = enc.get("offset_mapping", [])

    # Extract top-K logprobs per generated token vectorized on GPU
    topk_logprobs = []
    if out.scores:
        all_scores = torch.cat(out.scores, dim=0).float()
        probs = torch.softmax(all_scores, dim=-1)
        topk = torch.topk(probs, min(top_k_logprobs, probs.shape[-1]), dim=-1)
        topk_indices = topk.indices.cpu().numpy()
        topk_values = topk.values.cpu().numpy()

        unique_ids = np.unique(topk_indices)
        id_to_token = {int(uid): tokenizer.convert_ids_to_tokens(int(uid)) for uid in unique_ids}

        for row_inds, row_vals in zip(topk_indices, topk_values):
            topk_logprobs.append({
                "token_ids": [int(x) for x in row_inds],
                "tokens": [id_to_token[int(x)] for x in row_inds],
                "probs": [round(float(p), 6) for p in row_vals],
            })

    return {
        "generated_code": cleaned_code,
        "topk_logprobs": topk_logprobs,
        "prompt_token_count": prompt_len,
        "generated_token_count": len(generated_ids),
        "offset_mapping": gen_offset_mapping,
    }


def load_tasks(pilot: bool = False) -> list[dict]:
    """Load tasks from the appropriate JSON file."""
    path = PILOT_TASKS if pilot else FULL_TASKS
    if not path.exists():
        print(f"  ERROR: Task file not found: {path}")
        print(f"  Run 'python src/filter_tasks.py --pilot 10' first.")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"  Loaded {len(tasks)} tasks from {path.name}")
    return tasks


def get_checkpoint_path(pilot: bool, model_key: str) -> Path:
    """Get the checkpoint file path for resume support."""
    suffix = "pilot" if pilot else "full"
    return GENERATIONS_DIR / f"checkpoint_{suffix}_{model_key}.json"


def load_checkpoint(checkpoint_path: Path, output_dir: Path) -> set:
    """Load completed task IDs from checkpoint and existing output files."""
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, encoding="utf-8") as f:
            checkpoint = json.load(f)
        completed.update(checkpoint.get("completed_task_ids", []))

    # Also scan output directory for already saved files
    if output_dir.exists():
        for f in output_dir.glob("*.json"):
            # Task IDs were sanitized with _ replacing /
            name = f.stem
            if "_" in name:
                # e.g. BigCodeBench_708 -> BigCodeBench/708
                parts = name.split("_", 1)
                task_id = f"{parts[0]}/{parts[1]}"
                completed.add(task_id)

    if completed:
        print(f"  Found {len(completed)} previously completed tasks (will skip)")
    return completed


def save_checkpoint(checkpoint_path: Path, completed_ids: list[str]):
    """Save checkpoint with completed task IDs."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump({"completed_task_ids": completed_ids}, f)


def save_task_generations(task_id: str, generations: list[dict], output_dir: Path):
    """Save generations for a single task to a separate JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize task_id for filename (e.g., "BigCodeBench/708" -> "BigCodeBench_708")
    safe_id = task_id.replace("/", "_").replace("\\", "_")
    path = output_dir / f"{safe_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate code completions with per-token logprobs"
    )
    parser.add_argument("--pilot", action="store_true",
                        help="Use pilot task subset (10 tasks)")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="1.5B",
                        help="Model size (default: 1.5B)")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_PER_TASK,
                        help=f"Samples per task (default: {N_SAMPLES_PER_TASK})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Max new tokens per completion (default: {MAX_NEW_TOKENS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing generations with new token budget")
    parser.add_argument("--quantize", action="store_true",
                        help="Force 4-bit quantization (for smaller GPUs)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to process (e.g. 50)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Code Generation Pipeline")
    print("=" * 60)

    # Override quantization if requested
    if args.quantize:
        MODELS[args.model]["quantize"] = True

    # Load tasks
    tasks = load_tasks(pilot=args.pilot)
    if args.limit is not None and args.limit > 0:
        tasks = tasks[:args.limit]

    # Setup output directory
    mode_label = "pilot" if args.pilot else "full"
    output_dir = GENERATIONS_DIR / f"{mode_label}_{args.model}"

    # Handle checkpoint/resume
    checkpoint_path = get_checkpoint_path(args.pilot, args.model)
    if args.overwrite:
        completed_ids = set()
        print("  [--overwrite active] Re-generating all tasks with new token budget.")
    else:
        completed_ids = load_checkpoint(checkpoint_path, output_dir)

    remaining_tasks = [t for t in tasks if t["task_id"] not in completed_ids]
    if not remaining_tasks:
        print("  All tasks already completed! Use --overwrite to regenerate.")
        return

    print(f"  Mode: {mode_label}")
    print(f"  Tasks remaining: {len(remaining_tasks)}/{len(tasks)}")
    print(f"  Samples per task: {args.n_samples}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output: {output_dir}")
    total_generations = len(remaining_tasks) * args.n_samples
    print(f"  Total generations to produce: {total_generations}")

    # Load model
    tokenizer, model = load_model(args.model)

    # Generate
    all_completed_ids = list(completed_ids)
    start_time = time.time()
    total_tokens = 0

    task_pbar = tqdm(remaining_tasks, desc="Tasks", unit="task")
    for task in task_pbar:
        task_id = task["task_id"]
        prompt = task["complete_prompt"]
        task_pbar.set_postfix_str(task_id)

        task_generations = []
        for sample_idx in range(args.n_samples):
            result = generate_with_logprobs(
                tokenizer, model, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            generation = {
                "task_id": task_id,
                "sample_index": sample_idx,
                "generated_code": result["generated_code"],
                "topk_logprobs": result["topk_logprobs"],
                "offset_mapping": result["offset_mapping"],
                "prompt_token_count": result["prompt_token_count"],
                "generated_token_count": result["generated_token_count"],
                "temperature": args.temperature,
                "model": MODELS[args.model]["name"],
            }
            task_generations.append(generation)
            total_tokens += result["generated_token_count"]

        # Save this task's generations (crash-safe: one file per task)
        save_task_generations(task_id, task_generations, output_dir)

        # Update checkpoint
        all_completed_ids.append(task_id)
        save_checkpoint(checkpoint_path, all_completed_ids)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    elapsed = time.time() - start_time

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Generation Complete")
    print(f"{'=' * 60}")
    print(f"  Tasks completed: {len(all_completed_ids)}")
    print(f"  Total generations: {len(all_completed_ids) * args.n_samples}")
    print(f"  Total tokens generated: {total_tokens:,}")
    print(f"  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if total_generations > 0:
        print(f"  Avg time per generation: {elapsed/total_generations:.2f}s")
        print(f"  Throughput: {total_tokens/elapsed:.0f} tokens/sec")
    print(f"  Output directory: {output_dir}")
    print(f"\n  Next step: run sandbox_harness.py to get PASS/FAIL labels")


if __name__ == "__main__":
    main()
