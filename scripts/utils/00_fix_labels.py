"""
Regenerates scores for each prediction-answer pair using DeepSeek API.

Usage:
    python scripts/00_fix_labels.py --input data/processed/deepseek/amc-aime/final.json --output data/processed/deepseek/amc-aime/final_regraded.json
    
    # Or with specific response_ids
    python scripts/00_fix_labels.py --input final.json --output regraded.json --response_ids vfm_37358_60KZndTp vfm_12345_abc123
"""
import sys
import os
import re
import json
import ast
import argparse
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from src.data.llm_client import create_llm_client


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the final answer from \\boxed{...} in the response."""
    # Handle nested braces by finding matching closing brace
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return None
    
    # Take the last \boxed{} occurrence (usually the final answer)
    last_match = matches[-1]
    start = last_match.end()
    
    # Find matching closing brace
    brace_count = 1
    pos = start
    while pos < len(text) and brace_count > 0:
        if text[pos] == '{':
            brace_count += 1
        elif text[pos] == '}':
            brace_count -= 1
        pos += 1
    
    if brace_count == 0:
        return text[start:pos-1].strip()
    return None


def load_verification_info(response_ids: list[str]) -> dict[str, str]:
    """
    Load verification_info.ground_truth from SYNTHETIC-1 for given response_ids.
    
    Returns:
        Dict mapping response_id -> ground_truth
    """
    dataset = load_dataset("PrimeIntellect/SYNTHETIC-1", split="train", streaming=True)
    
    response_id_set = set(response_ids)
    ground_truths = {}
    
    print(f"Loading ground truth for {len(response_ids)} responses from SYNTHETIC-1...")
    
    for example in tqdm(dataset, desc="Scanning dataset"):
        rid = example.get("response_id")
        if rid in response_id_set:
            verification_info = example.get("verification_info", {})
            if isinstance(verification_info, str):
                verification_info = ast.literal_eval(verification_info)
            ground_truths[rid] = verification_info.get("ground_truth", "")
            
            # Early exit if we found all
            if len(ground_truths) == len(response_id_set):
                break
    
    print(f"Found ground truth for {len(ground_truths)}/{len(response_ids)} responses")
    return ground_truths


def grade_answer(
    llm_client,
    extracted_answer: str,
    ground_truth: str
) -> tuple[str, int, int]:
    """
    Use LLM to grade if extracted_answer matches ground_truth.
    
    Returns:
        Tuple of (score "0" or "1", input_tokens, output_tokens)
    """
    prompt = f"""Compare the following answer to the ground truth. 
Return ONLY "1" if the answer is correct (equivalent, allowing for formatting differences), or "0" if incorrect.

Answer: {extracted_answer}
Ground truth: {ground_truth}

Your response (just "0" or "1"):"""

    messages = [{"role": "user", "content": prompt}]
    
    try:
        response, in_tokens, out_tokens = llm_client.generate(messages=messages, max_tokens=8)
        # Extract just 0 or 1 from response
        score = "1" if "1" in response.strip() else "0"
        return score, in_tokens, out_tokens
    except Exception as e:
        print(f"Grading failed: {e}")
        return "0", 0, 0


def process_samples(
    input_path: str,
    output_path: str,
    response_ids: Optional[list[str]] = None,
    config_path: str = "./config.json",
    max_workers: int = 10,
    dry_run: bool = False
):
    """
    Process samples from input JSON, regrade them, and save to output.
    
    Args:
        input_path: Path to input final.json
        output_path: Path for output regraded JSON
        response_ids: Optional list to filter specific response_ids (if None, process all)
        config_path: Path to config.json with API credentials
        max_workers: Number of concurrent grading requests
        dry_run: If True, just show what would be processed without API calls
    """
    # Load input data
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        # Handle both JSON array and JSON lines format
        content = f.read().strip()
        if content.startswith('['):
            samples = json.loads(content)
        else:
            samples = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    print(f"Loaded {len(samples)} samples")
    
    # Filter by response_ids if provided
    if response_ids:
        response_id_set = set(response_ids)
        samples = [s for s in samples if s.get("response_id") in response_id_set]
        print(f"Filtered to {len(samples)} samples matching provided response_ids")
    
    if not samples:
        print("No samples to process!")
        return
    
    # Get unique response_ids to fetch ground truths
    unique_rids = list(set(s.get("response_id") for s in samples if s.get("response_id")))
    ground_truths = load_verification_info(unique_rids)
    
    # Load config and create LLM client
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    llm_client = create_llm_client(backend="deepseek-v3.2", config=config.get("deepseek-v3.2", {}))
    
    # Extract answers and prepare for grading
    grading_tasks = []
    for i, sample in enumerate(samples):
        extracted = extract_boxed_answer(sample.get("prediction", ""))
        rid = sample.get("response_id")
        gt = ground_truths.get(rid, "")
        
        if not extracted:
            print(f"Warning: Could not extract boxed answer for sample {i} (response_id: {rid})")
        
        grading_tasks.append({
            "index": i,
            "extracted_answer": extracted or "",
            "ground_truth": gt,
            "response_id": rid
        })
    
    if dry_run:
        print("\n=== DRY RUN ===")
        for task in grading_tasks[:5]:
            print(f"\nResponse: {task['response_id']}")
            print(f"  Extracted: {task['extracted_answer'][:100]}..." if len(task['extracted_answer']) > 100 else f"  Extracted: {task['extracted_answer']}")
            print(f"  Ground truth: {task['ground_truth']}")
        print(f"\n... and {len(grading_tasks) - 5} more")
        return
    
    # Grade concurrently
    print(f"\nGrading {len(grading_tasks)} samples with {max_workers} workers...")
    results = [None] * len(samples)
    total_in_tokens = 0
    total_out_tokens = 0
    
    def grade_task(task):
        score, in_tok, out_tok = grade_answer(
            llm_client,
            task["extracted_answer"],
            task["ground_truth"]
        )
        return task["index"], score, in_tok, out_tok
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(grade_task, task) for task in grading_tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Grading"):
            idx, score, in_tok, out_tok = future.result()
            results[idx] = score
            total_in_tokens += in_tok
            total_out_tokens += out_tok
    
    # Update samples with new scores
    for i, sample in enumerate(samples):
        sample["score"] = results[i]
    
    # Save output as JSONL
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    # Summary
    correct = sum(1 for s in results if s == "1")
    print(f"\n=== Grading Complete ===")
    print(f"Results saved to: {output_path}")
    print(f"Accuracy: {correct}/{len(samples)} ({100*correct/len(samples):.1f}%)")
    print(f"Token usage: {total_in_tokens:,} in / {total_out_tokens:,} out")


def main():
    parser = argparse.ArgumentParser(description="Regrade predictions using DeepSeek API")
    parser.add_argument("--input", "-i", required=True, help="Input final.json path")
    parser.add_argument("--output", "-o", required=True, help="Output path for regraded JSON")
    parser.add_argument("--response_ids", nargs="*", help="Optional: specific response_ids to process")
    parser.add_argument("--config", default="./config.json", help="Path to config.json")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent workers")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be processed without API calls")
    
    args = parser.parse_args()
    
    process_samples(
        input_path=args.input,
        output_path=args.output,
        response_ids=args.response_ids,
        config_path=args.config,
        max_workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
