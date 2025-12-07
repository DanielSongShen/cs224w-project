"""Step 3: Assign Thoughts to Reasoning Steps

Uses LLM to map individual thoughts to their corresponding reasoning steps.
"""
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path

from ..config import STEP3_SYSTEM_MESSAGE, SPLIT_WORDS
from ..utils import (
    normalize_thought_keys,
    extract_and_parse_json,
    extract_reasoning_dict,
    clean_thought_text,
    save_jsonl,
    normalize_thought_id
)


def process_assign(
    input_data: List[Dict[str, Any]],
    llm_client,
    output_dir: Path,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Assign thoughts to reasoning steps using LLM.
    
    Args:
        input_data: Items with 'thoughts_list' and 'reasoning_sketch'
        llm_client: LLM client instance
        output_dir: Directory to save intermediate results
        max_workers: Number of parallel workers (sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing
        debug: Enable debug logging
    
    Returns:
        Items with added 'assigned_step' field
    """
    print("\n=== Step 3: Assigning thoughts to reasoning steps ===")
    start_time = time.time()
    
    def process_item(item):
        # Normalize keys to strings
        thought_list = normalize_thought_keys(item["thoughts_list"])
        thoughts = [thought_list[str(i)] for i in range(len(thought_list))]
        
        # Remove split words from start
        thoughts = [clean_thought_text(t, SPLIT_WORDS) for t in thoughts]
        
        # Get reasoning steps
        reasoning_dict = extract_reasoning_dict(item.get("reasoning_sketch", ""))
        
        # Build user content
        reasoning_str = "\n".join([f"Step {k}. {v}" for k, v in reasoning_dict.items()])
        thoughts_str = "\n".join([
            f"Thought {i}: {t[:200]}..." if len(t) > 200 else f"Thought {i}: {t}" 
            for i, t in enumerate(thoughts)
        ])
        
        user_content = f"Reasoning Steps:\n{reasoning_str}\n\nThoughts:\n{thoughts_str}"
        
        messages = [
            {"role": "system", "content": STEP3_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response, in_tokens, out_tokens, cache_hits = llm_client.generate(messages=messages)
            parsed = extract_and_parse_json(response)
            
            if parsed:
                # Normalize the assignment keys
                all_assignments = {}
                for key, value in parsed.items():
                    # Extract thought number from key
                    thought_num = normalize_thought_id(key)
                    if thought_num is not None:
                        all_assignments[thought_num] = value
                
                # Coverage guarantee: Ensure every thought ID exists
                for i in range(len(thoughts)):
                    if i not in all_assignments:
                        # Inherit from previous thought, or default to Step 1
                        if i > 0:
                            all_assignments[i] = all_assignments.get(i-1, [1])
                        else:
                            all_assignments[i] = [1]
                        
                        if debug:
                            print(f"  Warning: Thought {i} missing from LLM output. Inherited step {all_assignments[i]}")
            
            else:
                # Fallback: assign each thought to step based on order
                all_assignments = {}
                step_count = len(reasoning_dict)
                for i in range(len(thoughts)):
                    step = min(int(i * step_count / len(thoughts)) + 1, step_count)
                    all_assignments[i] = [step]
            
            item["assigned_step"] = all_assignments
            item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
            item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            item["cache_hit_tokens"] = item.get("cache_hit_tokens", 0) + cache_hits
            
        except Exception as e:
            print(f"Error assigning steps for {item['tag']}: {e}")
            # Fallback assignment
            reasoning_dict = extract_reasoning_dict(item.get("reasoning_sketch", ""))
            step_count = len(reasoning_dict) if reasoning_dict else 1
            all_assignments = {}
            for i in range(len(thoughts)):
                step = min(int(i * step_count / len(thoughts)) + 1, step_count)
                all_assignments[i] = [step]
            item["assigned_step"] = all_assignments
        
        return item
    
    # Process in parallel (sync or async)
    if use_async:
        results = asyncio.run(_process_batch_async(input_data, process_item, batch_size))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_item, input_data))
    
    # Save
    output_path = output_dir / "process3.json"
    save_jsonl(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"Saved {len(results)} items to {output_path}")
    print(f"⏱️  Step 3 completed in {elapsed:.2f} seconds")
    
    return results


async def _process_batch_async(items: List[Any], process_func, batch_size: int) -> List[Any]:
    """Process items in async batches using thread-based concurrency."""
    results = []
    total = len(items)
    
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
        
        batch_results = await asyncio.gather(*[
            asyncio.to_thread(process_func, item)
            for item in batch
        ])
        
        results.extend(batch_results)
    
    return results