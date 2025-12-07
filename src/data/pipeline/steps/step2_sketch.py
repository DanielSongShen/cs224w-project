"""Step 2: Extract Reasoning Sketch

Uses LLM to extract high-level reasoning steps from the full reasoning text.
"""
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path

from ..config import STEP2_SKETCH_PROMPT
from ..utils import save_jsonl


def process_sketch(
    input_data: List[Dict[str, Any]],
    llm_client,
    output_dir: Path,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Extract reasoning sketch using LLM.
    
    Args:
        input_data: Items with 'thoughts_list'
        llm_client: LLM client instance
        output_dir: Directory to save intermediate results
        max_workers: Number of parallel workers (sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing
    
    Returns:
        Items with added 'reasoning_sketch' field
    """
    print("\n=== Step 2: Extracting reasoning sketch ===")
    start_time = time.time()
    
    def process_item(item):
        text = item["prediction"]
        if text.startswith("<think>"):
            text = (text.split("<think>")[1]).split("</think>")[0]
        else:
            text = text.split("</think>")[0]
        
        prompt = STEP2_SKETCH_PROMPT.format(text=text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response, in_tokens, out_tokens, cache_hits = llm_client.generate(messages=messages)
            item["reasoning_sketch"] = response
            item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
            item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            item["cache_hit_tokens"] = item.get("cache_hit_tokens", 0) + cache_hits
        except Exception as e:
            print(f"Error extracting sketch for {item['tag']}: {e}")
            item["reasoning_sketch"] = ""
        
        return item
    
    # Process in parallel (sync or async)
    if use_async:
        results = asyncio.run(_process_batch_async(input_data, process_item, batch_size))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_item, input_data))
    
    # Save
    output_path = output_dir / "process2.json"
    save_jsonl(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"Saved {len(results)} items to {output_path}")
    print(f"⏱️  Step 2 completed in {elapsed:.2f} seconds")
    
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