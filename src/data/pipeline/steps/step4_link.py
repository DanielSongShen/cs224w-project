"""Step 4: Assign Parent Relationships

The most expensive step. Implements:
- Prefix Caching Optimization (static data first in JSON)
- Incremental Saving (immediate write-and-flush)
- Robust Fallback (gap bridging + LCoT structural)
- Async Support (batched processing like Steps 2 & 3)
"""
import json
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any
from pathlib import Path

from ..config import (
    STEP4_SYSTEM_MESSAGE,
    SPLIT_WORDS,
    CATEGORY_MAP,
    MAX_THOUGHT_LENGTH,
    MAX_CANDIDATE_LENGTH
)
from ..utils import (
    normalize_thought_keys,
    extract_and_parse_json,
    extract_reasoning_dict,
    clean_thought_text,
    build_step_to_thoughts_mapping,
    save_jsonl,
    load_jsonl,
    truncate_text
)


def process_link(
    input_data: List[Dict[str, Any]],
    llm_client,
    output_dir: Path,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Assign parent relationships for each thought.
    
    Follows same pattern as Steps 2-3 but with incremental saving.
    
    Args:
        input_data: Items with 'thoughts_list' and 'assigned_step'
        llm_client: LLM client instance
        output_dir: Directory to save intermediate results
        max_workers: Number of parallel workers (sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing
        debug: Enable debug logging
    
    Returns:
        Items with added 'thought_relations' field
    """
    print("\n=== Step 4: Assigning parent relationships (LCoT Structural) ===")
    start_time = time.time()
    
    # INCREMENTAL SAVING: Check for existing progress
    incremental_path = output_dir / "process4_incremental.jsonl"
    
    processed_tags = set()
    results = []
    
    if incremental_path.exists():
        print(f"Found existing incremental file: {incremental_path}")
        existing_results = load_jsonl(incremental_path)
        for item in existing_results:
            processed_tags.add(item['tag'])
            results.append(item)
        print(f"Loaded {len(processed_tags)} already processed items")
    
    # FILTER: Only process what isn't done
    queue = [item for item in input_data if item['tag'] not in processed_tags]
    
    print(f"Processing {len(queue)} items ({len(processed_tags)} already done)")
    
    if not queue:
        print("All items already processed!")
        # Save final consolidated file
        output_path = output_dir / "process4.json"
        save_jsonl(results, output_path)
        return results
    
    # Define item processor (follows same pattern as steps 2-3)
    def process_item(item):
        return _process_single_item(item, llm_client, debug)
    
    # Process in parallel (sync or async) - SAME PATTERN AS STEPS 2-3
    new_results = []
    
    if use_async:
        # Async mode with batched processing
        new_results = asyncio.run(_process_batch_async(
            queue, process_item, batch_size, incremental_path
        ))
    else:
        # Sync mode with thread pool
        # Open file for incremental saving
        with open(incremental_path, 'a', encoding='utf-8') as f_out:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all items
                future_to_item = {
                    executor.submit(process_item, item): item 
                    for item in queue
                }
                
                # Process as they complete
                from concurrent.futures import as_completed
                for future in as_completed(future_to_item):
                    try:
                        result = future.result()
                        new_results.append(result)
                        
                        # INCREMENTAL SAVE (append + flush immediately)
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                        
                        if len(new_results) % 10 == 0:
                            total_done = len(processed_tags) + len(new_results)
                            print(f"  Processed {total_done}/{len(input_data)} items...")
                    
                    except Exception as e:
                        original_item = future_to_item[future]
                        print(f"Error processing {original_item.get('tag', 'unknown')}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Combine results
    results.extend(new_results)
    
    # Save final consolidated file
    output_path = output_dir / "process4.json"
    save_jsonl(results, output_path)
    
    # Statistics
    elapsed = time.time() - start_time
    total_api_calls = sum(
        sum(len(targets) for targets in item.get("thought_relations", {}).values())
        for item in new_results
    )
    
    print(f"Saved {len(results)} items to {output_path}")
    print(f"New parent selection queries: {total_api_calls}")
    print(f"⏱️  Step 4 completed in {elapsed:.2f} seconds")
    
    return results


async def _process_batch_async(
    items: List[Dict[str, Any]], 
    process_func, 
    batch_size: int,
    incremental_path: Path
) -> List[Any]:
    """
    Process items in async batches - SAME PATTERN AS STEPS 2-3.
    
    Added: Incremental saving to file.
    """
    results = []
    total = len(items)
    
    # Open file for incremental saving
    with open(incremental_path, 'a', encoding='utf-8') as f_out:
        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
            
            # Process batch concurrently - SAME AS STEPS 2-3
            batch_results = await asyncio.gather(*[
                asyncio.to_thread(process_func, item)
                for item in batch
            ])
            
            # Save batch results immediately
            for result in batch_results:
                results.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            f_out.flush()
    
    return results


def _process_single_item(
    item: Dict[str, Any], 
    llm_client, 
    debug: bool
) -> Dict[str, Any]:
    """
    Process a single item to assign parent relationships.
    
    This is the complex part with multiple LLM calls per item.
    Uses an inner thread pool (5 workers) for queries.
    
    Implements:
    - Coverage guarantee (sanitization)
    - Gap bridging (find nearest non-empty previous step)
    - LCoT structural fallback (connect to anchor)
    - Prefix caching optimization
    - Type normalization (handles old pipeline data)
    
    Args:
        item: Item with thoughts_list and assigned_step
        llm_client: LLM client instance
        debug: Enable debug logging
    
    Returns:
        Item with added thought_relations field
    """
    # Normalize keys to strings
    thought_list = normalize_thought_keys(item["thoughts_list"])
    thoughts = [thought_list[str(i)] for i in range(len(thought_list))]
    
    # Remove split words from start
    thoughts = [clean_thought_text(t, SPLIT_WORDS) for t in thoughts]
    
    # Get assigned_step mapping (thought_id -> [step_ids])
    assigned_step = item["assigned_step"]
    
    # NORMALIZE: Convert string keys from JSON to integers
    # This handles data from both old and new pipeline versions
    clean_assigned = {}
    for key, values in assigned_step.items():
        clean_key = int(key) if not isinstance(key, int) else key
        if not isinstance(values, list):
            values = [values]
        clean_values = [int(v) if not isinstance(v, int) else v for v in values]
        clean_assigned[clean_key] = clean_values
    
    # Build reverse mapping: step_id -> [thought_ids]
    step_to_thoughts = build_step_to_thoughts_mapping(clean_assigned)
    
    # --- COVERAGE GUARANTEE (Sanitization) ---
    # Ensures every thought ID from 0 to N-1 is assigned to a step
    all_thought_ids = set(range(len(thoughts)))
    assigned_ids = set()
    for ids in step_to_thoughts.values():
        assigned_ids.update(ids)
    
    missing_ids = sorted(list(all_thought_ids - assigned_ids))
    
    for t_id in missing_ids:
        # Inherit step from previous thought
        prev_id = t_id - 1
        found_step = 1  # Default
        
        if prev_id >= 0:
            for step_n, ids in step_to_thoughts.items():
                if prev_id in ids:
                    found_step = step_n
                    break
        
        if found_step not in step_to_thoughts:
            step_to_thoughts[found_step] = []
        step_to_thoughts[found_step].append(t_id)
        step_to_thoughts[found_step].sort()
    
    # Reconstruct assignments and save back to item
    new_assigned_step = {}
    for step_id, t_ids in step_to_thoughts.items():
        for t_id in t_ids:
            if t_id not in new_assigned_step:
                new_assigned_step[t_id] = []
            if step_id not in new_assigned_step[t_id]:
                new_assigned_step[t_id].append(step_id)
    
    item["assigned_step"] = new_assigned_step
    
    # Get reasoning steps
    reasoning_steps = extract_reasoning_dict(item.get("reasoning_sketch", ""))
    max_step = max(reasoning_steps.keys()) if reasoning_steps else 0
    
    # Initialize relations structure
    item["thought_relations"] = {}
    
    # Prepare all parent selection queries
    queries_to_process = []
    
    # For each reasoning step N (starting from step 1)
    for step_n in range(1, max_step + 1):
        if step_n not in step_to_thoughts:
            continue
        
        # --- GAP BRIDGING: Find nearest non-empty previous step ---
        prev_step_n = step_n - 1
        while prev_step_n > 0 and prev_step_n not in step_to_thoughts:
            prev_step_n -= 1
        
        thoughts_prev_step = step_to_thoughts.get(prev_step_n, [])
        thoughts_n = step_to_thoughts[step_n]
        
        # Calculate anchor of previous step
        anchor_prev_step = max(thoughts_prev_step) if thoughts_prev_step else None
        
        # For each thought in step N
        for thought_n in thoughts_n:
            if thought_n == 0:  # Skip T0 (root)
                continue
            if thought_n >= len(thoughts):
                continue
            
            text_n = truncate_text(thoughts[thought_n], MAX_THOUGHT_LENGTH)
            
            # Build candidate parents
            candidates = []
            for thought_prev in thoughts_prev_step:
                if thought_prev >= len(thoughts):
                    continue
                
                text_prev = truncate_text(thoughts[thought_prev], MAX_CANDIDATE_LENGTH)
                candidates.append({"id": thought_prev, "text": text_prev})
            
            if candidates:
                # PREFIX CACHING: Static data first
                user_content = json.dumps({
                    "candidate_parents": candidates,
                    "step_n": step_n,
                    "current_thought": text_n
                }, ensure_ascii=False)
                
                queries_to_process.append((
                    thought_n, candidates, user_content, anchor_prev_step
                ))
    
    # Process queries with INNER THREAD POOL (5 workers)
    def find_parents(query_data):
        thought_n, candidates, user_content, anchor_prev_step = query_data
        
        try:
            messages = [
                {"role": "system", "content": STEP4_SYSTEM_MESSAGE},
                {"role": "user", "content": user_content}
            ]
            response, in_tokens, out_tokens, cache_hits = llm_client.generate(messages=messages)
            parsed = extract_and_parse_json(response)
            
            parents = []
            llm_selected = False
            
            if parsed and "parents" in parsed:
                for parent_info in parsed["parents"]:
                    parent_id = parent_info.get("id")
                    category = parent_info.get("category", "Continuous Logic")
                    
                    # Verify valid candidate
                    if not any(c['id'] == parent_id for c in candidates):
                        continue
                    
                    cat_num = CATEGORY_MAP.get(category, 1)
                    parents.append((parent_id, cat_num))
                    llm_selected = True
            
            # LCoT STRUCTURAL FALLBACK
            if not parents and anchor_prev_step is not None:
                parents.append((anchor_prev_step, 5))
            elif not parents:
                parents.append((thought_n - 1, 5))
            
            return thought_n, parents, in_tokens, out_tokens, cache_hits
            
        except Exception as e:
            print(f"Error finding parents for thought {thought_n}: {e}")
            if anchor_prev_step is not None:
                return thought_n, [(anchor_prev_step, 5)], 0, 0, 0
            else:
                return thought_n, [(thought_n - 1, 5)], 0, 0, 0
    
    # Execute with controlled inner pool
    if queries_to_process:
        inner_workers = min(5, len(queries_to_process))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=inner_workers) as executor:
            query_results = list(executor.map(find_parents, queries_to_process))
        
        # Aggregate results
        for thought_n, parents, in_tokens, out_tokens, cache_hits in query_results:
            for parent_id, category in parents:
                if parent_id not in item["thought_relations"]:
                    item["thought_relations"][parent_id] = {}
                item["thought_relations"][parent_id][thought_n] = category
            
            item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
            item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            item["cache_hit_tokens"] = item.get("cache_hit_tokens", 0) + cache_hits
    
    return item