"""Step 4: Assign Parent Relationships (REFACTORED VERSION)

The most expensive step. Implements:
- Prefix Caching Optimization (static data first in JSON)
- Incremental Saving (immediate write-and-flush)
- Robust Fallback (gap bridging + LCoT structural)
- Async Support (batched processing like Steps 2 & 3)

CRITICAL FIX:
- Ensures Step 1 thoughts are connected to Root (0) when Step 0 is missing
- Prevents orphaned nodes in the final graph

REFACTORING (Dec 2024):
- Centralized causal anchor logic in main loop (Block A)
- Simplified worker function (no longer determines fallback)
- Pre-calculates best_fallback before worker execution
- Handles edge cases where causal filtering removes all parents

OPTIMIZATIONS (Dec 2024):
- Hidden Steps Fix: max_step accounts for both sketch and actual assignments
- Caching Optimization: Causal filter moved to output (post-hoc) so all thoughts
  in a step see identical candidate lists, maximizing prefix cache hits
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
    debug: bool = False,
    causal: bool = False
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
        causal: If True, only consider parents that occurred chronologically before

    Returns:
        Items with added 'thought_relations' field
    """
    print("\n=== Step 4: Assigning parent relationships (Refactored with Causal Anchor Logic) ===")
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
        return _process_single_item(item, llm_client, debug, causal)
    
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
    debug: bool,
    causal: bool = False
) -> Dict[str, Any]:
    """
    Process a single item to assign parent relationships.

    This is the complex part with multiple LLM calls per item.
    Uses an inner thread pool (5 workers) for queries.

    REFACTORED LOGIC:
    - Block A: Pre-calculates best fallback (Causal Anchor) in main loop
    - Block B: Builds LLM candidate list (CACHING OPTIMIZED - no causal filter at input)
    - Block C: Prepares payload for worker
    - Worker: Simplified - tries LLM, applies causal filter post-hoc, uses pre-calculated fallback

    Implements:
    - Coverage guarantee (sanitization)
    - Gap bridging (find nearest non-empty previous step)
    - **CRITICAL FIX**: Step 1 → Root connection when Step 0 is missing
    - **HIDDEN STEPS FIX**: Loop boundary accounts for both sketch and assignments
    - **CACHING OPTIMIZATION**: Causal filter applied post-hoc for maximum cache hits
    - LCoT structural fallback (connect to anchor)
    - Prefix caching optimization
    - Type normalization (handles old pipeline data)
    - Causal filtering (optional): Only consider chronologically earlier parents

    Args:
        item: Item with thoughts_list and assigned_step
        llm_client: LLM client instance
        debug: Enable debug logging
        causal: If True, only consider parents that occurred chronologically before

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
        found_step = -1  # Default
        
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
    
    # HIDDEN STEPS FIX: Calculate max step from both Sketch AND Actual Assignments
    # This prevents orphaned nodes when steps exist in assignments but not in sketch
    max_sketch_step = max(reasoning_steps.keys()) if reasoning_steps else 0
    max_assigned_step = max(step_to_thoughts.keys()) if step_to_thoughts else 0
    max_step = max(max_sketch_step, max_assigned_step)
    
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
        
        # ============================================================================
        # CRITICAL FIX: Ensure Step 1 thoughts connect to Root (0)
        # ============================================================================
        # When processing Step 1, if no Step 0 exists (thoughts_prev_step is empty),
        # we need to explicitly add Root (thought 0) as a candidate parent.
        # This prevents Step 1 thoughts from becoming orphaned nodes.
        if not thoughts_prev_step and step_n == 1:
            # Step 1 with no previous step → Use Root (0) as parent
            thoughts_prev_step = [0]
            if debug:
                print(f"  Step 1 Fix: Using Root (0) as parent for Step 1 thoughts")
        # ============================================================================
        
        thoughts_n = step_to_thoughts[step_n]
        
        # For each thought in step N
        for thought_n in thoughts_n:
            if thought_n == 0:  # Skip T0 (root)
                continue
            if thought_n >= len(thoughts):
                continue
            
            # =========================================================
            # LOGIC BLOCK A: DETERMINE FALLBACK (PRE-CALCULATION)
            # =========================================================
            # We determine the best fallback NOW, so the worker doesn't have to guess.
            # NOTE: We keep the causal filter here for fallback calculation because
            # this is internal logic, not part of the LLM prompt (no caching impact).
            
            # 1. Gather potential ancestors from previous step
            # If causal=True, we strictly filter for t < thought_n
            # If causal=False, we accept all (standard LCoT behavior)
            valid_ancestors = [
                t for t in thoughts_prev_step 
                if (not causal or t < thought_n)
            ]

            # 2. Select the Anchor
            # If we have ancestors, pick the latest one (Max).
            # If no ancestors exist (e.g. late thought in early step), fallback to Root (0).
            best_fallback = max(valid_ancestors) if valid_ancestors else 0

            if debug and causal and best_fallback == 0 and thought_n > 1:
                print(f"  Causal edge case: Thought {thought_n} has no valid ancestors, using Root (0)")

            # =========================================================
            # LOGIC BLOCK B: BUILD LLM CANDIDATES (CACHING OPTIMIZED)
            # =========================================================
            text_n = truncate_text(thoughts[thought_n], MAX_THOUGHT_LENGTH)
            
            # Build candidate parents for LLM
            # CACHING OPTIMIZATION: We include ALL previous step thoughts here
            # (no causal filter at input) so the candidate list is identical for
            # every thought in this step. This enables Prefix Caching.
            # The causal filter is applied POST-HOC in the worker on LLM output.
            candidates = []
            for thought_prev in thoughts_prev_step:
                if thought_prev >= len(thoughts):
                    continue

                text_prev = truncate_text(thoughts[thought_prev], MAX_CANDIDATE_LENGTH)
                candidates.append({"id": thought_prev, "text": text_prev})
            
            # =========================================================
            # LOGIC BLOCK C: PREPARE PAYLOAD
            # =========================================================
            user_content = None
            if candidates:
                # PREFIX CACHING: Static data first
                user_content = json.dumps({
                    "candidate_parents": candidates,
                    "step_n": step_n,
                    "current_thought": text_n
                }, ensure_ascii=False)
            
            # Queue the job with pre-calculated fallback
            # Note: We pass 'best_fallback' explicitly as 4th element
            queries_to_process.append((
                thought_n, candidates, user_content, best_fallback
            ))
    
    # =========================================================
    # SIMPLIFIED WORKER FUNCTION
    # =========================================================
    # The worker is now extremely clean. It has no logic about "causality" or "anchors"
    # - it just executes the LLM call and uses the pre-calculated fallback if needed.
    
    def find_parents(query_data):
        # Unpack the pre-calculated fallback_id
        thought_n, candidates, user_content, fallback_id = query_data
        
        parents = []
        in_tokens = 0
        out_tokens = 0
        cache_hits = 0

        # 1. ATTEMPT LLM (Only if content exists)
        if user_content:
            try:
                messages = [
                    {"role": "system", "content": STEP4_SYSTEM_MESSAGE},
                    {"role": "user", "content": user_content}
                ]
                response, in_tokens, out_tokens, cache_hits = llm_client.generate(messages=messages)
                parsed = extract_and_parse_json(response)
                
                if parsed and "parents" in parsed:
                    for p in parsed["parents"]:
                        pid = p.get("id")
                        
                        # CACHING OPTIMIZATION: Apply causal filter POST-HOC on output
                        # This allows input to be identical across thoughts (for caching)
                        # while still enforcing the causal constraint on final edges
                        if causal and pid >= thought_n:
                            if debug:
                                print(f"  Causal filter (post-hoc): Rejecting parent {pid} >= {thought_n}")
                            continue
                        
                        # Validate parent is in our candidates list
                        if any(c['id'] == pid for c in candidates):
                            category = p.get("category", "Continuous Logic")
                            cat_num = CATEGORY_MAP.get(category, 1)
                            parents.append((pid, cat_num))

            except Exception as e:
                print(f"LLM Error on thought {thought_n}: {e}")

        # 2. APPLY FALLBACK (If LLM failed, skipped, or returned nothing)
        if not parents:
            # We blindly trust the pre-calculated fallback from the main loop.
            # It already accounts for Step 1 orphans, Causal filtering, and Anchors.
            parents.append((fallback_id, 5))  # Category 5 = Fallback
        
        return thought_n, parents, in_tokens, out_tokens, cache_hits
    
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