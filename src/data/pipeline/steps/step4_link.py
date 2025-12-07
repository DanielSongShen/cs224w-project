"""Step 4: Assign Parent Relationships

The most expensive step. Implements:
- Prefix Caching Optimization (static data first in JSON)
- Incremental Saving (immediate write-and-flush)
- Robust Fallback (gap bridging + LCoT structural)
"""
import json
import time
import concurrent.futures
from concurrent.futures import as_completed
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
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Assign parent relationships for each thought.
    
    OPTIMIZED: Comparison between High-Level Steps with Robust Fallback.
    - Coverage Guarantee: Ensures all thoughts are assigned to steps
    - Gap Bridging: Finds nearest non-empty previous step
    - LCoT Structural Fallback: Connects to anchor of previous step
    - Prefix Caching: Static candidates first in JSON payload
    - Incremental Saving: Immediate write-and-flush on completion
    
    Args:
        input_data: Items with 'thoughts_list' and 'assigned_step'
        llm_client: LLM client instance
        output_dir: Directory to save intermediate results
        max_workers: Number of parallel workers
        debug: Enable detailed debug logging
    
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
    
    print(f"Resuming Step 4. Found {len(processed_tags)} done, {len(queue)} remaining.")
    
    if not queue:
        print("All items already processed!")
        return results
    
    # EXECUTION: Process with immediate write on completion
    # Open in 'a' (append) mode to preserve previous progress
    with open(incremental_path, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single_item, item, llm_client, debug): item 
                for item in queue
            }
            
            # Process completions as they arrive
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # IMMEDIATE SAVE & FLUSH
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    
                    if len(results) % 10 == 0:
                        print(f"  Processed {len(results)}/{len(input_data)} items...")
                
                except Exception as e:
                    original_item = futures[future]
                    print(f"Error processing item {original_item.get('tag', 'unknown')}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Calculate statistics
    elapsed = time.time() - start_time
    total_api_calls = sum(
        sum(len(targets) for targets in item.get("thought_relations", {}).values())
        for item in results
    )
    
    # Save final consolidated file
    output_path = output_dir / "process4.json"
    save_jsonl(results, output_path)
    
    print(f"Saved {len(results)} items to {output_path}")
    print(f"Total parent selection queries: {total_api_calls}")
    print(f"⏱️  Step 4 completed in {elapsed:.2f} seconds")
    
    return results


def _process_single_item(item: Dict[str, Any], llm_client, debug: bool) -> Dict[str, Any]:
    """
    Process a single item to assign parent relationships.
    
    Implements:
    - Coverage guarantee (sanitization)
    - Gap bridging (find nearest non-empty previous step)
    - LCoT structural fallback (connect to anchor)
    - Prefix caching optimization
    
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
    
    # Build reverse mapping: step_id -> [thought_ids]
    step_to_thoughts = build_step_to_thoughts_mapping(assigned_step)
    
    # --- COVERAGE GUARANTEE (Sanitization) ---
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
    
    # Debug: Print step assignments
    if debug:
        print(f"\n{'='*70}")
        print(f"DEBUG: Processing item '{item.get('tag', 'unknown')}'")
        print(f"{'='*70}")
        print(f"Total thoughts: {len(thoughts)}")
        print(f"Step assignments:")
        for step_id in sorted(step_to_thoughts.keys()):
            thought_ids = sorted(step_to_thoughts[step_id])
            print(f"  Step {step_id}: {thought_ids}")
    
    # Initialize relations structure: {parent_id: {child_id: category}}
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
        
        # Calculate anchor of previous step (last thought chronologically)
        anchor_prev_step = max(thoughts_prev_step) if thoughts_prev_step else None
        
        # For each thought in step N, find its parents from previous step
        for thought_n in thoughts_n:
            if thought_n == 0:  # Skip T0 (root node)
                continue
            if thought_n >= len(thoughts):
                continue
            
            text_n = thoughts[thought_n]
            text_n = truncate_text(text_n, MAX_THOUGHT_LENGTH)
            
            # Build candidate parents list
            candidates = []
            for thought_prev in thoughts_prev_step:
                if thought_prev >= len(thoughts):
                    continue
                
                text_prev = thoughts[thought_prev]
                text_prev = truncate_text(text_prev, MAX_CANDIDATE_LENGTH)
                
                candidates.append({
                    "id": thought_prev,
                    "text": text_prev
                })
            
            if candidates:
                # PREFIX CACHING OPTIMIZATION
                # Move static 'candidates' to VERY FRONT of JSON
                user_content = json.dumps({
                    "candidate_parents": candidates,  # <--- STATIC CONTENT FIRST
                    "step_n": step_n,                 # <--- DYNAMIC CONTENT LAST
                    "current_thought": text_n
                }, ensure_ascii=False)
                
                queries_to_process.append((
                    thought_n, 
                    candidates, 
                    user_content, 
                    step_n, 
                    anchor_prev_step
                ))
    
    # Process all queries in parallel
    def find_parents(query_data):
        thought_n, candidates, user_content, step_n, anchor_prev_step = query_data
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
                    
                    # Map category to integer
                    cat_num = CATEGORY_MAP.get(category, 1)
                    parents.append((parent_id, cat_num))
                    llm_selected = True
            
            # --- LCoT STRUCTURAL FALLBACK ---
            fallback_type = None
            if not parents and anchor_prev_step is not None:
                parents.append((anchor_prev_step, 5))  # Category 5 = Default
                fallback_type = "structural"
            elif not parents:
                # Absolute fallback if no previous step anchor exists
                parents.append((thought_n - 1, 5))
                fallback_type = "chronological"
            
            return thought_n, parents, in_tokens, out_tokens, cache_hits, llm_selected, fallback_type
            
        except Exception as e:
            print(f"Error finding parents for thought {thought_n}: {e}")
            # Fallback to anchor of previous step
            if anchor_prev_step is not None:
                return thought_n, [(anchor_prev_step, 5)], 0, 0, 0, False, "structural_error"
            else:
                return thought_n, [(thought_n - 1, 5)], 0, 0, 0, False, "chronological_error"
    
    # Execute in parallel
    if queries_to_process:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries_to_process), 20)) as executor:
            query_results = list(executor.map(find_parents, queries_to_process))
        
        # Aggregate results into thought_relations
        for thought_n, parents, in_tokens, out_tokens, cache_hits, llm_selected, fallback_type in query_results:
            for parent_id, category in parents:
                if parent_id not in item["thought_relations"]:
                    item["thought_relations"][parent_id] = {}
                item["thought_relations"][parent_id][thought_n] = category
            
            item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
            item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            item["cache_hit_tokens"] = item.get("cache_hit_tokens", 0) + cache_hits
    
    return item