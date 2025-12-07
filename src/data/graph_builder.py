"""Chain-of-Thought Reasoning Graph Pipeline

Converts reasoning traces into structured graphs with the following features:
- Splits reasoning text into atomic thoughts
- Extracts high-level reasoning steps
- Maps thoughts to reasoning steps
- Identifies parent-child relationships between thoughts
- Builds DAG representation supporting multiple parents per node

Categories:
1. Continuous Logic - Direct continuation of reasoning
2. Exploration - Alternative paths or branches
3. Backtracking - Revisions or corrections
4. Validation - Supporting evidence
5. Default - Automatic fallback connections (structural)

Optimizations:
- O(N) API calls instead of O(N²) by comparing only adjacent reasoning steps
- Parallel processing with configurable workers
- Async batch processing support
- Guaranteed connectivity with structural fallback mechanisms

ROBUST FALLBACK IMPROVEMENTS:
- Coverage Guarantee: Ensures all thoughts are assigned to steps (prevents disconnected roots)
- Gap Bridging: Finds nearest non-empty previous step
- LCoT Structural Fallback: Connects to anchor of previous step (creates hierarchical trees, not linear chains)
"""
import os
import sys
import json
import re
import time
import asyncio
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional

# Try to import create_llm_client - will be used if model_backend is passed
try:
    from .llm_client import create_llm_client
except ImportError:
    try:
        from llm_client import create_llm_client
    except ImportError:
        create_llm_client = None


class ReasoningGraphPipeline:
    """Pipeline for processing reasoning traces into graph structures (DAG support)"""
    
    # Split words for thought segmentation
    SPLIT_WORDS = [
        "Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify",
        "let's verify", "Or wait", "To verify", "Wait", "Verify",
        "Let's confirm", "Let's check", "Another example", "But let's",
        "No:", "no:", "\n\n"
    ]
    
    @staticmethod
    def normalize_thought_keys(thought_list: Dict[Any, Any]) -> Dict[str, Any]:
        """
        Normalize thought_list keys to strings.
        
        Handles the case where keys might be integers or strings.
        JSON serialization converts int keys to strings, but in-memory
        dicts might have int keys from enumerate().
        """
        if isinstance(thought_list, str):
            thought_list = json.loads(thought_list)
        
        # Convert all keys to strings
        return {str(k): v for k, v in thought_list.items()}
    
    def __init__(
        self,
        llm_client,
        output_dir: str,
        max_workers: int = 50,
        use_async: bool = False,
        batch_size: int = 10,
        debug: bool = False
    ):
        """
        Initialize reasoning graph pipeline.
        
        Args:
            llm_client: LLM client for API calls (must have .generate(messages) method)
            output_dir: Directory to store intermediate and final outputs
            max_workers: Number of parallel workers for LLM calls (sync mode)
            use_async: Whether to use async batch processing
            batch_size: Batch size for async processing
            debug: Enable detailed debug logging
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.use_async = use_async
        self.batch_size = batch_size
        self.debug = debug
    
    def split_text(self, text: str, split_words: List[str]) -> List[str]:
        """Split text into parts based on split words"""
        parts = []
        current_part = ""
        i = 0
        while i < len(text):
            found = False
            for word in split_words:
                if text[i:].startswith(word) and len(current_part) > 30:
                    parts.append(current_part)
                    current_part = word
                    i += len(word)
                    found = True
                    break
            if not found:
                current_part += text[i]
                i += 1
        if current_part:
            parts.append(current_part)
        return parts
    
    def step1_split_thoughts(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 1: Split reasoning text into thoughts.
        
        Args:
            input_data: List of preprocessed items with 'prediction', 'tag', etc.
        
        Returns:
            List of items with added 'thoughts_list' field
        """
        print("\n=== Step 1: Splitting thoughts ===")
        start_time = time.time()
        results = []
        
        for item in input_data:
            text = item["prediction"]
            
            # Extract from think tags if present
            if text.startswith("<think>"):
                text = (text.split("<think>")[1]).split("</think>")[0]
            else:
                text = text.split("</think>")[0]
            
            # Split into thoughts
            thought_parts = self.split_text(text, self.SPLIT_WORDS)
            
            if len(thought_parts) == 0:
                print(f"Warning: No thoughts found for {item['tag']}")
                continue
            
            # Convert to dict with integer keys
            thoughts_dict = {i: part for i, part in enumerate(thought_parts)}
            item["thoughts_list"] = thoughts_dict
            results.append(item)
        
        # Save intermediate result
        output_path = self.output_dir / "process1.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        print(f"Saved {len(results)} items to {output_path}")
        print(f"⏱️  Step 1 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step2_extract_sketch(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 2: Extract reasoning sketch using LLM.
        
        Args:
            input_data: Items with 'thoughts_list'
        
        Returns:
            Items with added 'reasoning_sketch' field
        """
        print("\n=== Step 2: Extracting reasoning sketch ===")
        start_time = time.time()
        
        prompt_template = """Analyze the following reasoning text and extract a strictly ordered, atomic sequence of key reasoning steps. Focus on extracting the validated, logically essential progression of thoughts while excluding backtracking, rechecks, or redundant details.

        Reasoning text: 
        <reasoning_text>
        {text}
        </reasoning_text>

        Please read the entire text carefully and generate by following these rules:
        1. Find the key steps and the logical flow of reasoning.
        2. Each step must represent a single, indivisible logical action that directly advances the reasoning.
        3. Determine the correct version of the step, ignoring redundant information. A correct step should be able to push the reasoning logic forward and have no errors in itself.
        4. Do not skip steps. Do not merge steps. Use the original phrasing where possible.
        5. Do not include verification steps unless it introduces new constraints.
        6. Organize the steps into a coherent sequence of key reasoning steps and number it sequentially (1., 2., 3., ...).
        7. Maintain strict output format.

        Output format:
        <reasoning_process>
        Step 1. [concise statement]: [Details]
        Step 2. [concise statement]: [Details]
        Step 3. [concise statement]: [Details]
        ...
        </reasoning_process>

        Please list the key reasoning steps of the provided text.
        """
        
        def process_item(item):
            text = item["prediction"]
            if text.startswith("<think>"):
                text = (text.split("<think>")[1]).split("</think>")[0]
            else:
                text = text.split("</think>")[0]
            
            messages = [{"role": "user", "content": prompt_template.format(text=text)}]
            
            try:
                response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
                item["reasoning_sketch"] = response
                item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            except Exception as e:
                print(f"Error extracting sketch for {item['tag']}: {e}")
                item["reasoning_sketch"] = ""
            
            return item
        
        # Process in parallel (sync or async)
        if self.use_async:
            results = asyncio.run(self._process_batch_async(input_data, process_item))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process2.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        print(f"Saved {len(results)} items to {output_path}")
        print(f"⏱️  Step 2 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step3_assign_steps(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 3: Assign thoughts to reasoning steps using LLM.
        
        Args:
            input_data: Items with 'thoughts_list' and 'reasoning_sketch'
        
        Returns:
            Items with added 'assigned_step' field
        """
        print("\n=== Step 3: Assigning thoughts to reasoning steps ===")
        start_time = time.time()
        
        system_message = """Your task is to assign specific thoughts to reasoning steps.

        You will be given:
        1. A numbered reasoning sketch (Step 1, Step 2, etc.)
        2. A list of specific thoughts (Thought 0, Thought 1, etc.)

        For each thought, determine which reasoning step(s) it belongs to based on:
        - Semantic relevance
        - Logical progression
        - Content overlap

        Guidelines:
        - A thought can belong to multiple steps if it spans reasoning boundaries
        - Consider the full context of both the thought and the step
        - Be precise in your assignments

        Output Format:
        ```json
        {
          "Thought 0": [1, 2],
          "Thought 1": [2],
          "Thought 2": [3],
          ...
        }
        ```"""
        
        def extract_and_parse_json(text):
            """Extract and parse JSON from response"""
            text = re.sub(r'//.*', '', text)
            text = re.sub(r'#.*', '', text)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            
            json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
            json_match = json_pattern.search(text)
            
            if json_match:
                json_text = json_match.group(1)
            else:
                json_pattern = re.compile(r'\{\s*(.*?)\s*\}', re.DOTALL)
                json_match = json_pattern.search(text)
                if json_match:
                    json_text = "{" + json_match.group(1) + "}"
                else:
                    return None
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return None
        
        def extract_reasoning_dict(text):
            """Extract reasoning steps from sketch text"""
            start_index = text.find("<reasoning_process>")
            end_index = text.find("</reasoning_process>")
            if start_index == -1 or end_index == -1:
                reasoning_text = text
            else:
                reasoning_text = text[start_index + len("<reasoning_process>"):end_index]
            
            pattern = re.compile(r'Step (\d+)\.\s*(.*?)(?=(Step \d+\.)|$)', re.DOTALL)
            matches = pattern.findall(reasoning_text)
            reasoning_dict = {}
            for match in matches:
                key = int(match[0])
                value = match[1].strip()
                if key not in reasoning_dict:
                    reasoning_dict[key] = value
            return reasoning_dict
        
        def process_item(item):
            # Normalize keys to strings
            thought_list = self.normalize_thought_keys(item["thoughts_list"])
            thoughts = [thought_list[str(i)] for i in range(len(thought_list))]
            
            # Remove split words from start
            for i, thought in enumerate(thoughts):
                for word in self.SPLIT_WORDS:
                    if thought.startswith(word):
                        thoughts[i] = thought[len(word):].lstrip(' \t\n\r.,;:!?')
                        break
            
            # Get reasoning steps
            reasoning_dict = extract_reasoning_dict(item.get("reasoning_sketch", ""))
            
            # Build user content
            reasoning_str = "\n".join([f"Step {k}. {v}" for k, v in reasoning_dict.items()])
            thoughts_str = "\n".join([f"Thought {i}: {t[:200]}..." if len(t) > 200 else f"Thought {i}: {t}" 
                                     for i, t in enumerate(thoughts)])
            
            user_content = f"Reasoning Steps:\n{reasoning_str}\n\nThoughts:\n{thoughts_str}"
            
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            try:
                response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
                parsed = extract_and_parse_json(response)
                
                if parsed:
                    # Normalize the assignment keys
                    all_assignments = {}
                    for key, value in parsed.items():
                        # Extract thought number from key
                        thought_num = re.search(r'\d+', str(key))
                        if thought_num:
                            all_assignments[int(thought_num.group())] = value
                    
                    # === FIX START: Backfill missing thoughts ===
                    # Ensure every thought ID from 0 to len(thoughts)-1 exists
                    for i in range(len(thoughts)):
                        if i not in all_assignments:
                            # Inherit from previous thought, or default to Step 1 if it's the first thought
                            if i > 0:
                                all_assignments[i] = all_assignments.get(i-1, [1])
                            else:
                                all_assignments[i] = [1]
                            
                            # Optional: Mark as inferred for debugging
                            if self.debug:
                                print(f"  Warning: Thought {i} missing from LLM output. Inherited step {all_assignments[i]}")
                    # === FIX END ===

                else:
                    # Fallback: assign each thought to step based on order
                    all_assignments = {}
                    step_count = len(reasoning_dict)
                    for i in range(len(thoughts)):
                        step = min(int(i * step_count / len(thoughts)) + 1, step_count)
                        all_assignments[i] = [step]
                
                item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
                
            except Exception as e:
                print(f"Error assigning steps for {item['tag']}: {e}")
                # Fallback assignment
                step_count = len(reasoning_dict)
                all_assignments = {}
                for i in range(len(thoughts)):
                    step = min(int(i * step_count / len(thoughts)) + 1, step_count)
                    all_assignments[i] = [step]
            
            item["assigned_step"] = all_assignments
            return item
        
        # Process in parallel (sync or async)
        if self.use_async:
            results = asyncio.run(self._process_batch_async(input_data, process_item))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process3.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        print(f"Saved {len(results)} items to {output_path}")
        print(f"⏱️  Step 3 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step4_assign_functions(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4: Assign parent relationships for each thought.
        
        OPTIMIZED: Comparison between High-Level Steps with Robust Fallback.
        - Coverage Guarantee: Ensures all thoughts are assigned to steps (prevents disconnected roots)
        - Gap Bridging: Finds nearest non-empty previous step
        - LCoT Structural Fallback: Connects to anchor of previous step (creates hierarchical trees)
        
        The structural fallback connects thoughts to the last thought of the previous step,
        creating "bushy" tree structures where siblings share a parent, rather than
        linear chains from chronological fallback.
        
        Args:
            input_data: Items with 'thoughts_list' and 'assigned_step'
        
        Returns:
            Items with added 'thought_relations' field
        """
        print("\n=== Step 4: Assigning parent relationships (LCoT Structural) ===")
        start_time = time.time()

        system_message = """You are analyzing the thought dependencies in a chain of reasoning.

                        For the Current Thought from reasoning step N, you will be given a list of Candidate Parents from the previous logical step.
                        Your task: Identify which candidates are directly related to the current thought.

                        Relationship Categories:
                        1. Continuous Logic - Current thought directly continues or extends the parent's reasoning
                        2. Exploration - Current thought branches into alternative paths from the parent
                        3. Backtracking - Current thought revises or corrects the parent
                        4. Validation - Current thought validates or provides evidence for the parent

                        Guidelines:
                        - You MUST select at least ONE parent (the most relevant candidate)
                        - A thought can have MULTIPLE parents if there are multiple clear connections
                        - Select ALL parents with CLEAR, DIRECT connections

                        IMPORTANT: Always return at least one parent. Every thought builds on prior reasoning.

                        Output Format:
                        ```json
                        {
                        "parents": [
                            {"id": <candidate_id>, "category": "<category_name>"},
                            ...
                        ]
                        }
                        ```"""
        
        def extract_and_parse_json(text):
            """Extract and parse JSON from response"""
            text = re.sub(r'//.*', '', text)
            text = re.sub(r'#.*', '', text)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            
            json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
            json_match = json_pattern.search(text)
            
            if json_match:
                json_text = json_match.group(1)
            else:
                json_pattern = re.compile(r'\{\s*(.*?)\s*\}', re.DOTALL)
                json_match = json_pattern.search(text)
                if json_match:
                    json_text = "{" + json_match.group(1) + "}"
                else:
                    return None
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                return None
        
        def extract_reasoning_dict(text):
            """Extract reasoning steps from sketch text"""
            start_index = text.find("<reasoning_process>")
            end_index = text.find("</reasoning_process>")
            if start_index == -1 or end_index == -1:
                reasoning_text = text
            else:
                reasoning_text = text[start_index + len("<reasoning_process>"):end_index]
            
            pattern = re.compile(r'Step (\d+)\.\s*(.*?)(?=(Step \d+\.)|$)', re.DOTALL)
            matches = pattern.findall(reasoning_text)
            reasoning_dict = {}
            for match in matches:
                key = int(match[0])
                value = match[1].strip()
                if key not in reasoning_dict:
                    reasoning_dict[key] = value
            return reasoning_dict
        
        def process_item(item):
            # Normalize keys to strings
            thought_list = self.normalize_thought_keys(item["thoughts_list"])
            thoughts = [thought_list[str(i)] for i in range(len(thought_list))]
            
            # Remove split words from start
            for i, thought in enumerate(thoughts):
                for word in self.SPLIT_WORDS:
                    if thought.startswith(word):
                        thoughts[i] = thought[len(word):].lstrip(' \t\n\r.,;:!?')
                        break
            
            # Get assigned_step mapping (thought_id -> [step_ids])
            assigned_step = item["assigned_step"]
            # Transform to clean integer format
            clean_assigned = {}
            for key, values in assigned_step.items():
                clean_key = int(re.sub(r'[A-Za-z]', '', str(key)))
                clean_values = [int(re.sub(r'[A-Za-z]', '', str(v))) for v in values]
                clean_assigned[clean_key] = clean_values
            
            # Build reverse mapping: step_id -> [thought_ids]
            step_to_thoughts = {}
            for thought_id, step_ids in clean_assigned.items():
                for step_id in step_ids:
                    if step_id not in step_to_thoughts:
                        step_to_thoughts[step_id] = []
                    step_to_thoughts[step_id].append(thought_id)
            
            # --- COVERAGE GUARANTEE (Sanitization) ---
            # Ensures every thought ID [0...N] is assigned to a step.
            # Fixes the "Disconnected Root" issue where thoughts are skipped by LLM.
            all_thought_ids = set(range(len(thoughts)))
            assigned_ids = set()
            for ids in step_to_thoughts.values():
                assigned_ids.update(ids)
            
            missing_ids = sorted(list(all_thought_ids - assigned_ids))
            
            for t_id in missing_ids:
                # Inherit step from previous thought (t_id - 1)
                # If T1 is missing, it inherits T0's step (or defaults to 1)
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
                step_to_thoughts[found_step].sort()  # Maintain order
            # --------------------------------------------
            
            # =========================================================
            # [FIX START] SAVE THE SANITIZED ASSIGNMENTS BACK TO ITEM
            # =========================================================
            # Reconstruct the assignments dictionary from the sanitized step_to_thoughts
            # so the corrections are persisted to final.json
            new_assigned_step = {}
            for step_id, t_ids in step_to_thoughts.items():
                for t_id in t_ids:
                    if t_id not in new_assigned_step:
                        new_assigned_step[t_id] = []
                    # Ensure we don't duplicate steps if rebuilding
                    if step_id not in new_assigned_step[t_id]:
                        new_assigned_step[t_id].append(step_id)
            
            # Update the item
            item["assigned_step"] = new_assigned_step
            # =========================================================
            # [FIX END]
            
            # Get reasoning steps
            reasoning_steps = extract_reasoning_dict(item.get("reasoning_sketch", ""))
            max_step = max(reasoning_steps.keys()) if reasoning_steps else 0
            
            # Debug: Print step assignments
            if self.debug:
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
            
            # Debug tracking
            if self.debug:
                debug_stats = {
                    "llm_selected": 0,
                    "structural_fallback": 0,
                    "chronological_fallback": 0,
                    "edge_details": []
                }
            
            # Prepare all parent selection queries
            queries_to_process = []
            
            # For each reasoning step N (starting from step 1)
            for step_n in range(1, max_step + 1):
                if step_n not in step_to_thoughts:
                    continue
                
                # --- ROBUST IMPROVEMENT 1: Gap Bridging (Find Nearest Non-Empty Previous Step) ---
                # Look backwards until we find a step with thoughts
                prev_step_n = step_n - 1
                while prev_step_n > 0 and prev_step_n not in step_to_thoughts:
                    prev_step_n -= 1
                
                # If we went all the way back and found nothing, use empty list
                thoughts_prev_step = step_to_thoughts.get(prev_step_n, [])
                thoughts_n = step_to_thoughts[step_n]
                
                # Calculate anchor of previous step (last thought chronologically)
                # This is used for LCoT Structural Fallback
                anchor_prev_step = max(thoughts_prev_step) if thoughts_prev_step else None
                
                # Debug: Print anchor info
                if self.debug and thoughts_n:
                    print(f"\n  Processing Step {step_n} (thoughts: {thoughts_n})")
                    print(f"    Previous step: {prev_step_n}, thoughts: {thoughts_prev_step}")
                    print(f"    Anchor: T{anchor_prev_step}" if anchor_prev_step is not None else "    Anchor: None")
                
                # For each thought in step N, find its parents from previous step
                for thought_n in thoughts_n:
                    if thought_n == 0:  # Skip T0 (root node)
                        continue
                    if thought_n >= len(thoughts):
                        continue
                    
                    text_n = thoughts[thought_n]
                    
                    # Truncate very long current thought
                    if len(text_n) > 500:
                        text_n = text_n[:500] + "..."
                    
                    # Build candidate parents list
                    candidates = []
                    
                    # Add Semantic Candidates (from prev_step_n)
                    for thought_prev in thoughts_prev_step:
                        if thought_prev >= len(thoughts):
                            continue
                        
                        text_prev = thoughts[thought_prev]
                        if len(text_prev) > 300:
                            text_prev = text_prev[:300] + "..."
                        
                        candidates.append({
                            "id": thought_prev,
                            "text": text_prev
                        })
                    
                    if candidates:
                        # OPTIMIZED FOR PREFIX CACHING
                        user_content = json.dumps({
                            "candidate_parents": candidates,  # <--- MOVE TO TOP
                            "step_n": step_n,
                            "current_thought": text_n         # <--- MOVE TO BOTTOM
                        }, ensure_ascii=False)
                        
                        queries_to_process.append((thought_n, candidates, user_content, step_n, anchor_prev_step))
            
            # Process all queries in parallel
            def find_parents(query_data):
                thought_n, candidates, user_content, step_n, anchor_prev_step = query_data
                try:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ]
                    response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
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
                            category_map = {
                                "Continuous Logic": 1,
                                "Exploration": 2,
                                "Backtracking": 3,
                                "Validation": 4,
                            }
                            cat_num = category_map.get(category, 1)
                            parents.append((parent_id, cat_num))
                            llm_selected = True
                    
                    # --- ROBUST IMPROVEMENT 2: LCoT Structural Fallback ---
                    # Connect to anchor of previous step (creates hierarchical trees)
                    # instead of T(n-1) (which creates linear chains)
                    fallback_type = None
                    if not parents and anchor_prev_step is not None:
                        parents.append((anchor_prev_step, 5))  # Category 5 = Default (structural)
                        fallback_type = "structural"
                    elif not parents:
                        # Absolute fallback if no previous step anchor exists
                        parents.append((thought_n - 1, 5))
                        fallback_type = "chronological"
                    
                    return thought_n, parents, in_tokens, out_tokens, llm_selected, fallback_type
                    
                except Exception as e:
                    print(f"Error finding parents for thought {thought_n} in {item['tag']}: {e}")
                    # Fallback to anchor of previous step
                    if anchor_prev_step is not None:
                        return thought_n, [(anchor_prev_step, 5)], 0, 0, False, "structural_error"
                    else:
                        return thought_n, [(thought_n - 1, 5)], 0, 0, False, "chronological_error"
            
            # Execute in parallel
            if queries_to_process:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries_to_process), 20)) as executor:
                    results = list(executor.map(find_parents, queries_to_process))
                
                # Aggregate results into thought_relations
                # Structure: {parent_id: {child_id: category}}
                for thought_n, parents, in_tokens, out_tokens, llm_selected, fallback_type in results:
                    for parent_id, category in parents:
                        if parent_id not in item["thought_relations"]:
                            item["thought_relations"][parent_id] = {}
                        item["thought_relations"][parent_id][thought_n] = category
                        
                        # Debug tracking
                        if self.debug:
                            if llm_selected:
                                debug_stats["llm_selected"] += 1
                            elif fallback_type in ["structural", "structural_error"]:
                                debug_stats["structural_fallback"] += 1
                            elif fallback_type in ["chronological", "chronological_error"]:
                                debug_stats["chronological_fallback"] += 1
                            
                            edge_type = "LLM" if llm_selected else f"Fallback({fallback_type})"
                            debug_stats["edge_details"].append({
                                "child": thought_n,
                                "parent": parent_id,
                                "type": edge_type,
                                "category": category
                            })
                    
                    item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                    item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            
            # Debug: Print summary
            if self.debug:
                print(f"\n{'='*70}")
                print(f"DEBUG SUMMARY for '{item.get('tag', 'unknown')}'")
                print(f"{'='*70}")
                print(f"Total edges created: {len(debug_stats['edge_details'])}")
                print(f"  LLM selected: {debug_stats['llm_selected']}")
                print(f"  Structural fallback: {debug_stats['structural_fallback']}")
                print(f"  Chronological fallback: {debug_stats['chronological_fallback']}")
                print(f"\nEdge details:")
                for edge in debug_stats["edge_details"]:
                    cat_name = {1: "Continuous", 2: "Exploration", 3: "Backtracking", 
                               4: "Validation", 5: "Default"}
                    print(f"  T{edge['parent']} → T{edge['child']} [{edge['type']}] ({cat_name.get(edge['category'], edge['category'])})")
                
                # Analyze linear chains
                out_degree = {}
                for edge in debug_stats["edge_details"]:
                    parent = edge["parent"]
                    out_degree[parent] = out_degree.get(parent, 0) + 1
                
                linear_edges = sum(1 for deg in out_degree.values() if deg == 1)
                branching_nodes = sum(1 for deg in out_degree.values() if deg > 1)
                
                print(f"\nStructure Analysis:")
                print(f"  Nodes with 1 child (linear): {linear_edges}")
                print(f"  Nodes with 2+ children (branching): {branching_nodes}")
                print(f"  Max children per node: {max(out_degree.values()) if out_degree else 0}")
                print(f"{'='*70}\n")
            
            return item
        
        # Process in parallel (sync or async)
        if self.use_async:
            results = asyncio.run(self._process_batch_async(input_data, process_item))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process4.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        # Calculate actual API calls (1 per child thought with parents)
        total_api_calls = sum(
            sum(len(targets) for targets in item.get("thought_relations", {}).values())
            for item in results
        )
        print(f"Saved {len(results)} items to {output_path}")
        print(f"Total parent selection queries: {total_api_calls}")
        print(f"⏱️  Step 4 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step5_build_graph(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Build graph structure from parent-child relationships.
        
        Creates a clean edge list representation (DAG-friendly).
        All nodes from thoughts_list are included.
        All edges from thought_relations are preserved (including multiple parents).
        
        Args:
            input_data: Items with 'thought_relations' from step 4
        
        Returns:
            Items with added 'reasoning_graph' field
        """
        print("\n=== Step 5: Building graph structures ===")
        start_time = time.time()
        
        results = []
        for item in input_data:
            try:
                thought_relations = item.get("thought_relations", {})
                
                # Convert keys to integers if needed
                if isinstance(thought_relations, str):
                    thought_relations = json.loads(thought_relations)
                
                # Normalize to integer keys: {parent_id: {child_id: category}}
                relations = {}
                for src_key, targets in thought_relations.items():
                    src = int(src_key) if not isinstance(src_key, int) else src_key
                    relations[src] = {}
                    for tgt_key, category in targets.items():
                        tgt = int(tgt_key) if not isinstance(tgt_key, int) else tgt_key
                        relations[src][tgt] = category
                
                # Build edge list
                edges = []
                all_nodes = set()
                
                for parent_id, targets in relations.items():
                    all_nodes.add(parent_id)
                    for child_id, category in targets.items():
                        all_nodes.add(child_id)
                        edges.append({
                            "source": parent_id,
                            "target": child_id,
                            "category": category
                        })
                
                # Get all thoughts to ensure complete node list
                thought_list = self.normalize_thought_keys(item["thoughts_list"])
                total_thoughts = len(thought_list)
                
                # Create node list (all thoughts)
                nodes = list(range(total_thoughts))
                
                # Build graph structure
                item["reasoning_graph"] = {
                    "nodes": nodes,
                    "edges": edges
                }
                
                # Add statistics
                nodes_with_edges = len(all_nodes)
                isolated_nodes = [n for n in nodes if n not in all_nodes]
                
                # Calculate in-degree and out-degree
                in_degree = {}
                out_degree = {}
                for edge in edges:
                    src = edge["source"]
                    tgt = edge["target"]
                    out_degree[src] = out_degree.get(src, 0) + 1
                    in_degree[tgt] = in_degree.get(tgt, 0) + 1
                
                # Find nodes with multiple parents
                multi_parent_nodes = [node for node, degree in in_degree.items() if degree > 1]
                
                item["graph_stats"] = {
                    "total_nodes": total_thoughts,
                    "nodes_with_edges": nodes_with_edges,
                    "isolated_nodes": len(isolated_nodes),
                    "total_edges": len(edges),
                    "nodes_with_multiple_parents": len(multi_parent_nodes),
                    "avg_in_degree": sum(in_degree.values()) / max(nodes_with_edges, 1),
                    "avg_out_degree": sum(out_degree.values()) / max(nodes_with_edges, 1),
                    "max_in_degree": max(in_degree.values()) if in_degree else 0,
                    "max_out_degree": max(out_degree.values()) if out_degree else 0
                }
                
                results.append(item)
                
            except Exception as e:
                print(f"Error building graph for {item.get('tag', 'unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save
        output_path = self.output_dir / "final.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        
        # Print summary statistics
        total_edges = sum(r.get("graph_stats", {}).get("total_edges", 0) for r in results)
        total_nodes = sum(r.get("graph_stats", {}).get("total_nodes", 0) for r in results)
        isolated = sum(r.get("graph_stats", {}).get("isolated_nodes", 0) for r in results)
        multi_parent = sum(r.get("graph_stats", {}).get("nodes_with_multiple_parents", 0) for r in results)
        
        print(f"Saved {len(results)} graphs to {output_path}")
        print(f"Graph statistics:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total edges: {total_edges}")
        print(f"  Nodes with multiple parents: {multi_parent}")
        print(f"  Isolated nodes: {isolated}")
        print(f"  Average edges per graph: {total_edges/max(len(results), 1):.1f}")
        print(f"⏱️  Step 5 completed in {elapsed:.2f} seconds")
        
        return results
    
    def run_full_pipeline(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the complete reasoning graph pipeline.
        
        Args:
            input_data: List of preprocessed reasoning traces
        
        Returns:
            List of items with reasoning graphs
        """
        print(f"\n{'='*80}")
        print("Starting Reasoning Graph Pipeline (DAG Support with LCoT Structural Fallback)")
        print(f"Processing {len(input_data)} samples")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")
        
        pipeline_start = time.time()
        
        # Step 1: Split thoughts
        data = self.step1_split_thoughts(input_data)
        
        # Step 2: Extract sketch
        data = self.step2_extract_sketch(data)
        
        # Step 3: Assign steps
        data = self.step3_assign_steps(data)
        
        # Step 4: Assign parent relationships (with robust fallback)
        data = self.step4_assign_functions(data)
        
        # Step 5: Build graphs with DAG support
        data = self.step5_build_graph(data)
        
        total_elapsed = time.time() - pipeline_start
        
        print(f"\n{'='*80}")
        print(f"Pipeline complete! Final output: {self.output_dir / 'final.json'}")
        print(f"⏱️  Total pipeline time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        print(f"{'='*80}\n")
        
        return data
    
    async def _process_batch_async(self, items: List[Any], process_func) -> List[Any]:
        """
        Process items in async batches using thread-based concurrency.
        
        Uses asyncio.to_thread() to run synchronous LLM calls concurrently
        in batches. This provides good I/O concurrency for API calls while
        maintaining simple synchronous code in process_func.
        
        Args:
            items: List of items to process
            process_func: Synchronous function to process each item
        
        Returns:
            List of processed results
        """
        results = []
        total = len(items)
        
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total + self.batch_size - 1) // self.batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")
            
            # Process batch concurrently - run sync process_func in thread pool
            batch_results = await asyncio.gather(*[
                asyncio.to_thread(process_func, item)
                for item in batch
            ])
            
            results.extend(batch_results)
        
        return results
    
    def _save_jsonl(self, data: List[Dict[str, Any]], path: Path):
        """Save data as JSONL"""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')


def run_reasoning_pipeline(
    reasoning_traces: List[Dict[str, Any]],
    output_dir: str,
    model_backend: str = None,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    llm_client=None,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10,
    debug: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to run reasoning graph pipeline.
    
    Args:
        reasoning_traces: Preprocessed reasoning traces with 'prediction' and 'tag' fields
        output_dir: Output directory for results
        model_backend: LLM backend name (e.g., "gpt5-nano", "qwen3-4b", "deepseek-v3.2")
        model_name: Explicit model name (optional)
        config: Configuration dict for LLM (optional)
        llm_client: Pre-configured LLM client (optional, alternative to model_backend)
        max_workers: Number of parallel workers (for sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing (default: 10)
        debug: Enable detailed debug logging (default: False)
        **kwargs: Additional arguments for LLM client creation
    
    Returns:
        List of processed items with reasoning graphs
    """
    # Create LLM client if not provided
    if llm_client is None:
        if model_backend is None:
            raise ValueError("Either llm_client or model_backend must be provided")
        
        if create_llm_client is None:
            raise ImportError(
                "create_llm_client not available. Install or import llm_client module."
            )
        
        llm_client = create_llm_client(
            backend=model_backend,
            model_name=model_name,
            config=config,
            **kwargs
        )
    
    # Create and run pipeline
    pipeline = ReasoningGraphPipeline(
        llm_client=llm_client,
        output_dir=output_dir,
        max_workers=max_workers,
        use_async=use_async,
        batch_size=batch_size,
        debug=debug
    )
    
    return pipeline.run_full_pipeline(reasoning_traces)