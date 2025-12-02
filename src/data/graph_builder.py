"""Wrapper for LCoT2Tree pipeline with configurable LLM backend - OPTIMIZED VERSION

Key changes:
- Step 4: Parent selection approach (O(N) instead of O(N²) API calls)
  * For each thought in step N, select parents from candidates in step N-1
  * LLM must return at least one parent per thought
  * Supports multiple parents for complex reasoning dependencies
  * Uses "Default" category (5) for fallback connections
- Step 5: Guaranteed connectivity with fallback mechanism
  * All thoughts guaranteed to be in tree
  * Missing nodes automatically connected to most recent thought from previous step
  * Fallback connections use "Default" category (5)

Categories:
1. Continuous Logic - Direct continuation of reasoning
2. Exploration - Alternative paths or branches
3. Backtracking - Revisions or corrections
4. Validation - Supporting evidence
5. Default - Automatic fallback connections (when LLM fails or for missing nodes)
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

from .llm_client import LLMClient, create_llm_client


# Tree utility classes (from LCoT2Tree)
class TreeNode:
    """Tree node for representing Chain-of-Thought structure"""
    def __init__(self, value, level, text=None, is_critical=False, father=None, 
                 children=None, cate=None, thought_list=None):
        self.value = value
        self.level = level
        self.text = text
        self.is_critical = is_critical
        self.cate = cate
        self.father = father
        self.thought_list = thought_list
        self.children = children if children is not None else []


def tree_to_dict_with_cate(node):
    """Convert tree node to dictionary with category information"""
    return {
        "value": node.value,
        "level": node.level,
        "cate": node.cate,
        "thought_list": node.thought_list,
        "children": [tree_to_dict_with_cate(child) for child in node.children]
    }


class LCoT2TreePipeline:
    """Pipeline for processing reasoning traces into CoT trees"""
    
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
        llm_client: LLMClient,
        output_dir: str,
        max_workers: int = 50,
        use_async: bool = False,
        batch_size: int = 10
    ):
        """
        Initialize LCoT2Tree pipeline.
        
        Args:
            llm_client: LLM client for API calls
            output_dir: Directory to store intermediate and final outputs
            max_workers: Number of parallel workers for LLM calls (sync mode)
            use_async: Whether to use async batch processing
            batch_size: Batch size for async processing
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.use_async = use_async
        self.batch_size = batch_size
    
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
        
        OPTIMIZED: For each thought in reasoning step N, identify its parents from step N-1
        in a single LLM call (reduces from O(N²) to O(N) API calls).
        
        Args:
            input_data: Items with 'thoughts_list' and 'assigned_step'
        
        Returns:
            Items with added 'thought_relations' field
        """
        print("\n=== Step 4: Assigning parent relationships ===")
        start_time = time.time()

        # New prompt that selects parents from candidates
        system_message = """You are analyzing the thought dependencies in a chain of reasoning.

                        For the Current Thought from reasoning step N, you will be given a list of Candidate Parents from step N-1.
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
            
            # Get reasoning steps
            reasoning_steps = extract_reasoning_dict(item.get("reasoning_sketch", ""))
            max_step = max(reasoning_steps.keys()) if reasoning_steps else 0
            
            # Initialize relations structure: {parent_id: {child_id: category}}
            item["thought_relations"] = {}
            
            # Prepare all parent selection queries
            queries_to_process = []
            
            # For each reasoning step N (starting from step 2)
            for step_n in range(2, max_step + 1):
                if step_n not in step_to_thoughts:
                    continue
                if (step_n - 1) not in step_to_thoughts:
                    continue
                
                thoughts_n = step_to_thoughts[step_n]
                thoughts_n_minus_1 = step_to_thoughts[step_n - 1]
                
                # For each thought in step N, find its parents from step N-1
                for thought_n in thoughts_n:
                    if thought_n >= len(thoughts):
                        continue
                    
                    text_n = thoughts[thought_n]
                    
                    # Truncate very long current thought
                    if len(text_n) > 500:
                        text_n = text_n[:500] + "..."
                    
                    # Build candidate parents list
                    candidates = []
                    for idx, thought_prev in enumerate(thoughts_n_minus_1):
                        if thought_prev >= len(thoughts):
                            continue
                        text_prev = thoughts[thought_prev]
                        # Truncate very long candidate
                        if len(text_prev) > 300:
                            text_prev = text_prev[:300] + "..."
                        candidates.append({
                            "id": thought_prev,
                            "text": text_prev
                        })
                    
                    if candidates:
                        user_content = json.dumps({
                            "step_n": step_n,
                            "current_thought": text_n,
                            "candidate_parents": candidates
                        }, ensure_ascii=False)
                        
                        queries_to_process.append((thought_n, candidates, user_content, step_n))
            
            # Process all queries in parallel
            def find_parents(query_data):
                thought_n, candidates, user_content, step_n = query_data
                try:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ]
                    response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
                    parsed = extract_and_parse_json(response)
                    
                    parents = []
                    if parsed and "parents" in parsed:
                        for parent_info in parsed["parents"]:
                            parent_id = parent_info.get("id")
                            category = parent_info.get("category", "Continuous Logic")
                            
                            # Map category to integer
                            category_map = {
                                "Continuous Logic": 1,
                                "Exploration": 2,
                                "Backtracking": 3,
                                "Validation": 4,
                            }
                            cat_num = category_map.get(category, 1)
                            parents.append((parent_id, cat_num))
                    
                    # Fallback: if no parents selected, use most recent thought from previous step
                    if not parents:
                        # Get the last candidate (most recent from previous step)
                        most_recent = candidates[-1]["id"]
                        parents.append((most_recent, 5))  # Category 5 = Default (fallback)
                    
                    return thought_n, parents, in_tokens, out_tokens
                    
                except Exception as e:
                    print(f"Error finding parents for thought {thought_n} in {item['tag']}: {e}")
                    # Fallback to most recent from previous step
                    most_recent = candidates[-1]["id"] if candidates else 0
                    return thought_n, [(most_recent, 5)], 0, 0  # Category 5 = Default
            
            # Execute in parallel
            if queries_to_process:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(queries_to_process), 20)) as executor:
                    results = list(executor.map(find_parents, queries_to_process))
                
                # Aggregate results into thought_relations
                # Structure: {parent_id: {child_id: category}}
                for thought_n, parents, in_tokens, out_tokens in results:
                    for parent_id, category in parents:
                        if parent_id not in item["thought_relations"]:
                            item["thought_relations"][parent_id] = {}
                        item["thought_relations"][parent_id][thought_n] = category
                    
                    item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                    item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            
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
    
    def step5_build_tree(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Build graph structure from parent-child relationships.
        
        OPTIMIZED: Build edges from thought_relations with guaranteed connectivity.
        Each child node is guaranteed to have at least one parent.
        
        Args:
            input_data: Items with 'thought_relations' from step 4
        
        Returns:
            Items with added 'cot_tree' field
        """
        print("\n=== Step 5: Building trees with guaranteed connectivity ===")
        start_time = time.time()
        
        def build_relation_tree(item):
            """Build tree structure from thought relations"""
            thought_relations = item.get("thought_relations", {})
            
            # Convert keys to integers if needed
            if isinstance(thought_relations, str):
                thought_relations = json.loads(thought_relations)
            
            # Ensure keys are integers: {parent_id: {child_id: category}}
            relations = {}
            for src_key, targets in thought_relations.items():
                src = int(src_key) if not isinstance(src_key, int) else src_key
                relations[src] = {}
                for tgt_key, category in targets.items():
                    tgt = int(tgt_key) if not isinstance(tgt_key, int) else tgt_key
                    relations[src][tgt] = category
            
            # Build adjacency list: parent -> [(child, category)]
            edges = {}
            all_children = set()
            for parent, targets in relations.items():
                for child, category in targets.items():
                    if parent not in edges:
                        edges[parent] = []
                    edges[parent].append((child, category))
                    all_children.add(child)
            
            # Get assigned_step for fallback parent assignment
            assigned_step = item.get("assigned_step", {})
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
            
            # Build tree starting from thought 0 (root)
            root = TreeNode("0", 0, cate=0, thought_list=[0])
            
            # Track which thoughts have been added to avoid cycles
            added_thoughts = {0}
            
            # BFS to build tree structure
            queue = [(root, 0)]  # (node, thought_id)
            
            while queue:
                parent_node, parent_thought = queue.pop(0)
                
                # Get children of this thought
                if parent_thought in edges:
                    for child_thought, category in edges[parent_thought]:
                        if child_thought not in added_thoughts:
                            # Create child node
                            child_value = f"{child_thought}-0"
                            child_level = parent_node.level + 1
                            
                            child_node = TreeNode(
                                child_value,
                                child_level,
                                father=parent_node,
                                cate=category,
                                thought_list=[child_thought]
                            )
                            
                            parent_node.children.append(child_node)
                            added_thoughts.add(child_thought)
                            queue.append((child_node, child_thought))
            
            # CONNECTIVITY GUARANTEE: Ensure all thoughts are in the tree
            # If any thought is missing, connect it to the most recent thought from previous reasoning step
            thought_list = self.normalize_thought_keys(item["thoughts_list"])
            total_thoughts = len(thought_list)
            
            missing_thoughts = set(range(total_thoughts)) - added_thoughts
            
            if missing_thoughts:
                # For each missing thought, find its reasoning step and connect to previous step
                for missing_id in sorted(missing_thoughts):
                    # Find which reasoning step this thought belongs to
                    thought_step = None
                    for step_id, thoughts in step_to_thoughts.items():
                        if missing_id in thoughts:
                            thought_step = step_id
                            break
                    
                    if thought_step is None or thought_step == 1:
                        # If not found or in first step, connect to root
                        fallback_parent_node = root
                        fallback_parent_thought = 0
                    else:
                        # Connect to most recent thought from previous step
                        prev_step_thoughts = step_to_thoughts.get(thought_step - 1, [0])
                        fallback_parent_thought = prev_step_thoughts[-1]  # Most recent
                        
                        # Find the node for this parent
                        fallback_parent_node = None
                        search_queue = [root]
                        while search_queue:
                            node = search_queue.pop(0)
                            if node.thought_list and node.thought_list[0] == fallback_parent_thought:
                                fallback_parent_node = node
                                break
                            search_queue.extend(node.children)
                        
                        if fallback_parent_node is None:
                            fallback_parent_node = root
                    
                    # Add missing thought as child
                    child_value = f"{missing_id}-0"
                    child_level = fallback_parent_node.level + 1
                    
                    child_node = TreeNode(
                        child_value,
                        child_level,
                        father=fallback_parent_node,
                        cate=5,  # Category 5 = Default (fallback connection)
                        thought_list=[missing_id]
                    )
                    
                    fallback_parent_node.children.append(child_node)
                    added_thoughts.add(missing_id)
            
            return root
        
        results = []
        for item in input_data:
            try:
                tree_root = build_relation_tree(item)
                tree_dict = tree_to_dict_with_cate(tree_root)
                item["cot_tree"] = tree_dict
                
                # Add statistics
                total_edges = sum(
                    len(targets) 
                    for targets in item.get("thought_relations", {}).values()
                )
                
                item["relation_stats"] = {
                    "total_edges": total_edges,
                    "tree_nodes": self._count_nodes(tree_dict)
                }
                
                results.append(item)
            except Exception as e:
                print(f"Error building tree for {item['tag']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save
        output_path = self.output_dir / "final.json"
        self._save_jsonl(results, output_path)
        
        elapsed = time.time() - start_time
        
        # Print summary statistics
        total_edges = sum(r.get("relation_stats", {}).get("total_edges", 0) for r in results)
        total_nodes = sum(r.get("relation_stats", {}).get("tree_nodes", 0) for r in results)
        
        print(f"Saved {len(results)} trees to {output_path}")
        print(f"Relation statistics:")
        print(f"  Total edges: {total_edges}")
        print(f"  Average nodes per tree: {total_nodes/max(len(results), 1):.1f}")
        print(f"⏱️  Step 5 completed in {elapsed:.2f} seconds")
        
        return results
    
    def _count_nodes(self, tree_dict):
        """Recursively count nodes in tree"""
        count = 1
        for child in tree_dict.get("children", []):
            count += self._count_nodes(child)
        return count
    
    def run_full_pipeline(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the complete LCoT2Tree pipeline.
        
        Args:
            input_data: List of preprocessed reasoning traces
        
        Returns:
            List of items with CoT trees
        """
        print(f"\n{'='*80}")
        print("Starting LCoT2Tree Pipeline (Optimized Version)")
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
        
        # Step 4: Assign parent relationships (OPTIMIZED)
        data = self.step4_assign_functions(data)
        
        # Step 5: Build trees with connectivity guarantee (OPTIMIZED)
        data = self.step5_build_tree(data)
        
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


def run_lcot2tree_pipeline(
    reasoning_traces: List[Dict[str, Any]],
    output_dir: str,
    model_backend: str = "gpt5-nano",
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to run LCoT2Tree pipeline.
    
    Args:
        reasoning_traces: Preprocessed reasoning traces
        output_dir: Output directory for results
        model_backend: LLM backend ("gpt5-nano", "qwen3-4b", "qwen3-32b", "openai", "huggingface")
        model_name: Explicit model name (optional)
        config: Configuration dict for LLM (optional)
        max_workers: Number of parallel workers (for sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing (default: 10)
        **kwargs: Additional arguments for LLM client
    
    Returns:
        List of processed items with CoT trees
    """
    # Create LLM client
    llm_client = create_llm_client(
        backend=model_backend,
        model_name=model_name,
        config=config,
        **kwargs
    )
    
    # Create and run pipeline
    pipeline = LCoT2TreePipeline(
        llm_client=llm_client,
        output_dir=output_dir,
        max_workers=max_workers,
        use_async=use_async,
        batch_size=batch_size
    )
    
    return pipeline.run_full_pipeline(reasoning_traces)