"""Wrapper for LCoT2Tree pipeline with configurable LLM backend - MODIFIED VERSION

Key changes:
- Step 4: Pairwise comparison of thoughts between consecutive reasoning steps
- Step 4: Added "Unrelated" as 5th category (encouraged to be used liberally)
- Step 5: Build edges based on thought relations, skip "Unrelated" connections
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
    # SPLIT_WORDS = [
    #     "Alternatively", "Wait, no", "Hmm", "But wait", "Let me verify",
    #     "let's verify", "Or wait", "To verify", "Wait", "Verify",
    #     "Let's confirm", "Let's check", "Another example", "But let's",
    #     "No:", "no:"
    # ]
    SPLIT_WORDS = [
        "\n\n"
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
            
            prompt = prompt_template.format(text=text)
            
            try:
                response, in_tokens, out_tokens = self.llm_client.generate(prompt)
                item["reasoning_sketch"] = response
                item["in_token_cost"] = in_tokens
                item["out_token_cost"] = out_tokens
            except Exception as e:
                print(f"Error processing {item['tag']}: {e}")
                item["reasoning_sketch"] = ""
                item["in_token_cost"] = 0
                item["out_token_cost"] = 0
            
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
        total_api_calls = len(results)
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
        print("\n=== Step 3: Assigning thoughts to steps ===")
        start_time = time.time()
        
        # System message (cacheable instructions)
        system_message = """Your task is to match each reasoning thought from List B to corresponding step number(s) in List A. Follow the following process:

        1. FIRST UNDERSTAND LIST B:
        - For each thought in List B, identify if it describes some SPECIFIC CALCULATION PROCESSes (mathematical operation, logical transformation, or data manipulation)
        - Ignore the descriptions that only state conclusions, concepts without showing the actual processing detail

        2. THEN MATCH TO LIST A:
        - For each thought from List B, find all steps in List A that:
            * Show the same underlying calculation (even with different numbers/words)
            * Represent the partial or same reasoning process
        - Ignore superficial wording differences - focus on logical equivalence

        3. OUTPUT REQUIREMENTS:
        - Return ALL plausible matches where computational processes align
        - Never return empty arrays (except for thought B0 if needed)
        - Multiple matches are encouraged when justified
        - Maintain strict JSON format

        Output Format (strict JSON):
        ```json
        {{
        "B0": ["A1"],
        "B1": ["A3"],
        "B2": ["A1", "A4"],
        ...
        }}```

        Please match the reasoning thoughts in List B to steps in List A."""
        
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
        
        def extract_and_parse_json(text):
            """Extract and parse JSON from response"""
            # Remove comments
            text = re.sub(r'//.*', '', text)
            text = re.sub(r'#.*', '', text)
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
            
            # Try to extract from code block
            json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
            json_match = json_pattern.search(text)
            
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON object
                json_pattern = re.compile(r'\{\s*(.*?)\s*\}', re.DOTALL)
                json_match = json_pattern.search(text)
                if json_match:
                    json_text = "{" + json_match.group(1) + "}"
                else:
                    return None
            
            try:
                return json.loads(json_text)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {json_text[:100]}")
                return None
        
        def process_item(item):
            # Normalize keys to strings
            thought_list = self.normalize_thought_keys(item["thoughts_list"])
            
            reasoning_sketch = extract_reasoning_dict(item["reasoning_sketch"])
            reasoning_text = json.dumps(reasoning_sketch, ensure_ascii=False)
            
            # First pass: identify all chunks to process
            chunks_to_process = []
            new_dict = {}
            thought_num = len(thought_list)
            
            for i in range(thought_num):
                new_dict[i] = thought_list[str(i)]
                thought_seg = json.dumps(new_dict, ensure_ascii=False)
                
                # Process when chunk is large enough or at end
                if len(thought_seg.split(" ")) > 600 or i == thought_num - 1:
                    # Create user message with variable data
                    user_content = f"""- List A (Detailed Steps): 
                    <list_a>
                    {reasoning_text}
                    </list_a>

                    - List B (Reasoning Thoughts): 
                    <list_b>
                    {thought_seg}
                    </list_b>"""
                    chunks_to_process.append(user_content)
                    new_dict = {}
            
            # Second pass: process all chunks in parallel
            def process_single_chunk(user_content):
                try:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ]
                    response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
                    assignments = extract_and_parse_json(response)
                    return assignments if assignments else {}, in_tokens, out_tokens
                except Exception as e:
                    print(f"Error processing chunk for {item['tag']}: {e}")
                    return {}, 0, 0
            
            # Execute all chunks in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks_to_process), 10)) as executor:
                chunk_results = list(executor.map(process_single_chunk, chunks_to_process))
            
            # Aggregate all results
            all_assignments = {}
            for assignments, in_tokens, out_tokens in chunk_results:
                all_assignments.update(assignments)
                item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
            
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
        # Estimate API calls (1-3 per sample, typically 2)
        est_api_calls = len(results) * 2
        print(f"Saved {len(results)} items to {output_path}")
        print(f"⏱️  Step 3 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step4_assign_functions(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4: Assign pairwise relationships between thoughts across reasoning steps.
        
        MODIFIED: For each reasoning step N, compare all thoughts assigned to step N with
        all thoughts assigned to step N-1 pairwise. Added "Unrelated" as 5th category.
        
        Args:
            input_data: Items with 'thoughts_list' and 'assigned_step'
        
        Returns:
            Items with added 'thought_relations' field
        """
        print("\n=== Step 4: Assigning pairwise thought relations ===")
        start_time = time.time()
        
        # System message with 5 categories including "Unrelated"
        system_message = """Your task is to classify the relationship between Text2 (from reasoning step N) and Text1 (from reasoning step N-1).

        CRITICAL: Only classify thoughts as related if there is a CLEAR, DIRECT connection. When in doubt, choose "Unrelated".

        Categories:
        1. Continuous Logic - Text2 directly continues or extends Text1's reasoning flow with the same line of thought
        2. Exploration - Text2 introduces alternative reasoning paths or explores parallel concepts branching from Text1
        3. Backtracking - Text2 revises, corrects, or adjusts something stated in Text1
        4. Validation - Text2 provides supporting evidence, verification, or examples specifically for Text1's claims
        5. Unrelated - Text2 has no meaningful connection to Text1, or the connection is too weak/indirect (USE THIS LIBERALLY)

        Guidelines:
        - Be conservative: if the connection isn't obvious and direct, classify as "Unrelated"
        - Most thought pairs should be "Unrelated" - only connect when there's a clear semantic link
        - Different topics, different sub-problems, or independent calculations should be "Unrelated"
        - Mere proximity in the text does not imply a relationship

        Output Format:
        Return only JSON format ```json{{"Category": "Name of Category"}}```"""
        
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
            
            # Initialize relations structure: {thought_i: {thought_j: category}}
            item["thought_relations"] = {}
            
            # Prepare all pairwise comparisons
            comparisons_to_process = []
            
            # For each reasoning step N (starting from step 2)
            for step_n in range(2, max_step + 1):
                if step_n not in step_to_thoughts:
                    continue
                if (step_n - 1) not in step_to_thoughts:
                    continue
                
                thoughts_n = step_to_thoughts[step_n]
                thoughts_n_minus_1 = step_to_thoughts[step_n - 1]
                
                # Compare each thought in step N with each thought in step N-1
                for thought_n in thoughts_n:
                    for thought_prev in thoughts_n_minus_1:
                        if thought_n >= len(thoughts) or thought_prev >= len(thoughts):
                            continue
                        
                        text_prev = thoughts[thought_prev]
                        text_n = thoughts[thought_n]
                        
                        # Truncate very long thoughts to avoid token limits
                        if len(text_prev) > 500:
                            text_prev = text_prev[:500] + "..."
                        if len(text_n) > 500:
                            text_n = text_n[:500] + "..."
                        
                        user_content = f'{{"Text1 (Step {step_n-1})": "{text_prev}", "Text2 (Step {step_n})": "{text_n}"}}'
                        comparisons_to_process.append((thought_prev, thought_n, user_content))
            
            # Process all comparisons in parallel
            def classify_pair(comparison_data):
                thought_prev, thought_n, user_content = comparison_data
                try:
                    messages = [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ]
                    response, in_tokens, out_tokens = self.llm_client.generate(messages=messages)
                    parsed = extract_and_parse_json(response)
                    
                    if parsed and "Category" in parsed:
                        category = parsed["Category"]
                        category_map = {
                            "Continuous Logic": 1,
                            "Exploration": 2,
                            "Backtracking": 3,
                            "Validation": 4,
                            "Unrelated": 5,
                        }
                        result = category_map.get(category, 5)  # Default to Unrelated
                    else:
                        result = 5  # Default to Unrelated
                    
                    return thought_prev, thought_n, result, in_tokens, out_tokens
                    
                except Exception as e:
                    print(f"Error classifying pair ({thought_prev}, {thought_n}) for {item['tag']}: {e}")
                    return thought_prev, thought_n, 5, 0, 0
            
            # Execute in parallel
            if comparisons_to_process:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(comparisons_to_process), 20)) as executor:
                    results = list(executor.map(classify_pair, comparisons_to_process))
                
                # Aggregate results
                for thought_prev, thought_n, category, in_tokens, out_tokens in results:
                    if thought_prev not in item["thought_relations"]:
                        item["thought_relations"][thought_prev] = {}
                    item["thought_relations"][thought_prev][thought_n] = category
                    
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
        # Calculate actual API calls
        total_api_calls = sum(
            sum(len(relations) for relations in item.get("thought_relations", {}).values())
            for item in results
        )
        print(f"Saved {len(results)} items to {output_path}")
        print(f"Total pairwise comparisons: {total_api_calls}")
        print(f"⏱️  Step 4 completed in {elapsed:.2f} seconds")
        
        return results
    
    def step5_build_tree(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Build graph structure from thought relations.
        
        MODIFIED: Build edges based on thought_relations, skipping "Unrelated" (category 5).
        The structure maintains tree format for backward compatibility but is built from relations.
        
        Args:
            input_data: Items with 'thought_relations' from step 4
        
        Returns:
            Items with added 'cot_tree' field
        """
        print("\n=== Step 5: Building relation-based trees ===")
        start_time = time.time()
        
        def build_relation_tree(item):
            """Build tree structure from thought relations"""
            thought_relations = item.get("thought_relations", {})
            
            # Convert keys to integers if needed
            if isinstance(thought_relations, str):
                thought_relations = json.loads(thought_relations)
            
            # Ensure keys are integers
            relations = {}
            for src_key, targets in thought_relations.items():
                src = int(src_key) if not isinstance(src_key, int) else src_key
                relations[src] = {}
                for tgt_key, category in targets.items():
                    tgt = int(tgt_key) if not isinstance(tgt_key, int) else tgt_key
                    relations[src][tgt] = category
            
            # Build adjacency list of related thoughts (exclude category 5 = Unrelated)
            edges = {}
            for src, targets in relations.items():
                for tgt, category in targets.items():
                    if category != 5:  # Skip Unrelated
                        if src not in edges:
                            edges[src] = []
                        edges[src].append((tgt, category))
            
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
                            # Use format: "thought_id-position" for uniqueness
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
            
            return root
        
        results = []
        for item in input_data:
            try:
                tree_root = build_relation_tree(item)
                tree_dict = tree_to_dict_with_cate(tree_root)
                item["cot_tree"] = tree_dict
                
                # Add statistics
                total_relations = sum(
                    len(targets) 
                    for targets in item.get("thought_relations", {}).values()
                )
                related_edges = sum(
                    sum(1 for cat in targets.values() if cat != 5)
                    for targets in item.get("thought_relations", {}).values()
                )
                unrelated_count = total_relations - related_edges
                
                item["relation_stats"] = {
                    "total_comparisons": total_relations,
                    "related_edges": related_edges,
                    "unrelated_count": unrelated_count,
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
        total_comparisons = sum(r.get("relation_stats", {}).get("total_comparisons", 0) for r in results)
        total_related = sum(r.get("relation_stats", {}).get("related_edges", 0) for r in results)
        total_unrelated = sum(r.get("relation_stats", {}).get("unrelated_count", 0) for r in results)
        
        print(f"Saved {len(results)} trees to {output_path}")
        print(f"Relation statistics:")
        print(f"  Total comparisons: {total_comparisons}")
        print(f"  Related edges: {total_related} ({100*total_related/max(total_comparisons, 1):.1f}%)")
        print(f"  Unrelated: {total_unrelated} ({100*total_unrelated/max(total_comparisons, 1):.1f}%)")
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
        print("Starting LCoT2Tree Pipeline (Modified Version)")
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
        
        # Step 4: Assign pairwise relations (MODIFIED)
        data = self.step4_assign_functions(data)
        
        # Step 5: Build relation-based trees (MODIFIED)
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