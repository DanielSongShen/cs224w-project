"""Wrapper for LCoT2Tree pipeline with configurable LLM backend"""
import os
import sys
import json
import re
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
        "No:", "no:"
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
        max_workers: int = 50
    ):
        """
        Initialize LCoT2Tree pipeline.
        
        Args:
            llm_client: LLM client for API calls
            output_dir: Directory to store intermediate and final outputs
            max_workers: Number of parallel workers for LLM calls
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
    
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
        print(f"Saved {len(results)} items to {output_path}")
        
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
Step 1. concise statement: Detail step
Step 2. concise statement: Detail step
Step 3. concise statement: Detail step
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
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process2.json"
        self._save_jsonl(results, output_path)
        print(f"Saved {len(results)} items to {output_path}")
        
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
        
        prompt_template = """Your task is to match each reasoning thought from List B to corresponding step number(s) in the List A. Follow the following process:

1. FIRST UNDERSTAND LIST B:
   - For each thought in List B, identify if it describes some SPECIFIC CALCULATION PROCESSes (mathematical operation, logical transformation, or data manipulation)
   - Ignore the describation that only state conclusions, concepts without showing the actual processing detail

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

Input:
- List A (Detailed Steps): 
<list_a>
{reasoning_step}
</list_a>

- List B (Reasoning Thoughts): 
<list_b>
{thoughts}
</list_b>

Output Format (strict JSON):
```json
{{
  "B0": ["A1"],
  "B1": ["A3"],
  "B2": ["A1", "A4"],
  ...
}}```

Please match the reasoning thoughts in List B to step in the List A.
"""
        
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
            
            # Process thoughts in chunks
            new_dict = {}
            thought_num = len(thought_list)
            
            all_assignments = {}
            
            for i in range(thought_num):
                new_dict[i] = thought_list[str(i)]
                thought_seg = json.dumps(new_dict, ensure_ascii=False)
                
                # Process when chunk is large enough or at end
                if len(thought_seg.split(" ")) > 600 or i == thought_num - 1:
                    prompt = prompt_template.format(
                        reasoning_step=reasoning_text,
                        thoughts=thought_seg
                    )
                    
                    try:
                        response, in_tokens, out_tokens = self.llm_client.generate(prompt)
                        assignments = extract_and_parse_json(response)
                        
                        if assignments:
                            all_assignments.update(assignments)
                        
                        item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                        item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
                    except Exception as e:
                        print(f"Error assigning steps for {item['tag']}: {e}")
                    
                    new_dict = {}
            
            item["assigned_step"] = all_assignments
            return item
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process3.json"
        self._save_jsonl(results, output_path)
        print(f"Saved {len(results)} items to {output_path}")
        
        return results
    
    def step4_assign_functions(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 4: Assign function types to thoughts using LLM.
        
        Args:
            input_data: Items with 'thoughts_list'
        
        Returns:
            Items with added 'thoughts_function' field
        """
        print("\n=== Step 4: Assigning thought functions ===")
        
        prompt_template = """Your task is to classify Text2's purpose relative to Text1 using these categories:

Categories:
1. Continuous Logic - Direct continuation/extension of Text1's reasoning flow
2. Exploration - Introduces parallel/unrelated concepts from Text1, alternative reasoning paths, or new topics
3. Backtracking - Revises, corrects, or adjusts previous step
4. Validation - Provides supporting evidence, logical justification, or examples for Text1's claims

Input:
{{
  "Text1": "{TEXT1}",
  "Text2": "{TEXT2}"
}}

Output Format:
Return only JSON format ```json{{"Category": "Name of Category"}}```
"""
        
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
        
        def process_item(item):
            # Normalize keys to strings
            thought_list = self.normalize_thought_keys(item["thoughts_list"])
            
            # Convert to list
            thoughts = [thought_list[str(i)] for i in range(len(thought_list))]
            
            # Remove split words from start
            for i, thought in enumerate(thoughts):
                for word in self.SPLIT_WORDS:
                    if thought.startswith(word):
                        thoughts[i] = thought[len(word):].lstrip(' \t\n\r.,;:!?')
                        break
            
            # Initialize function assignments
            item["thoughts_function"] = {0: 0, 1: 1}
            
            last_result = 0
            last_context = ""
            
            # Classify each thought
            for i in range(1, len(thoughts) - 1):
                if last_result == 0 or last_result == 1:
                    text1 = last_context + thoughts[i]
                else:
                    text1 = thoughts[i]
                text2 = thoughts[i + 1]
                
                prompt = prompt_template.format(TEXT1=text1, TEXT2=text2)
                
                try:
                    response, in_tokens, out_tokens = self.llm_client.generate(prompt)
                    parsed = extract_and_parse_json(response)
                    
                    if parsed and "Category" in parsed:
                        category = parsed["Category"]
                        index_map = {
                            "Continuous Logic": 1,
                            "Exploration": 2,
                            "Backtracking": 3,
                            "Validation": 4,
                        }
                        result = index_map.get(category, 1)
                    else:
                        result = 1  # Default to Continuous Logic
                    
                    item["in_token_cost"] = item.get("in_token_cost", 0) + in_tokens
                    item["out_token_cost"] = item.get("out_token_cost", 0) + out_tokens
                    
                except Exception as e:
                    print(f"Error classifying thought for {item['tag']}: {e}")
                    result = 1
                
                item["thoughts_function"][i + 1] = result
                last_context = text1
                last_result = result
            
            return item
        
        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_item, input_data))
        
        # Save
        output_path = self.output_dir / "process4.json"
        self._save_jsonl(results, output_path)
        print(f"Saved {len(results)} items to {output_path}")
        
        return results
    
    def step5_build_tree(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Step 5: Build tree structure from processed data.
        
        Args:
            input_data: Items with all processing complete
        
        Returns:
            Items with added 'cot_tree' field
        """
        print("\n=== Step 5: Building trees ===")
        
        def transform_dict(input_dict):
            """Transform assigned_step dict to clean integer format"""
            result = {}
            for key, values in input_dict.items():
                clean_key = int(re.sub(r'[A-Za-z]', '', str(key)))
                clean_values = []
                for value in values:
                    clean_value = int(re.sub(r'[A-Za-z]', '', str(value)))
                    clean_values.append(clean_value)
                result[clean_key] = clean_values
            return result
        
        def generate_tree_with_cate(item):
            """Generate tree structure with categories"""
            assigned_step = transform_dict(item["assigned_step"])
            
            # Normalize thoughts_function keys to strings
            thoughts_function = item["thoughts_function"]
            if isinstance(thoughts_function, str):
                thoughts_function = json.loads(thoughts_function)
            # Ensure all keys are strings for consistent access
            thoughts_function = {str(k): v for k, v in thoughts_function.items()}
            
            root = TreeNode("0", 0, cate=0, thought_list=[0])
            curr_node = root
            
            for i in range(len(thoughts_function)):
                if i not in assigned_step:
                    continue
                if len(assigned_step[i]) == 0:
                    continue
                
                while curr_node.level >= assigned_step[i][0]:
                    curr_node = curr_node.father
                
                for t, j in enumerate(assigned_step[i]):
                    curr_node.children.append(
                        TreeNode(
                            f"{i}-{t}", j, father=curr_node,
                            cate=thoughts_function[str(i)],
                            thought_list=[i]
                        )
                    )
                    curr_node = curr_node.children[-1]
            
            return root
        
        results = []
        for item in input_data:
            try:
                tree_root = generate_tree_with_cate(item)
                tree_dict = tree_to_dict_with_cate(tree_root)
                item["cot_tree"] = tree_dict
                results.append(item)
            except Exception as e:
                print(f"Error building tree for {item['tag']}: {e}")
        
        # Save
        output_path = self.output_dir / "final.json"
        self._save_jsonl(results, output_path)
        print(f"Saved {len(results)} trees to {output_path}")
        
        return results
    
    def run_full_pipeline(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run the complete LCoT2Tree pipeline.
        
        Args:
            input_data: List of preprocessed reasoning traces
        
        Returns:
            List of items with CoT trees
        """
        print(f"\n{'='*80}")
        print("Starting LCoT2Tree Pipeline")
        print(f"Processing {len(input_data)} samples")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")
        
        # Step 1: Split thoughts
        data = self.step1_split_thoughts(input_data)
        
        # Step 2: Extract sketch
        data = self.step2_extract_sketch(data)
        
        # Step 3: Assign steps
        data = self.step3_assign_steps(data)
        
        # Step 4: Assign functions
        data = self.step4_assign_functions(data)
        
        # Step 5: Build trees
        data = self.step5_build_tree(data)
        
        print(f"\n{'='*80}")
        print(f"Pipeline complete! Final output: {self.output_dir / 'final.json'}")
        print(f"{'='*80}\n")
        
        return data
    
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
        max_workers: Number of parallel workers
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
        max_workers=max_workers
    )
    
    return pipeline.run_full_pipeline(reasoning_traces)

