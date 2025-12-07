"""Utility functions for LCoT2Tree Pipeline

Contains:
- Text processing helpers
- Data normalization functions
- File I/O operations
- JSON parsing utilities
"""
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional


def normalize_thought_keys(thought_list: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Normalize thought_list keys to strings.
    
    Handles the case where keys might be integers or strings.
    JSON serialization converts int keys to strings, but in-memory
    dicts might have int keys from enumerate().
    
    Args:
        thought_list: Dictionary with thought IDs as keys
        
    Returns:
        Dictionary with normalized string keys
    """
    if isinstance(thought_list, str):
        thought_list = json.loads(thought_list)
    
    # Convert all keys to strings
    return {str(k): v for k, v in thought_list.items()}


def split_text(text: str, split_words: List[str]) -> List[str]:
    """
    Split text into parts based on split words.
    
    Args:
        text: Input text to split
        split_words: List of marker words/phrases that indicate thought boundaries
        
    Returns:
        List of text segments (thoughts)
    """
    parts = []
    current_part = ""
    i = 0
    
    while i < len(text):
        found = False
        for word in split_words:
            # Only split if we have substantial content (>30 chars) already
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


def extract_and_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from LLM response text.
    
    Handles:
    - JSON within markdown code blocks (```json ... ```)
    - JSON objects without code blocks
    - Comments (// and /* */)
    
    Args:
        text: Response text containing JSON
        
    Returns:
        Parsed JSON dictionary, or None if parsing fails
    """
    # Remove comments
    text = re.sub(r'//.*', '', text)
    text = re.sub(r'#.*', '', text)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Try to find JSON in code block first
    json_pattern = re.compile(r'```json\s*(.*?)\s*```', re.DOTALL)
    json_match = json_pattern.search(text)
    
    if json_match:
        json_text = json_match.group(1)
    else:
        # Try to find standalone JSON object
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


def extract_reasoning_dict(text: str) -> Dict[int, str]:
    """
    Extract reasoning steps from sketch text.
    
    Parses text in format:
    <reasoning_process>
    Step 1. Description...
    Step 2. Description...
    </reasoning_process>
    
    Args:
        text: Reasoning sketch text
        
    Returns:
        Dictionary mapping step number (int) to step description (str)
    """
    # Extract content between tags if present
    start_index = text.find("<reasoning_process>")
    end_index = text.find("</reasoning_process>")
    
    if start_index != -1 and end_index != -1:
        reasoning_text = text[start_index + len("<reasoning_process>"):end_index]
    else:
        reasoning_text = text
    
    # Parse steps using regex
    pattern = re.compile(r'Step (\d+)\.\s*(.*?)(?=(Step \d+\.)|$)', re.DOTALL)
    matches = pattern.findall(reasoning_text)
    
    reasoning_dict = {}
    for match in matches:
        step_num = int(match[0])
        step_text = match[1].strip()
        if step_num not in reasoning_dict:
            reasoning_dict[step_num] = step_text
    
    return reasoning_dict


def normalize_thought_id(thought_key: Any) -> Optional[int]:
    """
    Normalize a thought ID to integer.
    
    Handles various formats:
    - Integer: 5 -> 5
    - String with number: "5" -> 5
    - Mixed format: "Thought5" -> 5
    
    Args:
        thought_key: Thought identifier in any format
        
    Returns:
        Integer thought ID, or None if invalid
    """
    try:
        return int(re.sub(r'[A-Za-z]', '', str(thought_key)))
    except (ValueError, TypeError):
        return None


def normalize_step_id(step_key: Any) -> Optional[int]:
    """
    Normalize a step ID to integer.
    
    Similar to normalize_thought_id but for reasoning steps.
    
    Args:
        step_key: Step identifier in any format
        
    Returns:
        Integer step ID, or None if invalid
    """
    try:
        return int(re.sub(r'[A-Za-z]', '', str(step_key)))
    except (ValueError, TypeError):
        return None


def save_jsonl(data: List[Dict[str, Any]], path: Path, encoding: str = 'utf-8'):
    """
    Save data as JSON Lines format.
    
    Each item is written as a single line of JSON.
    
    Args:
        data: List of dictionaries to save
        path: Output file path
        encoding: Text encoding (default: utf-8)
    """
    with open(path, 'w', encoding=encoding) as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_jsonl(path: Path, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
    """
    Load data from JSON Lines format.
    
    Args:
        path: Input file path
        encoding: Text encoding (default: utf-8)
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(path, 'r', encoding=encoding) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line: {e}")
                    continue
    return data


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to append if truncated (default: "...")
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + suffix


def clean_thought_text(text: str, split_words: List[str]) -> str:
    """
    Clean thought text by removing split word prefixes.
    
    Args:
        text: Thought text
        split_words: List of split words to remove from start
        
    Returns:
        Cleaned text
    """
    for word in split_words:
        if text.startswith(word):
            return text[len(word):].lstrip(' \t\n\r.,;:!?')
    return text


def build_step_to_thoughts_mapping(assigned_step: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Build reverse mapping from step_id -> list of thought_ids.
    
    ROBUST: Normalizes string keys from JSON to integers.
    
    Args:
        assigned_step: Dictionary mapping thought_id -> [step_ids]
        
    Returns:
        Dictionary mapping step_id -> [thought_ids]
    """
    step_to_thoughts = {}
    
    for thought_key, step_ids in assigned_step.items():
        # NORMALIZE thought_id to integer (handles JSON string keys)
        thought_id = int(thought_key) if not isinstance(thought_key, int) else thought_key
        
        # Ensure step_ids is a list
        if not isinstance(step_ids, list):
            step_ids = [step_ids]
        
        for step_key in step_ids:
            # NORMALIZE step_id to integer
            step_id = int(step_key) if not isinstance(step_key, int) else step_key
            
            if step_id not in step_to_thoughts:
                step_to_thoughts[step_id] = []
            step_to_thoughts[step_id].append(thought_id)  # Now always int
    
    # Sort thought lists for each step
    for step_id in step_to_thoughts:
        step_to_thoughts[step_id].sort()
    
    return step_to_thoughts


def format_candidates_for_prompt(thoughts: List[str], thought_ids: List[int], max_length: int) -> List[Dict[str, Any]]:
    """
    Format candidate parent thoughts for LLM prompt.
    
    Args:
        thoughts: Full list of thought texts
        thought_ids: IDs of thoughts to include as candidates
        max_length: Maximum length for each thought text
        
    Returns:
        List of candidate dictionaries with 'id' and 'text' fields
    """
    candidates = []
    
    for thought_id in thought_ids:
        if thought_id >= len(thoughts):
            continue
        
        text = thoughts[thought_id]
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        candidates.append({
            "id": thought_id,
            "text": text
        })
    
    return candidates


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path