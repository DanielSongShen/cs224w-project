"""Step 1: Split Thoughts

Splits reasoning text into atomic thoughts using split words as markers.
"""
import time
from typing import List, Dict, Any
from pathlib import Path

from ..config import SPLIT_WORDS
from ..utils import split_text, save_jsonl


def process_split(
    input_data: List[Dict[str, Any]],
    output_dir: Path,
    split_words: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Split reasoning text into atomic thoughts.
    
    Args:
        input_data: List of items with 'prediction' field
        output_dir: Directory to save intermediate results
        split_words: Custom split words (optional, uses default from config)
    
    Returns:
        List of items with added 'thoughts_list' field
    """
    print("\n=== Step 1: Splitting thoughts ===")
    start_time = time.time()
    
    if split_words is None:
        split_words = SPLIT_WORDS
    
    results = []
    
    for item in input_data:
        text = item["prediction"]
        
        # Extract from think tags if present
        if text.startswith("<think>"):
            text = (text.split("<think>")[1]).split("</think>")[0]
        else:
            text = text.split("</think>")[0]
        
        # Split into thoughts
        thought_parts = split_text(text, split_words)
        
        if len(thought_parts) == 0:
            print(f"Warning: No thoughts found for {item['tag']}")
            continue
        
        # Convert to dict with integer keys
        thoughts_dict = {i: part for i, part in enumerate(thought_parts)}
        item["thoughts_list"] = thoughts_dict
        results.append(item)
    
    # Save intermediate result
    output_path = output_dir / "process1.json"
    save_jsonl(results, output_path)
    
    elapsed = time.time() - start_time
    print(f"Saved {len(results)} items to {output_path}")
    print(f"⏱️  Step 1 completed in {elapsed:.2f} seconds")
    
    return results