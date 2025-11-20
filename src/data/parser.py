"""Parse reasoning traces from SYNTHETIC-1 dataset"""
import json
from typing import List, Dict, Any, Optional


def preprocess_for_lcot2tree(
    samples: List[Dict[str, Any]],
    dataset_name: str = "synthetic1"
) -> List[Dict[str, Any]]:
    """
    Preprocess SYNTHETIC-1 samples into LCoT2Tree format.
    
    SYNTHETIC-1 Schema:
        - response_id: unique ID for the response
        - problem_id: unique ID for the problem
        - hf_dataset_name: the HuggingFace dataset name
        - prompt: the question/prompt
        - gold_standard_solution: the ground truth answer
        - llm_response: the model's response (reasoning trace)
        - score: correctness score (float, 0.0 or 1.0)
        - task_type, source, verification_info, metadata: additional fields
    
    Args:
        samples: List of samples from load_filtered_samples (SYNTHETIC-1 format)
        dataset_name: Name prefix for tags (default: "synthetic1")
    
    Returns:
        List of preprocessed examples in LCoT2Tree format with fields:
        - tag: unique identifier
        - prediction: reasoning text (from llm_response)
        - gold: list of ground truth answers (from gold_standard_solution)
        - score: correctness score ("0" or "1", from score field)
        - full_prompt: the original prompt
        - response_id, problem_id, hf_dataset_name: preserved metadata
    """
    preprocessed = []
    
    for idx, sample in enumerate(samples):
        # Extract reasoning text from llm_response (SYNTHETIC-1 field)
        reasoning_text = sample.get("llm_response", "")
        
        # Handle if llm_response is not a string (edge case)
        if not isinstance(reasoning_text, str):
            reasoning_text = str(reasoning_text)
        
        # Check if reasoning is wrapped in think tags
        # If not, wrap it (LCoT2Tree expects <think> tags)
        if reasoning_text.strip() and not reasoning_text.startswith("<think>"):
            reasoning_text = f"<think>{reasoning_text}</think>"
        
        # Extract ground truth from gold_standard_solution (SYNTHETIC-1 field)
        ground_truth = sample.get("gold_standard_solution", "")
        
        # Ensure ground_truth is a list
        if ground_truth is None or ground_truth == "":
            gold_list = [""]
        elif isinstance(ground_truth, list):
            gold_list = [str(g) for g in ground_truth]
        else:
            gold_list = [str(ground_truth)]
        
        # Extract correctness score (SYNTHETIC-1 uses float: 0.0 or 1.0)
        score_value = sample.get("score", 0.0)
        
        # Convert to string "0" or "1" as expected by LCoT2Tree
        if isinstance(score_value, (int, float)):
            score = "1" if score_value > 0.5 else "0"
        elif isinstance(score_value, str):
            # If already a string, try to parse as float
            try:
                score = "1" if float(score_value) > 0.5 else "0"
            except (ValueError, TypeError):
                score = "0"
        else:
            score = "0"
        
        # Create unique tag using response_id if available, otherwise use index
        if "response_id" in sample:
            tag = f"{dataset_name}_{sample['response_id']}"
        else:
            tag = f"{dataset_name}_{idx:05d}"
        
        # Build preprocessed item
        item = {
            "tag": tag,
            "prediction": reasoning_text,
            "gold": gold_list,
            "score": score,
            "id": idx,
        }
        
        # Include prompt (always available in SYNTHETIC-1)
        if "prompt" in sample:
            item["full_prompt"] = sample["prompt"]
        
        # Preserve important metadata from SYNTHETIC-1
        if "response_id" in sample:
            item["response_id"] = sample["response_id"]
        if "problem_id" in sample:
            item["problem_id"] = sample["problem_id"]
        if "hf_dataset_name" in sample:
            item["hf_dataset_name"] = sample["hf_dataset_name"]
        if "task_type" in sample:
            item["task_type"] = sample["task_type"]
        if "source" in sample:
            item["source"] = sample["source"]
        
        preprocessed.append(item)
    
    return preprocessed


def preprocess_openmath_reasoning_for_lcot2tree(
    samples: List[Dict[str, Any]],
    dataset_name: str = "openmath",
    min_pass_rate: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Preprocess OpenMathReasoning samples into LCoT2Tree format.
    
    OpenMathReasoning Schema:
        - problem: the math problem/question
        - generated_solution: the model's reasoning trace (CoT)
        - expected_answer: the ground truth answer
        - problem_type: type of problem (e.g., "math_word_problem")
        - generation_model: model used to generate solution
        - pass_rate_72b_tir: accuracy metric (float 0.0-1.0)
        - inference_mode: generation mode (e.g., "cot", "tir")
        - other metadata fields
    
    Args:
        samples: List of samples from OpenMathReasoning dataset
        dataset_name: Name prefix for tags (default: "openmath")
        min_pass_rate: Optional minimum pass rate threshold for filtering
    
    Returns:
        List of preprocessed examples in LCoT2Tree format with fields:
        - tag: unique identifier
        - prediction: reasoning text (from generated_solution)
        - gold: list of ground truth answers (from expected_answer)
        - score: correctness score ("0" or "1", from pass_rate_72b_tir)
        - full_prompt: the original problem
        - problem_type, generation_model, inference_mode: preserved metadata
    """
    preprocessed = []
    
    for idx, sample in enumerate(samples):
        # Apply pass rate filter if specified
        if min_pass_rate is not None:
            pass_rate = sample.get("pass_rate_72b_tir", 0.0)
            if pass_rate < min_pass_rate:
                continue
        
        # Extract reasoning text from generated_solution
        reasoning_text = sample.get("generated_solution", "")
        
        # Handle if generated_solution is not a string (edge case)
        if not isinstance(reasoning_text, str):
            reasoning_text = str(reasoning_text)
        
        # Wrap in <think> tags if not already wrapped
        if reasoning_text.strip() and not reasoning_text.startswith("<think>"):
            reasoning_text = f"<think>{reasoning_text}</think>"
        
        # Extract ground truth from expected_answer
        ground_truth = sample.get("expected_answer", "")
        
        # Ensure ground_truth is a list
        if ground_truth is None or ground_truth == "":
            gold_list = [""]
        elif isinstance(ground_truth, list):
            gold_list = [str(g) for g in ground_truth]
        else:
            gold_list = [str(ground_truth)]
        
        # Extract correctness score from pass_rate_72b_tir
        pass_rate = sample.get("pass_rate_72b_tir", 0.0)
        
        # Convert to binary score: >0.5 = correct, <=0.5 = incorrect
        if isinstance(pass_rate, (int, float)):
            score = "1" if pass_rate > 0.5 else "0"
        elif isinstance(pass_rate, str):
            try:
                score = "1" if float(pass_rate) > 0.5 else "0"
            except (ValueError, TypeError):
                score = "0"
        else:
            score = "0"
        
        # Create unique tag
        tag = f"{dataset_name}_{idx:06d}"
        
        # Build preprocessed item
        item = {
            "tag": tag,
            "prediction": reasoning_text,
            "gold": gold_list,
            "score": score,
            "id": idx,
        }
        
        # Include problem as full_prompt
        if "problem" in sample:
            item["full_prompt"] = sample["problem"]
        
        # Preserve important metadata from OpenMathReasoning
        if "problem_type" in sample:
            item["problem_type"] = sample["problem_type"]
        if "generation_model" in sample:
            item["generation_model"] = sample["generation_model"]
        if "inference_mode" in sample:
            item["inference_mode"] = sample["inference_mode"]
        if "pass_rate_72b_tir" in sample:
            item["pass_rate_72b_tir"] = sample["pass_rate_72b_tir"]
        
        preprocessed.append(item)
    
    return preprocessed


def save_preprocessed_data(
    preprocessed: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save preprocessed data to JSON file (one item per line).
    
    Args:
        preprocessed: List of preprocessed examples
        output_path: Path to output JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in preprocessed:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(preprocessed)} preprocessed samples to {output_path}")


def load_preprocessed_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load preprocessed data from JSON file.
    
    Args:
        input_path: Path to input JSON file
    
    Returns:
        List of preprocessed examples
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return data