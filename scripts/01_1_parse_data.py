"""Script to parse SYNTHETIC-1 and OpenMathReasoning datasets into graph representations"""
from datasets import load_dataset
import sys
import os
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our modules
from src.data.parser import (
    preprocess_for_lcot2tree,
    preprocess_openmath_reasoning_for_lcot2tree,
    save_preprocessed_data
)
from src.data.lcot2tree_wrapper import run_lcot2tree_pipeline


def load_filtered_samples(
    n: int,
    target_dataset: str,
    verbose: bool = False,
    balanced: bool = False,
    sources: list = None,
):
    """
    Load n samples from SYNTHETIC-1 dataset filtered by dataset name.
    
    Args:
        n: Number of examples to load. If balanced=True, this is the number per class.
        target_dataset: Name of the dataset to filter for (e.g., 'PrimeIntellect/verifiable-math-problems')
        verbose: Whether to print progress and sample details
        balanced: If True, load equal positive/negative samples based on 'score' field
        sources: List of source names to filter for (default: ["math"])
    
    Returns:
        List of examples matching the filter criteria
    """
    # Load SYNTHETIC-1 dataset with streaming to avoid downloading entire dataset
    dataset = load_dataset(
        "PrimeIntellect/SYNTHETIC-1",
        split="train",
        streaming=True
    )
    
    good_sources = sources if sources is not None else ["math"]  # amc_aime, olympiads, orca_math, math
    
    if balanced:
        # Balanced mode: collect equal positive and negative samples
        positive_samples = []
        negative_samples = []
        n_per_class = n
        
        if verbose:
            print(f"Loading balanced dataset: {n_per_class} samples per class from: {good_sources}")
            print("Searching through dataset...\n")
        
        for i, example in enumerate(dataset):
            if example.get("source") not in good_sources:
                continue
            
            # Determine label from score field
            score = example.get("score", None)
            if score is None:
                continue
            
            # score can be string "1"/"0" or int 1/0 or bool
            if isinstance(score, str):
                is_positive = score == "1" or score.lower() == "true"
            else:
                is_positive = bool(score)
            
            if is_positive and len(positive_samples) < n_per_class:
                positive_samples.append(example)
                if verbose:
                    print(f"Positive #{len(positive_samples)} (row {i}): score={score}")
            elif not is_positive and len(negative_samples) < n_per_class:
                negative_samples.append(example)
                if verbose:
                    print(f"Negative #{len(negative_samples)} (row {i}): score={score}")
            
            # Check if we have enough of both
            if len(positive_samples) >= n_per_class and len(negative_samples) >= n_per_class:
                break
            
            # Progress update every 5000 rows
            if verbose and (i + 1) % 5000 == 0:
                print(f"Scanned {i + 1} rows, pos={len(positive_samples)}, neg={len(negative_samples)}")
        
        sample_data = positive_samples + negative_samples
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Balanced dataset loaded:")
            print(f"  Positive samples: {len(positive_samples)}")
            print(f"  Negative samples: {len(negative_samples)}")
            print(f"  Total: {len(sample_data)}")
            print(f"{'='*80}")
    else:
        # Original unbalanced mode
        sample_data = []
        count = 0
        
        if verbose:
            print(f"Loading {n} samples from: {good_sources}")
            print("Searching through dataset...\n")
        
        for i, example in enumerate(dataset):
            if example.get("source") not in good_sources:
                continue
            sample_data.append(example)
            count += 1
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Match #{count} (overall row {i}):")
                print(f"{'='*80}")
                for key, value in example.items():
                    print(f"\n{key}:")
                    if isinstance(value, str) and len(value) > 500:
                        print(f"{value[:500]}... [truncated]")
                    else:
                        print(value)
            
            if count >= n:
                break
            
            if verbose and (i + 1) % 1000 == 0:
                print(f"Scanned {i + 1} rows, found {count} matches so far...")
        
        if verbose:
            print(f"\n\n{'='*80}")
            print(f"Successfully loaded {len(sample_data)} examples from {target_dataset}")
            if len(sample_data) > 0:
                print(f"Keys in each example: {list(sample_data[0].keys())}")
            print(f"{'='*80}")
    
    return sample_data


def load_openmath_reasoning_samples(
    n: int,
    min_pass_rate: float = None,
    verbose: bool = False,
    balanced: bool = False,
):
    """
    Load n samples from nvidia/OpenMathReasoning dataset.
    
    Args:
        n: Number of examples to load. If balanced=True, this is the number per class.
        min_pass_rate: Optional minimum pass rate threshold for filtering (0.0-1.0)
        verbose: Whether to print progress and sample details
        balanced: If True, load equal positive/negative samples based on 'score' field
    
    Returns:
        List of examples from OpenMathReasoning dataset
    """
    # Load OpenMathReasoning dataset with streaming
    dataset = load_dataset(
        "nvidia/OpenMathReasoning",
        split="cot",  # Use chain-of-thought split
        streaming=True
    )
    
    scanned = 0
    
    if balanced:
        # Balanced mode: collect equal positive and negative samples
        positive_samples = []
        negative_samples = []
        n_per_class = n
        
        if verbose:
            print(f"Loading balanced dataset: {n_per_class} samples per class from OpenMathReasoning")
            if min_pass_rate is not None:
                print(f"Filtering for pass_rate_72b_tir >= {min_pass_rate}")
            print("Searching through dataset...\n")
        
        for i, example in enumerate(dataset):
            scanned += 1
            
            # Apply pass rate filter if specified
            if min_pass_rate is not None:
                pass_rate = example.get("pass_rate_72b_tir", 0.0)
                if pass_rate < min_pass_rate:
                    continue
            
            # Determine label from score field
            score = example.get("score", None)
            if score is None:
                # If no score field, try to infer from pass_rate
                pass_rate = example.get("pass_rate_72b_tir", 0.0)
                is_positive = pass_rate >= 0.5  # Use 0.5 as threshold
            elif isinstance(score, str):
                is_positive = score == "1" or score.lower() == "true"
            else:
                is_positive = bool(score)
            
            if is_positive and len(positive_samples) < n_per_class:
                positive_samples.append(example)
                if verbose:
                    print(f"Positive #{len(positive_samples)} (row {i}): score={score}")
            elif not is_positive and len(negative_samples) < n_per_class:
                negative_samples.append(example)
                if verbose:
                    print(f"Negative #{len(negative_samples)} (row {i}): score={score}")
            
            # Check if we have enough of both
            if len(positive_samples) >= n_per_class and len(negative_samples) >= n_per_class:
                break
            
            # Progress update every 5000 rows
            if verbose and (scanned) % 5000 == 0:
                print(f"Scanned {scanned} rows, pos={len(positive_samples)}, neg={len(negative_samples)}")
        
        sample_data = positive_samples + negative_samples
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Balanced dataset loaded:")
            print(f"  Positive samples: {len(positive_samples)}")
            print(f"  Negative samples: {len(negative_samples)}")
            print(f"  Total: {len(sample_data)}")
            print(f"{'='*80}")
    else:
        # Original unbalanced mode
        sample_data = []
        count = 0
        
        if verbose:
            print(f"Loading {n} samples from nvidia/OpenMathReasoning (cot split)")
            if min_pass_rate is not None:
                print(f"Filtering for pass_rate_72b_tir >= {min_pass_rate}")
            print("Searching through dataset...\n")
        
        for i, example in enumerate(dataset):
            scanned += 1
            
            # Apply pass rate filter if specified
            if min_pass_rate is not None:
                pass_rate = example.get("pass_rate_72b_tir", 0.0)
                if pass_rate < min_pass_rate:
                    continue
            
            sample_data.append(example)
            count += 1
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Match #{count} (scanned {scanned} samples):")
                print(f"{'='*80}")
                
                print(f"\nProblem: {example.get('problem', '')[:200]}...")
                print(f"\nGenerated Solution: {example.get('generated_solution', '')[:300]}...")
                print(f"\nExpected Answer: {example.get('expected_answer', '')}")
                print(f"\nPass Rate (72B TIR): {example.get('pass_rate_72b_tir', 'N/A')}")
                print(f"Problem Type: {example.get('problem_type', 'N/A')}")
                print(f"Generation Model: {example.get('generation_model', 'N/A')}")
            
            if count >= n:
                break
            
            if verbose and (scanned) % 1000 == 0:
                print(f"Scanned {scanned} samples, found {count} matches so far...")
        
        if verbose:
            print(f"\n\n{'='*80}")
            print(f"Successfully loaded {len(sample_data)} examples")
            if len(sample_data) > 0:
                print(f"Keys in each example: {list(sample_data[0].keys())}")
            print(f"{'='*80}")
    
    return sample_data


def test_lcot2tree_pipeline(
    n_samples: int = 3,
    dataset_type: str = "SYNTHETIC-1",
    target_dataset: str = "PrimeIntellect/verifiable-math-problems",
    min_pass_rate: float = None,
    model_backend: str = "gpt5-nano",
    output_dir: str = "./data/processed/lcot2tree_test",
    config_path: str = "./config.json",
    use_async: bool = False,
    batch_size: int = 10,
    verbose: bool = False,
    balanced: bool = False,
    sources: list = None,
):
    """
    Test the complete LCoT2Tree pipeline with a small sample of data.
    
    Args:
        n_samples: Number of samples to process. If balanced=True, this is per class.
        dataset_type: Dataset to use ("SYNTHETIC-1" or "OpenMathReasoning")
        target_dataset: Dataset name to filter for (SYNTHETIC-1 only)
        min_pass_rate: Minimum pass rate for OpenMathReasoning (optional float 0.0-1.0)
        model_backend: LLM backend to use ("gpt5-nano", "qwen3-4b", "qwen3-32b", "deepseek-v3.2")
        output_dir: Output directory for results
        config_path: Path to config.json with API keys
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing (default: 10)
        verbose: Whether to print progress
        balanced: If True, load equal positive/negative samples based on 'score' field
        sources: List of source names to filter for (SYNTHETIC-1 only)
    
    Returns:
        List of processed items with CoT trees
    """
    print(f"\n{'='*80}")
    print("Testing LCoT2Tree Pipeline")
    print(f"{'='*80}\n")
    
    # Step 1: Load samples based on dataset type
    if balanced:
        print(f"Step 1: Loading {n_samples} samples PER CLASS (balanced) from {dataset_type}...")
    else:
        print(f"Step 1: Loading {n_samples} samples from {dataset_type}...")
    
    if dataset_type == "SYNTHETIC-1":
        samples = load_filtered_samples(
            n_samples,
            target_dataset,
            verbose=verbose,
            balanced=balanced,
            sources=sources,
        )
        dataset_name_prefix = "synthetic1"
    elif dataset_type == "OpenMathReasoning":
        samples = load_openmath_reasoning_samples(
            n_samples,
            min_pass_rate=min_pass_rate,
            verbose=verbose,
            balanced=balanced,
        )
        dataset_name_prefix = "openmath"
    else:
        print(f"ERROR: Unknown dataset type: {dataset_type}")
        print("Must be 'SYNTHETIC-1' or 'OpenMathReasoning'")
        return None
    
    if len(samples) == 0:
        print("ERROR: No samples found!")
        return None
    
    print(f"✓ Loaded {len(samples)} samples\n")
    
    # Step 2: Preprocess for LCoT2Tree
    print("Step 2: Preprocessing samples for LCoT2Tree format...")
    
    if dataset_type == "SYNTHETIC-1":
        preprocessed = preprocess_for_lcot2tree(samples, dataset_name=dataset_name_prefix)
    else:  # OpenMathReasoning
        preprocessed = preprocess_openmath_reasoning_for_lcot2tree(
            samples,
            dataset_name=dataset_name_prefix,
            min_pass_rate=min_pass_rate
        )
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed.json")
    save_preprocessed_data(preprocessed, preprocessed_path)
    print(f"✓ Preprocessed and saved {len(preprocessed)} samples\n")
    
    # Print sample preprocessed item
    if verbose and len(preprocessed) > 0:
        print("Sample preprocessed item:")
        sample_item = preprocessed[0].copy()
        # Truncate long fields for readability
        if "prediction" in sample_item and len(sample_item["prediction"]) > 200:
            sample_item["prediction"] = sample_item["prediction"][:200] + "... [truncated]"
        print(json.dumps(sample_item, indent=2))
        print()
    
    # Step 3: Load config
    print("Step 3: Loading configuration...")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    model_config = config.get(model_backend, {})
    print(f"✓ Using backend: {model_backend}")
    print(f"  Model: {model_config.get('model_id', 'default')}")
    print(f"  Mode: {'Async batched' if use_async else 'Sync parallel'}")
    if use_async:
        print(f"  Batch size: {batch_size}")
    print()
    
    # Step 4: Run LCoT2Tree pipeline
    print("Step 4: Running LCoT2Tree pipeline...")
    print("This may take several minutes depending on the LLM backend...\n")
    
    try:
        results = run_lcot2tree_pipeline(
            reasoning_traces=preprocessed,
            output_dir=output_dir,
            model_backend=model_backend,
            config=model_config,
            max_workers=config.get("lcot2tree", {}).get("max_workers", 10),
            use_async=use_async,
            batch_size=batch_size
        )
        
        print(f"\n✓ Pipeline complete! Processed {len(results)} samples")
        print(f"✓ Results saved to: {output_dir}/final.json")
        
        # Print statistics
        total_in_tokens = sum(item.get("in_token_cost", 0) for item in results)
        total_out_tokens = sum(item.get("out_token_cost", 0) for item in results)
        
        print(f"\nToken usage statistics:")
        print(f"  Input tokens: {total_in_tokens:,}")
        print(f"  Output tokens: {total_out_tokens:,}")
        print(f"  Total tokens: {total_in_tokens + total_out_tokens:,}")
        
        # Print sample tree
        if verbose and len(results) > 0 and "cot_tree" in results[0]:
            print("\nSample CoT tree structure:")
            print(json.dumps(results[0]["cot_tree"], indent=2)[:500] + "... [truncated]")
        
        print(f"\n{'='*80}")
        print("Test completed successfully!")
        print(f"{'='*80}\n")
        
        return results
    
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test LCoT2Tree pipeline")
    parser.add_argument(
        "--n_samples", type=int, default=3,
        help="Number of samples to process. If --balanced, this is per class (default: 3)"
    )
    parser.add_argument(
        "--dataset", type=str, default="SYNTHETIC-1",
        choices=["SYNTHETIC-1", "OpenMathReasoning"],
        help="Dataset to use (default: SYNTHETIC-1)"
    )
    parser.add_argument(
        "--target_dataset", type=str,
        default="PrimeIntellect/verifiable-math-problems",
        help="Target dataset name (SYNTHETIC-1 only)"
    )
    parser.add_argument(
        "--sources", type=str, nargs="+",
        default=None,
        help="Source names to filter for in SYNTHETIC-1 (e.g., math amc_aime olympiads orca_math)"
    )
    parser.add_argument(
        "--min_pass_rate", type=float, default=None,
        help="Minimum pass rate threshold for OpenMathReasoning (0.0-1.0, optional)"
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="Load balanced dataset with equal positive/negative samples based on 'score' field"
    )
    parser.add_argument(
        "--backend", type=str, default="deepseek-v3.2",
        choices=["gpt5-nano", "gpt5-mini", "qwen3-4b", "qwen3-32b", "deepseek", "deepseek-v3.2"],
        help="LLM backend to use (default: gpt5-nano)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./data/processed/lcot2tree_test",
        help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="./config.json",
        help="Path to config.json"
    )
    parser.add_argument(
        "--async", dest="use_async", action="store_true",
        help="Use async batch processing (default: False)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size for async processing (default: 10)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose output (default: False)"
    )
    
    args = parser.parse_args()
    
    test_lcot2tree_pipeline(
        n_samples=args.n_samples,
        dataset_type=args.dataset,
        target_dataset=args.target_dataset,
        min_pass_rate=args.min_pass_rate,
        model_backend=args.backend,
        output_dir=args.output_dir,
        config_path=args.config,
        use_async=args.use_async,
        batch_size=args.batch_size,
        verbose=args.verbose,
        balanced=args.balanced,
        sources=args.sources,
    )


if __name__ == "__main__":
    main()