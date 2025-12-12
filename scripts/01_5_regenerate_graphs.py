#!/usr/bin/env python3
"""
Regenerate reasoning graphs from final_regraded.json using refactored pipeline.

This script loads existing reasoning traces and rebuilds their graph representations
using the modular LCoT2Tree pipeline with restart capability.

MODIFICATIONS FOR OLD PIPELINE DATA COMPATIBILITY:
- Added adapter logic for start_step=4 to handle old lcot2tree_wrapper.py data
- Converts "B0" -> 0 and "A1" -> 1 format to pure integers
- Handles legacy split words not in new config
- Creates process3.json to seed the pipeline

Usage:
    # Full pipeline run
    python 01_6_regenerate_graphs_new.py --input data/processed/final_regraded.json --output_dir ./output
    
    # Restart from step 4 WITH OLD DATA (main use case)
    python 01_6_regenerate_graphs_new.py \
      --input data/processed/final_regraded.json \
      --output_dir ./data/processed/graphs_regenerated \
      --start_step 4 \
      --backend deepseek-v3.2
    
    # With specific backend and config
    python 01_6_regenerate_graphs_new.py --input final_regraded.json --output_dir ./output --backend deepseek-v3.2 --config ./config.json
    
    # Limit samples for testing
    python 01_6_regenerate_graphs_new.py --input final_regraded.json --output_dir ./output --max_samples 5 --verbose
"""

import sys
import os
import json
import argparse
import re
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import refactored pipeline
from src.data.pipeline import run_reasoning_pipeline


def load_regraded_data(input_path: str, verbose: bool = False):
    """
    Load final_regraded.json using hybrid JSON/JSONL logic.
    
    Args:
        input_path: Path to final_regraded.json
        verbose: Whether to print loading details
    
    Returns:
        List of samples from the file
    """
    if verbose:
        print(f"Loading data from {input_path}...")
    
    with open(input_path, 'r') as f:
        # Handle both JSON array and JSON lines format
        content = f.read().strip()
        if content.startswith('['):
            samples = json.loads(content)
        else:
            samples = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    if verbose:
        print(f"✓ Loaded {len(samples)} samples")
        if len(samples) > 0:
            print(f"  Keys in each sample: {list(samples[0].keys())}")
    
    return samples


def prepare_for_pipeline(samples: list, verbose: bool = False):
    """
    Transform loaded data into clean format for pipeline.
    
    Extracts only the necessary fields:
    - prediction: The reasoning text
    - tag: Identifier (mapped from response_id)
    
    Args:
        samples: List of samples from final_regraded.json
        verbose: Whether to print transformation details
    
    Returns:
        List of clean dictionaries with 'prediction' and 'tag'
    """
    if verbose:
        print("\nPreparing data for pipeline...")
    
    clean_data = []
    
    for i, sample in enumerate(samples):
        # Extract prediction and tag
        prediction = sample.get("prediction", "")
        response_id = sample.get("response_id", f"sample_{i}")
        
        # Create clean dictionary with only required fields
        clean_item = {
            "prediction": prediction,
            "tag": response_id
        }
        
        clean_data.append(clean_item)
        
        if verbose and i < 3:
            print(f"\n  Sample {i}:")
            print(f"    tag: {clean_item['tag']}")
            print(f"    prediction length: {len(clean_item['prediction'])} chars")
    
    if verbose:
        print(f"\n✓ Prepared {len(clean_data)} clean samples for pipeline")
    
    return clean_data


def strip_prefix(text: str, prefix: str) -> int:
    """
    Strip a letter prefix from a string and return integer.
    
    Args:
        text: String like "B0", "A1", etc.
        prefix: Prefix to strip (e.g., "B", "A")
    
    Returns:
        Integer value after stripping prefix
    
    Examples:
        strip_prefix("B0", "B") -> 0
        strip_prefix("A1", "A") -> 1
    """
    text_str = str(text)
    if text_str.startswith(prefix):
        return int(text_str[len(prefix):])
    # If no prefix, try to convert directly
    return int(re.sub(r'[A-Za-z]', '', text_str))


def adapt_old_pipeline_data(samples: list, verbose: bool = False):
    """
    Adapt old pipeline data (from lcot2tree_wrapper.py) to new pipeline format.
    
    Key transformations:
    1. Convert assigned_step: "B0" -> 0, ["A1"] -> [1]
    2. Normalize thoughts_list keys to strings
    3. Add legacy split words to thought cleaning
    4. Verify reasoning_sketch exists
    
    Args:
        samples: List of samples from old pipeline
        verbose: Whether to print adaptation details
    
    Returns:
        List of adapted samples ready for Step 4
    """
    if verbose:
        print("\nAdapting OLD pipeline data for New Step 4...")
        print("=" * 80)
    
    adapted_data = []
    warnings = []
    
    for idx, item in enumerate(samples):
        # 1. SANITIZE 'assigned_step': Convert "B0"->"A1" to 0->1
        if "assigned_step" in item:
            old_assign = item["assigned_step"]
            new_assign = {}
            
            for k, v in old_assign.items():
                try:
                    # Strip 'B' from key (thought index)
                    # Handle both "B0" format and plain integers
                    clean_k = strip_prefix(k, "B")
                    
                    # Strip 'A' from values (step indices)
                    # Ensure v is a list
                    if not isinstance(v, list):
                        v = [v]
                    
                    clean_v = [strip_prefix(val, "A") for val in v]
                    new_assign[clean_k] = clean_v
                    
                except (ValueError, TypeError) as e:
                    if verbose:
                        warnings.append(f"Item {item.get('tag', idx)}: Error converting assigned_step key '{k}': {e}")
                    continue
            
            item["assigned_step"] = new_assign
            
            if verbose and idx < 3:
                print(f"\nSample {idx} ({item.get('tag')}):")
                print(f"  Old assigned_step (first 3): {dict(list(old_assign.items())[:3])}")
                print(f"  New assigned_step (first 3): {dict(list(new_assign.items())[:3])}")
        
        # 2. NORMALIZE 'thoughts_list' keys to strings
        # (New pipeline's normalize_thought_keys expects string keys)
        if "thoughts_list" in item:
            thought_list = item["thoughts_list"]
            if isinstance(thought_list, str):
                thought_list = json.loads(thought_list)
            # Convert all keys to strings
            item["thoughts_list"] = {str(k): v for k, v in thought_list.items()}
        
        # 3. VERIFY 'reasoning_sketch' exists (Step 4 needs it for gap bridging)
        if "reasoning_sketch" not in item:
            warnings.append(f"Item {item.get('tag', idx)}: Missing reasoning_sketch")
        
        # 4. CLEAN legacy split words from thoughts if present
        # (This is optional - the new pipeline will also clean, but we can pre-clean here)
        # We'll let the pipeline handle this since it has access to SPLIT_WORDS
        
        adapted_data.append(item)
    
    if verbose:
        print(f"\n✓ Adapted {len(adapted_data)} items")
        if warnings:
            print(f"\n⚠ Warnings ({len(warnings)}):")
            for warn in warnings[:5]:  # Show first 5 warnings
                print(f"  - {warn}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more warnings")
        print("=" * 80)
    
    return adapted_data


def regenerate_graphs(
    input_path: str,
    output_dir: str,
    model_backend: str = "deepseek-v3.2",
    config_path: str = "./config.json",
    max_samples: int = None,
    use_async: bool = False,
    batch_size: int = 10,
    debug: bool = False,
    verbose: bool = False,
    start_step: int = 1,
    causal: bool = False
):
    """
    Regenerate reasoning graphs from final_regraded.json.

    Args:
        input_path: Path to final_regraded.json
        output_dir: Output directory for new graphs
        model_backend: LLM backend to use
        config_path: Path to config.json
        max_samples: Maximum number of samples to process (None for all)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing
        debug: Enable debug logging in pipeline
        verbose: Enable verbose output
        start_step: Step to start from (1-5)
        causal: If True, only consider parents that occurred chronologically before

    Returns:
        List of processed items with reasoning graphs
    """
    print(f"\n{'='*80}")
    print("Regenerating Reasoning Graphs (Refactored Pipeline)")
    if start_step > 1:
        print(f"RESTART MODE: Starting from Step {start_step}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STEP 1: Load and Adapter Logic
    # ========================================================================
    if start_step == 1:
        print(f"Step 1: Loading raw data...")
        samples = load_regraded_data(input_path, verbose=verbose)
        
        if len(samples) == 0:
            print("ERROR: No samples found in input file!")
            return None
        
        # Limit samples if requested
        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
            print(f"\nLimited to first {max_samples} samples for testing")
        
        print(f"✓ Will process {len(samples)} samples\n")
        
        # Transform to clean format
        print("Step 2: Transforming data to pipeline format...")
        clean_data = prepare_for_pipeline(samples, verbose=verbose)
        
    elif start_step == 4:
        print(f"Step 1: Adapting OLD Pipeline data for New Step 4...")
        print("=" * 80)
        print("This mode handles data from lcot2tree_wrapper.py:")
        print("  - Converts 'B0' -> 0, 'A1' -> 1 in assigned_step")
        print("  - Normalizes thoughts_list keys")
        print("  - Verifies reasoning_sketch exists")
        print("  - Creates process3.json for pipeline")
        print("=" * 80 + "\n")
        
        # Load the OLD full file
        samples = load_regraded_data(input_path, verbose=verbose)
        
        if len(samples) == 0:
            print("ERROR: No samples found in input file!")
            return None
        
        # Limit samples if requested (BEFORE adaptation)
        if max_samples is not None and max_samples < len(samples):
            samples = samples[:max_samples]
            print(f"Limited to first {max_samples} samples for testing\n")
        
        # Adapt old format to new format
        adapted_data = adapt_old_pipeline_data(samples, verbose=verbose)
        
        # FORCE SAVE as process3.json
        # This tricks the pipeline into thinking Step 3 just finished
        p3_path = Path(output_dir) / "process3.json"
        p3_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving adapted data to {p3_path}...")
        with open(p3_path, 'w', encoding='utf-8') as f:
            for item in adapted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ Saved {len(adapted_data)} items to process3.json")
        print("  Pipeline will load this file and start from Step 4\n")
        
        # Pass data to pipeline (it will also load from process3.json)
        clean_data = adapted_data
    
    elif start_step in [2, 3]:
        print(f"Skipping data loading (starting from step {start_step})")
        print("Pipeline will load from intermediate files\n")
        clean_data = []
    
    elif start_step == 5:
        print(f"Skipping data loading (starting from step {start_step})")
        print("Pipeline will load from process4.json\n")
        clean_data = []
    
    else:
        print(f"ERROR: Invalid start_step {start_step}")
        return None
    
    # ========================================================================
    # STEP 2: Load Configuration
    # ========================================================================
    print("Step 2: Loading configuration...")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        print(f"WARNING: Config file not found at {config_path}")
        print("Proceeding with default configuration")
    
    model_config = config.get(model_backend, {})
    max_workers = config.get("reasoning_graph", {}).get("max_workers", 50)
    
    print(f"✓ Using backend: {model_backend}")
    print(f"  Model: {model_config.get('model_id', 'default')}")
    print(f"  Mode: {'Async batched' if use_async else 'Sync parallel'}")
    print(f"  Max workers: {max_workers}")
    if use_async:
        print(f"  Batch size: {batch_size}")
    if debug:
        print(f"  Debug mode: ENABLED")
    if start_step > 1:
        print(f"  🔄 Restart from step: {start_step}")
    print()
    
    # ========================================================================
    # STEP 3: Run Pipeline
    # ========================================================================
    print("Step 3: Running reasoning graph pipeline...")
    if start_step == 4:
        print("  → Step 4 will use process3.json (adapted old data)")
        print("  → Steps 1-3 results preserved from old pipeline")
        print("  → Only generating NEW edges (Step 4)\n")
    print("This may take several minutes depending on the number of samples...\n")
    
    try:
        results = run_reasoning_pipeline(
            reasoning_traces=clean_data,
            output_dir=output_dir,
            model_backend=model_backend,
            config=model_config,
            max_workers=max_workers,
            use_async=use_async,
            batch_size=batch_size,
            debug=debug,
            start_step=start_step,
            max_samples=max_samples,
            causal=causal
        )
        
        return results
    
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate reasoning graphs from final_regraded.json (Refactored Pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline run
  python 01_6_regenerate_graphs_new.py --input final_regraded.json --output_dir ./output
  
  # Restart from step 4 WITH OLD DATA (converts B0/A1 format to integers)
  python 01_6_regenerate_graphs_new.py \\
    --input data/processed/final_regraded.json \\
    --output_dir ./data/processed/graphs_regenerated \\
    --start_step 4 \\
    --backend deepseek-v3.2
  
  # Test with 5 samples
  python 01_6_regenerate_graphs_new.py --input final_regraded.json --output_dir ./output --max_samples 5
        """
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/final_regraded.json",
        help="Path to final_regraded.json"
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="./data/processed/graphs_regenerated",
        help="Output directory for new graphs (default: ./data/processed/graphs_regenerated)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="deepseek-v3.2",
        choices=["gpt5-nano", "gpt5-mini", "qwen3-4b", "qwen3-32b", "deepseek", "deepseek-v3.2"],
        help="LLM backend to use (default: deepseek-v3.2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.json",
        help="Path to config.json (default: ./config.json)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Use async batch processing (default: False)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for async processing (default: 10)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging in pipeline (default: False)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (default: False)"
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Step to start from (1-5). Use 4 to run new edges on old data (default: 1)"
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Only consider parents that occurred chronologically before (default: False)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists (only if starting from step 1 or 4)
    if args.start_step in [1, 4] and not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    # Run regeneration
    results = regenerate_graphs(
        input_path=args.input,
        output_dir=args.output_dir,
        model_backend=args.backend,
        config_path=args.config,
        max_samples=args.max_samples,
        use_async=args.use_async,
        batch_size=args.batch_size,
        debug=args.debug,
        verbose=args.verbose,
        start_step=args.start_step,
        causal=args.causal
    )
    
    if results is None:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())