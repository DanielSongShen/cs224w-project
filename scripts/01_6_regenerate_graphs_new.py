#!/usr/bin/env python3
"""
Regenerate reasoning graphs from final_regraded.json using refactored pipeline.

This script loads existing reasoning traces and rebuilds their graph representations
using the modular LCoT2Tree pipeline with restart capability.

Usage:
    # Full pipeline run
    python 02_regenerate_graphs.py --input data/processed/final_regraded.json --output_dir ./output
    
    # Restart from step 4 (if steps 1-3 already completed)
    python 02_regenerate_graphs.py --input data/processed/final_regraded.json --output_dir ./output --start_step 4
    
    # With specific backend and config
    python 02_regenerate_graphs.py --input final_regraded.json --output_dir ./output --backend deepseek-v3.2 --config ./config.json
    
    # Limit samples for testing
    python 02_regenerate_graphs.py --input final_regraded.json --output_dir ./output --max_samples 5 --verbose
"""

import sys
import os
import json
import argparse
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
    start_step: int = 1
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
    
    Returns:
        List of processed items with reasoning graphs
    """
    print(f"\n{'='*80}")
    print("Regenerating Reasoning Graphs (Refactored Pipeline)")
    if start_step > 1:
        print(f"RESTART MODE: Starting from Step {start_step}")
    print(f"{'='*80}\n")
    
    # Step 1: Load existing data
    if start_step == 1:
        print(f"Step 1: Loading existing reasoning traces...")
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
    else:
        print(f"Skipping data loading (starting from step {start_step})")
        # Pipeline will load from intermediate files
        clean_data = []
    
    # Load config
    print("\nStep 3: Loading configuration...")
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
    
    # Run pipeline
    print("Step 4: Running reasoning graph pipeline...")
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
            max_samples=max_samples
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
  python 02_regenerate_graphs.py --input final_regraded.json --output_dir ./output
  
  # Restart from step 4 (if crash occurred)
  python 02_regenerate_graphs.py --input final_regraded.json --output_dir ./output --start_step 4
  
  # Test with 5 samples
  python 02_regenerate_graphs.py --input final_regraded.json --output_dir ./output --max_samples 5
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
        help="Step to start from (1-5). Use this to restart from a specific step. (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists (only if starting from step 1)
    if args.start_step == 1 and not os.path.exists(args.input):
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
        start_step=args.start_step
    )
    
    if results is None:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())