#!/usr/bin/env python3
"""
Script to continue LCoT2Tree pipeline from a specific step.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.llm_client import create_llm_client
from src.data.lcot2tree_wrapper import LCoT2TreePipeline


def continue_from_step(
    step: int,
    input_file: str,
    output_dir: str,
    backend: str = "deepseek-v3.2",
    config_path: str = "./config.json",
    use_async: bool = True,
    batch_size: int = 10,
):
    """
    Continue pipeline from a specific step.
    
    Args:
        step: Step number to start from (1-5)
        input_file: Path to the input JSON file (e.g., process3.json)
        output_dir: Output directory
        backend: LLM backend to use
        config_path: Path to config.json
        use_async: Whether to use async processing
        batch_size: Batch size for async processing
    """
    print(f"\n{'='*80}")
    print(f"Continuing LCoT2Tree Pipeline from Step {step}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Backend: {backend}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load input data
    print(f"Loading data from {input_file}...")
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"Loaded {len(data)} items\n")
    # Create LLM client
    llm_client = create_llm_client(
        backend=backend,
        model_name=None,
        config=config.get(backend)
    )
    # Create pipeline
    pipeline = LCoT2TreePipeline(
        llm_client=llm_client,
        output_dir=Path(output_dir),
        max_workers=10,
        use_async=use_async,
        batch_size=batch_size
    )
    
    # Run from specified step
    if step == 4:
        data = pipeline.step4_assign_functions(data)
        data = pipeline.step5_build_tree(data)
    elif step == 5:
        data = pipeline.step5_build_tree(data)
    else:
        raise ValueError(f"Invalid step: {step}. Can only continue from steps 4 or 5.")
    
    print(f"\n{'='*80}")
    print("Pipeline completed successfully!")
    print(f"Final output: {output_dir}/final.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Continue LCoT2Tree pipeline from a specific step")
    parser.add_argument(
        "--step", type=int, required=True,
        choices=[4, 5],
        help="Step to start from (4 or 5)"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input JSON file (e.g., data/processed/deepseek/amc-aime/process3.json)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory (e.g., data/processed/deepseek/amc-aime)"
    )
    parser.add_argument(
        "--backend", type=str, default="deepseek-v3.2",
        help="LLM backend to use (default: deepseek-v3.2)"
    )
    parser.add_argument(
        "--config", type=str, default="./config.json",
        help="Path to config.json"
    )
    parser.add_argument(
        "--async", dest="use_async", action="store_true", default=True,
        help="Use async batch processing (default: True)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10,
        help="Batch size for async processing (default: 10)"
    )
    
    args = parser.parse_args()
    
    continue_from_step(
        step=args.step,
        input_file=args.input,
        output_dir=args.output_dir,
        backend=args.backend,
        config_path=args.config,
        use_async=args.use_async,
        batch_size=args.batch_size
    )

