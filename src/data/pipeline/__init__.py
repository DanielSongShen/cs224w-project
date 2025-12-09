"""LCoT2Tree Reasoning Graph Pipeline

Refactored modular pipeline for converting reasoning traces into graph structures.

Main entry point:
    run_reasoning_pipeline() - Convenience function to run the full pipeline

Features:
- Modular architecture with single-responsibility modules
- DeepSeek prefix caching optimization
- Incremental checkpointing with restart capability
- Robust fallback mechanisms
- DAG support with multiple parents per node
"""
from typing import List, Dict, Any, Optional

from .builder import ReasoningGraphPipeline
from .llm_client import create_llm_client

__version__ = "2.0.0"
__all__ = ['run_reasoning_pipeline', 'ReasoningGraphPipeline', 'create_llm_client']


def run_reasoning_pipeline(
    reasoning_traces: List[Dict[str, Any]],
    output_dir: str,
    model_backend: str = None,
    model_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    llm_client=None,
    max_workers: int = 50,
    use_async: bool = False,
    batch_size: int = 10,
    debug: bool = False,
    start_step: int = 1,
    max_samples: Optional[int] = None,
    causal: bool = False,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to run reasoning graph pipeline.

    Args:
        reasoning_traces: Preprocessed reasoning traces with 'prediction' and 'tag' fields
        output_dir: Output directory for results
        model_backend: LLM backend name (e.g., "gpt5-nano", "qwen3-4b", "deepseek-v3.2")
        model_name: Explicit model name (optional)
        config: Configuration dict for LLM (optional)
        llm_client: Pre-configured LLM client (optional, alternative to model_backend)
        max_workers: Number of parallel workers (for sync mode)
        use_async: Whether to use async batch processing
        batch_size: Batch size for async processing (default: 10)
        debug: Enable detailed debug logging (default: False)
        start_step: Step to start from (1-5). Use this to restart from a specific step.
        max_samples: Maximum number of samples to process (applies to restarts too)
        causal: If True, only consider parents that occurred chronologically before (default: False)
        **kwargs: Additional arguments for LLM client creation

    Returns:
        List of processed items with reasoning graphs
    
    Example:
        >>> from src.pipeline import run_reasoning_pipeline
        >>> 
        >>> # Prepare data
        >>> data = [
        ...     {"prediction": "<think>...</think>", "tag": "sample_1"},
        ...     {"prediction": "<think>...</think>", "tag": "sample_2"},
        ... ]
        >>> 
        >>> # Run pipeline
        >>> results = run_reasoning_pipeline(
        ...     reasoning_traces=data,
        ...     output_dir="./output",
        ...     model_backend="deepseek-v3.2",
        ...     config={"api_key": "...", "url": "..."}
        ... )
        >>> 
        >>> # Restart from step 4 if needed
        >>> results = run_reasoning_pipeline(
        ...     reasoning_traces=data,
        ...     output_dir="./output",
        ...     model_backend="deepseek-v3.2",
        ...     config={"api_key": "...", "url": "..."},
        ...     start_step=4
        ... )
    """
    # Create LLM client if not provided
    if llm_client is None:
        if model_backend is None:
            raise ValueError("Either llm_client or model_backend must be provided")
        
        llm_client = create_llm_client(
            backend=model_backend,
            model_name=model_name,
            config=config,
            **kwargs
        )
    
    # Create and run pipeline
    pipeline = ReasoningGraphPipeline(
        llm_client=llm_client,
        output_dir=output_dir,
        max_workers=max_workers,
        use_async=use_async,
        batch_size=batch_size,
        debug=debug,
        causal=causal
    )
    
    return pipeline.run_full_pipeline(
        input_data=reasoning_traces,
        start_step=start_step,
        max_samples=max_samples
    )