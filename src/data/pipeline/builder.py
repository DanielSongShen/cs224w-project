"""Reasoning Graph Pipeline Orchestrator

Main pipeline class that coordinates all 5 steps with:
- Restart capability from any step
- Progress tracking and statistics
- Cache hit rate reporting
- Sample limiting for testing
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import DEFAULT_MAX_WORKERS, DEFAULT_BATCH_SIZE
from .utils import save_jsonl, load_jsonl, ensure_directory
from .steps import (
    process_split,
    process_sketch,
    process_assign,
    process_link,
    process_graph
)


class ReasoningGraphPipeline:
    """Pipeline for processing reasoning traces into graph structures (DAG support)"""
    
    def __init__(
        self,
        llm_client,
        output_dir: str,
        max_workers: int = DEFAULT_MAX_WORKERS,
        use_async: bool = False,
        batch_size: int = DEFAULT_BATCH_SIZE,
        debug: bool = False
    ):
        """
        Initialize reasoning graph pipeline.
        
        Args:
            llm_client: LLM client for API calls (must have .generate(messages) method)
            output_dir: Directory to store intermediate and final outputs
            max_workers: Number of parallel workers for LLM calls (sync mode)
            use_async: Whether to use async batch processing
            batch_size: Batch size for async processing
            debug: Enable detailed debug logging
        """
        self.llm_client = llm_client
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        self.max_workers = max_workers
        self.use_async = use_async
        self.batch_size = batch_size
        self.debug = debug
    
    def run_full_pipeline(
        self, 
        input_data: List[Dict[str, Any]], 
        start_step: int = 1,
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run the complete reasoning graph pipeline with restart capability.
        
        Args:
            input_data: List of preprocessed reasoning traces
            start_step: Step to start from (1-5). If > 1, loads output from previous step.
            max_samples: Maximum number of samples to process (applies to restarts too)
        
        Returns:
            List of items with reasoning graphs
        """
        print(f"\n{'='*80}")
        print("Starting Reasoning Graph Pipeline (DAG Support with LCoT Structural Fallback)")
        if start_step > 1:
            print(f"RESTART MODE: Starting from Step {start_step}")
        if max_samples is not None:
            print(f"Sample limit: {max_samples}")
        print(f"Processing {len(input_data)} samples")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}")
        
        pipeline_start = time.time()
        
        # Track if we actually made API calls this run
        initial_sample_count = len(input_data)
        
        data = input_data
        
        # ========================================================================
        # STEP 1: Split Thoughts
        # ========================================================================
        if start_step <= 1:
            data = process_split(
                input_data=data,
                output_dir=self.output_dir
            )
        elif start_step == 2:
            print("\n=== Skipping Step 1, loading process1.json ===")
            data = load_jsonl(self.output_dir / "process1.json")
            print(f"Loaded {len(data)} items from Step 1")
            
            # Apply max_samples limit if restarting
            if max_samples is not None and len(data) > max_samples:
                data = data[:max_samples]
                print(f"Limited to first {max_samples} samples for testing")
        
        # ========================================================================
        # STEP 2: Extract Reasoning Sketch
        # ========================================================================
        if start_step <= 2:
            data = process_sketch(
                input_data=data,
                llm_client=self.llm_client,
                output_dir=self.output_dir,
                max_workers=self.max_workers,
                use_async=self.use_async,
                batch_size=self.batch_size
            )
        elif start_step == 3:
            print("\n=== Skipping Step 2, loading process2.json ===")
            data = load_jsonl(self.output_dir / "process2.json")
            print(f"Loaded {len(data)} items from Step 2")
            
            # Apply max_samples limit if restarting
            if max_samples is not None and len(data) > max_samples:
                data = data[:max_samples]
                print(f"Limited to first {max_samples} samples for testing")
        
        # ========================================================================
        # STEP 3: Assign Thoughts to Steps
        # ========================================================================
        if start_step <= 3:
            data = process_assign(
                input_data=data,
                llm_client=self.llm_client,
                output_dir=self.output_dir,
                max_workers=self.max_workers,
                use_async=self.use_async,
                batch_size=self.batch_size,
                debug=self.debug
            )
        elif start_step == 4:
            print("\n=== Skipping Step 3, loading process3.json ===")
            data = load_jsonl(self.output_dir / "process3.json")
            print(f"Loaded {len(data)} items from Step 3")
            
            # Apply max_samples limit if restarting
            if max_samples is not None and len(data) > max_samples:
                data = data[:max_samples]
                print(f"Limited to first {max_samples} samples for testing")
        
        # ========================================================================
        # STEP 4: Assign Parent Relationships (Critical Path)
        # ========================================================================
        if start_step <= 4:
            data = process_link(
                input_data=data,
                llm_client=self.llm_client,
                output_dir=self.output_dir,
                max_workers=self.max_workers,
                debug=self.debug
            )
        elif start_step == 5:
            print("\n=== Skipping Step 4, loading process4.json ===")
            # Try loading from process4.json first, fallback to incremental
            process4_path = self.output_dir / "process4.json"
            incremental_path = self.output_dir / "process4_incremental.jsonl"
            
            if process4_path.exists():
                data = load_jsonl(process4_path)
                print(f"Loaded {len(data)} items from Step 4 (consolidated)")
            elif incremental_path.exists():
                data = load_jsonl(incremental_path)
                print(f"Loaded {len(data)} items from Step 4 (incremental)")
            else:
                raise FileNotFoundError(
                    f"Cannot find process4.json or process4_incremental.jsonl in {self.output_dir}"
                )
            
            # Apply max_samples limit if restarting
            if max_samples is not None and len(data) > max_samples:
                data = data[:max_samples]
                print(f"Limited to first {max_samples} samples for testing")
        
        # ========================================================================
        # STEP 5: Build Graph Structures
        # ========================================================================
        if start_step <= 5:
            data = process_graph(
                input_data=data,
                output_dir=self.output_dir
            )
        # Note: No need to load after step 5 since it's the final step
        
        # ========================================================================
        # FINAL STATISTICS
        # ========================================================================
        total_elapsed = time.time() - pipeline_start
        
        self._print_final_statistics(data, total_elapsed, start_step)
        
        return data
    
    def _print_final_statistics(self, data: List[Dict[str, Any]], total_elapsed: float, start_step: int):
        """
        Print comprehensive pipeline statistics including cache hit rate.
        
        Args:
            data: Final processed data
            total_elapsed: Total pipeline execution time
            start_step: Which step the pipeline started from
        """
        print(f"\n{'='*80}")
        print(f"Pipeline complete! Final output: {self.output_dir / 'final.json'}")
        print(f"⏱️  Total pipeline time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        print(f"{'='*80}")
        
        # Token usage statistics
        total_in_tokens = sum(item.get("in_token_cost", 0) for item in data)
        total_out_tokens = sum(item.get("out_token_cost", 0) for item in data)
        total_cache_hits = sum(item.get("cache_hit_tokens", 0) for item in data)
        
        # Determine if this was purely a restart with no new API calls
        # Steps 1 and 5 never make API calls
        api_calling_steps = {2, 3, 4}
        made_new_calls = start_step in api_calling_steps or start_step == 1
        
        # For step 4 specifically, check if incremental file existed
        if start_step == 4:
            incremental_path = self.output_dir / "process4_incremental.jsonl"
            if incremental_path.exists():
                # If all items were already processed, no new calls were made
                # This is indicated by checking if any new data was written
                # (We can infer this from whether results differ from loaded data)
                pass  # Let the user see the historical stats
        
        print(f"\n📊 Token Usage Statistics:")
        
        # If restarting from a late step or if tokens are 0, clarify this is historical
        if start_step > 1 or total_in_tokens == 0:
            if total_in_tokens > 0:
                print(f"  ⚠️  NOTE: Token statistics are from loaded checkpoint data")
                print(f"  (May include historical usage from previous runs)")
                print()
            else:
                print(f"  ⚠️  No API calls made in this run")
                print()
        
        print(f"  Input tokens: {total_in_tokens:,}")
        print(f"  Output tokens: {total_out_tokens:,}")
        print(f"  Cache hit tokens: {total_cache_hits:,}")
        print(f"  Total tokens: {total_in_tokens + total_out_tokens:,}")
        
        # Calculate cache hit rate
        if total_in_tokens > 0:
            cache_hit_rate = (total_cache_hits / total_in_tokens) * 100
            print(f"  💰 Cache Hit Rate: {cache_hit_rate:.2f}%")
            
            # Cost savings estimate (assuming cache hits are free)
            effective_tokens = total_in_tokens - total_cache_hits
            savings_pct = (total_cache_hits / total_in_tokens) * 100 if total_in_tokens > 0 else 0
            print(f"  💵 Effective Input Tokens (after cache): {effective_tokens:,}")
            print(f"  💵 Cost Savings: ~{savings_pct:.1f}% reduction in input token costs")
        
        # Graph statistics
        if len(data) > 0 and "graph_stats" in data[0]:
            total_nodes = sum(r.get("graph_stats", {}).get("total_nodes", 0) for r in data)
            total_edges = sum(r.get("graph_stats", {}).get("total_edges", 0) for r in data)
            multi_parent = sum(r.get("graph_stats", {}).get("nodes_with_multiple_parents", 0) for r in data)
            isolated = sum(r.get("graph_stats", {}).get("isolated_nodes", 0) for r in data)
            
            print(f"\n🔗 Graph Statistics:")
            print(f"  Total nodes: {total_nodes:,}")
            print(f"  Total edges: {total_edges:,}")
            print(f"  Nodes with multiple parents (DAG structure): {multi_parent:,}")
            print(f"  Isolated nodes: {isolated:,}")
            print(f"  Avg nodes per graph: {total_nodes/len(data):.1f}")
            print(f"  Avg edges per graph: {total_edges/len(data):.1f}")
        
        # Sample graph structure (if verbose debug mode)
        if self.debug and len(data) > 0 and "reasoning_graph" in data[0]:
            print(f"\n🔍 Sample Reasoning Graph Structure:")
            import json
            sample_graph = json.dumps(data[0]["reasoning_graph"], indent=2)
            if len(sample_graph) > 500:
                print(sample_graph[:500] + "... [truncated]")
            else:
                print(sample_graph)
        
        print(f"\n{'='*80}")
        print("Regeneration completed successfully!")
        print(f"{'='*80}\n")