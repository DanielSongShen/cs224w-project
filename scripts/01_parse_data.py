"""Script to parse SYNTHETIC-1 dataset into graph representations"""
from datasets import load_dataset
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.parser import (
    split_into_nodes, 
    parse_reasoning_trace, 
    ReasoningGraph,
    create_edge_prompt,
    parse_llm_edges
)

# Add graph_of_thoughts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'graph_of_thoughts_examples'))
from graph_of_thoughts import language_models


def load_filtered_samples(n: int, target_dataset: str, verbose: bool = True):
    """
    Load n samples from SYNTHETIC-1 dataset filtered by dataset name.
    
    Args:
        n: Number of examples to load
        target_dataset: Name of the dataset to filter for (e.g., 'PrimeIntellect/verifiable-math-problems')
        verbose: Whether to print progress and sample details
    
    Returns:
        List of examples matching the filter criteria
    """
    # Load SYNTHETIC-1 dataset with streaming to avoid downloading entire dataset
    dataset = load_dataset(
        "PrimeIntellect/SYNTHETIC-1",
        split="train",
        streaming=True
    )
    
    sample_data = []
    count = 0
    
    if verbose:
        print(f"Loading {n} samples from: {target_dataset}")
        print("Searching through dataset...\n")
    
    for i, example in enumerate(dataset):
        # Check if this example is from the target dataset
        if example.get("hf_dataset_name") == target_dataset:
            sample_data.append(example)
            count += 1
            
            if verbose:
                print(f"\n{'='*80}")
                print(f"Match #{count} (overall row {i}):")
                print(f"{'='*80}")
                for key, value in example.items():
                    print(f"\n{key}:")
                    # Truncate long values for readability
                    if isinstance(value, str) and len(value) > 500:
                        print(f"{value[:500]}... [truncated]")
                    else:
                        print(value)
            
            # Stop after finding n matches
            if count >= n:
                break
        
        # Progress update every 1000 rows
        if verbose and (i + 1) % 1000 == 0:
            print(f"Scanned {i + 1} rows, found {count} matches so far...")
    
    if verbose:
        print(f"\n\n{'='*80}")
        print(f"Successfully loaded {len(sample_data)} examples from {target_dataset}")
        if len(sample_data) > 0:
            print(f"Keys in each example: {list(sample_data[0].keys())}")
        print(f"{'='*80}")
    
    return sample_data


# ============================================================================
# Test Functions for Reasoning Trace Parser
# ============================================================================

class LLMClient:
    """Wrapper around graph_of_thoughts ChatGPT for edge creation"""
    
    def __init__(self, config_path: str, model_name: str = "gpt5-nano"):
        """Initialize the LLM client
        
        Args:
            config_path: Path to config.json
            model_name: Model name in config (default: gpt5-nano)
        """
        self.lm = language_models.ChatGPT(config_path, model_name=model_name)
        self.call_count = 0
    
    def query(self, prompt: str) -> str:
        """Query the LLM with a prompt
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response text from LLM
        """
        self.call_count += 1
        response = self.lm.query(prompt, num_responses=1)
        return response[0] if isinstance(response, list) else response


def test_node_splitting():
    """Test splitting reasoning traces into nodes"""
    print("\n" + "="*80)
    print("TEST: Node Splitting")
    print("="*80)
    
    # Test case 1: Simple trace with discourse markers
    trace1 = """First, let's calculate x = 5. Next, we use x in the equation y = 2x. 
    Wait, I made a mistake. x should be 6. Therefore, y = 12."""
    
    nodes1 = split_into_nodes(trace1, min_length=10)
    
    print(f"\nTest 1: Simple trace with markers")
    print(f"Input: {trace1[:100]}...")
    print(f"Number of nodes: {len(nodes1)}")
    for node in nodes1:
        print(f"\nNode {node.node_id} [{node.marker_category}]:")
        print(f"  Markers: {node.markers}")
        print(f"  Text: {node.text[:100]}...")
    
    # Test case 2: Trace with no markers
    trace2 = "Calculate the sum of 1 + 2 + 3 which equals 6."
    nodes2 = split_into_nodes(trace2, min_length=10)
    
    print(f"\n\nTest 2: Trace with no markers")
    print(f"Input: {trace2}")
    print(f"Number of nodes: {len(nodes2)}")
    print(f"Node 0 text: {nodes2[0].text}")
    
    # Test case 3: Trace with backtracking
    trace3 = """Let's solve for x. We have x^2 = 16. So x = 4. 
    Actually, we need to consider both roots. So x = 4 or x = -4. 
    Therefore, the solution set is {-4, 4}."""
    
    nodes3 = split_into_nodes(trace3, min_length=15)
    
    print(f"\n\nTest 3: Trace with backtracking")
    print(f"Number of nodes: {len(nodes3)}")
    for node in nodes3:
        print(f"\nNode {node.node_id} [{node.marker_category}]:")
        print(f"  Markers: {node.markers}")
        print(f"  Text: {node.text[:80]}...")
    
    return nodes1, nodes2, nodes3


def test_edge_creation(config_path: str = "config.json"):
    """Test LLM-based edge creation"""
    print("\n" + "="*80)
    print("TEST: Edge Creation with GPT-5-nano")
    print("="*80)
    
    trace = """Step 1: Calculate x = 5. Next, use x in equation. 
    Wait, x should be 6. Recall from earlier that we need to check. Therefore final answer is 12."""
    
    nodes = split_into_nodes(trace, min_length=10)
    
    print(f"\nCreated {len(nodes)} nodes")
    
    # Test with real LLM
    llm_client = LLMClient(config_path, model_name="gpt5-nano")
    
    # Create prompt
    prompt = create_edge_prompt(nodes)
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"\nPrompt preview:\n{prompt[:300]}...")
    
    # Get response
    print("\nQuerying LLM for edge structure...")
    response = llm_client.query(prompt)
    print(f"\nLLM response:\n{response}")
    
    # Parse edges
    edges = parse_llm_edges(response, len(nodes))
    
    print(f"\nParsed {len(edges)} edges:")
    for edge in edges:
        print(f"  {edge.from_node} -> {edge.to_node} ({edge.edge_type})")
    
    return nodes, edges


def test_full_pipeline(config_path: str = "config.json"):
    """Test the complete parsing pipeline"""
    print("\n" + "="*80)
    print("TEST: Full Parsing Pipeline")
    print("="*80)
    
    trace = """Let's solve the equation 2x + 5 = 15. 
    First, subtract 5 from both sides: 2x = 10. 
    Then, divide by 2: x = 5. 
    Wait, let me verify this. 
    Substituting x = 5 back: 2(5) + 5 = 10 + 5 = 15. 
    Actually, that's correct. 
    Therefore, the solution is x = 5."""
    
    print(f"\nInput trace:\n{trace}\n")
    
    llm_client = LLMClient(config_path, model_name="gpt5-nano")
    graph = parse_reasoning_trace(trace, llm_client, min_node_length=15)
    
    print(f"Graph Statistics:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  LLM calls: {llm_client.call_count}")
    
    print(f"\nNodes:")
    for node in graph.nodes:
        print(f"  {node.node_id}: {node.text[:60]}... [{node.marker_category}]")
    
    print(f"\nEdges:")
    for edge in graph.edges:
        from_text = graph.nodes[edge.from_node].text[:40]
        to_text = graph.nodes[edge.to_node].text[:40]
        print(f"  {edge.from_node} -> {edge.to_node} ({edge.edge_type})")
        print(f"    From: {from_text}...")
        print(f"    To: {to_text}...")
    
    return graph


def test_with_real_data(num_samples=3, config_path: str = "config.json"):
    """Test parser with actual dataset samples"""
    print("\n" + "="*80)
    print("TEST: Parsing Real Dataset Samples")
    print("="*80)
    
    print(f"\nLoading {num_samples} samples from dataset...")
    samples = load_filtered_samples(
        n=num_samples,
        target_dataset="PrimeIntellect/verifiable-math-problems",
        verbose=False
    )
    
    if not samples:
        print("No samples loaded!")
        return
    
    llm_client = LLMClient(config_path, model_name="gpt5-nano")
    
    for i, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*80}")
        
        # Get the reasoning trace (could be in different fields)
        trace = sample.get('reasoning_trace') or sample.get('solution') or sample.get('text', '')
        
        if not trace:
            print("No reasoning trace found in sample")
            continue
        
        print(f"\nOriginal trace length: {len(trace)} chars")
        print(f"Preview: {trace[:200]}...\n")
        
        # Parse into graph
        try:
            graph = parse_reasoning_trace(trace, llm_client, min_node_length=20)
            
            print(f"Parsing Results:")
            print(f"  Nodes: {len(graph.nodes)}")
            print(f"  Edges: {len(graph.edges)}")
            
            print(f"\nFirst 3 nodes:")
            for node in graph.nodes[:3]:
                print(f"  Node {node.node_id} [{node.marker_category}]: {node.text[:80]}...")
            
            if len(graph.nodes) > 3:
                print(f"  ... and {len(graph.nodes) - 3} more nodes")
            
            print(f"\nEdge structure:")
            edge_types = {}
            for edge in graph.edges:
                edge_types[edge.edge_type] = edge_types.get(edge.edge_type, 0) + 1
            for edge_type, count in edge_types.items():
                print(f"  {edge_type}: {count} edges")
                
        except Exception as e:
            print(f"Error parsing sample: {e}")
            import traceback
            traceback.print_exc()


def run_all_tests(config_path: str = "config.json"):
    """Run all test functions
    
    Args:
        config_path: Path to config.json file
    """
    print("\n" + "="*80)
    print("RUNNING ALL PARSER TESTS")
    print(f"Using config: {config_path}")
    print("="*80)
    
    try:
        # Test 1: Node splitting (no LLM needed)
        test_node_splitting()
        
        # Test 2: Edge creation (uses LLM)
        test_edge_creation(config_path)
        
        # Test 3: Full pipeline (uses LLM)
        test_full_pipeline(config_path)
        
        # Test 4: Real data (optional, uses LLM and API)
        # test_with_real_data(num_samples=2, config_path=config_path)
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"TEST FAILED: {e}")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse reasoning traces from SYNTHETIC-1 dataset')
    parser.add_argument('--test', action='store_true', help='Run parser tests')
    parser.add_argument('--analyze', action='store_true', help='Analyze score distribution')
    parser.add_argument('--samples', type=int, default=10000, help='Number of samples to load')
    
    args = parser.parse_args()
    
    if args.test:
        # Run parser tests
        run_all_tests()
    elif args.analyze:
        # Original score distribution analysis
        samples = load_filtered_samples(
            n=args.samples,
            target_dataset="PrimeIntellect/verifiable-math-problems",
            verbose=False
        )
        
        # Analyze score distribution
        print(f"\n{'='*80}")
        print(f"Score Distribution Analysis")
        print(f"{'='*80}")
        
        scores = [sample['score'] for sample in samples]
        
        # Count unique scores
        from collections import Counter
        score_counts = Counter(scores)
        
        print(f"\nTotal samples: {len(samples)}")
        print(f"\nScore distribution:")
        for score in sorted(score_counts.keys()):
            count = score_counts[score]
            percentage = (count / len(samples)) * 100
            print(f"  Score {score}: {count:4d} samples ({percentage:5.2f}%)")
        
        print(f"\nUnique scores: {sorted(score_counts.keys())}")
        
        # Analyze token length of reasoning traces
        print(f"\n{'='*80}")
        print(f"Token Length Analysis")
        print(f"{'='*80}")
        
        # Extract llm_response lengths
        trace_lengths_chars = []
        for sample in samples:
            llm_response = sample.get('llm_response', '')
            if llm_response:
                trace_lengths_chars.append(len(llm_response))
        
        if trace_lengths_chars:
            # Approximate token count: ~4 characters per token for English
            CHARS_PER_TOKEN = 4.0
            trace_lengths_tokens = [length / CHARS_PER_TOKEN for length in trace_lengths_chars]
            
            import statistics
            avg_chars = statistics.mean(trace_lengths_chars)
            avg_tokens = statistics.mean(trace_lengths_tokens)
            median_chars = statistics.median(trace_lengths_chars)
            median_tokens = statistics.median(trace_lengths_tokens)
            min_chars = min(trace_lengths_chars)
            max_chars = max(trace_lengths_chars)
            min_tokens = min(trace_lengths_tokens)
            max_tokens = max(trace_lengths_tokens)
            
            print(f"\nCharacter lengths:")
            print(f"  Average: {avg_chars:.1f} chars")
            print(f"  Median:  {median_chars:.1f} chars")
            print(f"  Min:     {min_chars} chars")
            print(f"  Max:     {max_chars} chars")
            
            print(f"\nEstimated token counts (using ~{CHARS_PER_TOKEN} chars/token):")
            print(f"  Average: {avg_tokens:.1f} tokens")
            print(f"  Median:  {median_tokens:.1f} tokens")
            print(f"  Min:     {min_tokens:.1f} tokens")
            print(f"  Max:     {max_tokens:.1f} tokens")
            
            # Calculate percentiles for better understanding
            sorted_tokens = sorted(trace_lengths_tokens)
            p25 = sorted_tokens[len(sorted_tokens) // 4]
            p75 = sorted_tokens[3 * len(sorted_tokens) // 4]
            p90 = sorted_tokens[9 * len(sorted_tokens) // 10]
            p95 = sorted_tokens[95 * len(sorted_tokens) // 100]
            
            print(f"\nToken count percentiles:")
            print(f"  25th: {p25:.1f} tokens")
            print(f"  75th: {p75:.1f} tokens")
            print(f"  90th: {p90:.1f} tokens")
            print(f"  95th: {p95:.1f} tokens")
        else:
            print("\nNo llm_response found in samples")
        
        print(f"{'='*80}")
    else:
        print("Please specify --test or --analyze")
        print("Usage: python 01_parse_data.py --test  # Run parser tests")
        print("       python 01_parse_data.py --analyze --samples 1000  # Analyze score distribution")
