"""Parse reasoning traces from SYNTHETIC-1 dataset"""

"""
IMPLEMENTATION SPEC: Reasoning Trace Node Parser
================================================

Purpose:
    Convert linear CoT reasoning traces into graph nodes by splitting on discourse markers
    that indicate reasoning transitions, backtracking, or alternative exploration.

Input:
    - reasoning_trace: str (raw CoT text from dataset)

Output:
    - List[Node] where each Node contains:
        - text: str (content of reasoning step)
        - node_id: int (unique identifier)
        - markers: List[str] (discourse markers that preceded this node)

Splitting Strategy:
    Split on discourse markers that indicate reasoning structure:
    
    Backtracking: "wait", "actually", "no", "hold on", "that's wrong", "mistake"
    Alternatives: "alternatively", "instead", "or we could", "another way", "what if"
    Continuation: "next", "then", "now", "so", "therefore", "thus"
    Refinement: "better yet", "more precisely", "to clarify", "correcting"
    Reference: "recall", "as we found", "from earlier", "using this"
    
Algorithm:
    1. Normalize text (lowercase for matching, preserve original for content)
    2. Identify all marker positions in text
    3. Split text at marker boundaries
    4. Create node for each segment with associated marker type
    5. Clean and validate nodes (remove empty, merge too-short segments)
    
Edge cases:
    - Multiple markers in sequence → prioritize by importance (backtrack > alternative > continuation)
    - Very short segments (<20 chars) → merge with previous/next
    - No markers found → return single node with full text

Edge Creation (LLM-based):
    Use LLM to identify semantic relationships between nodes for edge creation.
    
    Approach: Single-prompt batch processing for efficiency
    
    Input to LLM:
        - List of nodes with IDs and text: [(0, "text0"), (1, "text1"), ...]
        - Prompt: "Identify which reasoning steps should be connected. Create edges for:
                   1. Sequential flow (one step leads to next)
                   2. Backtracking (revisiting/correcting earlier step)
                   3. References (using result from earlier step)
                   4. Dependencies (step requires earlier step's output)"
    
    Output from LLM (structured JSON):
        {
            "edges": [
                {"from": 0, "to": 1, "type": "sequential"},
                {"from": 3, "to": 1, "type": "backtrack"},
                {"from": 4, "to": 2, "type": "reference"}
            ]
        }
    
    Efficiency optimizations:
        - Process entire node list in single prompt (avoid O(n²) calls)
        - Use structured output (JSON) for reliable parsing
        - Set max_tokens based on expected edge count (~50 tokens per edge)
        - For very long traces (>20 nodes), use sliding window approach:
          * Window size W=10 nodes, overlap O=3 nodes
          * Process windows separately, merge edge lists
          * Only consider edges within window (prevents quadratic blowup)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json


# Discourse marker categories
DISCOURSE_MARKERS = {
    "backtrack": ["wait", "actually", "no", "hold on", "that's wrong", "mistake"],
    "alternative": ["alternatively", "instead", "or we could", "another way", "what if"],
    "continuation": ["next", "then", "now", "so", "therefore", "thus"],
    "refinement": ["better yet", "more precisely", "to clarify", "correcting"],
    "reference": ["recall", "as we found", "from earlier", "using this"],
}

# Flatten all markers for easy lookup
ALL_MARKERS = {marker: category for category, markers in DISCOURSE_MARKERS.items() for marker in markers}


@dataclass
class ReasoningNode:
    """Represents a single reasoning step in the trace"""
    node_id: int
    text: str
    markers: List[str] = field(default_factory=list)
    marker_category: Optional[str] = None


@dataclass
class ReasoningEdge:
    """Represents a connection between reasoning steps"""
    from_node: int
    to_node: int
    edge_type: Optional[str] = None


@dataclass
class ReasoningGraph:
    """Complete graph representation of a reasoning trace"""
    nodes: List[ReasoningNode]
    edges: List[ReasoningEdge]
    original_trace: str


def find_marker_positions(text: str) -> List[Tuple[int, str, str]]:
    """
    Find all discourse marker positions in text.
    
    Args:
        text: Input reasoning trace
        
    Returns:
        List of (position, marker_text, category) tuples sorted by position
    """
    text_lower = text.lower()
    positions = []
    
    for marker, category in ALL_MARKERS.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(marker) + r'\b'
        for match in re.finditer(pattern, text_lower):
            positions.append((match.start(), marker, category))
    
    # Sort by position and remove duplicates (keep first marker at same position)
    positions.sort(key=lambda x: x[0])
    return positions


def split_into_nodes(text: str, min_length: int = 20) -> List[ReasoningNode]:
    """
    Split reasoning trace into nodes based on discourse markers.
    
    Args:
        text: Input reasoning trace
        min_length: Minimum character length for a valid node
        
    Returns:
        List of ReasoningNode objects
    """
    marker_positions = find_marker_positions(text)
    
    if not marker_positions:
        # No markers found, return single node
        return [ReasoningNode(node_id=0, text=text.strip())]
    
    nodes = []
    node_id = 0
    last_pos = 0
    last_marker = None
    last_category = None
    
    for i, (pos, marker, category) in enumerate(marker_positions):
        # Extract text segment before this marker
        segment = text[last_pos:pos].strip()
        
        if segment and len(segment) >= min_length:
            # Create node for segment with the marker that preceded it
            nodes.append(ReasoningNode(
                node_id=node_id,
                text=segment,
                markers=[last_marker] if last_marker else [],
                marker_category=last_category
            ))
            node_id += 1
        
        # Move to position after the marker
        marker_end = pos + len(marker)
        last_pos = marker_end
        last_marker = marker
        last_category = category
    
    # Add remaining text after last marker
    final_segment = text[last_pos:].strip()
    if final_segment and len(final_segment) >= min_length:
        nodes.append(ReasoningNode(
            node_id=node_id,
            text=final_segment,
            markers=[last_marker] if last_marker else [],
            marker_category=last_category
        ))
    
    # If no valid nodes created, return full text as single node
    if not nodes:
        return [ReasoningNode(node_id=0, text=text.strip())]
    
    return nodes


def create_edge_prompt(nodes: List[ReasoningNode]) -> str:
    """
    Create prompt for LLM to identify edges between nodes.
    
    Args:
        nodes: List of reasoning nodes
        
    Returns:
        Formatted prompt string
    """
    nodes_text = "\n".join([f"{node.node_id}: {node.text}" for node in nodes])
    
    prompt = f"""Analyze this mathematical reasoning trace broken into steps. Identify which steps should be connected based on their logical relationships.

Reasoning Steps:
{nodes_text}

Create edges between steps for these relationships:
1. Sequential flow: One step naturally leads to the next
2. Backtracking: A step revisits or corrects an earlier step
3. Reference: A step uses a result or insight from an earlier step
4. Dependency: A step requires information from an earlier step

Return ONLY a JSON object with this exact format:
{{
    "edges": [
        {{"from": 0, "to": 1, "type": "sequential"}},
        {{"from": 3, "to": 1, "type": "backtrack"}}
    ]
}}

Edges should be directed. Include all meaningful connections, not just sequential ones."""
    
    return prompt


def parse_llm_edges(llm_response: str, num_nodes: int) -> List[ReasoningEdge]:
    """
    Parse LLM response into edge list.
    
    Args:
        llm_response: JSON string from LLM
        num_nodes: Number of nodes (for validation)
        
    Returns:
        List of ReasoningEdge objects
    """
    try:
        # Try to extract JSON from response
        # LLM might include extra text, so find JSON block
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(llm_response)
        
        edges = []
        for edge_dict in data.get("edges", []):
            from_node = edge_dict["from"]
            to_node = edge_dict["to"]
            edge_type = edge_dict.get("type")
            
            # Validate node IDs
            if 0 <= from_node < num_nodes and 0 <= to_node < num_nodes:
                edges.append(ReasoningEdge(
                    from_node=from_node,
                    to_node=to_node,
                    edge_type=edge_type
                ))
        
        return edges
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Failed to parse LLM response: {e}")
        return create_fallback_edges(num_nodes)


def create_fallback_edges(num_nodes: int) -> List[ReasoningEdge]:
    """
    Create simple sequential edges as fallback.
    
    Args:
        num_nodes: Number of nodes
        
    Returns:
        List of sequential edges
    """
    edges = []
    for i in range(num_nodes - 1):
        edges.append(ReasoningEdge(
            from_node=i,
            to_node=i + 1,
            edge_type="sequential"
        ))
    return edges


def create_edges_with_llm(nodes: List[ReasoningNode], llm_client, window_size: int = 20) -> List[ReasoningEdge]:
    """
    Use LLM to create edges between nodes.
    
    Args:
        nodes: List of reasoning nodes
        llm_client: LLM client object with query method
        window_size: Maximum nodes per prompt (for long traces)
        
    Returns:
        List of ReasoningEdge objects
    """
    if len(nodes) <= window_size:
        # Process all nodes in single prompt
        prompt = create_edge_prompt(nodes)
        response = llm_client.query(prompt)
        return parse_llm_edges(response, len(nodes))
    
    else:
        # Use sliding window for long traces
        all_edges = []
        overlap = 3
        
        for start in range(0, len(nodes), window_size - overlap):
            end = min(start + window_size, len(nodes))
            window_nodes = nodes[start:end]
            
            # Adjust node IDs for window
            prompt = create_edge_prompt(window_nodes)
            response = llm_client.query(prompt)
            window_edges = parse_llm_edges(response, len(window_nodes))
            
            # Adjust edge node IDs back to global IDs
            for edge in window_edges:
                edge.from_node += start
                edge.to_node += start
                all_edges.append(edge)
            
            if end >= len(nodes):
                break
        
        # Remove duplicate edges
        unique_edges = []
        seen = set()
        for edge in all_edges:
            key = (edge.from_node, edge.to_node)
            if key not in seen:
                seen.add(key)
                unique_edges.append(edge)
        
        return unique_edges


def parse_reasoning_trace(trace: str, llm_client, min_node_length: int = 20) -> ReasoningGraph:
    """
    Main function to parse reasoning trace into graph structure.
    
    Args:
        trace: Raw reasoning trace text
        llm_client: LLM client for edge creation
        min_node_length: Minimum character length for nodes
        
    Returns:
        ReasoningGraph object with nodes and edges
    """
    # Step 1: Split into nodes
    nodes = split_into_nodes(trace, min_length=min_node_length)
    
    # Step 2: Create edges using LLM
    edges = create_edges_with_llm(nodes, llm_client)
    
    # Step 3: Return complete graph
    return ReasoningGraph(
        nodes=nodes,
        edges=edges,
        original_trace=trace
    )

