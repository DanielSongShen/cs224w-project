"""Step 5: Build Graph Structure

Converts parent-child relationships into clean DAG representation.
Includes all nodes and edges with comprehensive statistics.
"""
import json
import time
from typing import List, Dict, Any
from pathlib import Path

from ..utils import normalize_thought_keys, save_jsonl


def process_graph(
    input_data: List[Dict[str, Any]],
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Build graph structure from parent-child relationships.
    
    Creates a clean edge list representation (DAG-friendly).
    All nodes from thoughts_list are included.
    All edges from thought_relations are preserved (including multiple parents).
    
    Args:
        input_data: Items with 'thought_relations' from step 4
        output_dir: Directory to save final results
    
    Returns:
        Items with added 'reasoning_graph' and 'graph_stats' fields
    """
    print("\n=== Step 5: Building graph structures ===")
    start_time = time.time()
    
    results = []
    for item in input_data:
        try:
            thought_relations = item.get("thought_relations", {})
            
            # Convert keys to integers if needed
            if isinstance(thought_relations, str):
                thought_relations = json.loads(thought_relations)
            
            # Normalize to integer keys: {parent_id: {child_id: category}}
            relations = {}
            for src_key, targets in thought_relations.items():
                src = int(src_key) if not isinstance(src_key, int) else src_key
                relations[src] = {}
                for tgt_key, category in targets.items():
                    tgt = int(tgt_key) if not isinstance(tgt_key, int) else tgt_key
                    relations[src][tgt] = category
            
            # Build edge list
            edges = []
            all_nodes = set()
            
            for parent_id, targets in relations.items():
                all_nodes.add(parent_id)
                for child_id, category in targets.items():
                    all_nodes.add(child_id)
                    edges.append({
                        "source": parent_id,
                        "target": child_id,
                        "category": category
                    })
            
            # Get all thoughts to ensure complete node list
            thought_list = normalize_thought_keys(item["thoughts_list"])
            total_thoughts = len(thought_list)
            
            # Create node list (all thoughts)
            nodes = list(range(total_thoughts))
            
            # Build graph structure
            item["reasoning_graph"] = {
                "nodes": nodes,
                "edges": edges
            }
            
            # Add statistics
            nodes_with_edges = len(all_nodes)
            isolated_nodes = [n for n in nodes if n not in all_nodes]
            
            # Calculate in-degree and out-degree
            in_degree = {}
            out_degree = {}
            for edge in edges:
                src = edge["source"]
                tgt = edge["target"]
                out_degree[src] = out_degree.get(src, 0) + 1
                in_degree[tgt] = in_degree.get(tgt, 0) + 1
            
            # Find nodes with multiple parents
            multi_parent_nodes = [node for node, degree in in_degree.items() if degree > 1]
            
            item["graph_stats"] = {
                "total_nodes": total_thoughts,
                "nodes_with_edges": nodes_with_edges,
                "isolated_nodes": len(isolated_nodes),
                "total_edges": len(edges),
                "nodes_with_multiple_parents": len(multi_parent_nodes),
                "avg_in_degree": sum(in_degree.values()) / max(nodes_with_edges, 1),
                "avg_out_degree": sum(out_degree.values()) / max(nodes_with_edges, 1),
                "max_in_degree": max(in_degree.values()) if in_degree else 0,
                "max_out_degree": max(out_degree.values()) if out_degree else 0
            }
            
            results.append(item)
            
        except Exception as e:
            print(f"Error building graph for {item.get('tag', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save
    output_path = output_dir / "final.json"
    save_jsonl(results, output_path)
    
    elapsed = time.time() - start_time
    
    # Print summary statistics
    total_edges = sum(r.get("graph_stats", {}).get("total_edges", 0) for r in results)
    total_nodes = sum(r.get("graph_stats", {}).get("total_nodes", 0) for r in results)
    isolated = sum(r.get("graph_stats", {}).get("isolated_nodes", 0) for r in results)
    multi_parent = sum(r.get("graph_stats", {}).get("nodes_with_multiple_parents", 0) for r in results)
    
    print(f"Saved {len(results)} graphs to {output_path}")
    print(f"Graph statistics:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Total edges: {total_edges}")
    print(f"  Nodes with multiple parents: {multi_parent}")
    print(f"  Isolated nodes: {isolated}")
    print(f"  Average edges per graph: {total_edges/max(len(results), 1):.1f}")
    print(f"⏱️  Step 5 completed in {elapsed:.2f} seconds")
    
    return results