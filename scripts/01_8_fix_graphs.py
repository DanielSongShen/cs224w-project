#!/usr/bin/env python3
"""
Fix orphaned Step 1 thoughts in completed reasoning graphs (final.json).

This script works on the FINAL output (after Step 5) where graphs are complete.
It adds missing Root (0) → Step 1 edges and recalculates graph statistics.

Usage:
    python fix_final_graphs.py \
        --input data/processed/graphs/final.json \
        --output data/processed/graphs/final_fixed.json
    
    # With verbose output
    python fix_final_graphs.py \
        --input data/processed/graphs/final.json \
        --output data/processed/graphs/final_fixed.json \
        --verbose
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict


def load_data(path: Path) -> List[Dict[str, Any]]:
    """Load data from JSONL format."""
    print(f"Loading data from {path}...")
    
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"  Loaded {len(data)} items")
    return data


def save_data(data: List[Dict[str, Any]], path: Path):
    """Save data in JSONL format."""
    print(f"Saving fixed data to {path}...")
    
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"  Saved {len(data)} items")


def normalize_key(key: Any) -> int:
    """Normalize keys to integers."""
    if isinstance(key, str):
        return int(key)
    return int(key)


def find_step1_thoughts(item: Dict[str, Any]) -> List[int]:
    """
    Find all thought IDs assigned to Step 1.
    Uses precise integer matching to avoid false positives (e.g., Step 10, 11, 21).
    
    Args:
        item: Data item with 'assigned_step' field
    
    Returns:
        List of thought IDs belonging to Step 1
    """
    assigned_step = item.get("assigned_step", {})
    step1_thoughts = []
    
    for thought_id, steps in assigned_step.items():
        # Normalize thought_id to int
        t_id = normalize_key(thought_id)
        
        # Skip Root Node (ID 0)
        if t_id == 0:
            continue
        
        # Normalize steps to a list of integers
        if not isinstance(steps, list):
            steps = [steps]
        
        # Robustly convert all step identifiers to int
        step_ints = set()
        for s in steps:
            try:
                # Handle both int and string formats, strip any non-numeric suffixes
                step_ints.add(int(str(s).replace("A", "").replace("B", "")))
            except (ValueError, TypeError):
                continue
        
        # Precise check: Is integer 1 in the set?
        if 1 in step_ints:
            step1_thoughts.append(t_id)
    
    return sorted(step1_thoughts)


def calculate_graph_stats(nodes: List[int], edges: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive graph statistics.
    
    Args:
        nodes: List of node IDs
        edges: List of edge dictionaries with 'source', 'target', 'category'
    
    Returns:
        Dictionary with graph statistics
    """
    num_nodes = len(nodes)
    num_edges = len(edges)
    
    # Build adjacency structures
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    parents = defaultdict(set)
    
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        
        out_degree[src] += 1
        in_degree[tgt] += 1
        parents[tgt].add(src)
    
    # Calculate statistics
    isolated_nodes = sum(1 for node in nodes if in_degree[node] == 0 and out_degree[node] == 0)
    nodes_with_multiple_parents = sum(1 for node in nodes if len(parents[node]) > 1)
    nodes_with_edges = sum(1 for node in nodes if in_degree[node] > 0 or out_degree[node] > 0)
    
    return {
        "total_nodes": num_nodes,
        "total_edges": num_edges,
        "isolated_nodes": isolated_nodes,
        "nodes_with_multiple_parents": nodes_with_multiple_parents,
        "avg_in_degree": sum(in_degree.values()) / max(nodes_with_edges, 1),
        "avg_out_degree": sum(out_degree.values()) / max(nodes_with_edges, 1),
        "max_in_degree": max(in_degree.values()) if in_degree else 0,
        "max_out_degree": max(out_degree.values()) if out_degree else 0
    }


def fix_item_graph(item: Dict[str, Any], verbose: bool = False) -> Dict[str, int]:
    """
    Fix root connections for a single item's reasoning graph.
    
    Args:
        item: Data item with 'reasoning_graph' and 'assigned_step'
        verbose: Print details for each item
    
    Returns:
        Dictionary with statistics: 'step1_thoughts', 'already_connected', 'newly_connected'
    """
    stats = {
        'step1_thoughts': 0,
        'already_connected': 0,
        'newly_connected': 0
    }
    
    # Check if item has reasoning_graph
    if "reasoning_graph" not in item:
        return stats
    
    # Find all Step 1 thoughts
    step1_thoughts = find_step1_thoughts(item)
    stats['step1_thoughts'] = len(step1_thoughts)
    
    if not step1_thoughts:
        return stats
    
    # Get reasoning graph
    reasoning_graph = item["reasoning_graph"]
    nodes = reasoning_graph.get("nodes", [])
    edges = reasoning_graph.get("edges", [])
    
    # Build set of existing edges from Root (0)
    root_children = set()
    for edge in edges:
        if edge['source'] == 0:
            root_children.add(edge['target'])
    
    # Add missing edges
    new_edges_added = []
    for t_id in step1_thoughts:
        if t_id in root_children:
            stats['already_connected'] += 1
            if verbose:
                print(f"    Thought {t_id}: Already connected to Root")
        else:
            # Add edge: Root (0) -> Thought (t_id) with Category 1 (Continuous Logic)
            new_edge = {
                "source": 0,
                "target": t_id,
                "category": 1
            }
            edges.append(new_edge)
            new_edges_added.append(t_id)
            stats['newly_connected'] += 1
            if verbose:
                print(f"    Thought {t_id}: ✓ Connected to Root")
    
    # Update reasoning graph
    item["reasoning_graph"]["edges"] = edges
    
    # Recalculate graph statistics
    if stats['newly_connected'] > 0:
        item["graph_stats"] = calculate_graph_stats(nodes, edges)
    
    return stats


def analyze_orphans(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze orphan statistics before fixing.
    
    Args:
        data: List of items to analyze
    
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*80)
    print("ANALYZING ORPHAN STATISTICS")
    print("="*80)
    
    total_items = len(data)
    items_with_graphs = 0
    items_with_step1 = 0
    total_step1_thoughts = 0
    orphaned_step1_thoughts = 0
    
    for item in data:
        if "reasoning_graph" not in item:
            continue
        
        items_with_graphs += 1
        
        step1_thoughts = find_step1_thoughts(item)
        if not step1_thoughts:
            continue
        
        items_with_step1 += 1
        total_step1_thoughts += len(step1_thoughts)
        
        # Check which are orphaned
        edges = item["reasoning_graph"].get("edges", [])
        root_children = {edge['target'] for edge in edges if edge['source'] == 0}
        
        for t_id in step1_thoughts:
            if t_id not in root_children:
                orphaned_step1_thoughts += 1
    
    print(f"\nTotal items: {total_items}")
    print(f"Items with reasoning graphs: {items_with_graphs}")
    print(f"Items with Step 1 thoughts: {items_with_step1}")
    print(f"Total Step 1 thoughts: {total_step1_thoughts}")
    print(f"Orphaned Step 1 thoughts: {orphaned_step1_thoughts}")
    
    if total_step1_thoughts > 0:
        orphan_rate = (orphaned_step1_thoughts / total_step1_thoughts) * 100
        print(f"Orphan rate: {orphan_rate:.1f}%")
    
    return {
        'total_items': total_items,
        'items_with_graphs': items_with_graphs,
        'items_with_step1': items_with_step1,
        'total_step1_thoughts': total_step1_thoughts,
        'orphaned_step1_thoughts': orphaned_step1_thoughts
    }


def fix_graphs(input_path: str, output_path: str, verbose: bool = False):
    """
    Main function to fix root connections in completed graphs.
    
    Args:
        input_path: Path to input final.json
        output_path: Path to output final_fixed.json
        verbose: Print detailed progress for each item
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_data(input_file)
    
    # Analyze before fixing
    pre_stats = analyze_orphans(data)
    
    # Fix each item
    print("\n" + "="*80)
    print("FIXING ROOT CONNECTIONS")
    print("="*80 + "\n")
    
    total_stats = defaultdict(int)
    items_modified = 0
    
    for idx, item in enumerate(data):
        item_stats = fix_item_graph(item, verbose=verbose)
        
        # Aggregate statistics
        for key, value in item_stats.items():
            total_stats[key] += value
        
        # Track modified items
        if item_stats['newly_connected'] > 0:
            items_modified += 1
        
        # Progress indicator
        if (idx + 1) % 100 == 0 or verbose:
            tag = item.get('tag', f'item_{idx}')
            if item_stats['newly_connected'] > 0:
                print(f"[{idx+1}/{len(data)}] {tag}: Fixed {item_stats['newly_connected']} orphans")
    
    # Print final statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nItems processed: {len(data)}")
    print(f"Items modified: {items_modified}")
    print(f"\nStep 1 thoughts found: {total_stats['step1_thoughts']}")
    print(f"Already connected: {total_stats['already_connected']}")
    print(f"Newly connected: {total_stats['newly_connected']}")
    
    if total_stats['step1_thoughts'] > 0:
        connection_rate = (total_stats['newly_connected'] / total_stats['step1_thoughts']) * 100
        print(f"\nPercentage fixed: {connection_rate:.1f}%")
    
    # Save fixed data
    print()
    save_data(data, output_file)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFixed graphs saved to: {output_file}")
    print("\nNext steps:")
    print(f"  1. Verify the fix: python verify_final_graphs.py --input {output_file}")
    print(f"  2. Generate PyG dataset: python src/data/dataset.py {output_file} --format graph")
    print(f"  3. Train models: python scripts/02_train_model.py --pt-file ...")
    print()


def verify_fix(input_path: str):
    """
    Verify that all Step 1 thoughts are connected to Root.
    
    Args:
        input_path: Path to fixed file
    """
    print("\n" + "="*80)
    print("VERIFYING FIX")
    print("="*80 + "\n")
    
    input_file = Path(input_path)
    data = load_data(input_file)
    
    total_step1 = 0
    orphaned = 0
    items_with_orphans = []
    
    for idx, item in enumerate(data):
        if "reasoning_graph" not in item:
            continue
        
        step1_thoughts = find_step1_thoughts(item)
        if not step1_thoughts:
            continue
        
        edges = item["reasoning_graph"].get("edges", [])
        root_children = {edge['target'] for edge in edges if edge['source'] == 0}
        
        item_orphans = []
        for t_id in step1_thoughts:
            total_step1 += 1
            if t_id not in root_children:
                orphaned += 1
                item_orphans.append(t_id)
        
        if item_orphans:
            tag = item.get('tag', f'item_{idx}')
            items_with_orphans.append((tag, item_orphans))
    
    print(f"Total Step 1 thoughts: {total_step1}")
    print(f"Orphaned: {orphaned}")
    
    if orphaned == 0:
        print("\n✓ SUCCESS: All Step 1 thoughts are connected to Root!")
    else:
        print(f"\n✗ WARNING: {orphaned} orphans still exist")
        print("\nItems with orphans:")
        for tag, orphan_list in items_with_orphans[:5]:  # Show first 5
            print(f"  {tag}: {orphan_list}")
        if len(items_with_orphans) > 5:
            print(f"  ... and {len(items_with_orphans) - 5} more")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix orphaned Step 1 thoughts in completed reasoning graphs (final.json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix completed graphs
  python fix_final_graphs.py \\
    --input data/processed/graphs/final.json \\
    --output data/processed/graphs/final_fixed.json
  
  # With verbose output
  python fix_final_graphs.py \\
    --input data/processed/graphs/final.json \\
    --output data/processed/graphs/final_fixed.json \\
    --verbose
  
  # Verify the fix
  python fix_final_graphs.py \\
    --verify \\
    --input data/processed/graphs/final_fixed.json
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input final.json"
    )
    parser.add_argument(
        "--output",
        help="Path to output final_fixed.json",
        default = "data\\processed\\final_fixed.json"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress for each item"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that fix was successful (uses --output path)"
    )
    
    args = parser.parse_args()
    
    fix_graphs(args.input, args.output, args.verbose)
    verify_fix(args.output)