#!/usr/bin/env python3
"""
Visualize Chain-of-Thought graphs from reasoning pipeline output.

Updated for graph representation with hierarchical tree layout.
COMPLETELY NON-RECURSIVE VERSION - Fixed all recursion depth issues.
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, Any, List, Tuple
from collections import Counter

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Category colors for different thought types
CATEGORY_COLORS = {
    0: '#808080',  # Root - Gray
    1: '#4CAF50',  # Continuous Logic - Green
    2: '#FF9800',  # Exploration - Orange
    3: '#2196F3',  # Backtracking - Blue
    4: '#FFC107',  # Validation - Amber
    5: '#9C27B0',  # Default - Purple (fallback connections)
}

CATEGORY_NAMES = {
    0: 'Root',
    1: 'Continuous Logic',
    2: 'Exploration',
    3: 'Backtracking',
    4: 'Validation',
    5: 'Default',
}


def graph_from_edge_list(reasoning_graph: Dict[str, Any]) -> nx.DiGraph:
    """
    Create NetworkX directed graph from edge list format.
    
    Args:
        reasoning_graph: Graph dict with 'nodes' and 'edges' fields
            nodes: List of node IDs
            edges: List of dicts with 'source', 'target', 'category'
    
    Returns:
        NetworkX directed graph
    """
    graph = nx.DiGraph()
    
    # Add all nodes
    nodes = reasoning_graph.get('nodes', [])
    for node_id in nodes:
        # Compute level based on shortest path from node 0 (root)
        # We'll do this after adding all edges
        graph.add_node(
            node_id,
            thought_num=node_id,
            thought_list=[node_id]
        )
    
    # Add all edges
    edges = reasoning_graph.get('edges', [])
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        category = edge.get('category', 0)
        
        if source is not None and target is not None:
            graph.add_edge(source, target, category=category)
    
    # Compute levels using BFS from root (node 0)
    if 0 in graph:
        levels = {0: 0}
        queue = [0]
        
        while queue:
            node = queue.pop(0)
            current_level = levels[node]
            
            for child in graph.successors(node):
                if child not in levels:
                    levels[child] = current_level + 1
                    queue.append(child)
        
        # Set level and subset attributes for layout
        for node in graph.nodes():
            level = levels.get(node, -1)
            graph.nodes[node]['level'] = level
            graph.nodes[node]['subset'] = level
            graph.nodes[node]['category'] = 0  # Default category for nodes
    else:
        # No root node, set all to level 0
        for node in graph.nodes():
            graph.nodes[node]['level'] = 0
            graph.nodes[node]['subset'] = 0
            graph.nodes[node]['category'] = 0
    
    return graph


def add_all_nodes_and_edges(graph: nx.DiGraph, item_data: Dict[str, Any], show_isolated: bool = True) -> nx.DiGraph:
    """
    Add all nodes from thoughts_list and all edges from thought_relations.
    
    Args:
        graph: Existing graph from reasoning_graph structure
        item_data: Full item data with thought_relations (legacy field)
        show_isolated: Whether to include isolated nodes
    
    Returns:
        Graph with all nodes and edges added
    """
    # Build thought_num to node_id mapping
    thought_to_node = {}
    for node_id, data in graph.nodes(data=True):
        thought_num = data.get('thought_num', node_id)
        thought_to_node[thought_num] = node_id
    
    # Add isolated nodes only if show_isolated=True
    if show_isolated and 'thoughts_list' in item_data:
        thoughts_list = item_data['thoughts_list']
        if isinstance(thoughts_list, str):
            thoughts_list = json.loads(thoughts_list)
        
        for thought_id in thoughts_list.keys():
            thought_num = int(thought_id) if not isinstance(thought_id, int) else thought_id
            if thought_num not in thought_to_node:
                node_id = thought_num
                graph.add_node(
                    node_id,
                    level=-1,
                    category=0,
                    thought_num=thought_num,
                    thought_list=[thought_num],
                    subset=-1
                )
                thought_to_node[thought_num] = node_id
    
    # Add edges from thought_relations (legacy field, if present)
    if 'thought_relations' not in item_data:
        return graph
    
    relations = item_data['thought_relations']
    
    for src_thought, targets in relations.items():
        src_thought = int(src_thought) if not isinstance(src_thought, int) else src_thought
        
        for tgt_thought, category in targets.items():
            tgt_thought = int(tgt_thought) if not isinstance(tgt_thought, int) else tgt_thought
            
            src_node = thought_to_node.get(src_thought)
            tgt_node = thought_to_node.get(tgt_thought)
            
            if src_node is not None and tgt_node is not None:
                if not graph.has_edge(src_node, tgt_node):
                    graph.add_edge(src_node, tgt_node, category=category)
    
    return graph


def get_multipartite_layout(graph: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
    """
    Create hierarchical layout using multipartite_layout (NON-RECURSIVE).
    
    This uses NetworkX's multipartite_layout which is based on linear algebra
    and never hits recursion limits. It works for any graph structure including
    DAGs with multiple parents and even cycles.
    
    Args:
        graph: Directed graph
    
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    # Ensure every node has a 'subset' attribute for the layout
    for n, d in graph.nodes(data=True):
        if 'subset' not in d:
            d['subset'] = d.get('level', 0)

    try:
        # subset_key tells nx to group nodes by their 'level'
        # align='horizontal' makes the layers run Left->Right
        pos = nx.multipartite_layout(graph, subset_key="subset", align='horizontal')
        
        # ROTATE the layout: standard multipartite is Left->Right. 
        # We want Top->Down. So we swap (x,y) -> (y, -x)
        # x becomes the horizontal spread, y becomes the negative level (downwards)
        pos = {node: (y, -x) for node, (x, y) in pos.items()}
    except Exception as e:
        print(f"Layout fallback due to: {e}")
        # Fallback to spring layout if multipartite fails
        pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        
    return pos


def get_relation_stats(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract statistics from thought_relations field (legacy) or reasoning_graph.
    
    Args:
        item_data: Full item data
        
    Returns:
        Dictionary with relation statistics
    """
    stats = {
        'has_relations': False,
        'total_edges': 0,
        'category_dist': {},
    }
    
    # Try reasoning_graph first (new format)
    if 'reasoning_graph' in item_data:
        graph = item_data['reasoning_graph']
        edges = graph.get('edges', [])
        
        if edges:
            stats['has_relations'] = True
            stats['total_edges'] = len(edges)
            
            all_categories = [edge.get('category', 0) for edge in edges]
            category_counts = Counter(all_categories)
            stats['category_dist'] = dict(category_counts)
        
        return stats
    
    # Fall back to thought_relations (legacy format)
    if 'thought_relations' not in item_data:
        return stats

    relations = item_data['thought_relations']
    stats['has_relations'] = True

    all_categories = []
    for targets in relations.values():
        for category in targets.values():
            all_categories.append(category)

    stats['total_edges'] = len(all_categories)
    category_counts = Counter(all_categories)
    stats['category_dist'] = dict(category_counts)
    
    return stats


def visualize_graph(reasoning_graph: Dict[str, Any], item_data: Dict[str, Any], 
                    output_path: str, show_thoughts: bool = False,
                    show_edge_labels: bool = False, show_isolated: bool = False):
    """
    Visualize a single reasoning graph with hierarchical layout.
    
    Args:
        reasoning_graph: Graph structure from reasoning_graph field
        item_data: Full item data including thoughts_list, tag, etc.
        output_path: Path to save the visualization
        show_thoughts: Whether to show thought text in node labels
        show_edge_labels: Whether to show edge categories
        show_isolated: Whether to show isolated nodes
    """
    # 1. Build Graph from Edge List (NO RECURSION)
    graph = graph_from_edge_list(reasoning_graph)
    graph = add_all_nodes_and_edges(graph, item_data, show_isolated)
    
    # 2. Compute Layout (NON-RECURSIVE multipartite_layout)
    pos = get_multipartite_layout(graph)
    
    relation_stats = get_relation_stats(item_data)
    has_stats = relation_stats['has_relations']
    
    if has_stats:
        fig = plt.figure(figsize=(18, 10))
        ax_main = plt.subplot(1, 2, 1)
        ax_stats = plt.subplot(1, 2, 2)
    else:
        fig, ax_main = plt.subplots(figsize=(16, 10))
    
    categories = nx.get_node_attributes(graph, 'category')
    thought_nums = nx.get_node_attributes(graph, 'thought_num')
    levels = nx.get_node_attributes(graph, 'level')
    
    node_colors = []
    for node in graph.nodes():
        cat = categories.get(node, 0)
        level = levels.get(node, 0)
        color = CATEGORY_COLORS.get(cat, '#808080')
        if level == -1:
            color = '#D3D3D3'
        node_colors.append(color)
    
    if show_thoughts and 'thoughts_list' in item_data:
        thoughts = item_data['thoughts_list']
        if isinstance(thoughts, str):
            thoughts = json.loads(thoughts)
        thoughts = {str(k): v for k, v in thoughts.items()}
        
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, node)
            thought_text = thoughts.get(str(thought_num), "")
            thought_text = thought_text.replace('\\', '/')
            if len(thought_text) > 30:
                thought_text = thought_text[:30] + "..."
            labels[node] = f"T{thought_num}\n{thought_text}"
    else:
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, node)
            labels[node] = f"T{thought_num}"
    
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, 
        node_size=800, alpha=0.9, ax=ax_main
    )
    
    edge_categories = nx.get_edge_attributes(graph, 'category')
    
    if show_edge_labels:
        for cat_id in range(6):
            edges_in_cat = [
                (u, v) for (u, v), cat in edge_categories.items() 
                if cat == cat_id
            ]
            if edges_in_cat:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=edges_in_cat,
                    edge_color=CATEGORY_COLORS[cat_id],
                    width=2, alpha=0.6, arrows=True,
                    arrowsize=15, ax=ax_main
                )
    else:
        nx.draw_networkx_edges(
            graph, pos, edge_color='gray',
            width=1.5, alpha=0.5, arrows=True,
            arrowsize=15, ax=ax_main
        )
    
    nx.draw_networkx_labels(
        graph, pos, labels, font_size=9,
        font_weight='bold', ax=ax_main
    )
    
    tag = item_data.get('tag', 'Unknown')
    ax_main.set_title(f"Chain-of-Thought Graph: {tag}", fontsize=14, pad=20)
    ax_main.axis('off')
    
    legend_elements = [
        mpatches.Patch(color=CATEGORY_COLORS[i], label=CATEGORY_NAMES[i], alpha=0.7)
        for i in sorted(CATEGORY_COLORS.keys())
    ]
    ax_main.legend(
        handles=legend_elements, loc='upper left',
        fontsize=10, framealpha=0.9
    )
    
    if has_stats:
        ax_stats.axis('off')
        
        stats_text = "Relation Statistics\n" + "="*30 + "\n\n"
        stats_text += f"Total Edges: {relation_stats['total_edges']}\n\n"
        
        stats_text += "Category Distribution:\n"
        for cat_id in sorted(relation_stats['category_dist'].keys()):
            cat_name = CATEGORY_NAMES.get(cat_id, f"Unknown ({cat_id})")
            count = relation_stats['category_dist'][cat_id]
            pct = 100 * count / relation_stats['total_edges']
            stats_text += f"  {cat_name}: {count} ({pct:.1f}%)\n"
        
        stats_text += f"\nGraph Metrics:\n"
        stats_text += f"  Avg In-Degree: {sum(d for n, d in graph.in_degree()) / graph.number_of_nodes():.2f}\n"
        stats_text += f"  Avg Out-Degree: {sum(d for n, d in graph.out_degree()) / graph.number_of_nodes():.2f}\n"
        
        try:
            stats_text += f"  Longest Path: {nx.dag_longest_path_length(graph)}\n"
        except:
            stats_text += f"  Longest Path: N/A\n"
        
        ax_stats.text(0.1, 0.9, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        cats = sorted([c for c in relation_stats['category_dist'].keys()])
        if cats:
            counts = [relation_stats['category_dist'][c] for c in cats]
            colors = [CATEGORY_COLORS[c] for c in cats]
            names = [CATEGORY_NAMES[c] for c in cats]
            
            ax_bar = fig.add_axes([0.57, 0.15, 0.35, 0.25])
            bars = ax_bar.bar(range(len(cats)), counts, color=colors, alpha=0.7)
            ax_bar.set_xticks(range(len(cats)))
            ax_bar.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax_bar.set_ylabel('Count', fontsize=9)
            ax_bar.set_title('Edge Categories', fontsize=10)
            ax_bar.grid(axis='y', alpha=0.3)
    
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed for {item_data.get('tag', 'unknown')}: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def visualize_all_graphs(input_path: str, output_dir: str, max_graphs: int = None, 
                         show_thoughts: bool = False, show_edge_labels: bool = False,
                         summary_stats: bool = True, show_isolated: bool = False):
    """
    Visualize all graphs from final.json output.
    
    Args:
        input_path: Path to final.json
        output_dir: Directory to save visualizations
        max_graphs: Maximum number of graphs to visualize (None for all)
        show_thoughts: Whether to show thought text in labels
        show_edge_labels: Whether to color edges by category
        summary_stats: Whether to print summary statistics
        show_isolated: Whether to show isolated nodes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading graphs from: {input_path}")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Found {len(data)} items")
    
    has_new_format = any('reasoning_graph' in item for item in data)
    has_old_format = any('cot_tree' in item for item in data)
    
    if has_new_format:
        print("✓ Detected new graph format (reasoning_graph)")
    elif has_old_format:
        print("⚠ Warning: Old format detected (cot_tree). This visualizer expects reasoning_graph format.")
        print("  Visualization may not work correctly.")
    else:
        print("✗ Error: No graph data found in input file")
        return
    
    if max_graphs is not None:
        data = data[:max_graphs]
        print(f"Visualizing first {len(data)} graphs")
    
    all_stats = []
    
    for i, item in enumerate(data):
        if 'reasoning_graph' not in item:
            print(f"Warning: Item {i} has no reasoning_graph, skipping")
            continue
        
        tag = item.get('tag', f'item_{i}')
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{safe_tag}.png"
        
        print(f"\nVisualizing graph {i+1}/{len(data)}: {tag}")
        
        try:
            visualize_graph(
                item['reasoning_graph'], 
                item, 
                str(output_path),
                show_thoughts=show_thoughts,
                show_edge_labels=show_edge_labels,
                show_isolated=show_isolated
            )
            
            stats = get_relation_stats(item)
            all_stats.append(stats)
                
        except Exception as e:
            print(f"Error visualizing graph {tag}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Visualization complete! Images saved to: {output_dir}")
    print(f"{'='*80}\n")
    
    if summary_stats and all_stats and any(s['has_relations'] for s in all_stats):
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        total_edges = sum(s['total_edges'] for s in all_stats)
        
        print(f"\nOverall:")
        print(f"  Total edges: {total_edges:,}")
        
        all_categories = Counter()
        for stats in all_stats:
            all_categories.update(stats['category_dist'])
        
        print(f"\nCategory Distribution (all samples):")
        for cat_id in sorted(all_categories.keys()):
            cat_name = CATEGORY_NAMES.get(cat_id, f"Unknown ({cat_id})")
            count = all_categories[cat_id]
            pct = 100 * count / total_edges if total_edges > 0 else 0
            print(f"  {cat_name}: {count:,} ({pct:.1f}%)")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Chain-of-Thought graphs from reasoning pipeline - NON-RECURSIVE"
    )
    parser.add_argument(
        'input_path',
        nargs='?',
        default='./data/processed/reasoning_graph_test/final.json',
        help='Path to final.json file (default: ./data/processed/reasoning_graph_test/final.json)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='./outputs/visualizations/reasoning_graphs',
        help='Output directory for visualizations (default: ./outputs/visualizations/reasoning_graphs)'
    )
    parser.add_argument(
        '--max_graphs', '-n',
        type=int,
        default=None,
        help='Maximum number of graphs to visualize (default: all)'
    )
    parser.add_argument(
        '--show_thoughts', '-t',
        action='store_true',
        help='Show thought text in node labels (default: just show thought numbers)'
    )
    parser.add_argument(
        '--show_edge_labels', '-e',
        action='store_true',
        help='Color edges by category (default: gray edges)'
    )
    parser.add_argument(
        '--show_isolated', '-i',
        action='store_true',
        help='Show isolated nodes not connected to graph (default: False)'
    )
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Disable summary statistics printing'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        print("\nUsage examples:")
        print("  python visualize_reasoning_graphs.py")
        print("  python visualize_reasoning_graphs.py path/to/final.json")
        print("  python visualize_reasoning_graphs.py --max_graphs 5 --show_thoughts")
        print("  python visualize_reasoning_graphs.py --show_edge_labels")
        print("  python visualize_reasoning_graphs.py --show_isolated")
        return 1
    
    visualize_all_graphs(
        args.input_path,
        args.output_dir,
        max_graphs=args.max_graphs,
        show_thoughts=args.show_thoughts,
        show_edge_labels=args.show_edge_labels,
        summary_stats=not args.no_summary,
        show_isolated=args.show_isolated
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())