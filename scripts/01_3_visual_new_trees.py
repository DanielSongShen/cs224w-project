#!/usr/bin/env python3
"""
Visualize Chain-of-Thought graphs from LCoT2Tree output (Parent Selection Version).

Updated for parent selection approach with hierarchical tree layout.
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


def tree_to_graph(tree_dict: Dict[str, Any], parent_id: str = None, graph: nx.DiGraph = None) -> nx.DiGraph:
    """
    Convert tree dictionary to NetworkX directed graph ITERATIVELY.
    
    Args:
        tree_dict: Tree node dictionary with value, level, cate, thought_list, children
        parent_id: ID of parent node (for edge creation)
        graph: Existing graph to add to (creates new if None)
    
    Returns:
        NetworkX directed graph representing the tree/graph
    """
    if graph is None:
        graph = nx.DiGraph()
    
    # Use a stack to avoid recursion depth issues
    # Stack items: (current_node_dict, parent_id_for_this_node)
    stack = [(tree_dict, parent_id)]
    
    while stack:
        curr, pid = stack.pop()
        node_id = curr['value']
        
        thought_list = curr.get('thought_list', [])
        thought_num = thought_list[0] if thought_list else 0
        
        # Add the node with subset attribute for multipartite_layout
        graph.add_node(
            node_id,
            level=curr['level'],
            category=curr.get('cate', 0),
            thought_num=thought_num,
            thought_list=thought_list,
            subset=curr['level']  # 'subset' attribute is needed for multipartite_layout
        )
        
        # Add edge from parent
        if pid is not None:
            edge_category = curr.get('cate', 0)
            graph.add_edge(pid, node_id, category=edge_category)
        
        # Add children to stack
        # (Reverse order so they pop in original order)
        for child in reversed(curr.get('children', [])):
            stack.append((child, node_id))
            
    return graph


def add_all_nodes_and_edges(graph: nx.DiGraph, item_data: Dict[str, Any], show_isolated: bool = True) -> nx.DiGraph:
    """
    Add all nodes from thoughts_list and all edges from thought_relations.
    
    Args:
        graph: Existing graph from tree structure
        item_data: Full item data with thought_relations
        show_isolated: Whether to include isolated nodes (always adds edges)
    
    Returns:
        Graph with all nodes and edges added
    """
    # Build thought_num to node_id mapping
    thought_to_node = {}
    for node_id, data in graph.nodes(data=True):
        thought_num = data.get('thought_num', 0)
        thought_to_node[thought_num] = node_id
    
    # Add isolated nodes only if show_isolated=True
    if show_isolated and 'thoughts_list' in item_data:
        thoughts_list = item_data['thoughts_list']
        if isinstance(thoughts_list, str):
            import json
            thoughts_list = json.loads(thoughts_list)
        
        for thought_id in thoughts_list.keys():
            thought_num = int(thought_id) if not isinstance(thought_id, int) else thought_id
            if thought_num not in thought_to_node:
                node_id = f"isolated_{thought_num}"
                graph.add_node(
                    node_id,
                    level=-1,
                    category=0,
                    thought_num=thought_num,
                    thought_list=[thought_num],
                    subset=-1
                )
                thought_to_node[thought_num] = node_id
    
    # ALWAYS add edges from thought_relations (to show multiple parents)
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


def get_multipartite_layout(graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
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
    Extract statistics from thought_relations field.
    
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


def visualize_tree(tree_dict: Dict[str, Any], item_data: Dict[str, Any], 
                   output_path: str, show_thoughts: bool = False,
                   show_edge_labels: bool = False, show_isolated: bool = False):
    """
    Visualize a single CoT tree/graph with hierarchical layout.
    
    Args:
        tree_dict: Tree structure from cot_tree field
        item_data: Full item data including thoughts_list, tag, etc.
        output_path: Path to save the visualization
        show_thoughts: Whether to show thought text in node labels
        show_edge_labels: Whether to show edge categories
        show_isolated: Whether to show isolated nodes
    """
    # 1. Build Graph Iteratively (NO RECURSION)
    graph = tree_to_graph(tree_dict)
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
            import json
            thoughts = json.loads(thoughts)
        thoughts = {str(k): v for k, v in thoughts.items()}
        
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, 0)
            thought_text = thoughts.get(str(thought_num), "")
            thought_text = thought_text.replace('\\', '/')
            if len(thought_text) > 30:
                thought_text = thought_text[:30] + "..."
            labels[node] = f"T{thought_num}\n{thought_text}"
    else:
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, 0)
            labels[node] = f"T{thought_num}" if levels.get(node, 0) == -1 else str(node)
    
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


def visualize_all_trees(input_path: str, output_dir: str, max_trees: int = None, 
                        show_thoughts: bool = False, show_edge_labels: bool = False,
                        summary_stats: bool = True, show_isolated: bool = False):
    """
    Visualize all trees from final.json output.
    
    Args:
        input_path: Path to final.json
        output_dir: Directory to save visualizations
        max_trees: Maximum number of trees to visualize (None for all)
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
    
    has_relations = any('thought_relations' in item for item in data)
    if has_relations:
        print("✓ Detected parent selection format (with thought_relations)")
    else:
        print("✓ Detected original pipeline format")
    
    if max_trees is not None:
        data = data[:max_trees]
        print(f"Visualizing first {len(data)} graphs")
    
    all_stats = []
    
    for i, item in enumerate(data):
        if 'cot_tree' not in item:
            print(f"Warning: Item {i} has no cot_tree, skipping")
            continue
        
        tag = item.get('tag', f'item_{i}')
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{safe_tag}.png"
        
        print(f"\nVisualizing graph {i+1}/{len(data)}: {tag}")
        
        try:
            visualize_tree(
                item['cot_tree'], 
                item, 
                str(output_path),
                show_thoughts=show_thoughts,
                show_edge_labels=show_edge_labels,
                show_isolated=show_isolated
            )
            
            if has_relations:
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
    
    if summary_stats and all_stats and has_relations:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (Parent Selection)")
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
            pct = 100 * count / total_edges
            print(f"  {cat_name}: {count:,} ({pct:.1f}%)")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Chain-of-Thought graphs from LCoT2Tree output (Parent Selection Version) - NON-RECURSIVE"
    )
    parser.add_argument(
        '--input_path',
        nargs='?',
        default='./data/processed/lcot2tree_test/final.json',
        help='Path to final.json file (default: ./data/processed/lcot2tree_test/final.json)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='./outputs/visualizations/cot_graphs',
        help='Output directory for visualizations (default: ./outputs/visualizations/cot_graphs)'
    )
    parser.add_argument(
        '--max_trees', '-n',
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
        help='Show isolated nodes not connected to tree (default: False, but multiple parents always shown)'
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
        print("  python visualize_cot_graphs.py")
        print("  python visualize_cot_graphs.py path/to/final.json")
        print("  python visualize_cot_graphs.py --max_trees 5 --show_thoughts")
        print("  python visualize_cot_graphs.py --show_edge_labels")
        print("  python visualize_cot_graphs.py --show_isolated  # Show isolated nodes")
        return 1
    
    visualize_all_trees(
        args.input_path,
        args.output_dir,
        max_trees=args.max_trees,
        show_thoughts=args.show_thoughts,
        show_edge_labels=args.show_edge_labels,
        summary_stats=not args.no_summary,
        show_isolated=args.show_isolated
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())