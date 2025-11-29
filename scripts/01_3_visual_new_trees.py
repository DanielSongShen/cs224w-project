#!/usr/bin/env python3
"""
Visualize Chain-of-Thought graphs from modified LCoT2Tree output.

This script handles both old (tree-based) and new (relation-based) formats.
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
    5: '#E91E63',  # Unrelated - Pink
}

CATEGORY_NAMES = {
    0: 'Root',
    1: 'Continuous Logic',
    2: 'Exploration',
    3: 'Backtracking',
    4: 'Validation',
    5: 'Unrelated',
}


def tree_to_graph(tree_dict: Dict[str, Any], parent_id: str = None, graph: nx.DiGraph = None) -> nx.DiGraph:
    """
    Convert tree dictionary to NetworkX directed graph.
    
    Args:
        tree_dict: Tree node dictionary with value, level, cate, thought_list, children
        parent_id: ID of parent node (for edge creation)
        graph: Existing graph to add to (creates new if None)
    
    Returns:
        NetworkX directed graph representing the tree/graph
    """
    if graph is None:
        graph = nx.DiGraph()
    
    # Create unique node ID
    node_id = tree_dict['value']
    
    # Extract thought number(s) from thought_list for labeling
    thought_list = tree_dict.get('thought_list', [])
    if thought_list:
        # Use first thought number as the primary label
        thought_num = thought_list[0]
    else:
        thought_num = 0
    
    # Add node with attributes
    graph.add_node(
        node_id,
        level=tree_dict['level'],
        category=tree_dict.get('cate', 0),
        thought_num=thought_num,
        thought_list=thought_list
    )
    
    # Add edge from parent if exists (store category as edge attribute)
    if parent_id is not None:
        edge_category = tree_dict.get('cate', 0)
        graph.add_edge(parent_id, node_id, category=edge_category)
    
    # Recursively add children
    for child in tree_dict.get('children', []):
        tree_to_graph(child, node_id, graph)
    
    return graph


def add_all_nodes_and_edges(graph: nx.DiGraph, item_data: Dict[str, Any], show_unrelated: bool = True) -> nx.DiGraph:
    """
    Add all nodes from thoughts_list and all edges from thought_relations.
    
    Args:
        graph: Existing graph from tree structure
        item_data: Full item data with thought_relations
        show_unrelated: Whether to include unrelated edges and isolated nodes
    
    Returns:
        Graph with all nodes and edges added
    """
    if not show_unrelated:
        return graph
    
    # First, ensure all thoughts are represented as nodes
    if 'thoughts_list' in item_data:
        thoughts_list = item_data['thoughts_list']
        if isinstance(thoughts_list, str):
            import json
            thoughts_list = json.loads(thoughts_list)
        
        # Get existing nodes by thought_num
        thought_to_node = {}
        for node_id, data in graph.nodes(data=True):
            thought_num = data.get('thought_num', 0)
            thought_to_node[thought_num] = node_id
        
        # Add missing nodes
        for thought_id in thoughts_list.keys():
            thought_num = int(thought_id) if not isinstance(thought_id, int) else thought_id
            if thought_num not in thought_to_node:
                # Create a new isolated node
                node_id = f"isolated_{thought_num}"
                graph.add_node(
                    node_id,
                    level=-1,  # Mark as isolated
                    category=0,  # Default category
                    thought_num=thought_num,
                    thought_list=[thought_num]
                )
                thought_to_node[thought_num] = node_id
    else:
        # Build mapping from existing nodes
        thought_to_node = {}
        for node_id, data in graph.nodes(data=True):
            thought_num = data.get('thought_num', 0)
            thought_to_node[thought_num] = node_id
    
    # Add all edges from thought_relations
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
                # Only add if edge doesn't already exist
                if not graph.has_edge(src_node, tgt_node):
                    graph.add_edge(src_node, tgt_node, category=category)
    
    return graph


def get_hierarchical_pos(graph: nx.DiGraph, root: str = "0", width: float = 1.0, 
                         vert_gap: float = 0.2, vert_loc: float = 0, 
                         xcenter: float = 0.5, pos: Dict = None, 
                         parent: str = None) -> Dict[str, Tuple[float, float]]:
    """
    Create hierarchical layout for tree visualization.
    
    This positions nodes in a tree layout where levels are horizontal
    and children are spread vertically under parents.
    """
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)
    
    children = list(graph.neighbors(root))
    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos = get_hierarchical_pos(
                graph, child, width=dx, vert_gap=vert_gap,
                vert_loc=vert_loc-vert_gap, xcenter=nextx, 
                pos=pos, parent=root
            )
    return pos


def get_relation_stats(item_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract statistics from thought_relations field (modified pipeline).
    
    Args:
        item_data: Full item data
        
    Returns:
        Dictionary with relation statistics
    """
    stats = {
        'has_relations': False,
        'total_comparisons': 0,
        'related_count': 0,
        'unrelated_count': 0,
        'category_dist': {},
        'sparsity': 0.0
    }
    
    if 'thought_relations' not in item_data:
        return stats

    relations = item_data['thought_relations']
    stats['has_relations'] = True

    # Flatten nested dict structure: {thought_prev: {thought_n: category}}
    all_categories = []
    for targets in relations.values():
        for category in targets.values():
            all_categories.append(category)

    stats['total_comparisons'] = len(all_categories)

    category_counts = Counter(all_categories)
    stats['category_dist'] = dict(category_counts)

    stats['unrelated_count'] = category_counts.get(5, 0)
    stats['related_count'] = stats['total_comparisons'] - stats['unrelated_count']
    
    if stats['total_comparisons'] > 0:
        stats['sparsity'] = stats['related_count'] / stats['total_comparisons']
    
    return stats


def visualize_tree(tree_dict: Dict[str, Any], item_data: Dict[str, Any], 
                   output_path: str, show_thoughts: bool = False,
                   show_edge_labels: bool = False, show_unrelated: bool = False):
    """
    Visualize a single CoT tree/graph.
    
    Args:
        tree_dict: Tree structure from cot_tree field
        item_data: Full item data including thoughts_list, tag, etc.
        output_path: Path to save the visualization
        show_thoughts: Whether to show thought text in node labels
        show_edge_labels: Whether to show edge categories
        show_unrelated: Whether to show unrelated edges and isolated nodes
    """
    # Convert tree to graph
    graph = tree_to_graph(tree_dict)
    
    # Add all nodes and edges from thought_relations (if enabled)
    graph = add_all_nodes_and_edges(graph, item_data, show_unrelated)
    
    # Use spring layout instead of hierarchical (handles cycles)
    pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
    
    # Create figure with subplots if we have relation stats
    relation_stats = get_relation_stats(item_data)
    has_stats = relation_stats['has_relations']
    
    if has_stats:
        fig = plt.figure(figsize=(18, 10))
        ax_main = plt.subplot(1, 2, 1)
        ax_stats = plt.subplot(1, 2, 2)
    else:
        fig, ax_main = plt.subplots(figsize=(16, 10))
    
    # Get node attributes
    categories = nx.get_node_attributes(graph, 'category')
    thought_nums = nx.get_node_attributes(graph, 'thought_num')
    levels = nx.get_node_attributes(graph, 'level')
    
    # Prepare node colors based on category (use lighter color for isolated nodes)
    node_colors = []
    for node in graph.nodes():
        cat = categories.get(node, 0)
        level = levels.get(node, 0)
        color = CATEGORY_COLORS.get(cat, '#808080')
        # Make isolated nodes lighter
        if level == -1:
            color = '#D3D3D3'  # Light gray for isolated nodes
        node_colors.append(color)
    
    # Prepare node labels
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
    
    # Draw the graph
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, 
        node_size=800, alpha=0.9, ax=ax_main
    )
    
    nx.draw_networkx_labels(
        graph, pos, labels, font_size=9, 
        font_weight='bold', ax=ax_main
    )
    
    # Get edge attributes
    edge_attrs = nx.get_edge_attributes(graph, 'category')
    
    # Separate edges by category
    related_edges = [(u, v) for u, v in graph.edges() if edge_attrs.get((u, v), 0) != 5]
    unrelated_edges = [(u, v) for u, v in graph.edges() if edge_attrs.get((u, v), 0) == 5]
    
    # Draw related edges
    if show_edge_labels and edge_attrs:
        for edge in related_edges:
            edge_cat = edge_attrs.get(edge, 0)
            edge_color = CATEGORY_COLORS.get(edge_cat, '#808080')
            nx.draw_networkx_edges(
                graph, pos, edgelist=[edge],
                edge_color=edge_color,
                arrows=True, arrowsize=20, 
                width=2, alpha=0.6, ax=ax_main
            )
    else:
        nx.draw_networkx_edges(
            graph, pos, edgelist=related_edges,
            edge_color='gray', 
            arrows=True, arrowsize=20, 
            width=2, alpha=0.6, ax=ax_main
        )
    
    # Draw unrelated edges only if show_unrelated is True
    if show_unrelated and unrelated_edges:
        nx.draw_networkx_edges(
            graph, pos, edgelist=unrelated_edges,
            edge_color='#E91E63',
            arrows=True, arrowsize=15, 
            width=1, alpha=0.2, style='dashed', ax=ax_main
        )
    
    # Add title with metadata
    title = f"CoT Graph: {item_data.get('tag', 'Unknown')}\n"
    title += f"Score: {item_data.get('score', 'N/A')} | "
    title += f"Nodes: {graph.number_of_nodes()} | "
    title += f"Edges: {graph.number_of_edges()}"
    
    if has_stats:
        title += f" | Sparsity: {relation_stats['sparsity']:.1%}"
    
    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend for node categories
    legend_elements = [
        mpatches.Patch(color=CATEGORY_COLORS[cat], label=CATEGORY_NAMES[cat])
        for cat in sorted(CATEGORY_COLORS.keys())
        if cat in categories.values() or (cat == 5 and show_unrelated)
    ]
    ax_main.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax_main.axis('off')
    
    # Add statistics panel if available
    if has_stats:
        ax_stats.axis('off')
        
        # Create statistics text
        stats_text = "Relation Statistics\n" + "="*30 + "\n\n"
        stats_text += f"Total Comparisons: {relation_stats['total_comparisons']}\n"
        stats_text += f"Related Edges: {relation_stats['related_count']}\n"
        stats_text += f"Unrelated: {relation_stats['unrelated_count']}\n"
        stats_text += f"Graph Sparsity: {relation_stats['sparsity']:.1%}\n\n"
        
        stats_text += "Category Distribution:\n"
        for cat_id in sorted(relation_stats['category_dist'].keys()):
            cat_name = CATEGORY_NAMES.get(cat_id, f"Unknown ({cat_id})")
            count = relation_stats['category_dist'][cat_id]
            pct = 100 * count / relation_stats['total_comparisons']
            stats_text += f"  {cat_name}: {count} ({pct:.1f}%)\n"
        
        # Add graph metrics
        stats_text += f"\nGraph Metrics:\n"
        stats_text += f"  Avg In-Degree: {sum(d for n, d in graph.in_degree()) / graph.number_of_nodes():.2f}\n"
        stats_text += f"  Avg Out-Degree: {sum(d for n, d in graph.out_degree()) / graph.number_of_nodes():.2f}\n"
        
        try:
            stats_text += f"  Longest Path: {nx.dag_longest_path_length(graph)}\n"
        except:
            stats_text += f"  Longest Path: N/A\n"
        
        # Display statistics
        ax_stats.text(0.1, 0.9, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add bar chart of category distribution
        cats = sorted([c for c in relation_stats['category_dist'].keys()])
        if cats:
            counts = [relation_stats['category_dist'][c] for c in cats]
            colors = [CATEGORY_COLORS[c] for c in cats]
            names = [CATEGORY_NAMES[c] for c in cats]
            
            # Create inset axes for bar chart
            ax_bar = fig.add_axes([0.57, 0.15, 0.35, 0.25])
            bars = ax_bar.bar(range(len(cats)), counts, color=colors, alpha=0.7)
            ax_bar.set_xticks(range(len(cats)))
            ax_bar.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
            ax_bar.set_ylabel('Count', fontsize=9)
            ax_bar.set_title('Edge Categories', fontsize=10)
            ax_bar.grid(axis='y', alpha=0.3)
    
    # Use tight_layout with error handling
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed for {item_data.get('tag', 'unknown')}: {e}")
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def visualize_all_trees(input_path: str, output_dir: str, max_trees: int = None, 
                        show_thoughts: bool = False, show_edge_labels: bool = False,
                        summary_stats: bool = True, show_unrelated: bool = False):
    """
    Visualize all trees from final.json output.
    
    Args:
        input_path: Path to final.json
        output_dir: Directory to save visualizations
        max_trees: Maximum number of trees to visualize (None for all)
        show_thoughts: Whether to show thought text in labels
        show_edge_labels: Whether to color edges by category
        summary_stats: Whether to print summary statistics
        show_unrelated: Whether to show unrelated edges and isolated nodes
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (JSONL format)
    print(f"Loading graphs from: {input_path}")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Found {len(data)} items")
    
    # Check if data uses new format
    has_relations = any('thought_relations' in item for item in data)
    if has_relations:
        print("✓ Detected modified pipeline format (with thought_relations)")
    else:
        print("✓ Detected original pipeline format")
    
    # Limit if requested
    if max_trees is not None:
        data = data[:max_trees]
        print(f"Visualizing first {len(data)} graphs")
    
    # Collect statistics
    all_stats = []
    
    # Visualize each tree
    for i, item in enumerate(data):
        if 'cot_tree' not in item:
            print(f"Warning: Item {i} has no cot_tree, skipping")
            continue
        
        # Generate output filename
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
                show_unrelated=show_unrelated
            )
            
            # Collect stats
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
    
    # Print summary statistics
    if summary_stats and all_stats and has_relations:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS (Modified Pipeline)")
        print("="*80)
        
        total_comparisons = sum(s['total_comparisons'] for s in all_stats)
        total_related = sum(s['related_count'] for s in all_stats)
        total_unrelated = sum(s['unrelated_count'] for s in all_stats)
        
        print(f"\nOverall:")
        print(f"  Total pairwise comparisons: {total_comparisons:,}")
        print(f"  Related edges: {total_related:,} ({100*total_related/total_comparisons:.1f}%)")
        print(f"  Unrelated: {total_unrelated:,} ({100*total_unrelated/total_comparisons:.1f}%)")
        
        # Average sparsity
        avg_sparsity = sum(s['sparsity'] for s in all_stats) / len(all_stats)
        print(f"  Average graph sparsity: {avg_sparsity:.1%}")
        
        # Aggregate category distribution
        all_categories = Counter()
        for stats in all_stats:
            all_categories.update(stats['category_dist'])
        
        print(f"\nCategory Distribution (all samples):")
        for cat_id in sorted(all_categories.keys()):
            cat_name = CATEGORY_NAMES.get(cat_id, f"Unknown ({cat_id})")
            count = all_categories[cat_id]
            pct = 100 * count / total_comparisons
            print(f"  {cat_name}: {count:,} ({pct:.1f}%)")
        
        print("\n" + "="*80 + "\n")


def main():
    """Main entry point with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Chain-of-Thought graphs from LCoT2Tree output (supports both old and new formats)"
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
        '--show_unrelated', '-u',
        action='store_true',
        help='Show unrelated edges and isolated nodes (default: False)'
    )
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Disable summary statistics printing'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        print("\nUsage examples:")
        print("  python visualize_with_unrelated.py")
        print("  python visualize_with_unrelated.py path/to/final.json")
        print("  python visualize_with_unrelated.py --max_trees 5 --show_thoughts")
        print("  python visualize_with_unrelated.py --show_edge_labels")
        print("  python visualize_with_unrelated.py --show_unrelated  # Show all edges including unrelated")
        return 1
    
    # Visualize trees
    visualize_all_trees(
        args.input_path,
        args.output_dir,
        max_trees=args.max_trees,
        show_thoughts=args.show_thoughts,
        show_edge_labels=args.show_edge_labels,
        summary_stats=not args.no_summary,
        show_unrelated=args.show_unrelated
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())