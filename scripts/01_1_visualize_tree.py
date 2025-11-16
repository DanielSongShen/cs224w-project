#!/usr/bin/env python3
"""
Visualize Chain-of-Thought trees from LCoT2Tree output.

This script loads the final.json output and creates tree visualizations
using networkx and matplotlib.
"""

import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import Dict, Any, List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Category colors for different thought types
CATEGORY_COLORS = {
    0: '#808080',  # Root - Gray
    1: '#4CAF50',  # Continuous Logic - Green
    2: '#FF9800',  # Exploration - Orange
    3: '#2196F3',  # Backtracking - Blue
    4: '#FFC107',  # Validation - Amber
}

CATEGORY_NAMES = {
    0: 'Root',
    1: 'Continuous Logic',
    2: 'Exploration',
    3: 'Backtracking',
    4: 'Validation',
}


def tree_to_graph(tree_dict: Dict[str, Any], parent_id: str = None, graph: nx.DiGraph = None) -> nx.DiGraph:
    """
    Convert tree dictionary to NetworkX directed graph.
    
    Args:
        tree_dict: Tree node dictionary with value, level, cate, thought_list, children
        parent_id: ID of parent node (for edge creation)
        graph: Existing graph to add to (creates new if None)
    
    Returns:
        NetworkX directed graph representing the tree
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
    
    # Add edge from parent if exists
    if parent_id is not None:
        graph.add_edge(parent_id, node_id)
    
    # Recursively add children
    for child in tree_dict.get('children', []):
        tree_to_graph(child, node_id, graph)
    
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


def visualize_tree(tree_dict: Dict[str, Any], item_data: Dict[str, Any], 
                   output_path: str, show_thoughts: bool = False):
    """
    Visualize a single CoT tree.
    
    Args:
        tree_dict: Tree structure from cot_tree field
        item_data: Full item data including thoughts_list, tag, etc.
        output_path: Path to save the visualization
        show_thoughts: Whether to show thought text in node labels
    """
    # Convert tree to graph
    graph = tree_to_graph(tree_dict)
    
    # Get hierarchical layout
    pos = get_hierarchical_pos(graph)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Get node attributes
    categories = nx.get_node_attributes(graph, 'category')
    thought_nums = nx.get_node_attributes(graph, 'thought_num')
    
    # Prepare node colors based on category
    node_colors = [CATEGORY_COLORS.get(categories[node], '#808080') for node in graph.nodes()]
    
    # Prepare node labels - show node values (unique IDs)
    if show_thoughts and 'thoughts_list' in item_data:
        thoughts = item_data['thoughts_list']
        # Convert keys to strings if needed
        thoughts = {str(k): v for k, v in thoughts.items()}
        
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, 0)
            thought_text = thoughts.get(str(thought_num), "")
            # Sanitize LaTeX characters that matplotlib might try to parse
            # Replace backslashes to prevent LaTeX parsing errors
            thought_text = thought_text.replace('\\', '/')
            # Truncate long thoughts
            if len(thought_text) > 30:
                thought_text = thought_text[:30] + "..."
            # Show node value (unique ID) with thought text
            labels[node] = f"{node}\n{thought_text}"
    else:
        # Show node values (unique IDs like "0", "1-0", "2-1", etc.)
        labels = {node: str(node) for node in graph.nodes()}
    
    # Draw the graph
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, 
        node_size=800, alpha=0.9, ax=ax
    )
    
    nx.draw_networkx_labels(
        graph, pos, labels, font_size=9, 
        font_weight='bold', ax=ax
    )
    
    nx.draw_networkx_edges(
        graph, pos, edge_color='gray', 
        arrows=True, arrowsize=20, 
        width=2, alpha=0.6, ax=ax
    )
    
    # Add title with metadata
    title = f"CoT Tree: {item_data.get('tag', 'Unknown')}\n"
    title += f"Score: {item_data.get('score', 'N/A')} | "
    title += f"Nodes: {graph.number_of_nodes()} | "
    title += f"Max Step: {max(nx.get_node_attributes(graph, 'level').values())} | "
    # Calculate actual tree depth (longest path from root)
    try:
        tree_depth = nx.dag_longest_path_length(graph)
        title += f"Tree Depth: {tree_depth}"
    except:
        title += f"Tree Depth: N/A"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=CATEGORY_COLORS[cat], label=CATEGORY_NAMES[cat])
        for cat in sorted(CATEGORY_COLORS.keys())
        if cat in categories.values()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Add prompt as text box if available
    if 'full_prompt' in item_data:
        prompt_text = item_data['full_prompt']
        # Sanitize LaTeX characters
        prompt_text = prompt_text.replace('\\', '/')
        if len(prompt_text) > 150:
            prompt_text = prompt_text[:150] + "..."
        ax.text(
            0.02, 0.02, f"Prompt: {prompt_text}", 
            transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            wrap=True
        )
    
    ax.axis('off')
    
    # Use tight_layout with error handling
    try:
        plt.tight_layout()
    except Exception as e:
        print(f"Warning: tight_layout failed for {item_data.get('tag', 'unknown')}: {e}")
        # Continue without tight_layout
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def visualize_all_trees(input_path: str, output_dir: str, max_trees: int = None, 
                        show_thoughts: bool = False):
    """
    Visualize all trees from final.json output.
    
    Args:
        input_path: Path to final.json
        output_dir: Directory to save visualizations
        max_trees: Maximum number of trees to visualize (None for all)
        show_thoughts: Whether to show thought text in labels
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (JSONL format: one JSON object per line)
    print(f"Loading trees from: {input_path}")
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Found {len(data)} items")
    
    # Limit if requested
    if max_trees is not None:
        data = data[:max_trees]
        print(f"Visualizing first {len(data)} trees")
    
    # Visualize each tree
    for i, item in enumerate(data):
        if 'cot_tree' not in item:
            print(f"Warning: Item {i} has no cot_tree, skipping")
            continue
        
        # Generate output filename
        tag = item.get('tag', f'item_{i}')
        safe_tag = tag.replace('/', '_').replace('\\', '_')
        output_path = output_dir / f"{safe_tag}.png"
        
        print(f"\nVisualizing tree {i+1}/{len(data)}: {tag}")
        
        try:
            visualize_tree(
                item['cot_tree'], 
                item, 
                str(output_path),
                show_thoughts=show_thoughts
            )
        except Exception as e:
            print(f"Error visualizing tree {tag}: {str(e)[:100]}")
            # Skip traceback for cleaner output, just continue with next tree
            continue
    
    print(f"\n{'='*80}")
    print(f"Visualization complete! Images saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    """Main entry point with command-line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize Chain-of-Thought trees from LCoT2Tree output"
    )
    parser.add_argument(
        '--input_path',
        nargs='?',
        default='./data/processed/lcot2tree_test/final.json',
        help='Path to final.json file (default: ./data/processed/lcot2tree_test/final.json)'
    )
    parser.add_argument(
        '--output_dir', '-o',
        default='./outputs/visualizations/cot_trees',
        help='Output directory for visualizations (default: ./outputs/visualizations/cot_trees)'
    )
    parser.add_argument(
        '--max_trees', '-n',
        type=int,
        default=None,
        help='Maximum number of trees to visualize (default: all)'
    )
    parser.add_argument(
        '--show_thoughts', '-t',
        action='store_true',
        help='Show thought text in node labels (default: just show thought numbers)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        print("\nUsage examples:")
        print("  python scripts/01_1_visualize_tree.py")
        print("  python scripts/01_1_visualize_tree.py path/to/final.json")
        print("  python scripts/01_1_visualize_tree.py --max_trees 5 --show_thoughts")
        return 1
    
    # Visualize trees
    visualize_all_trees(
        args.input_path,
        args.output_dir,
        max_trees=args.max_trees,
        show_thoughts=args.show_thoughts
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

