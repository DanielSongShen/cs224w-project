"""Visualize the graph structure after tree_to_graph conversion"""

import json
import sys
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from typing import List, Dict
import numpy as np

# Add project paths
sys.path.append(str(Path(__file__).parent.parent / "LCoT2Tree" / "src" / "gnn"))

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not installed")
    AutoTokenizer = None


def tree_to_graph(root: dict, tokens_list: List[int], edge_type: str = "11_1"):
    """
    Convert a CoT tree structure to a PyTorch Geometric graph.
    (Same function as in 02_train_model.py)
    """
    nodes = []
    nodes_dict = {}
    edges = []
    edge_features = []
    level_cnt = {0: 1}
    
    def dict_to_tree(tree_dict, index):
        """Recursively traverse tree and build graph"""
        for num, child_dict in enumerate(tree_dict["children"]):
            dict_to_tree(child_dict, num)
        
        value = tree_dict["value"]
        level = tree_dict["level"]
        
        if level in level_cnt:
            level_cnt[level] += 1
        else:
            level_cnt[level] = 1
        
        # Parse node value
        node_v = value.split(",")[-1].split("-")[0] if str(value) != "0" else "0"
        node_s = value.split(",")[-1].split("-")[1] if str(value) != "0" else "0"
        node_c = len(value.split(","))
        distance = (len(tokens_list) - int(node_v)) / len(tokens_list) if len(tokens_list) > 0 else 0
        
        # Calculate token lengths
        current_token_len = sum([tokens_list[int(v.split("-")[0])] for v in value.split(",")])
        prev_token_len = sum([t for t in tokens_list[:int(node_v)]]) / 100 if int(node_v) < len(tokens_list) else 0
        children_count = len(tree_dict["children"])
        level_node_count = level_cnt[level]
        depth = level
        
        # Node features
        nodes.append([
            int(node_v),           # 0: thought ID
            int(node_s),           # 1: segment ID
            level,                 # 2: depth
            current_token_len,     # 3: current tokens
            node_c,                # 4: component count
            prev_token_len,        # 5: previous tokens
            index,                 # 6: child index
            children_count,        # 7: children count
            level_node_count,      # 8: level node count
            distance               # 9: distance from end
        ])
        nodes_dict[value] = len(nodes) - 1
        
        # Add edges to children
        for child_dict in tree_dict["children"]:
            child_value = child_dict["value"]
            
            edge_feat = [
                child_dict["cate"] if edge_type[2] == "1" or edge_type[2] == "3" else 1
            ]
            
            # Parent to child edge
            edges.append([nodes_dict[value], nodes_dict[child_value]])
            edge_features.append(edge_feat)
            
            # Add bidirectional edge if specified
            if edge_type[1] == "1":
                edges.append([nodes_dict[child_value], nodes_dict[value]])
                edge_features.append([-child_dict["cate"] if edge_type[2] == "2" or edge_type[2] == "3" else -1])
    
    dict_to_tree(root, 0)
    
    # Return as numpy arrays for easier visualization
    return np.array(nodes), edges, edge_features, nodes_dict


def visualize_graph(
    nodes: np.ndarray,
    edges: List[List[int]],
    edge_features: List[List[float]],
    item: Dict,
    output_path: str = None
):
    """Visualize the graph structure using networkx and matplotlib"""
    
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for i, node_feats in enumerate(nodes):
        G.add_node(i, 
                   thought_id=int(node_feats[0]),
                   depth=int(node_feats[2]),
                   tokens=node_feats[3],
                   children_count=int(node_feats[7]))
    
    # Add edges
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # Subplot 1: Graph structure with hierarchical layout
    ax1 = plt.subplot(2, 3, (1, 4))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Color nodes by depth
    depths = [G.nodes[n]['depth'] for n in G.nodes()]
    max_depth = max(depths) if depths else 1
    node_colors = [plt.cm.viridis(d / max_depth) for d in depths]
    
    # Size nodes by token count
    node_sizes = [100 + G.nodes[n]['tokens'] * 2 for n in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                          alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                          arrowsize=20, alpha=0.5, ax=ax1)
    
    # Add labels
    labels = {n: f"{n}\nT{G.nodes[n]['thought_id']}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax1)
    
    ax1.set_title(f"Graph Structure - {item.get('tag', 'unknown')}\n"
                  f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}, "
                  f"Score: {item.get('score', '?')}", 
                  fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Subplot 2: Node feature heatmap
    ax2 = plt.subplot(2, 3, 2)
    feature_names = ['Thought ID', 'Segment ID', 'Depth', 'Curr Tokens',
                    'Components', 'Prev Tokens', 'Child Idx', 'Children',
                    'Level Nodes', 'Distance']
    
    # Normalize features for visualization
    nodes_norm = nodes.copy()
    for i in range(nodes.shape[1]):
        col = nodes[:, i]
        if col.max() > col.min():
            nodes_norm[:, i] = (col - col.min()) / (col.max() - col.min())
    
    im = ax2.imshow(nodes_norm.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels(feature_names, fontsize=9)
    ax2.set_xlabel('Node Index', fontsize=10)
    ax2.set_title('Node Features (Normalized)', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # Subplot 3: Depth distribution
    ax3 = plt.subplot(2, 3, 3)
    depth_counts = {}
    for n in G.nodes():
        d = G.nodes[n]['depth']
        depth_counts[d] = depth_counts.get(d, 0) + 1
    
    depths_list = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in depths_list]
    
    bars = ax3.bar(depths_list, counts, color=plt.cm.viridis(np.array(depths_list) / max(depths_list)))
    ax3.set_xlabel('Depth Level', fontsize=10)
    ax3.set_ylabel('Number of Nodes', fontsize=10)
    ax3.set_title('Node Distribution by Depth', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Subplot 4: Degree distribution
    ax4 = plt.subplot(2, 3, 5)
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    out_degrees = [G.out_degree(n) for n in G.nodes()]
    
    x = np.arange(len(G.nodes()))
    width = 0.35
    
    ax4.bar(x - width/2, in_degrees, width, label='In-degree', alpha=0.8, color='skyblue')
    ax4.bar(x + width/2, out_degrees, width, label='Out-degree', alpha=0.8, color='salmon')
    ax4.set_xlabel('Node Index', fontsize=10)
    ax4.set_ylabel('Degree', fontsize=10)
    ax4.set_title('Node Degree Distribution', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # Subplot 5: Feature statistics
    ax5 = plt.subplot(2, 3, 6)
    feature_stats = []
    for i, name in enumerate(feature_names):
        mean_val = nodes[:, i].mean()
        std_val = nodes[:, i].std()
        feature_stats.append((name, mean_val, std_val))
    
    # Display as table
    ax5.axis('tight')
    ax5.axis('off')
    
    table_data = []
    for name, mean, std in feature_stats[:7]:  # Show first 7 features
        table_data.append([name, f'{mean:.2f}', f'{std:.2f}'])
    
    table = ax5.table(cellText=table_data,
                     colLabels=['Feature', 'Mean', 'Std'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax5.set_title('Feature Statistics', fontsize=11, fontweight='bold', pad=20)
    
    # Add overall information
    fig.suptitle(f'Graph Visualization After tree_to_graph Conversion', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    
    plt.show()
    
    return G


def print_graph_info(nodes: np.ndarray, edges: List, edge_features: List, item: Dict):
    """Print detailed information about the graph"""
    
    print("\n" + "="*80)
    print("GRAPH STRUCTURE INFORMATION")
    print("="*80)
    
    print(f"\nExample ID: {item.get('tag', 'unknown')}")
    print(f"Correct Answer: {item.get('score', '?')}")
    
    print(f"\n{'='*80}")
    print("GRAPH STATISTICS")
    print(f"{'='*80}")
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")
    print(f"Number of thoughts: {len(set(nodes[:, 0]))}")
    print(f"Max depth: {int(nodes[:, 2].max())}")
    print(f"Average node degree: {len(edges) / len(nodes):.2f}")
    
    print(f"\n{'='*80}")
    print("NODE FEATURES (first 5 nodes)")
    print(f"{'='*80}")
    
    feature_names = ['Thought ID', 'Segment', 'Depth', 'Tokens', 'Components',
                    'Prev Tokens', 'Child Idx', 'Children', 'Level Nodes', 'Distance']
    
    print(f"{'Node':<6}", end='')
    for name in feature_names:
        print(f"{name:<12}", end='')
    print()
    print("-" * 130)
    
    for i in range(min(5, len(nodes))):
        print(f"{i:<6}", end='')
        for j in range(len(feature_names)):
            print(f"{nodes[i, j]:<12.2f}", end='')
        print()
    
    if len(nodes) > 5:
        print(f"... ({len(nodes) - 5} more nodes)")
    
    print(f"\n{'='*80}")
    print("EDGE INFORMATION (first 10 edges)")
    print(f"{'='*80}")
    print(f"{'From':<8}{'To':<8}{'Feature':<12}Type")
    print("-" * 40)
    
    for i in range(min(10, len(edges))):
        edge = edges[i]
        feat = edge_features[i][0] if edge_features else 0
        edge_type = "Parent→Child" if i % 2 == 0 else "Child→Parent"
        print(f"{edge[0]:<8}{edge[1]:<8}{feat:<12.2f}{edge_type}")
    
    if len(edges) > 10:
        print(f"... ({len(edges) - 10} more edges)")
    
    print(f"\n{'='*80}")
    print("REASONING THOUGHTS")
    print(f"{'='*80}")
    
    thoughts_list = item.get('thoughts_list', {})
    if isinstance(thoughts_list, str):
        thoughts_list = json.loads(thoughts_list)
    
    for thought_id in sorted(thoughts_list.keys(), key=int)[:3]:
        text = thoughts_list[thought_id]
        preview = text[:150] + "..." if len(text) > 150 else text
        print(f"\nThought {thought_id}:")
        print(f"  {preview}")
    
    if len(thoughts_list) > 3:
        print(f"\n... ({len(thoughts_list) - 3} more thoughts)")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <data_path> [example_index] [output_path]")
        print("\nExample:")
        print("  python scripts/visualize_graph.py deepseek/final.json")
        print("  python scripts/visualize_graph.py deepseek/final.json 0 outputs/graph_viz_0.png")
        sys.exit(1)
    
    data_path = sys.argv[1]
    example_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load tokenizer
    if AutoTokenizer is None:
        print("Error: transformers not installed. Run: pip install transformers")
        sys.exit(1)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {data_path}")
    
    if not Path(data_path).exists():
        print(f"Error: File not found: {data_path}")
        sys.exit(1)
    
    with open(data_path, "r") as f:
        lines = f.readlines()
    
    if example_idx >= len(lines):
        print(f"Error: Example index {example_idx} out of range (file has {len(lines)} examples)")
        sys.exit(1)
    
    # Parse example
    item = json.loads(lines[example_idx])
    
    # Get thoughts
    thoughts_list = item["thoughts_list"]
    if isinstance(thoughts_list, str):
        thoughts_list = json.loads(thoughts_list)
    
    # Count tokens
    tokens_list = [len(tokenizer.encode(text)) for text in thoughts_list.values()]
    
    # Get tree
    cot_tree = item['cot_tree']
    
    # Convert to graph
    print(f"\nConverting tree to graph...")
    nodes, edges, edge_features, nodes_dict = tree_to_graph(cot_tree, tokens_list, edge_type="11_1")
    
    # Print information
    print_graph_info(nodes, edges, edge_features, item)
    
    # Visualize
    print(f"\nCreating visualization...")
    G = visualize_graph(nodes, edges, edge_features, item, output_path)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nGraph has been visualized successfully!")
    print(f"  - Nodes are colored by depth (darker = deeper)")
    print(f"  - Node size represents token count")
    print(f"  - Arrows show edge direction (parent → child)")
    
    if output_path:
        print(f"\nVisualization saved to: {output_path}")
    else:
        print(f"\nVisualization displayed (not saved)")
        print(f"To save, run: python scripts/visualize_graph.py {data_path} {example_idx} output.png")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

