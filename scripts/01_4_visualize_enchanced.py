#!/usr/bin/env python3
"""
Visualize Chain-of-Thought graphs from reasoning pipeline output.

Nodes are aligned by their assigned reasoning steps, not graph depth.
Edges are color-coded by relationship category.
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
import re

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


def extract_reasoning_steps(reasoning_sketch: str) -> Dict[int, str]:
    """
    Extract reasoning steps from sketch text.
    
    Args:
        reasoning_sketch: Text containing reasoning steps
    
    Returns:
        Dictionary mapping step number to step description
    """
    if not reasoning_sketch:
        return {}
    
    # Extract content between tags if present
    start_index = reasoning_sketch.find("<reasoning_process>")
    end_index = reasoning_sketch.find("</reasoning_process>")
    
    if start_index != -1 and end_index != -1:
        reasoning_text = reasoning_sketch[start_index + len("<reasoning_process>"):end_index]
    else:
        reasoning_text = reasoning_sketch
    
    # Parse steps using regex
    pattern = re.compile(r'Step (\d+)\.\s*(.*?)(?=(Step \d+\.)|$)', re.DOTALL)
    matches = pattern.findall(reasoning_text)
    
    reasoning_dict = {}
    for match in matches:
        step_num = int(match[0])
        step_text = match[1].strip()
        reasoning_dict[step_num] = step_text
    
    return reasoning_dict


def build_step_to_thoughts_mapping(assigned_step: Dict) -> Dict[int, List[int]]:
    """
    Build reverse mapping from step_id -> list of thought_ids.
    
    Args:
        assigned_step: Dictionary mapping thought_id -> [step_ids]
    
    Returns:
        Dictionary mapping step_id -> [thought_ids]
    """
    step_to_thoughts = {}
    
    for thought_key, step_ids in assigned_step.items():
        # Normalize thought_id to integer
        try:
            thought_id = int(re.sub(r'[A-Za-z]', '', str(thought_key)))
        except (ValueError, TypeError):
            continue  # Skip invalid thought IDs
        
        # Normalize step_ids to integers
        if not isinstance(step_ids, list):
            step_ids = [step_ids]
        
        for step_key in step_ids:
            try:
                step_id = int(re.sub(r'[A-Za-z]', '', str(step_key)))
                
                if step_id not in step_to_thoughts:
                    step_to_thoughts[step_id] = []
                step_to_thoughts[step_id].append(thought_id)
            except (ValueError, TypeError):
                continue  # Skip invalid step IDs
    
    # Sort thought lists for each step
    for step_id in step_to_thoughts:
        step_to_thoughts[step_id].sort()
    
    return step_to_thoughts


def get_thought_primary_step(thought_id: int, assigned_step: Dict) -> int:
    """
    Get the primary (earliest) reasoning step for a thought.
    
    Args:
        thought_id: The thought ID
        assigned_step: Dictionary mapping thought_id -> [step_ids]
    
    Returns:
        The earliest step ID this thought is assigned to (defaults to 0 if not found)
    """
    # Find this thought's assignment
    for thought_key, step_ids in assigned_step.items():
        try:
            tid = int(re.sub(r'[A-Za-z]', '', str(thought_key)))
        except (ValueError, TypeError):
            continue
            
        if tid == thought_id:
            if not isinstance(step_ids, list):
                step_ids = [step_ids]
            
            # Normalize step_ids to integers
            normalized_steps = []
            for s in step_ids:
                try:
                    normalized = int(re.sub(r'[A-Za-z]', '', str(s)))
                    normalized_steps.append(normalized)
                except (ValueError, TypeError):
                    pass  # Skip invalid step IDs
            
            # Return minimum if we have valid steps, otherwise default to 0
            if normalized_steps:
                return min(normalized_steps)
            else:
                # Empty or invalid step assignment, default to step 0
                print(f"Warning: Thought {thought_id} has empty step assignment, defaulting to step 0")
                return 0
    
    # Default to step 0 if not found in assigned_step
    return 0


def print_step_assignments(item_data: Dict[str, Any], show_thought_text: bool = False):
    """
    Print thought assignments to reasoning steps in a readable format.
    
    Args:
        item_data: Full item data with reasoning_sketch and assigned_step
        show_thought_text: Whether to show snippet of thought text
    """
    tag = item_data.get('tag', 'Unknown')
    print(f"\n{'='*80}")
    print(f"THOUGHT ASSIGNMENTS: {tag}")
    print(f"{'='*80}\n")
    
    # Extract reasoning steps
    reasoning_sketch = item_data.get('reasoning_sketch', '')
    reasoning_steps = extract_reasoning_steps(reasoning_sketch)
    
    if not reasoning_steps:
        print("No reasoning steps found in sketch.")
        return
    
    # Build step -> thoughts mapping
    assigned_step = item_data.get('assigned_step', {})
    step_to_thoughts = build_step_to_thoughts_mapping(assigned_step)
    
    # Get thoughts text if needed
    thoughts_text = {}
    if show_thought_text and 'thoughts_list' in item_data:
        thoughts_list = item_data['thoughts_list']
        if isinstance(thoughts_list, str):
            thoughts_list = json.loads(thoughts_list)
        thoughts_text = {int(k): v for k, v in thoughts_list.items()}
    
    # Print each step with its assigned thoughts
    for step_num in sorted(reasoning_steps.keys()):
        step_desc = reasoning_steps[step_num]
        
        # Truncate long descriptions
        if len(step_desc) > 100:
            step_desc = step_desc[:100] + "..."
        
        print(f"Step {step_num}: {step_desc}")
        print(f"{'-'*80}")
        
        # Get thoughts for this step
        thought_ids = step_to_thoughts.get(step_num, [])
        
        if not thought_ids:
            print("  (No thoughts assigned)")
        else:
            print(f"  Assigned thoughts: {thought_ids}")
            
            if show_thought_text:
                for thought_id in thought_ids:
                    thought = thoughts_text.get(thought_id, "")
                    # Truncate and clean
                    thought = thought.replace('\n', ' ').strip()
                    if len(thought) > 80:
                        thought = thought[:80] + "..."
                    print(f"    T{thought_id}: {thought}")
        
        print()
    
    # Summary statistics
    total_thoughts = len(assigned_step)
    thoughts_with_assignments = len(set(
        thought_id for thoughts in step_to_thoughts.values() 
        for thought_id in thoughts
    ))
    
    print(f"Summary:")
    print(f"  Total reasoning steps: {len(reasoning_steps)}")
    print(f"  Total thoughts: {total_thoughts}")
    print(f"  Thoughts with assignments: {thoughts_with_assignments}")
    
    # Find thoughts assigned to multiple steps
    multi_step_thoughts = []
    for thought_key, step_ids in assigned_step.items():
        try:
            thought_id = int(re.sub(r'[A-Za-z]', '', str(thought_key)))
            if isinstance(step_ids, list) and len(step_ids) > 1:
                multi_step_thoughts.append((thought_id, step_ids))
        except (ValueError, TypeError):
            continue
    
    if multi_step_thoughts:
        print(f"  Thoughts spanning multiple steps: {len(multi_step_thoughts)}")
        for thought_id, step_ids in multi_step_thoughts[:5]:  # Show first 5
            print(f"    T{thought_id} → Steps {step_ids}")
    
    print(f"\n{'='*80}\n")


def graph_from_edge_list(reasoning_graph: Dict[str, Any], item_data: Dict[str, Any]) -> nx.DiGraph:
    """
    Create NetworkX directed graph from edge list format.
    Uses assigned_step for node positioning instead of graph depth.
    
    Args:
        reasoning_graph: Graph dict with 'nodes' and 'edges' fields
        item_data: Full item data with assigned_step mapping
    
    Returns:
        NetworkX directed graph with step-based attributes
    """
    graph = nx.DiGraph()
    
    # Get step assignments
    assigned_step = item_data.get('assigned_step', {})
    
    # Add all nodes with their reasoning step assignments
    nodes = reasoning_graph.get('nodes', [])
    for node_id in nodes:
        # Get primary (earliest) reasoning step for this thought
        primary_step = get_thought_primary_step(node_id, assigned_step)
        
        graph.add_node(
            node_id,
            thought_num=node_id,
            thought_list=[node_id],
            reasoning_step=primary_step,  # Use this for layout instead of graph level
            subset=primary_step  # For multipartite layout
        )
    
    # Add all edges
    edges = reasoning_graph.get('edges', [])
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        category = edge.get('category', 0)
        
        if source is not None and target is not None:
            graph.add_edge(source, target, category=category)
    
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
    # Get step assignments for new nodes
    assigned_step = item_data.get('assigned_step', {})
    
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
                primary_step = get_thought_primary_step(thought_num, assigned_step)
                
                graph.add_node(
                    node_id,
                    reasoning_step=primary_step,
                    thought_num=thought_num,
                    thought_list=[thought_num],
                    subset=primary_step
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


def get_step_based_layout(graph: nx.DiGraph, reasoning_steps: Dict[int, str]) -> Dict[int, Tuple[float, float]]:
    """
    Create layout based on reasoning step assignments.
    Nodes are arranged in rows by their assigned reasoning step.
    
    Args:
        graph: Directed graph with 'reasoning_step' node attribute
        reasoning_steps: Dictionary of step_id -> step description
    
    Returns:
        Dictionary mapping node IDs to (x, y) positions
    """
    pos = {}
    
    # Group nodes by reasoning step
    step_to_nodes = {}
    for node, data in graph.nodes(data=True):
        step = data.get('reasoning_step', 0)
        if step not in step_to_nodes:
            step_to_nodes[step] = []
        step_to_nodes[step].append(node)
    
    # Get all steps (including those in reasoning_steps even if no nodes)
    all_steps = set(step_to_nodes.keys()) | set(reasoning_steps.keys())
    if not all_steps:
        all_steps = {0}  # Default to step 0 if no steps found
    
    max_step = max(all_steps)
    
    # Layout parameters
    vertical_spacing = 2.0  # Space between reasoning steps
    horizontal_spacing = 1.5  # Space between nodes in same step
    
    # Position nodes
    for step_idx, step_num in enumerate(sorted(all_steps)):
        nodes_in_step = step_to_nodes.get(step_num, [])
        
        if not nodes_in_step:
            continue
        
        # Sort nodes by ID for consistent layout
        nodes_in_step.sort()
        
        # Calculate y position (negative so it goes down)
        y_pos = -step_idx * vertical_spacing
        
        # Calculate x positions (centered)
        num_nodes = len(nodes_in_step)
        total_width = (num_nodes - 1) * horizontal_spacing
        start_x = -total_width / 2
        
        for i, node in enumerate(nodes_in_step):
            x_pos = start_x + i * horizontal_spacing
            pos[node] = (x_pos, y_pos)
    
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
                    show_isolated: bool = False,
                    show_assignments: bool = True, show_step_labels: bool = True):
    """
    Visualize a single reasoning graph with step-based layout.
    Edges are always color-coded by their relationship category.
    
    Args:
        reasoning_graph: Graph structure from reasoning_graph field
        item_data: Full item data including thoughts_list, tag, etc.
        output_path: Path to save the visualization
        show_thoughts: Whether to show thought text in node labels
        show_isolated: Whether to show isolated nodes
        show_assignments: Whether to print step assignments to console
        show_step_labels: Whether to show reasoning step labels on the graph
    """
    # Print step assignments if requested
    if show_assignments:
        print_step_assignments(item_data, show_thought_text=True)
    
    # Extract reasoning steps
    reasoning_sketch = item_data.get('reasoning_sketch', '')
    reasoning_steps = extract_reasoning_steps(reasoning_sketch)
    
    # Build Graph with step-based attributes
    graph = graph_from_edge_list(reasoning_graph, item_data)
    graph = add_all_nodes_and_edges(graph, item_data, show_isolated)
    
    # Check if graph is empty
    if graph.number_of_nodes() == 0:
        print(f"Warning: Graph is empty for {item_data.get('tag', 'unknown')}, skipping visualization")
        return
    
    # Compute Layout based on reasoning steps
    pos = get_step_based_layout(graph, reasoning_steps)
    
    relation_stats = get_relation_stats(item_data)
    has_stats = relation_stats['has_relations']
    
    # Create figure
    if has_stats:
        fig = plt.figure(figsize=(20, 12))
        ax_main = plt.subplot(1, 2, 1)
        ax_stats = plt.subplot(1, 2, 2)
    else:
        fig, ax_main = plt.subplots(figsize=(18, 12))
    
    # Get node attributes
    thought_nums = nx.get_node_attributes(graph, 'thought_num')
    reasoning_step_attrs = nx.get_node_attributes(graph, 'reasoning_step')
    
    # Color nodes by their reasoning step
    node_colors = []
    step_colors_map = {}
    color_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', 
                     '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788']
    
    for node in graph.nodes():
        step = reasoning_step_attrs.get(node, 0)
        if step not in step_colors_map:
            step_colors_map[step] = color_palette[step % len(color_palette)]
        node_colors.append(step_colors_map[step])
    
    # Create node labels
    if show_thoughts and 'thoughts_list' in item_data:
        thoughts = item_data['thoughts_list']
        if isinstance(thoughts, str):
            thoughts = json.loads(thoughts)
        thoughts = {str(k): v for k, v in thoughts.items()}
        
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, node)
            thought_text = thoughts.get(str(thought_num), "")
            thought_text = thought_text.replace('\\', '/').replace('\n', ' ')
            if len(thought_text) > 30:
                thought_text = thought_text[:30] + "..."
            labels[node] = f"T{thought_num}\n{thought_text}"
    else:
        labels = {}
        for node in graph.nodes():
            thought_num = thought_nums.get(node, node)
            labels[node] = f"T{thought_num}"
    
    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos, node_color=node_colors, 
        node_size=1000, alpha=0.9, ax=ax_main,
        edgecolors='black', linewidths=1.5
    )
    
    # Draw edges COLOR-CODED BY CATEGORY (always)
    edge_categories = nx.get_edge_attributes(graph, 'category')
    
    # Draw edges grouped by category for consistent coloring
    for cat_id in range(6):
        edges_in_cat = [
            (u, v) for (u, v), cat in edge_categories.items() 
            if cat == cat_id
        ]
        if edges_in_cat:
            nx.draw_networkx_edges(
                graph, pos, edgelist=edges_in_cat,
                edge_color=CATEGORY_COLORS[cat_id],
                width=2.5, alpha=0.7, arrows=True,
                arrowsize=20, arrowstyle='->', ax=ax_main,
                connectionstyle='arc3,rad=0.1'
            )
    
    # Draw labels
    nx.draw_networkx_labels(
        graph, pos, labels, font_size=9,
        font_weight='bold', ax=ax_main
    )
    
    # Add reasoning step labels on the left side
    if show_step_labels and reasoning_steps:
        step_to_nodes = {}
        for node, data in graph.nodes(data=True):
            step = data.get('reasoning_step', 0)
            if step not in step_to_nodes:
                step_to_nodes[step] = []
            step_to_nodes[step].append(node)
        
        for step_idx, step_num in enumerate(sorted(reasoning_steps.keys())):
            if step_num not in step_to_nodes:
                continue
            
            # Get y position from first node in this step
            nodes_in_step = step_to_nodes[step_num]
            if nodes_in_step and nodes_in_step[0] in pos:
                _, y_pos = pos[nodes_in_step[0]]
                
                # Get step description
                step_desc = reasoning_steps[step_num]
                if len(step_desc) > 50:
                    step_desc = step_desc[:50] + "..."
                
                # Find leftmost x position
                x_positions = [pos[n][0] for n in nodes_in_step if n in pos]
                if x_positions:
                    left_x = min(x_positions) - 2.5
                    
                    # Add step label
                    ax_main.text(
                        left_x, y_pos, 
                        f"Step {step_num}:\n{step_desc}",
                        fontsize=8,
                        verticalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=step_colors_map.get(step_num, '#EEEEEE'), 
                                alpha=0.7, edgecolor='black'),
                        wrap=True
                    )
    
    # Title
    tag = item_data.get('tag', 'Unknown')
    ax_main.set_title(f"Chain-of-Thought Graph: {tag}\n(Nodes by step, Edges by relationship)", 
                      fontsize=14, pad=20, fontweight='bold')
    ax_main.axis('off')
    
    # Legend for edge categories (ALWAYS shown)
    legend_elements = [
        mpatches.Patch(color=CATEGORY_COLORS[i], label=CATEGORY_NAMES[i], alpha=0.7)
        for i in sorted(CATEGORY_COLORS.keys())
    ]
    ax_main.legend(
        handles=legend_elements, loc='upper right',
        fontsize=10, framealpha=0.9, title="Edge Categories"
    )
    
    # Statistics panel
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
        stats_text += f"  Nodes: {graph.number_of_nodes()}\n"
        stats_text += f"  Edges: {graph.number_of_edges()}\n"
        stats_text += f"  Avg In-Degree: {sum(d for n, d in graph.in_degree()) / max(graph.number_of_nodes(), 1):.2f}\n"
        stats_text += f"  Avg Out-Degree: {sum(d for n, d in graph.out_degree()) / max(graph.number_of_nodes(), 1):.2f}\n"
        
        try:
            stats_text += f"  Longest Path: {nx.dag_longest_path_length(graph)}\n"
        except:
            stats_text += f"  Longest Path: N/A\n"
        
        stats_text += f"\nReasoning Steps: {len(reasoning_steps)}\n"
        
        ax_stats.text(0.1, 0.9, stats_text, 
                     transform=ax_stats.transAxes,
                     fontsize=11, verticalalignment='top',
                     fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Bar chart
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
        print(f"Warning: tight_layout failed: {e}")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def visualize_all_graphs(input_path: str, output_dir: str, max_graphs: int = None, 
                         show_thoughts: bool = False,
                         summary_stats: bool = True, show_isolated: bool = False,
                         show_assignments: bool = True, show_step_labels: bool = True):
    """
    Visualize all graphs from final.json output.
    
    Args:
        input_path: Path to final.json
        output_dir: Directory to save visualizations
        max_graphs: Maximum number of graphs to visualize (None for all)
        show_thoughts: Whether to show thought text in labels
        summary_stats: Whether to print summary statistics
        show_isolated: Whether to show isolated nodes
        show_assignments: Whether to print step assignments to console
        show_step_labels: Whether to show reasoning step labels on graphs
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
                show_isolated=show_isolated,
                show_assignments=show_assignments,
                show_step_labels=show_step_labels
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
        description="Visualize Chain-of-Thought graphs with step-based layout and color-coded edges"
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
        '--show_isolated', '-i',
        action='store_true',
        help='Show isolated nodes not connected to graph (default: False)'
    )
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Disable summary statistics printing'
    )
    parser.add_argument(
        '--no_assignments',
        action='store_true',
        help='Disable printing thought assignments to reasoning steps'
    )
    parser.add_argument(
        '--no_step_labels',
        action='store_true',
        help='Disable reasoning step labels on graphs'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        print("\nUsage examples:")
        print("  python visualize_reasoning_graphs.py")
        print("  python visualize_reasoning_graphs.py path/to/final.json")
        print("  python visualize_reasoning_graphs.py --max_graphs 5 --show_thoughts")
        print("  python visualize_reasoning_graphs.py --no_step_labels")
        return 1
    
    visualize_all_graphs(
        args.input_path,
        args.output_dir,
        max_graphs=args.max_graphs,
        show_thoughts=args.show_thoughts,
        summary_stats=not args.no_summary,
        show_isolated=args.show_isolated,
        show_assignments=not args.no_assignments,
        show_step_labels=not args.no_step_labels
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())