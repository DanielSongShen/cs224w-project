"""PyTorch Geometric Dataset for reasoning trace graphs"""

import json
import os
import os.path as osp
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from collections import defaultdict

import torch
from torch_geometric.data import HeteroData, InMemoryDataset, download_url
from torch_geometric.loader import DataLoader


# Edge type names corresponding to category values
CATEGORY_TO_EDGE_TYPE = {
    0: "root",
    1: "continuous_logic",
    2: "exploration",
    3: "backtracking",
    4: "validation",
}


def _traverse_tree(
    node: Dict[str, Any],
    node_list: List[Dict[str, Any]],
    edges_by_type: Dict[str, List[Tuple[int, int]]],
    node_id_map: Dict[str, int],
) -> None:
    """
    Recursively traverse the tree and collect nodes and edges.
    
    Args:
        node: Current tree node
        node_list: List to accumulate node data
        edges_by_type: Dict mapping edge type to list of (src, dst) tuples
        node_id_map: Maps node value string to integer node ID
    """
    # Get or assign node ID for current node
    node_value = str(node["value"])
    if node_value not in node_id_map:
        node_id_map[node_value] = len(node_list)
        node_list.append({
            "value": node_value,
            "level": node["level"],
            "cate": node["cate"],
            "thought_list": node["thought_list"],
        })
    
    parent_id = node_id_map[node_value]
    
    # Process children
    for child in node.get("children", []):
        child_value = str(child["value"])
        
        # Assign ID to child if not already assigned
        if child_value not in node_id_map:
            node_id_map[child_value] = len(node_list)
            node_list.append({
                "value": child_value,
                "level": child["level"],
                "cate": child["cate"],
                "thought_list": child["thought_list"],
            })
        
        child_id = node_id_map[child_value]
        
        # Add edge based on child's category
        child_cate = child["cate"]
        edge_type = CATEGORY_TO_EDGE_TYPE.get(child_cate, "continuous_logic")
        edges_by_type[edge_type].append((parent_id, child_id))
        
        # Recurse into child
        _traverse_tree(child, node_list, edges_by_type, node_id_map)


def convert_json_to_hetero_graph(
    data: Dict[str, Any],
    include_reverse_edges: bool = True,
) -> HeteroData:
    """
    Convert a JSON object from the LCoT2Tree format to a heterogeneous PyTorch Geometric graph.
    
    The graph has a single node type "thought" and multiple edge types based on
    the relationship category between thoughts:
        - root: Initial/default edges (cate=0)
        - continuous_logic: Direct continuation of reasoning (cate=1)
        - exploration: Alternative reasoning paths (cate=2)
        - backtracking: Revisions/corrections (cate=3)
        - validation: Supporting evidence/justification (cate=4)
    
    Args:
        data: JSON object containing 'cot_tree', 'thoughts_list', and 'score'
        include_reverse_edges: Whether to add reverse edges for message passing
        
    Returns:
        HeteroData object with node features and typed edges
    """
    cot_tree = data["cot_tree"]
    thoughts_list = data.get("thoughts_list", {})
    score = data.get("score", "0")
    
    # Convert score to integer label
    label = int(score) if isinstance(score, (int, float)) else int(score == "1")
    
    # Traverse tree to collect nodes and edges
    node_list: List[Dict[str, Any]] = []
    edges_by_type: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    node_id_map: Dict[str, int] = {}
    
    _traverse_tree(cot_tree, node_list, edges_by_type, node_id_map)
    
    num_nodes = len(node_list)
    
    # Create HeteroData object
    hetero_data = HeteroData()
    
    # Node features for "thought" nodes
    # Features: [level, category, thought_index]
    node_features = []
    for node in node_list:
        level = node["level"]
        cate = node["cate"]
        # Use first thought in thought_list, or -1 if empty
        thought_idx = node["thought_list"][0] if node["thought_list"] else -1
        node_features.append([level, cate, thought_idx])
    
    hetero_data["thought"].x = torch.tensor(node_features, dtype=torch.float)
    hetero_data["thought"].num_nodes = num_nodes
    
    # Add edges for each type
    # IMPORTANT: All graphs must have identical edge type schemas for InMemoryDataset
    # to correctly collate and separate them. Always add all edge types, even if empty.
    for edge_type_name in CATEGORY_TO_EDGE_TYPE.values():
        edges = edges_by_type.get(edge_type_name, [])

        if edges:
            src_nodes = [e[0] for e in edges]
            dst_nodes = [e[1] for e in edges]
            edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            reverse_edge_index = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
        else:
            # Empty edge index - still must be added for consistent schema
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            reverse_edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        hetero_data["thought", edge_type_name, "thought"].edge_index = edge_index
        
        # Always add reverse edges for consistent schema (even if empty)
        if include_reverse_edges:
            hetero_data["thought", f"rev_{edge_type_name}", "thought"].edge_index = reverse_edge_index
    
    # Store label as graph-level attribute
    hetero_data.y = torch.tensor([label], dtype=torch.long)
    
    return hetero_data


def load_jsonl_to_graphs(
    filepath: str,
    include_reverse_edges: bool = True,
) -> List[HeteroData]:
    """
    Load a JSONL file and convert each line to a HeteroData graph.
    
    Args:
        filepath: Path to JSONL file
        include_reverse_edges: Whether to add reverse edges
        
    Returns:
        List of HeteroData objects
    """
    graphs = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                graph = convert_json_to_hetero_graph(data, include_reverse_edges)
                graphs.append(graph)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to parse line: {e}")
                continue
    
    return graphs


class ReasoningTraceDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for reasoning trace graphs.
    
    Each sample is a heterogeneous graph representing a chain-of-thought reasoning trace,
    with the task being binary classification (correct vs incorrect reasoning).
    
    Args:
        root: Root directory where the dataset should be saved.
        raw_filepath: Path to the raw JSONL file containing reasoning traces.
        include_reverse_edges: Whether to include reverse edges for bidirectional 
            message passing. Default: True
        transform: A function/transform that takes in a Data object and returns a 
            transformed version. Default: None
        pre_transform: A function/transform that takes in a Data object and returns a
            transformed version. Applied before saving to disk. Default: None
        pre_filter: A function that takes in a Data object and returns a boolean value,
            indicating whether the data object should be included. Default: None
        force_reload: If True, re-process the dataset even if it exists. Default: False
    
    Example:
        >>> dataset = ReasoningTraceDataset(
        ...     root='./data/processed',
        ...     raw_filepath='./data/raw/final.json'
        ... )
        >>> len(dataset)
        100
        >>> dataset[0]
        HeteroData(...)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        root: str,
        raw_filepath: str,
        include_reverse_edges: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.raw_filepath = osp.abspath(raw_filepath)
        self.include_reverse_edges = include_reverse_edges
        
        # Store the raw filename for processed file naming
        self._raw_filename = osp.basename(raw_filepath).replace('.json', '').replace('.jsonl', '')
        
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        self.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return raw file names (not used since we specify raw_filepath directly)."""
        return [osp.basename(self.raw_filepath)]
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return processed file names."""
        suffix = "_with_rev" if self.include_reverse_edges else ""
        return [f'{self._raw_filename}{suffix}.pt']
    
    def download(self):
        """Download is not needed - we use local files."""
        pass
    
    def process(self):
        """Process raw JSONL file into graph data objects."""
        print(f"Processing {self.raw_filepath}...")
        
        # Load and convert all graphs
        data_list = load_jsonl_to_graphs(
            self.raw_filepath,
            include_reverse_edges=self.include_reverse_edges,
        )
        
        print(f"Loaded {len(data_list)} graphs")
        
        # Apply pre_filter if specified
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
            print(f"After filtering: {len(data_list)} graphs")
        
        # Apply pre_transform if specified
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Save processed data
        self.save(data_list, self.processed_paths[0])
        print(f"Saved processed dataset to {self.processed_paths[0]}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the dataset."""
        num_graphs = len(self)
        
        if num_graphs == 0:
            return {"num_graphs": 0}
        
        # Collect statistics in a single pass
        labels_list = []
        num_nodes_list = []
        edge_counts = defaultdict(list)
        
        for i in range(num_graphs):
            try:
                data = self[i]
            except:
                breakpoint()
            labels_list.append(data.y.item())
            num_nodes_list.append(data["thought"].num_nodes)
            for edge_type in data.edge_types:
                edge_counts[edge_type].append(data[edge_type].edge_index.shape[1])
        
        num_positive = sum(labels_list)
        num_negative = num_graphs - num_positive
        
        return {
            "num_graphs": num_graphs,
            "num_positive": num_positive,
            "num_negative": num_negative,
            "class_ratio": num_positive / num_graphs if num_graphs > 0 else 0,
            "avg_nodes": sum(num_nodes_list) / num_graphs,
            "min_nodes": min(num_nodes_list),
            "max_nodes": max(num_nodes_list),
            "edge_types": list(edge_counts.keys()),
            "avg_edges_per_type": {
                str(k): sum(v) / len(v) for k, v in edge_counts.items()
            },
        }
    
    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}({len(self)} graphs, '
            f'raw_file={osp.basename(self.raw_filepath)})'
        )


def create_train_val_test_split(
    dataset: ReasoningTraceDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits for the dataset.
    
    Args:
        dataset: The ReasoningTraceDataset to split
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        stratify: Whether to stratify by label (maintain class balance)
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n = len(dataset)
    
    if stratify:
        # Get labels
        labels = [dataset[i].y.item() for i in range(n)]
        
        # Separate indices by class
        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]
        
        # Shuffle each class
        torch.manual_seed(seed)
        pos_perm = torch.randperm(len(pos_indices)).tolist()
        neg_perm = torch.randperm(len(neg_indices)).tolist()
        
        pos_indices = [pos_indices[i] for i in pos_perm]
        neg_indices = [neg_indices[i] for i in neg_perm]
        
        # Split each class
        def split_indices(indices):
            n_cls = len(indices)
            n_train = int(n_cls * train_ratio)
            n_val = int(n_cls * val_ratio)
            return (
                indices[:n_train],
                indices[n_train:n_train + n_val],
                indices[n_train + n_val:],
            )
        
        pos_train, pos_val, pos_test = split_indices(pos_indices)
        neg_train, neg_val, neg_test = split_indices(neg_indices)
        
        train_indices = pos_train + neg_train
        val_indices = pos_val + neg_val
        test_indices = pos_test + neg_test
        
        # Shuffle combined indices
        torch.manual_seed(seed + 1)
        train_indices = [train_indices[i] for i in torch.randperm(len(train_indices)).tolist()]
        val_indices = [val_indices[i] for i in torch.randperm(len(val_indices)).tolist()]
        test_indices = [test_indices[i] for i in torch.randperm(len(test_indices)).tolist()]
    else:
        # Simple random split
        torch.manual_seed(seed)
        perm = torch.randperm(n).tolist()
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        train_indices = perm[:n_train]
        val_indices = perm[n_train:n_train + n_val]
        test_indices = perm[n_train + n_val:]
    
    return train_indices, val_indices, test_indices


def get_dataloaders(
    dataset: ReasoningTraceDataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
    stratify: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        dataset: The ReasoningTraceDataset
        batch_size: Batch size for all loaders
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation  
        test_ratio: Fraction of data for testing
        seed: Random seed for reproducibility
        num_workers: Number of workers for data loading
        stratify: Whether to stratify splits by label
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_idx, val_idx, test_idx = create_train_val_test_split(
        dataset, train_ratio, val_ratio, test_ratio, seed, stratify
    )
    
    train_dataset = dataset[train_idx]
    val_dataset = dataset[val_idx]
    test_dataset = dataset[test_idx]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build reasoning trace dataset")
    parser.add_argument(
        "raw_filepath", 
        type=str, 
        help="Path to raw JSONL file"
    )
    parser.add_argument(
        "--root", 
        type=str, 
        default="./data/processed_graphs",
        help="Root directory to store processed dataset"
    )
    parser.add_argument(
        "--no-reverse-edges", 
        action="store_true",
        help="Don't include reverse edges"
    )
    parser.add_argument(
        "--force-reload", 
        action="store_true",
        help="Force reprocessing even if processed file exists"
    )
    
    args = parser.parse_args()
    
    print(f"Building dataset from {args.raw_filepath}")
    print(f"Saving to {args.root}")
    
    dataset = ReasoningTraceDataset(
        root=args.root,
        raw_filepath=args.raw_filepath,
        include_reverse_edges=not args.no_reverse_edges,
        force_reload=args.force_reload,
    )
    
    print(f"\nDataset: {dataset}")
    
    summary = dataset.get_summary()
    print(f"\nSummary:")
    print(f"  Total graphs: {summary['num_graphs']}")
    print(f"  Positive (correct): {summary['num_positive']}")
    print(f"  Negative (incorrect): {summary['num_negative']}")
    print(f"  Class ratio (positive): {summary['class_ratio']:.2%}")
    print(f"  Avg nodes per graph: {summary['avg_nodes']:.1f}")
    print(f"  Node range: [{summary['min_nodes']}, {summary['max_nodes']}]")
    print(f"\n  Edge types and avg counts:")
    for edge_type, avg_count in summary['avg_edges_per_type'].items():
        print(f"    {edge_type}: {avg_count:.1f}")
    
    # Test data loading
    print("\nTesting DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset, 
        batch_size=2,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Show a sample batch
    for batch in train_loader:
        print(f"\nSample batch:")
        print(f"  Batch type: {type(batch)}")
        print(f"  Node features shape: {batch['thought'].x.shape}")
        print(f"  Labels: {batch.y.flatten().tolist()}")
        print(f"  Edge types: {batch.edge_types}")
        break
