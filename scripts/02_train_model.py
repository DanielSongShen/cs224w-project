"""Script to train GAT model on LCoT2Tree graphs for answer correctness prediction"""

import json
import sys
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import numpy as np
import random
from pathlib import Path
import argparse
from typing import List, Tuple, Dict
import copy

# Add LCoT2Tree to path
sys.path.append(str(Path(__file__).parent.parent / "LCoT2Tree" / "src" / "gnn"))
sys.path.append(str(Path(__file__).parent.parent / "LCoT2Tree" / "src" / "cot2tree"))

try:
    from networks import GATv2GraphClassifier
except ImportError as e:
    print(f"Error importing GATv2GraphClassifier: {e}")
    print("Please check that LCoT2Tree/src/gnn/networks.py exists")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: transformers not installed. Using fallback tokenizer.")
    AutoTokenizer = None


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tree_to_graph(root: dict, tokens_list: List[int], edge_type: str = "11_1") -> Data:
    """
    Convert a CoT tree structure to a PyTorch Geometric graph.
    
    Args:
        root: Dictionary representing the tree structure with 'value', 'level', 'cate', 'thought_list', 'children'
        tokens_list: List of token counts for each thought
        edge_type: String encoding edge configuration (e.g., "11_1" means bidirectional with edge attributes)
        
    Returns:
        PyTorch Geometric Data object representing the graph
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
        
        child_index = len(nodes)
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
        
        # Node features: [thought_id, segment_id, level, current_tokens, component_count, 
        #                 prev_tokens, index, children_count, level_node_count, distance]
        nodes.append([
            int(node_v),           # 0: thought ID
            int(node_s),           # 1: segment ID within thought
            level,                 # 2: depth in tree
            current_token_len,     # 3: number of tokens in current thought
            node_c,                # 4: number of components
            prev_token_len,        # 5: normalized previous token count
            index,                 # 6: child index
            children_count,        # 7: number of children
            level_node_count,      # 8: number of nodes at this level
            distance               # 9: normalized distance from end
        ])
        nodes_dict[value] = len(nodes) - 1
        
        # Add edges to children
        for child_dict in tree_dict["children"]:
            child_value = child_dict["value"]
            
            # Edge features (category information)
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
    
    # Convert to tensors
    x = torch.tensor(nodes, dtype=torch.float)
    edge_index = torch.tensor([[edge[0], edge[1]] for edge in edges], dtype=torch.long).t().contiguous()
    
    if edge_type[0] == '1':
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        return Data(x=x, edge_index=edge_index)


def load_dataset(
    data_path: str,
    tokenizer,
    selected_features: List[int] = None,
    edge_type: str = "11_1"
) -> Tuple[List[Data], List[int], List[dict]]:
    """
    Load dataset from JSON file and convert to graphs.
    
    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer for counting tokens
        selected_features: List of feature indices to use (None = all)
        edge_type: Edge configuration string
        
    Returns:
        Tuple of (graphs, labels, items)
    """
    graphs = []
    labels = []
    items = []
    
    print(f"Loading data from {data_path}")
    
    with open(data_path, "r") as f:
        for line_num, line in enumerate(f):
            try:
                item = json.loads(line)
                
                # Parse thoughts_list
                if isinstance(item["thoughts_list"], str):
                    thought_list = json.loads(item["thoughts_list"])
                else:
                    thought_list = item["thoughts_list"]
                
                # Count tokens for each thought
                tokens_list = [len(tokenizer.encode(text)) for text in thought_list.values()]
                
                # Get tree structure and label
                cot_tree = item['cot_tree']
                score = float(item['score'])
                
                # Convert tree to graph
                graph = tree_to_graph(cot_tree, tokens_list, edge_type)
                
                graphs.append(graph)
                labels.append(score)
                items.append(item)
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs, labels, items


def train_epoch(model, loader, optimizer, device, selected_feat):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    num_batches = 0
    
    has_edge_attr = hasattr(loader.dataset[0], 'edge_attr') and loader.dataset[0].edge_attr is not None
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Select features
        x = data.x[:, selected_feat] if selected_feat is not None else data.x
        
        # Forward pass
        if has_edge_attr:
            out = model(x, data.edge_index, data.batch, data.edge_attr)
        else:
            out = model(x, data.edge_index, data.batch)
        
        # Compute loss
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        total += len(data.y)
        num_batches += 1
    
    # Handle empty loader case
    if num_batches == 0:
        return 0.0, 0.0
    
    return total_loss / num_batches, correct / total


def evaluate(model, loader, device, selected_feat):
    """Evaluate model on dataset"""
    model.eval()
    correct = 0
    total = 0
    predictions = []
    
    has_edge_attr = hasattr(loader.dataset[0], 'edge_attr') and loader.dataset[0].edge_attr is not None
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            # Select features
            x = data.x[:, selected_feat] if selected_feat is not None else data.x
            
            # Forward pass
            if has_edge_attr:
                out = model(x, data.edge_index, data.batch, data.edge_attr)
            else:
                out = model(x, data.edge_index, data.batch)
            
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += len(data.y)
            predictions.extend(pred.cpu().tolist())
    
    return correct / total, predictions


def main(args):
    """Main training function"""
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    if AutoTokenizer is None:
        print("Error: transformers not installed")
        print("Please install: pip install transformers")
        sys.exit(1)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer {args.tokenizer}: {e}")
        print("Using default tokenizer (gpt2)...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        except Exception as e2:
            print(f"Error loading gpt2 tokenizer: {e2}")
            sys.exit(1)
    
    # Load dataset
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data path {data_path} does not exist")
        return
    
    # Load data
    graphs, labels, items = load_dataset(
        str(data_path),
        tokenizer,
        args.selected_features,
        args.edge_type
    )
    
    if len(graphs) == 0:
        print("Error: No valid graphs loaded")
        return
    
    # Split data
    n_total = len(graphs)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)
    
    # Ensure at least 1 example in each split if possible
    if n_train == 0 and n_total > 0:
        n_train = 1
    if n_val == 0 and n_total > n_train:
        n_val = 1
    
    # Add labels to graphs
    for i, graph in enumerate(graphs):
        graph.y = torch.tensor([int(labels[i])], dtype=torch.long)
        graph = graph.to(device)
    
    # Create train/val/test splits
    train_dataset = graphs[:n_train]
    val_dataset = graphs[n_train:n_train + n_val]
    test_dataset = graphs[n_train + n_val:]
    
    print(f"Dataset split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Check if dataset is too small
    if len(train_dataset) < 20:
        print(f"\n{'!'*60}")
        print(f"WARNING: Very small dataset ({len(train_dataset)} training examples)")
        print("Consider:")
        print("  - Using smaller batch size (e.g., --batch_size 4)")
        print("  - Reducing model capacity (e.g., --hidden_dim 32)")
        print("  - Fewer training runs (e.g., --num_runs 3)")
        print("  - Results may not be reliable with such small data")
        print(f"{'!'*60}\n")
    
    # Adjust batch size if necessary
    effective_batch_size = min(args.batch_size, len(train_dataset))
    if effective_batch_size != args.batch_size:
        print(f"Note: Adjusting batch size from {args.batch_size} to {effective_batch_size} (dataset is small)")
    
    # Create data loaders
    # Only drop_last if we have enough samples
    drop_last = len(train_dataset) > effective_batch_size
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size, shuffle=False)
    
    # Setup model
    in_channels = len(args.selected_features) if args.selected_features else 10
    has_edge_attr = hasattr(train_dataset[0], 'edge_attr') and train_dataset[0].edge_attr is not None
    edge_dim = train_dataset[0].edge_attr.size(1) if has_edge_attr else None
    
    print(f"Model config - Input features: {in_channels}, Hidden: {args.hidden_dim}, Edge dim: {edge_dim}")
    
    # Run multiple training runs
    test_accuracies = []
    best_overall_model = None
    best_overall_acc = 0
    
    for run in range(args.num_runs):
        print(f"\n{'='*60}")
        print(f"Training run {run + 1}/{args.num_runs}")
        print(f"{'='*60}")
        
        # Initialize model
        model = GATv2GraphClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            out_channels=2,  # Binary classification
            edge_dim=edge_dim
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        # Training loop
        best_val_acc = 0
        best_model = copy.deepcopy(model)  # Initialize with current model
        patience_counter = 0
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, args.selected_features
            )
            
            # Only evaluate on validation set if it exists
            if len(val_dataset) > 0:
                val_acc, _ = evaluate(model, val_loader, device, args.selected_features)
            else:
                val_acc = train_acc  # Use training accuracy if no validation set
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Always keep the latest model if we haven't saved anything better
            if best_model is None:
                best_model = copy.deepcopy(model)
            
            # Print progress
            if epoch % args.print_every == 0:
                print(f"Epoch {epoch:03d} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if args.early_stopping and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Evaluate on test set
        model = best_model
        if len(test_dataset) > 0:
            test_acc, test_preds = evaluate(model, test_loader, device, args.selected_features)
        else:
            # If no test set, use validation accuracy
            test_acc = best_val_acc
            print("Warning: No test set available, using validation accuracy")
        
        test_accuracies.append(test_acc)
        
        print(f"Run {run + 1} - Best Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save best overall model
        if test_acc > best_overall_acc:
            best_overall_acc = test_acc
            best_overall_model = best_model
    
    # Print final results
    print(f"\n{'='*60}")
    print("Final Results")
    print(f"{'='*60}")
    print(f"Mean Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
    print(f"Best Test Accuracy: {max(test_accuracies):.4f}")
    
    # Save best model
    if args.save_model:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"gat_model_seed{args.seed}.pth"
        torch.save(best_overall_model.state_dict(), model_path)
        print(f"\nSaved best model to {model_path}")
        
        # Save results
        results = {
            "test_accuracies": test_accuracies,
            "mean_accuracy": float(np.mean(test_accuracies)),
            "std_accuracy": float(np.std(test_accuracies)),
            "best_accuracy": float(max(test_accuracies)),
            "args": vars(args)
        }
        
        results_path = output_dir / f"results_seed{args.seed}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GAT model on LCoT2Tree graphs")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="deepseek/final.json",
                       help="Path to JSONL data file")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer to use for counting tokens")
    parser.add_argument("--selected_features", type=int, nargs="+", default=[0, 2, 5, 7, 8],
                       help="List of feature indices to use (default: [0, 2, 5, 7, 8])")
    parser.add_argument("--edge_type", type=str, default="11_1",
                       help="Edge type configuration (default: 11_1)")
    
    # Data split parameters
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="Validation set ratio")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=64,
                       help="Hidden dimension size")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of training runs for averaging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--patience", type=int, default=10,
                       help="Patience for early stopping")
    parser.add_argument("--print_every", type=int, default=5,
                       help="Print progress every N epochs")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                       help="Directory to save models and results")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    
    args = parser.parse_args()
    
    print("Training GAT Model for Answer Correctness Prediction")
    print(f"{'='*60}")
    print(f"Data path: {args.data_path}")
    print(f"Features: {args.selected_features}")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    main(args)
