#!/usr/bin/env python3
"""
Train GNN model for graph-level binary classification on reasoning traces.

Usage:
    # Load from pre-processed .pt file (recommended)
    python scripts/02_train_model.py --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt

    # Load from raw JSON (will process and cache)
    python scripts/02_train_model.py --raw-filepath ./data/processed/deepseek/amc-aime/final.json

    # With custom parameters
    python scripts/02_train_model.py \
        --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt \
        --model-type hetero_gin \
        --hidden-channels 64 \
        --num-layers 3 \
        --epochs 100 \
        --batch-size 32 \
        --lr 0.001

    # With text embeddings for tree format (after running 03_precompute_embeddings.py)
    python scripts/02_train_model.py \
        --pt-file ./data/processed/deepseek/cn_k12/directed/final.pt \
        --embedding-dir ./data/processed/deepseek/cn_k12/ \
        --encoder-type text_aware \
        --model-type simple_gin

    # With text embeddings for graph format (after running 03_precompute_embeddings.py)
    python scripts/02_train_model.py \
        --pt-file ./data/processed/deepseek/cn_k12/graph_format/final.pt \
        --embedding-dir ./data/processed/deepseek/cn_k12/ \
        --encoder-type text_aware_graph \
        --model-type simple_gin

    # Cross-validation
    python scripts/02_train_model.py \
        --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt \
        --cross-validation --n-folds 5
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import ReasoningTraceDataset, PreProcessedDataset, get_dataloaders
from src.models.gin import create_model, HeteroGraphClassifier, SimpleHeteroGNN
from src.training.trainer import (
    GraphClassificationTrainer,
    TrainingConfig,
    train_with_cross_validation,
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train GNN for graph-level binary classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--pt-file",
        type=str,
        help="Path to pre-processed .pt file (recommended, faster)",
    )
    data_group.add_argument(
        "--raw-filepath",
        type=str,
        help="Path to raw JSONL file containing reasoning traces (will process and cache)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/processed_graphs",
        help="Root directory for processed dataset (only used with --raw-filepath)",
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reprocessing of dataset even if cached (only used with --raw-filepath)",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="Directory containing thought_embeddings.pt and embedding_index.json (for text-aware encoding)",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple_gin",
        choices=["hetero_gin", "hetero_gat", "simple_gin"],
        help="Type of GNN model to use",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=64,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of message passing layers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--no-node-encoder",
        action="store_true",
        help="Disable learned node encoder (use raw features with linear projection)",
    )
    parser.add_argument(
        "--encoder-type",
        type=str,
        default="tree",
        choices=["tree", "graph", "robust", "text_aware", "text_aware_graph"],
        help="Node encoder type: 'tree' (embedding), 'graph' (for graph format), 'robust' (MLP + LayerNorm), 'text_aware' (with sentence embeddings for tree format), 'text_aware_graph' (with sentence embeddings for graph format) (default: tree)",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default="add",
        choices=["mean", "add"],
        help="Global pooling strategy for hetero models (default: add)",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-3,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    
    # Split arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing",
    )
    
    # Cross-validation
    parser.add_argument(
        "--cross-validation",
        action="store_true",
        help="Use k-fold cross-validation instead of single split",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/models",
        help="Directory to save model checkpoints and results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N epochs",
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GNN Training for Graph Classification")
    print("=" * 60)
    
    # Load dataset
    if args.pt_file:
        print(f"\nLoading pre-processed dataset from: {args.pt_file}")
        if args.embedding_dir:
            print(f"Loading text embeddings from: {args.embedding_dir}")
        dataset = PreProcessedDataset(args.pt_file, embedding_dir=args.embedding_dir)
    else:
        print(f"\nLoading dataset from: {args.raw_filepath}")
        dataset = ReasoningTraceDataset(
            root=args.data_root,
            raw_filepath=args.raw_filepath,
            include_reverse_edges=True,
            force_reload=args.force_reload,
        )
    
    print(f"\nDataset: {dataset}")
    summary = dataset.get_summary()
    print(f"  Total graphs: {summary['num_graphs']}")
    print(f"  Positive (correct): {summary['num_positive']}")
    print(f"  Negative (incorrect): {summary['num_negative']}")
    print(f"  Class ratio: {summary['class_ratio']:.2%}")
    print(f"  Avg nodes: {summary['avg_nodes']:.1f}")
    print(f"  Edge types: {len(summary['edge_types'])}")
    
    # Get sample graph for metadata
    sample_graph = dataset[0]
    in_channels = sample_graph["thought"].x.shape[1]
    edge_types = sample_graph.edge_types
    
    print(f"\nGraph structure:")
    print(f"  Input features: {in_channels}")
    print(f"  Edge types: {edge_types}")
    
    # Create training config
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        log_interval=args.log_interval,
        checkpoint_dir=args.output_dir,
        save_best=True,
        device=args.device,
        verbose=True,
    )
    
    if args.cross_validation:
        # Cross-validation mode
        print(f"\n{'='*60}")
        print(f"Running {args.n_folds}-fold cross-validation")
        print(f"{'='*60}")
        
        def model_factory():
            return create_model(
                model_type=args.model_type,
                in_channels=in_channels,
                hidden_channels=args.hidden_channels,
                out_channels=2,
                edge_types=edge_types,
                num_layers=args.num_layers,
                dropout=args.dropout,
                pool=args.pool,
                use_node_encoder=not args.no_node_encoder,
                encoder_type=args.encoder_type,
            )
        
        cv_results = train_with_cross_validation(
            model_factory=model_factory,
            dataset=dataset,
            n_folds=args.n_folds,
            batch_size=args.batch_size,
            config=config,
            seed=args.seed,
            class_counts=(summary['num_negative'], summary['num_positive']),
        )
        
        # Save cross-validation results
        results_path = os.path.join(args.output_dir, f"cv_results_seed{args.seed}.json")
        with open(results_path, "w") as f:
            # Convert to serializable format
            cv_results_serializable = {
                "mean_acc": float(cv_results["mean_acc"]),
                "std_acc": float(cv_results["std_acc"]),
                "mean_f1": float(cv_results["mean_f1"]),
                "std_f1": float(cv_results["std_f1"]),
                "fold_results": [
                    {k: v for k, v in r.items() if k not in ["predictions", "labels", "probabilities"]}
                    for r in cv_results["fold_results"]
                ],
            }
            json.dump(cv_results_serializable, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
    else:
        # Single train/val/test split mode
        print(f"\n{'='*60}")
        print(f"Training with {args.train_ratio:.0%}/{args.val_ratio:.0%}/{args.test_ratio:.0%} split")
        print(f"{'='*60}")
        
        # Get data loaders
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            stratify=True,
        )
        
        print(f"\nData splits:")
        print(f"  Train batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
        print(f"  Val batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")
        print(f"  Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")
        
        # Create model
        print(f"\nCreating {args.model_type} model...")
        model = create_model(
            model_type=args.model_type,
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=2,
            edge_types=edge_types,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pool=args.pool,
            use_node_encoder=not args.no_node_encoder,
            encoder_type=args.encoder_type,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        # Create trainer and train
        # Pass class counts for balanced loss weighting
        class_counts = (summary['num_negative'], summary['num_positive'])
        trainer = GraphClassificationTrainer(model, config=config, class_counts=class_counts)
        
        print(f"\nTraining...")
        print("-" * 60)
        metrics = trainer.fit(train_loader, val_loader)
        
        # Evaluate on test set
        print(f"\n{'='*60}")
        print("Evaluating on test set")
        print(f"{'='*60}")
        test_results = trainer.evaluate(test_loader)
        
        # Save model
        model_path = os.path.join(args.output_dir, f"{args.model_type}_model_seed{args.seed}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save results
        results = {
            "config": {
                "model_type": args.model_type,
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
            "dataset": {
                "raw_filepath": args.raw_filepath,
                "num_graphs": summary["num_graphs"],
                "class_ratio": summary["class_ratio"],
            },
            "training": {
                "best_epoch": metrics.best_epoch,
                "best_val_acc": float(metrics.best_val_acc),
                "best_val_f1": float(metrics.best_val_f1),
            },
            "test_results": {
                "accuracy": float(test_results["test_acc"]),
                "precision": float(test_results["precision"]),
                "recall": float(test_results["recall"]),
                "f1": float(test_results["f1"]),
                "confusion_matrix": test_results["confusion_matrix"],
            },
        }
        
        results_path = os.path.join(args.output_dir, f"results_seed{args.seed}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"  Best Val F1: {metrics.best_val_f1:.4f} (Acc: {metrics.best_val_acc:.4f}, epoch {metrics.best_epoch})")
        print(f"  Test Accuracy: {test_results['test_acc']:.4f}")
        print(f"  Test F1: {test_results['f1']:.4f}")


if __name__ == "__main__":
    main()

