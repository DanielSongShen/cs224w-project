#!/usr/bin/env python3
"""Script to evaluate model performance and explanation quality

Usage:
    # Evaluate a specific model checkpoint
    python scripts/04_evaluate.py --model-path ./outputsNEW/encoder/amc-aime/undirected/models/simple_gin/simple_gin_model_seed42.pth --pt-file ./data/processed/deepseek/amc-aime/undirected/final_regraded_with_rev.pt --model-type simple_gin

    # Evaluate with custom parameters
    python scripts/04_evaluate.py --model-path ./outputs/models/simple_gin_model_seed42.pth --pt-file ./data/processed/deepseek/amc-aime/undirected/final_regraded_with_rev.pt --model-type simple_gin --batch-size 16 --test-ratio 0.1
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

from src.data.dataset import PreProcessedDataset, get_dataloaders
from src.models.gin import create_model
from src.training.trainer import GraphClassificationTrainer, TrainingConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_from_checkpoint(checkpoint_path: str, model_type: str, device: str = "auto"):
    """Load a model from a checkpoint or .pth file."""
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # This is a full checkpoint with metadata
        model_state = checkpoint["model_state_dict"]
        config = checkpoint.get("config", {})
    else:
        # This is just the model state dict
        model_state = checkpoint
        config = {}

    # Try to infer model configuration from state dict
    state_keys = set(model_state.keys())

    # Check if node encoder is used
    has_encoder = any("node_encoder" in k for k in state_keys)
    use_node_encoder = has_encoder

    print(f"  Detected node encoder: {use_node_encoder}")

    if use_node_encoder:
        # Try to infer encoder type
        if any("thought_embeddings" in k for k in state_keys):
            encoder_type = "text_aware"
        elif any("pe_scale" in k for k in state_keys):
            encoder_type = "robust"  # or "graph", we'll determine from input dim
        else:
            encoder_type = "tree"

        print(f"  Inferred encoder type: {encoder_type}")

        # Get input dimension from conv layer
        conv_keys = [k for k in state_keys if k.startswith("convs.0.nn.0.weight")]
        if conv_keys:
            weight_shape = model_state[conv_keys[0]].shape
            if len(weight_shape) >= 2:
                in_channels_inferred = weight_shape[1]
                print(f"  Inferred input channels: {in_channels_inferred}")
            else:
                in_channels_inferred = 32  # fallback
        else:
            in_channels_inferred = 32  # fallback
    else:
        encoder_type = "tree"  # doesn't matter
        # For no encoder, input channels should match the conv layer
        conv_keys = [k for k in state_keys if k.startswith("convs.0.nn.0.weight")]
        if conv_keys:
            weight_shape = model_state[conv_keys[0]].shape
            if len(weight_shape) >= 2:
                in_channels_inferred = weight_shape[1]
                print(f"  Inferred input channels (no encoder): {in_channels_inferred}")
            else:
                in_channels_inferred = 4  # fallback
        else:
            in_channels_inferred = 4  # fallback

    # Extract model parameters from config or use inferred values
    model_kwargs = {
        "model_type": model_type,
        "in_channels": in_channels_inferred,
        "hidden_channels": config.get("hidden_channels", 32),
        "out_channels": 2,  # Binary classification
        "edge_types": config.get("edge_types"),  # Will be set from sample
        "num_layers": config.get("num_layers", 3),
        "dropout": config.get("dropout", 0.1),
        "pool": config.get("pool", "add"),
        "use_node_encoder": use_node_encoder,
        "encoder_type": encoder_type,
    }

    # Create model
    model = create_model(**model_kwargs)
    model.load_state_dict(model_state)
    model.to(device)

    return model, config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GNN model on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and data arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth or .pt file)",
    )
    parser.add_argument(
        "--pt-file",
        type=str,
        required=True,
        help="Path to pre-processed dataset (.pt file)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="simple_gin",
        choices=["hetero_gin", "hetero_gat", "simple_gin"],
        help="Type of GNN model",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default=None,
        help="Directory containing text embeddings (for text-aware models)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for testing (should match training split)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: same as model directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed evaluation results",
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from: {args.pt_file}")
    if args.embedding_dir:
        print(f"Loading text embeddings from: {args.embedding_dir}")

    dataset = PreProcessedDataset(args.pt_file, embedding_dir=args.embedding_dir)

    print(f"\nDataset: {dataset}")
    summary = dataset.get_summary()
    print(f"  Total graphs: {summary['num_graphs']}")
    print(f"  Positive (correct): {summary['num_positive']}")
    print(f"  Negative (incorrect): {summary['num_negative']}")
    print(f"  Class ratio: {summary['class_ratio']:.2%}")

    # Get sample graph for model metadata
    sample_graph = dataset[0]
    in_channels = sample_graph["thought"].x.shape[1]
    edge_types = sample_graph.edge_types if hasattr(sample_graph, 'edge_types') else None

    print(f"\nGraph structure:")
    print(f"  Input features: {in_channels}")
    print(f"  Edge types: {edge_types}")

    # Load model
    print(f"\nLoading model from: {args.model_path}")
    try:
        model, model_config = load_model_from_checkpoint(
            args.model_path, args.model_type, args.device
        )
        print("  Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to create model with inferred parameters...")

        # Load state dict to inspect it
        state_dict = torch.load(args.model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
            config = state_dict.get("config", {})
        else:
            config = {}

        # Try to infer the correct model parameters
        model_kwargs = {
            "model_type": args.model_type,
            "in_channels": in_channels,  # Use actual dataset dimensions
            "hidden_channels": 32,  # Default
            "out_channels": 2,
            "edge_types": edge_types,
            "num_layers": 3,
            "dropout": 0.1,
            "pool": "add",
            "use_node_encoder": True,
            "encoder_type": "tree",  # Default
        }

        # Try different combinations of encoder settings
        model = None

        # First try with node encoder disabled (common for no_encoder models)
        try:
            model_kwargs["use_node_encoder"] = False
            model_kwargs["in_channels"] = in_channels  # Raw input features
            temp_model = create_model(**model_kwargs)
            temp_model.load_state_dict(state_dict)
            model = temp_model
            print(f"  Successfully created model with use_node_encoder=False")
        except Exception:
            pass

        # If that didn't work, try with node encoder enabled
        if model is None:
            encoder_types_to_try = ["tree", "graph", "robust", "text_aware", "text_aware_graph"]
            for encoder_type in encoder_types_to_try:
                try:
                    model_kwargs["use_node_encoder"] = True
                    model_kwargs["encoder_type"] = encoder_type
                    # Adjust input channels based on encoder type
                    if encoder_type in ["text_aware", "text_aware_graph"]:
                        # These use pre-computed embeddings, input should be embedding dim
                        model_kwargs["in_channels"] = 384  # Common embedding dimension
                    elif encoder_type == "tree":
                        model_kwargs["in_channels"] = in_channels
                    else:
                        model_kwargs["in_channels"] = in_channels

                    temp_model = create_model(**model_kwargs)
                    temp_model.load_state_dict(state_dict)
                    model = temp_model
                    print(f"  Successfully created model with encoder_type: {encoder_type}")
                    break
                except Exception:
                    continue

        if model is None:
            print(f"  Failed to load model with any configuration")
            print("  Model architecture may not match saved checkpoint")
            return

        model = model.to(args.device if args.device == "auto" else torch.device(args.device))

    # Get class counts for balanced evaluation
    class_counts = (summary['num_negative'], summary['num_positive'])

    # Create trainer for evaluation
    config = TrainingConfig(
        device=args.device,
        verbose=args.verbose,
    )
    trainer = GraphClassificationTrainer(model, config=config, class_counts=class_counts)

    # Create test data loader
    print(f"\nCreating test data loader (test_ratio={args.test_ratio})...")

    # We need to create a test loader with the same split as training
    # Since we don't have access to the exact split, we'll create a loader with test_ratio
    _, _, test_loader = get_dataloaders(
        dataset,
        batch_size=args.batch_size,
        train_ratio=0.8,  # Assuming standard split
        val_ratio=0.1,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=True,
    )

    print(f"  Test batches: {len(test_loader)} ({len(test_loader.dataset)} samples)")

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set")
    print(f"{'='*60}")

    test_results = trainer.evaluate(test_loader)

    # Save results
    results = {
        "model_path": args.model_path,
        "dataset_path": args.pt_file,
        "model_type": args.model_type,
        "evaluation_config": {
            "batch_size": args.batch_size,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "device": str(trainer.device),
        },
        "dataset_summary": {
            "num_graphs": summary["num_graphs"],
            "class_ratio": summary["class_ratio"],
            "num_positive": summary["num_positive"],
            "num_negative": summary["num_negative"],
        },
        "test_results": test_results,
    }

    # Save to JSON
    results_path = os.path.join(args.output_dir, f"evaluation_results_seed{args.seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Model: {args.model_type}")
    print(f"  Dataset: {os.path.basename(args.pt_file)}")
    print(f"  Test Samples: {len(test_loader.dataset)}")
    print(f"  Accuracy: {test_results['test_acc']:.4f}")
    print(f"  Precision: {test_results['precision']:.4f}")
    print(f"  Recall: {test_results['recall']:.4f}")
    print(f"  F1 Score: {test_results['f1']:.4f}")
    print(f"  Confusion Matrix: {test_results['confusion_matrix']}")

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()