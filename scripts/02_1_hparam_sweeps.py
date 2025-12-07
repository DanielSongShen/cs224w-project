#!/usr/bin/env python3
"""
Hyperparameter sweeps for GNN models on reasoning trace classification.

Usage:
    # Run grid search on a single dataset
    python scripts/02_1_hparam_sweeps.py \
        --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt \
        --search-type grid

    # Run random search with 50 trials
    python scripts/02_1_hparam_sweeps.py \
        --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt \
        --search-type random \
        --n-trials 50

    # Run on all three datasets
    python scripts/02_1_hparam_sweeps.py --all-datasets --search-type random --n-trials 30

    # Run specific model types only
    python scripts/02_1_hparam_sweeps.py \
        --pt-file ./data/processed/deepseek/amc-aime/undirected/processed/final_regraded_with_rev.pt \
        --models simple_gin hetero_gin
"""

import argparse
import itertools
import json
import os
import random
import signal
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch


class TrialTimeout(Exception):
    """Raised when a trial exceeds the time limit."""
    pass


def timeout_handler(signum, frame):
    raise TrialTimeout("Trial exceeded time limit")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import PreProcessedDataset, get_dataloaders
from src.models.gin import create_model
from src.training.trainer import GraphClassificationTrainer, TrainingConfig


# Default dataset paths
DATASET_PATHS = {
    "amc-aime": "./data/processed/deepseek/amc-aime/undirected/final_regraded_with_rev.pt",
    "olympiads": "./data/processed/deepseek/olympiads/undirected/final_regraded_with_rev.pt",
    "orca-math": "./data/processed/deepseek/orca-math/undirected/final_regraded_with_rev.pt",
    "combined": "./data/processed/deepseek/orca-math/undirected/encoder/final_regraded_with_rev.pt"
}

# Hyperparameter search spaces
HPARAM_GRID = {
    "hidden_channels": [32, 64],
    "num_layers": [3, 5],
    "dropout": [0.1, 0.5],
    "learning_rate": [1e-3],
    "batch_size": [32],
    "weight_decay": [0.0, 1e-4],
}

# Reduced grid for faster initial experiments
HPARAM_GRID_SMALL = {
    "hidden_channels": [32, 64],
    "num_layers": [2, 3],
    "dropout": [0.1, 0.3],
    "learning_rate": [1e-3, 1e-5],
    "batch_size": [32],
    "weight_decay": [0.0, 1e-4],
}

MODEL_TYPES = ["simple_gin", "hetero_gin", "hetero_gat"]


@dataclass
class HparamConfig:
    """Hyperparameter configuration for a single trial."""
    model_type: str
    hidden_channels: int
    num_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    weight_decay: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    config: HparamConfig
    dataset: str
    best_val_acc: float
    best_val_f1: float
    best_epoch: int
    test_acc: float
    test_f1: float
    test_precision: float
    test_recall: float
    train_time_seconds: float
    seed: int
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "config": self.config.to_dict(),
            "dataset": self.dataset,
            "best_val_acc": self.best_val_acc,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "test_acc": self.test_acc,
            "test_f1": self.test_f1,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "train_time_seconds": self.train_time_seconds,
            "seed": self.seed,
        }
        return result


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_grid_configs(
    model_types: List[str],
    hparam_grid: Dict[str, List],
) -> List[HparamConfig]:
    """Generate all combinations from the hyperparameter grid."""
    configs = []
    
    # Get all combinations of hyperparameters
    keys = list(hparam_grid.keys())
    values = [hparam_grid[k] for k in keys]
    
    for model_type in model_types:
        for combo in itertools.product(*values):
            hparams = dict(zip(keys, combo))
            config = HparamConfig(model_type=model_type, **hparams)
            configs.append(config)
    
    return configs


def generate_random_configs(
    model_types: List[str],
    hparam_grid: Dict[str, List],
    n_trials: int,
) -> List[HparamConfig]:
    """Generate random hyperparameter configurations."""
    configs = []
    
    for _ in range(n_trials):
        model_type = random.choice(model_types)
        hparams = {k: random.choice(v) for k, v in hparam_grid.items()}
        config = HparamConfig(model_type=model_type, **hparams)
        configs.append(config)
    
    return configs


def run_trial(
    config: HparamConfig,
    dataset: PreProcessedDataset,
    dataset_name: str,
    class_counts: Tuple[int, int],
    seed: int = 42,
    epochs: int = 100,
    patience: int = 20,
    device: str = "auto",
    verbose: bool = False,
    timeout_seconds: Optional[int] = None,
) -> TrialResult:
    """Run a single hyperparameter trial."""
    import time
    
    # Set up timeout if specified (Unix only)
    if timeout_seconds and hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    
    try:
        set_seed(seed)
        
        # Get sample graph for metadata
        sample_graph = dataset[0]
        in_channels = sample_graph["thought"].x.shape[1]
        edge_types = sample_graph.edge_types
        
        # Create data loaders
        train_loader, val_loader, test_loader = get_dataloaders(
            dataset,
            batch_size=config.batch_size,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=seed,
            stratify=True,
        )
        
        # Create model
        model = create_model(
            model_type=config.model_type,
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            out_channels=2,
            edge_types=edge_types,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        
        # Create training config
        train_config = TrainingConfig(
            epochs=epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=patience,
            log_interval=epochs + 1,  # Suppress logging
            verbose=verbose,
            device=device,
        )
        
        # Train with balanced loss
        trainer = GraphClassificationTrainer(model, config=train_config, class_counts=class_counts)
        
        start_time = time.time()
        metrics = trainer.fit(train_loader, val_loader)
        train_time = time.time() - start_time
        
        # Evaluate
        test_results = trainer.evaluate(test_loader)
        
        return TrialResult(
            config=config,
            dataset=dataset_name,
            best_val_acc=metrics.best_val_acc,
            best_val_f1=metrics.best_val_f1,
            best_epoch=metrics.best_epoch,
            test_acc=test_results["test_acc"],
            test_f1=test_results["f1"],
            test_precision=test_results["precision"],
            test_recall=test_results["recall"],
            train_time_seconds=train_time,
            seed=seed,
        )
    finally:
        # Cancel the alarm
        if timeout_seconds and hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def run_sweep(
    configs: List[HparamConfig],
    dataset_paths: Dict[str, str],
    output_dir: str,
    seed: int = 42,
    epochs: int = 100,
    patience: int = 20,
    device: str = "auto",
    verbose: bool = False,
    resume_from: int = 1,
    timeout_seconds: Optional[int] = None,
) -> List[TrialResult]:
    """Run hyperparameter sweep across all configs and datasets."""
    
    results = []
    total_trials = len(configs) * len(dataset_paths)
    trial_num = 0
    skipped = 0
    
    # Load all datasets once and compute class counts
    datasets = {}
    dataset_class_counts = {}
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            print(f"Loading dataset: {name}")
            ds = PreProcessedDataset(path)
            datasets[name] = ds
            # Compute class counts for balanced loss
            summary = ds.get_summary()
            dataset_class_counts[name] = (summary['num_negative'], summary['num_positive'])
            print(f"  Class counts: {dataset_class_counts[name]}")
        else:
            print(f"Warning: Dataset not found: {path}")
    
    if not datasets:
        raise ValueError("No datasets found!")
    
    print(f"\n{'='*60}")
    print(f"Starting hyperparameter sweep")
    print(f"  Configurations: {len(configs)}")
    print(f"  Datasets: {list(datasets.keys())}")
    print(f"  Total trials: {total_trials}")
    if resume_from > 1:
        print(f"  Resuming from trial: {resume_from}")
    if timeout_seconds:
        print(f"  Timeout per trial: {timeout_seconds}s ({timeout_seconds//60}min)")
    print(f"{'='*60}\n")
    
    # Run trials
    for dataset_name, dataset in datasets.items():
        for config in configs:
            trial_num += 1
            
            # Skip trials before resume point
            if trial_num < resume_from:
                skipped += 1
                continue
            
            print(f"Trial {trial_num}/{total_trials}: {config.model_type} on {dataset_name}")
            print(f"  Config: hidden={config.hidden_channels}, layers={config.num_layers}, "
                  f"dropout={config.dropout}, lr={config.learning_rate}, "
                  f"batch={config.batch_size}, wd={config.weight_decay}")
            
            try:
                result = run_trial(
                    config=config,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    class_counts=dataset_class_counts[dataset_name],
                    seed=seed,
                    epochs=epochs,
                    patience=patience,
                    device=device,
                    verbose=verbose,
                    timeout_seconds=timeout_seconds,
                )
                results.append(result)
                
                print(f"  Val F1: {result.best_val_f1:.4f}, Val Acc: {result.best_val_acc:.4f} (epoch {result.best_epoch})")
                print(f"  Test Acc: {result.test_acc:.4f}, F1: {result.test_f1:.4f}")
                print(f"  Time: {result.train_time_seconds:.1f}s\n")
                
            except TrialTimeout:
                print(f"  TIMEOUT: Trial exceeded {timeout_seconds}s limit, skipping\n")
                continue
            except Exception as e:
                print(f"  ERROR: {e}\n")
                continue
            
            # Save intermediate results
            save_results(results, output_dir)
    
    if skipped > 0:
        print(f"Skipped {skipped} trials (resumed from trial {resume_from})")
    
    return results


def save_results(results: List[TrialResult], output_dir: str):
    """Save sweep results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to dicts
    results_data = [r.to_dict() for r in results]
    
    # Save full results
    output_path = os.path.join(output_dir, "sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)


def analyze_results(results: List[TrialResult]) -> Dict[str, Any]:
    """Analyze sweep results and find best configurations."""
    
    if not results:
        return {}
    
    # Group by dataset
    by_dataset: Dict[str, List[TrialResult]] = {}
    for r in results:
        if r.dataset not in by_dataset:
            by_dataset[r.dataset] = []
        by_dataset[r.dataset].append(r)
    
    # Group by model type
    by_model: Dict[str, List[TrialResult]] = {}
    for r in results:
        if r.config.model_type not in by_model:
            by_model[r.config.model_type] = []
        by_model[r.config.model_type].append(r)
    
    analysis = {
        "total_trials": len(results),
        "datasets": list(by_dataset.keys()),
        "model_types": list(by_model.keys()),
        "best_per_dataset": {},
        "best_per_model": {},
        "overall_best": None,
    }
    
    # Best per dataset
    for dataset, dataset_results in by_dataset.items():
        best = max(dataset_results, key=lambda r: r.test_acc)
        analysis["best_per_dataset"][dataset] = {
            "test_acc": best.test_acc,
            "test_f1": best.test_f1,
            "config": best.config.to_dict(),
        }
    
    # Best per model type
    for model_type, model_results in by_model.items():
        best = max(model_results, key=lambda r: r.test_acc)
        analysis["best_per_model"][model_type] = {
            "test_acc": best.test_acc,
            "test_f1": best.test_f1,
            "config": best.config.to_dict(),
        }
    
    # Overall best
    overall_best = max(results, key=lambda r: r.test_acc)
    analysis["overall_best"] = {
        "test_acc": overall_best.test_acc,
        "test_f1": overall_best.test_f1,
        "dataset": overall_best.dataset,
        "config": overall_best.config.to_dict(),
    }
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis summary."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("=" * 60)
    
    print(f"\nTotal trials: {analysis['total_trials']}")
    print(f"Datasets: {analysis['datasets']}")
    print(f"Model types: {analysis['model_types']}")
    
    print("\n" + "-" * 60)
    print("Best per Dataset:")
    print("-" * 60)
    for dataset, best in analysis["best_per_dataset"].items():
        print(f"\n  {dataset}:")
        print(f"    Test Acc: {best['test_acc']:.4f}, F1: {best['test_f1']:.4f}")
        print(f"    Config: {best['config']['model_type']}, "
              f"hidden={best['config']['hidden_channels']}, "
              f"layers={best['config']['num_layers']}, "
              f"dropout={best['config']['dropout']}, "
              f"lr={best['config']['learning_rate']}")
    
    print("\n" + "-" * 60)
    print("Best per Model Type:")
    print("-" * 60)
    for model_type, best in analysis["best_per_model"].items():
        print(f"\n  {model_type}:")
        print(f"    Test Acc: {best['test_acc']:.4f}, F1: {best['test_f1']:.4f}")
        print(f"    Config: hidden={best['config']['hidden_channels']}, "
              f"layers={best['config']['num_layers']}, "
              f"dropout={best['config']['dropout']}, "
              f"lr={best['config']['learning_rate']}")
    
    print("\n" + "-" * 60)
    print("Overall Best:")
    print("-" * 60)
    best = analysis["overall_best"]
    print(f"  Dataset: {best['dataset']}")
    print(f"  Test Acc: {best['test_acc']:.4f}, F1: {best['test_f1']:.4f}")
    print(f"  Config: {best['config']['model_type']}, "
          f"hidden={best['config']['hidden_channels']}, "
          f"layers={best['config']['num_layers']}, "
          f"dropout={best['config']['dropout']}, "
          f"lr={best['config']['learning_rate']}, "
          f"batch={best['config']['batch_size']}, "
          f"wd={best['config']['weight_decay']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweeps for GNN models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument(
        "--pt-file",
        type=str,
        help="Path to single pre-processed .pt file",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run on all three datasets (amc-aime, olympiads, orca-math)",
    )
    
    # Search arguments
    parser.add_argument(
        "--search-type",
        type=str,
        default="grid",
        choices=["grid", "grid_small", "random"],
        help="Type of hyperparameter search",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials for random search",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=MODEL_TYPES,
        choices=MODEL_TYPES,
        help="Model types to include in sweep",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum epochs per trial",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="Resume from this trial number (1-indexed)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per trial in seconds (default: 7200 = 2 hours)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/sweeps",
        help="Directory to save sweep results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose training output",
    )
    
    args = parser.parse_args()
    
    # Determine datasets to use
    if args.all_datasets:
        dataset_paths = DATASET_PATHS
    elif args.pt_file:
        dataset_name = Path(args.pt_file).parent.parent.parent.name
        dataset_paths = {dataset_name: args.pt_file}
    else:
        parser.error("Must specify --pt-file or --all-datasets")
    
    # Generate configurations
    if args.search_type == "grid":
        configs = generate_grid_configs(args.models, HPARAM_GRID)
    elif args.search_type == "grid_small":
        configs = generate_grid_configs(args.models, HPARAM_GRID_SMALL)
    else:  # random
        configs = generate_random_configs(args.models, HPARAM_GRID, args.n_trials)
    
    print(f"Generated {len(configs)} configurations")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"sweep_{timestamp}")
    
    # Save sweep config
    os.makedirs(output_dir, exist_ok=True)
    sweep_config = {
        "search_type": args.search_type,
        "n_configs": len(configs),
        "models": args.models,
        "datasets": list(dataset_paths.keys()),
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(os.path.join(output_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_config, f, indent=2)
    
    # Run sweep
    results = run_sweep(
        configs=configs,
        dataset_paths=dataset_paths,
        output_dir=output_dir,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        device=args.device,
        verbose=args.verbose,
        resume_from=args.resume_from,
        timeout_seconds=args.timeout,
    )
    
    # Analyze and print results
    if results:
        analysis = analyze_results(results)
        print_analysis(analysis)
        
        # Save analysis
        with open(os.path.join(output_dir, "analysis.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    else:
        print("No results collected!")


if __name__ == "__main__":
    main()
