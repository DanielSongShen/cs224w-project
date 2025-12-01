"""Training loop for GNN models on heterogeneous graph classification"""

import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    
    # Training parameters
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    
    # Early stopping
    patience: int = 20
    min_delta: float = 1e-4
    
    # Logging
    log_interval: int = 10
    verbose: bool = True
    
    # Checkpointing
    checkpoint_dir: Optional[str] = None
    save_best: bool = True
    
    # Device
    device: str = "auto"  # "auto", "cuda", "cpu", or "mps"
    
    def get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


@dataclass
class TrainingMetrics:
    """Container for training metrics over time."""
    
    train_losses: List[float] = field(default_factory=list)
    train_accs: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accs: List[float] = field(default_factory=list)
    best_val_acc: float = 0.0
    best_epoch: int = 0
    
    # Per-class metrics
    val_class_accs: List[Dict[int, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "val_class_accs": self.val_class_accs,
        }


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4, mode: str = "max"):
        """
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: "max" for metrics like accuracy, "min" for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class GraphClassificationTrainer:
    """
    Trainer for graph-level binary classification on heterogeneous graphs.
    
    Handles the training loop, validation, early stopping, and checkpointing
    for GNN models that perform graph-level classification.
    
    Example:
        >>> from src.data.dataset import ReasoningTraceDataset, get_dataloaders
        >>> from src.models.hetero_gnn import HeteroGNN
        >>> 
        >>> dataset = ReasoningTraceDataset(root='./data', raw_filepath='./data/final.json')
        >>> train_loader, val_loader, test_loader = get_dataloaders(dataset, batch_size=32)
        >>> 
        >>> model = HeteroGNN(...)
        >>> trainer = GraphClassificationTrainer(model, config=TrainingConfig(epochs=100))
        >>> metrics = trainer.fit(train_loader, val_loader)
        >>> test_results = trainer.evaluate(test_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model for graph classification
            config: Training configuration (uses defaults if None)
            optimizer: Custom optimizer (creates Adam if None)
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function (uses CrossEntropyLoss if None)
        """
        self.config = config or TrainingConfig()
        self.device = self.config.get_device()
        self.model = model.to(self.device)
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            self.optimizer = optimizer
            
        self.scheduler = scheduler
        
        # Initialize criterion
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        # State tracking
        self.best_model_state: Optional[Dict] = None
        self.metrics = TrainingMetrics()
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            mode="max",
        )
        
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Run one training epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass - model should handle HeteroData
            out = self.model(batch)
            
            # Get labels - handle both batched and single graph cases
            if hasattr(batch, 'y'):
                labels = batch.y
                if labels.dim() > 1:
                    labels = labels.view(-1)
            else:
                raise ValueError("Batch does not have labels (y attribute)")
            
            # Compute loss
            loss = self.criterion(out, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict[int, float]]:
        """
        Run validation.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Tuple of (average loss, accuracy, per-class accuracy dict)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class tracking
        class_correct: Dict[int, int] = {}
        class_total: Dict[int, int] = {}
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            out = self.model(batch)
            
            # Get labels
            labels = batch.y
            if labels.dim() > 1:
                labels = labels.view(-1)
            
            # Compute loss
            loss = self.criterion(out, labels)
            
            # Track metrics
            total_loss += loss.item() * labels.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            # Per-class metrics
            for i in range(labels.size(0)):
                label = labels[i].item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_total[label] += 1
                if pred[i] == labels[i]:
                    class_correct[label] += 1
        
        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute per-class accuracy
        class_acc = {
            label: class_correct[label] / class_total[label]
            for label in class_total
            if class_total[label] > 0
        }
        
        return avg_loss, accuracy, class_acc
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingMetrics:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            
        Returns:
            TrainingMetrics containing training history
        """
        # Reset state
        self.metrics = TrainingMetrics()
        self.early_stopping.reset()
        self.best_model_state = None
        
        if self.config.verbose:
            print(f"Training on {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("-" * 60)
        
        for epoch in range(1, self.config.epochs + 1):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader)
            self.metrics.train_losses.append(train_loss)
            self.metrics.train_accs.append(train_acc)
            
            # Validation
            val_loss, val_acc, class_acc = self._validate(val_loader)
            self.metrics.val_losses.append(val_loss)
            self.metrics.val_accs.append(val_acc)
            self.metrics.val_class_accs.append(class_acc)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Track best model
            if val_acc > self.metrics.best_val_acc:
                self.metrics.best_val_acc = val_acc
                self.metrics.best_epoch = epoch
                if self.config.save_best:
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
            
            # Logging
            if self.config.verbose and epoch % self.config.log_interval == 0:
                class_acc_str = ", ".join(
                    f"Class {k}: {v:.4f}" for k, v in sorted(class_acc.items())
                )
                print(
                    f"Epoch {epoch:03d} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"{class_acc_str}"
                )
            
            # Early stopping check
            if self.early_stopping(val_acc):
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.config.verbose:
                print(f"Restored best model from epoch {self.metrics.best_epoch}")
        
        # Save checkpoint if configured
        if self.config.checkpoint_dir is not None:
            self.save_checkpoint(self.config.checkpoint_dir)
        
        if self.config.verbose:
            print("-" * 60)
            print(f"Training complete. Best Val Acc: {self.metrics.best_val_acc:.4f} at epoch {self.metrics.best_epoch}")
        
        return self.metrics
    
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Dictionary with test metrics
        """
        test_loss, test_acc, class_acc = self._validate(test_loader)
        
        # Collect all predictions and labels for additional metrics
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            probs = F.softmax(out, dim=1)
            
            labels = batch.y
            if labels.dim() > 1:
                labels = labels.view(-1)
            
            all_preds.extend(out.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs[:, 1].cpu().tolist())  # Probability of positive class
        
        # Compute additional metrics
        results = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "class_acc": class_acc,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "num_samples": len(all_labels),
        }
        
        # Compute confusion matrix components
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        
        results["confusion_matrix"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results["precision"] = precision
        results["recall"] = recall
        results["f1"] = f1
        
        if self.config.verbose:
            print(f"Test Results:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            for label, acc in sorted(class_acc.items()):
                print(f"  Class {label} Accuracy: {acc:.4f}")
        
        return results
    
    def save_checkpoint(self, path: str, filename: str = "checkpoint.pt"):
        """
        Save model checkpoint.
        
        Args:
            path: Directory to save checkpoint
            filename: Checkpoint filename
        """
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.to_dict(),
            "config": {
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "patience": self.config.patience,
            },
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if self.config.verbose:
            print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, path: str, filename: str = "checkpoint.pt"):
        """
        Load model checkpoint.
        
        Args:
            path: Directory containing checkpoint
            filename: Checkpoint filename
        """
        filepath = os.path.join(path, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.config.verbose:
            print(f"Checkpoint loaded from {filepath}")


def train_with_cross_validation(
    model_factory: Callable[[], nn.Module],
    dataset,
    n_folds: int = 5,
    batch_size: int = 32,
    config: Optional[TrainingConfig] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train model with k-fold cross-validation.
    
    Args:
        model_factory: Callable that returns a new model instance
        dataset: ReasoningTraceDataset or similar
        n_folds: Number of cross-validation folds
        batch_size: Batch size for data loaders
        config: Training configuration
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    config = config or TrainingConfig()
    
    # Get labels for stratification
    labels = [dataset[i].y.item() for i in range(len(dataset))]
    indices = np.arange(len(dataset))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Create data loaders for this fold
        train_subset = [dataset[i] for i in train_idx]
        val_subset = [dataset[i] for i in val_idx]
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Create fresh model
        model = model_factory()
        
        # Train
        trainer = GraphClassificationTrainer(model, config=config)
        metrics = trainer.fit(train_loader, val_loader)
        
        # Evaluate on validation fold
        results = trainer.evaluate(val_loader)
        results["fold"] = fold + 1
        results["train_size"] = len(train_idx)
        results["val_size"] = len(val_idx)
        fold_results.append(results)
    
    # Aggregate results
    all_accs = [r["test_acc"] for r in fold_results]
    all_f1s = [r["f1"] for r in fold_results]
    
    summary = {
        "fold_results": fold_results,
        "mean_acc": np.mean(all_accs),
        "std_acc": np.std(all_accs),
        "mean_f1": np.mean(all_f1s),
        "std_f1": np.std(all_f1s),
    }
    
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Accuracy: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
    print(f"F1 Score: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    
    return summary


if __name__ == "__main__":
    # Example usage demonstrating the trainer
    import argparse
    
    parser = argparse.ArgumentParser(description="Train GNN for graph classification")
    parser.add_argument("--data-root", type=str, default="./data/processed")
    parser.add_argument("--raw-filepath", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="./outputs/checkpoints")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Import dataset utilities
    from src.data.dataset import ReasoningTraceDataset, get_dataloaders
    
    print("Loading dataset...")
    dataset = ReasoningTraceDataset(
        root=args.data_root,
        raw_filepath=args.raw_filepath,
    )
    
    print(f"Dataset: {dataset}")
    print(dataset.get_summary())
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Note: You need to implement a model that handles HeteroData
    # Example model creation would look like:
    #
    # from src.models.hetero_gnn import HeteroGraphClassifier
    # model = HeteroGraphClassifier(
    #     in_channels=3,  # level, category, thought_idx
    #     hidden_channels=64,
    #     out_channels=2,  # binary classification
    #     metadata=dataset[0].metadata(),
    # )
    #
    # For now, print a message about the expected interface
    print("\nTo train, create a model that:")
    print("  1. Takes HeteroData as input")
    print("  2. Returns logits of shape [batch_size, num_classes]")
    print("  3. Example: model = HeteroGraphClassifier(...)")
    print("\nThen run:")
    print("  trainer = GraphClassificationTrainer(model, config=TrainingConfig(...))")
    print("  metrics = trainer.fit(train_loader, val_loader)")
    print("  results = trainer.evaluate(test_loader)")
