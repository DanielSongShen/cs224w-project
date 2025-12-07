#!/usr/bin/env python3
"""
Analyze hyperparameter sweep results and identify top configurations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results from both sweep files
sweep_files = [
    "outputs/sweeps/sweep_20251204_120249/sweep_results.json",
    "outputs/sweeps/sweep_20251205_120730/sweep_results.json",
]

all_results = []
for f in sweep_files:
    with open(f) as fp:
        all_results.extend(json.load(fp))

print(f"Total trials loaded: {len(all_results)}")

# Convert to DataFrame for easier analysis
rows = []
for r in all_results:
    row = {
        "hidden_channels": r["config"]["hidden_channels"],
        "num_layers": r["config"]["num_layers"],
        "dropout": r["config"]["dropout"],
        "learning_rate": r["config"]["learning_rate"],
        "batch_size": r["config"]["batch_size"],
        "weight_decay": r["config"]["weight_decay"],
        "test_acc": r["test_acc"],
        "test_f1": r["test_f1"],
        "test_precision": r["test_precision"],
        "test_recall": r["test_recall"],
        "best_val_acc": r["best_val_acc"],
        "best_val_f1": r["best_val_f1"],
        "best_epoch": r["best_epoch"],
        "train_time_seconds": r["train_time_seconds"],
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ============================================================
# TOP 10 CONFIGURATIONS
# ============================================================
print("\n" + "=" * 60)
print("TOP 10 CONFIGURATIONS BY TEST ACCURACY")
print("=" * 60)

top10 = df.nlargest(10, "test_acc")[
    ["hidden_channels", "num_layers", "dropout", "learning_rate", 
     "batch_size", "weight_decay", "test_acc", "test_f1"]
]
print(top10.to_string(index=False))

print("\n" + "=" * 60)
print("TOP 10 CONFIGURATIONS BY TEST F1")
print("=" * 60)

top10_f1 = df.nlargest(10, "test_f1")[
    ["hidden_channels", "num_layers", "dropout", "learning_rate", 
     "batch_size", "weight_decay", "test_acc", "test_f1"]
]
print(top10_f1.to_string(index=False))

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Best test accuracy: {df['test_acc'].max():.4f}")
print(f"Best test F1: {df['test_f1'].max():.4f}")
print(f"Mean test accuracy: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
print(f"Mean test F1: {df['test_f1'].mean():.4f} ± {df['test_f1'].std():.4f}")
print(f"Mean training time: {df['train_time_seconds'].mean():.1f}s")

# ============================================================
# HYPERPARAMETER ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("MEAN TEST ACCURACY BY HYPERPARAMETER")
print("=" * 60)

for col in ["hidden_channels", "num_layers", "dropout", "learning_rate", "batch_size", "weight_decay"]:
    print(f"\n{col}:")
    grouped = df.groupby(col)["test_acc"].agg(["mean", "std", "count"])
    for val, row in grouped.iterrows():
        print(f"  {val}: {row['mean']:.4f} ± {row['std']:.4f} (n={int(row['count'])})")

# ============================================================
# HELPER FUNCTION FOR PLOTS
# ============================================================
def create_plots_for_metric(df, metric_col, metric_name, suffix):
    """Create all plots for a given metric."""
    
    # Boxplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Hyperparameter Sweep Analysis - {metric_name}", fontsize=14, fontweight="bold")
    
    params = [
        ("hidden_channels", "Hidden Channels"),
        ("num_layers", "Number of Layers"),
        ("dropout", "Dropout"),
        ("learning_rate", "Learning Rate"),
        ("batch_size", "Batch Size"),
        ("weight_decay", "Weight Decay"),
    ]
    
    for idx, (col, title) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        df.boxplot(column=metric_col, by=col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel(metric_name)
        plt.suptitle("")
    
    plt.tight_layout()
    plt.savefig(f"outputs/sweeps/hparam_boxplots_{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved boxplots to outputs/sweeps/hparam_boxplots_{suffix}.png")
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    np.random.seed(42)  # For reproducible jitter
    scatter = ax.scatter(
        df["num_layers"] + np.random.uniform(-0.1, 0.1, len(df)),
        df["hidden_channels"] + np.random.uniform(-2, 2, len(df)),
        c=df[metric_col],
        cmap="RdYlGn",
        alpha=0.6,
        s=50,
    )
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Hidden Channels")
    ax.set_title(f"{metric_name} by Architecture (layers × hidden)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(metric_name)
    plt.savefig(f"outputs/sweeps/architecture_scatter_{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved scatter plot to outputs/sweeps/architecture_scatter_{suffix}.png")
    
    # Architecture heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    pivot = df.pivot_table(values=metric_col, index="hidden_channels", columns="num_layers", aggfunc="mean")
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Hidden Channels")
    ax.set_title(f"Mean {metric_name}: Layers × Hidden Channels")
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label=f"Mean {metric_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/sweeps/architecture_heatmap_{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved heatmap to outputs/sweeps/architecture_heatmap_{suffix}.png")
    
    # LR × Dropout heatmap
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot_lr = df.pivot_table(values=metric_col, index="learning_rate", columns="dropout", aggfunc="mean")
    im = ax.imshow(pivot_lr.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot_lr.columns)))
    ax.set_xticklabels(pivot_lr.columns)
    ax.set_yticks(range(len(pivot_lr.index)))
    ax.set_yticklabels([f"{x:.0e}" for x in pivot_lr.index])
    ax.set_xlabel("Dropout")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"Mean {metric_name}: Learning Rate × Dropout")
    
    for i in range(len(pivot_lr.index)):
        for j in range(len(pivot_lr.columns)):
            val = pivot_lr.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=12, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label=f"Mean {metric_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/sweeps/lr_dropout_heatmap_{suffix}.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved LR×Dropout heatmap to outputs/sweeps/lr_dropout_heatmap_{suffix}.png")
    
    # Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[metric_col], bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(df[metric_col].mean(), color="red", linestyle="--", label=f"Mean: {df[metric_col].mean():.3f}")
    ax.axvline(df[metric_col].max(), color="green", linestyle="--", label=f"Max: {df[metric_col].max():.3f}")
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {metric_name} Across All Trials")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/sweeps/{suffix}_distribution.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved distribution plot to outputs/sweeps/{suffix}_distribution.png")


# ============================================================
# CREATE PLOTS FOR ALL METRICS
# ============================================================
metrics = [
    ("test_acc", "Test Accuracy", "accuracy"),
    ("test_f1", "Test F1", "f1"),
    ("test_precision", "Test Precision", "precision"),
    ("test_recall", "Test Recall", "recall"),
]

for metric_col, metric_name, suffix in metrics:
    print(f"\n{'='*60}")
    print(f"CREATING PLOTS FOR {metric_name.upper()}")
    print(f"{'='*60}")
    create_plots_for_metric(df, metric_col, metric_name, suffix)

print("\n" + "=" * 60)
print("BEST OVERALL CONFIGURATION")
print("=" * 60)
best_idx = df["test_acc"].idxmax()
best = df.loc[best_idx]
print(f"  Hidden Channels: {int(best['hidden_channels'])}")
print(f"  Num Layers: {int(best['num_layers'])}")
print(f"  Dropout: {best['dropout']}")
print(f"  Learning Rate: {best['learning_rate']}")
print(f"  Batch Size: {int(best['batch_size'])}")
print(f"  Weight Decay: {best['weight_decay']}")
print(f"  --")
print(f"  Test Accuracy: {best['test_acc']:.4f}")
print(f"  Test F1: {best['test_f1']:.4f}")
print(f"  Test Precision: {best['test_precision']:.4f}")
print(f"  Test Recall: {best['test_recall']:.4f}")

plt.show()

