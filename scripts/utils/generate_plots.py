"""Generate training plots and other figures to visualize"""

import re
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path


def plot_training_loss(log_file_paths, labels=None, plot_train=True, plot_val=True, colors=None, line_styles=None):
    """
    Plot training and validation loss from one or more training log files.

    Args:
        log_file_paths (str, Path, or list): Path(s) to training log file(s). Can be a single path or list of paths.
        labels (list, optional): Custom labels for each log file. If None, uses directory names.
        plot_train (bool): Whether to plot training loss. Default True.
        plot_val (bool): Whether to plot validation loss. Default True.
    """
    # Handle single file input
    if isinstance(log_file_paths, (str, Path)):
        log_file_paths = [log_file_paths]

    # Convert to Path objects
    log_paths = [Path(p) for p in log_file_paths]

    # Generate labels if not provided
    if labels is None:
        labels = [p.parent.name for p in log_paths]
    elif len(labels) != len(log_paths):
        raise ValueError("Number of labels must match number of log files")

    # Regex pattern to match epoch lines
    # Format: "Epoch 001 | Train Loss: 0.6492 | Train Acc: 0.6385 | Val Loss: 0.6381 | ..."
    epoch_pattern = r'Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Train Acc:\s+[0-9.]+\s+\|\s+Val Loss:\s+([0-9.]+)'

    # Colors for different models
    if colors == None:
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'lightcyan', 'lightgray', 'lightyellow']
    if line_styles == None:
        line_styles = ['-']

    # Create the plot
    plt.figure(figsize=(12, 8))

    max_epochs = 0
    all_epochs = []

    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        if not log_path.exists():
            print(f"Warning: Log file not found: {log_path}")
            continue

        epochs = []
        train_losses = []
        val_losses = []

        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(epoch_pattern, line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))

                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)

        if not epochs:
            print(f"No training data found in {log_path}")
            continue

        color = colors[i % len(colors)]
        ls = line_styles[i % len(line_styles)]

        if plot_train:
            plt.plot(epochs, train_losses, color=color, linestyle=ls, linewidth=4,
                    label=f'{label}', marker='o', markersize=3, alpha=0.7)
        if plot_val:
            plt.plot(epochs, val_losses, color=color, linestyle='--', linewidth=2,
                    label=f'{label}Val', marker='s', markersize=3, alpha=0.7)

        max_epochs = max(max_epochs, max(epochs))
        all_epochs.extend(epochs)

    if not all_epochs:
        print("No valid training data found in any of the provided files")
        return

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss of Top Models')

    # Create legend with multiple columns if many lines
    num_lines = sum([plot_train, plot_val]) * len([p for p in log_paths if p.exists()])
    ncol = min(2, num_lines // 8 + 1)  # Use 2 columns if more than 8 lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol)

    plt.grid(True, alpha=0.3)

    # Set x-axis to show integer epochs
    unique_epochs = sorted(list(set(all_epochs)))
    plt.xticks(unique_epochs[::max(1, len(unique_epochs)//15)])  # Show ~15 ticks

    plt.tight_layout()
    plt.show()


def plot_test_metrics(results_file_paths, labels=None, colors=None, metrics=None, height_overrides=None, show_x_labels=True, sort_by_metric=None, baseline_label=None, gradient_base_color=None):
    """
    Plot test metrics (accuracy, precision, recall, F1) from results JSON files as bar charts.

    Args:
        results_file_paths (str, Path, or list): Path(s) to results JSON file(s). Can be a single path or list of paths.
        labels (list, optional): Custom labels for each results file. If None, uses directory names.
        colors (list, optional): Custom colors for each model. If None, uses default colors.
        metrics (list, optional): Which metrics to plot. If None, plots ['accuracy', 'precision', 'recall', 'f1'].
        height_overrides (dict, optional): Dictionary to override specific bar heights.
            Format: {'label': {'metric': value, ...}, ...}
        show_x_labels (bool, optional): Whether to show x-axis labels on subplots. Default True.
        sort_by_metric (str, optional): Metric name to sort bars by (e.g., 'accuracy'). If None, no sorting. Default None.
        baseline_label (str, optional): Label of model to always place first (leftmost). Other models sorted around it. Default None.
        gradient_base_color (str or tuple, optional): Base color for the left-to-right gradient (excluding baseline). If None, uses the first model's color. Default None.
    """
    # Handle single file input
    if isinstance(results_file_paths, (str, Path)):
        results_file_paths = [results_file_paths]

    # Convert to Path objects
    results_paths = [Path(p) for p in results_file_paths]

    # Default metrics to plot
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1']

    # Generate labels if not provided
    if labels is None:
        labels = [p.parent.parent.parent.name + '_' + p.parent.parent.name for p in results_paths]
    elif len(labels) != len(results_paths):
        raise ValueError("Number of labels must match number of results files")

    # Default colors
    if colors is None:
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink', 'lightcyan', 'lightgray', 'lightyellow']

    # Collect data from all files
    results_data = []

    for results_path, label in zip(results_paths, labels):
        if not results_path.exists():
            print(f"Warning: Results file not found: {results_path}")
            results_data.append({metric: 0 for metric in metrics})
            continue

        try:
            with open(results_path, 'r') as f:
                data = json.load(f)

            test_results = data.get('test_results', {})
            model_data = {}

            for metric in metrics:
                value = test_results.get(metric, 0)
                # Convert to percentage for better visualization
                model_data[metric] = value * 100 if isinstance(value, (int, float)) else 0

            # Apply height overrides if specified
            if height_overrides and label in height_overrides:
                overrides = height_overrides[label]
                for metric, override_value in overrides.items():
                    if metric in model_data:
                        model_data[metric] = override_value

            results_data.append(model_data)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading {results_path}: {e}")
            results_data.append({metric: 0 for metric in metrics})

    if not results_data:
        print("No valid results data found")
        return

    # Sort by specified metric if requested
    if sort_by_metric and sort_by_metric in metrics:
        if baseline_label and baseline_label in labels:
            # Find baseline index
            baseline_idx = labels.index(baseline_label)

            # Separate baseline from others
            baseline_data = results_data[baseline_idx]
            baseline_color = 'grey'

            # Remove baseline from lists
            other_results = results_data[:baseline_idx] + results_data[baseline_idx+1:]
            other_labels = labels[:baseline_idx] + labels[baseline_idx+1:]
            other_colors = colors[:baseline_idx] + colors[baseline_idx+1:]

            # Sort the other models
            sort_values = [(i, model_data[sort_by_metric]) for i, model_data in enumerate(other_results)]
            sort_values.sort(key=lambda x: x[1])

            # Reorder other models
            sorted_indices = [idx for idx, _ in sort_values]
            other_results = [other_results[i] for i in sorted_indices]
            other_labels = [other_labels[i] for i in sorted_indices]
            other_colors = [other_colors[i] for i in sorted_indices]

            # Create gradient colors for non-baseline models (darker from left to right)
            if len(other_colors) > 1:
                base_color = gradient_base_color if gradient_base_color is not None else other_colors[0]
                base_rgb = mcolors.to_rgb(base_color)

                # Create gradient from light to dark across models with more extreme differences
                other_colors = []
                for k in range(len(other_labels)):
                    # More extreme darkening: Start at 1.0, decrease by larger amounts
                    darkness_factor = 1.0 - (k * 0.25)  # Start at 1.0, decrease by 0.25 each time
                    darkness_factor = max(0.2, darkness_factor)  # Don't go too dark

                    gradient_color = tuple(max(0.0, c * darkness_factor) for c in base_rgb)
                    other_colors.append(gradient_color)

            # Combine: baseline first, then gradient-colored others
            results_data = [baseline_data] + other_results
            labels = [baseline_label] + other_labels
            colors = [baseline_color] + other_colors
        else:
            # No baseline specified, sort all models
            sort_values = [(i, model_data[sort_by_metric]) for i, model_data in enumerate(results_data)]
            sort_values.sort(key=lambda x: x[1])

            sorted_indices = [idx for idx, _ in sort_values]
            results_data = [results_data[i] for i in sorted_indices]
            labels = [labels[i] for i in sorted_indices]
            colors = [colors[i] for i in sorted_indices]

            # Create gradient colors for all models
            if len(colors) > 1:
                base_color = gradient_base_color if gradient_base_color is not None else colors[0]
                base_rgb = mcolors.to_rgb(base_color)

                colors = []
                for k in range(len(labels)):
                    # More extreme darkening: Start at 1.0, decrease by 0.25 each time
                    darkness_factor = 1.0 - (k * 0.25)  # Start at 1.0, decrease by 0.25 each time
                    darkness_factor = max(0.2, darkness_factor)  # Don't go too dark

                    gradient_color = tuple(max(0.0, c * darkness_factor) for c in base_rgb)
                    colors.append(gradient_color)

    # Create subplots for each metric
    num_metrics = len(metrics)
    if num_metrics == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]  # Make it a list for consistent indexing
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

    # Set width of bars and positions
    bar_width = 4 / len(results_paths)  # Width of each bar (made much thicker)
    x_positions = np.arange(len(results_paths))  # x positions for groups

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot bars for each model
        for j, (label, model_data) in enumerate(zip(labels, results_data)):
            value = model_data.get(metric, 0)
            x_pos = x_positions[j]

            ax.bar(x_pos, value, bar_width,
                   label=label if i == 0 else "",  # Only show legend on first subplot
                   color=colors[j % len(colors)], alpha=0.8)

            # Add value labels on bars
            ax.text(x_pos, value + 0.5, f'{value:.1f}', ha='center', va='bottom', fontsize=9)

    # Add horizontal baseline reference line if baseline exists
    if baseline_label and baseline_label in labels:
        baseline_idx = labels.index(baseline_label)
        baseline_value = results_data[baseline_idx][metric]

        # Add dashed red line across the entire plot for this metric
        axes[i].axhline(y=baseline_value, color='lightcoral', linestyle='--', linewidth=2, alpha=0.7,
                       label='Baseline' if i == 0 else "")

    # Customize subplot
        ax.set_title(f'{metric.capitalize()} (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_xticks(x_positions)
        if show_x_labels:
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        else:
            ax.set_xticklabels([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)  # Since we're showing percentages

    # Add overall title and legend
    fig.suptitle('Test Metrics of Main Methods', fontsize=14, fontweight='bold', y=0.98)

    # Create legend with wider layout
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.95),
               ncol=min(6, len(labels)), fontsize=10,
               bbox_transform=fig.transFigure)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Make room for the legend
    plt.show()


# Example usage:

# Plot single file
# plot_training_loss('outputsNEW/encoder64/combined/undirected/models/simple_gin/train.log')

# Plot multiple files for comparison
"""
log_files = [
    'outputsNEW/no_encoder/combined/undirected/models/simple_gin/train.log',
    'outputsNEW/encoder64/combined/undirected/models/simple_gin/train.log',
    'outputsNEW/no_encoder64/combined2/undirected/models/simple_gin/train.log',
    'outputsNEW/encoder64/combined2/undirected/models/simple_gin/train.log',
    'outputsNEW/no_encoder64/combined3/undirected/models/simple_gin/train.log',
    'outputsNEW/encoder64/combined3/undirected/models/simple_gin/train.log'
]

custom_labels = ['No Encoder (small)', 'Encoder (small)', 'No Encoder (medium)', 'Encoder (medium)', 'No Encoder (large)', 'Encoder (large)']
"""
"""custom_labels = ['Small Thoughts ']
log_files = [
    'outputsNEW/graph64/small_graph/undirected/models/simple_gin/train.log',
]
plot_training_loss(log_files, labels=custom_labels, plot_val=True)"""

log_files = [
    'outputsNEW/encoder64/combined/undirected/models/simple_gin/train.log',
    'outputsNEW/encoder64/combined/undirected/models/hetero_gin/train.log',
    'outputsNEW/graph64/big_graph/undirected/models/simple_gin/train.log',
    'outputsNEW/text64/combined/undirected/models/simple_gin/train.log',
    'outputsNEW/graph64/big_causal_graph/undirected/models/simple_gin/train.log',
]

custom_labels = ['HO-GIN (tree)', 'HE-GIN (tree)', 'HO-GIN (big thoughts)', 'HO-GIN (text embeddings)', 'HO-GIN (big causal thoughts)']
colors = ['skyblue', 'skyblue', 'palegreen', 'bisque', 'palegreen']
line_styles = ['--', ':', '--', '--', '--']
plot_training_loss(log_files, labels=custom_labels, plot_val=False, colors=colors, line_styles=line_styles)

# Example: Plot test metrics from results files
"""
results_files = [
    'outputsNEW/encoder/cn_k12/directed/models/hetero_gin/results_seed42.json',
    'outputsNEW/encoder64/combined/undirected/models/simple_gin/results_seed42.json',
    'outputsNEW/graph64/big_graph/undirected/models/simple_gin/results_seed42.json',
    'outputsNEW/graph64/big_causal_graph/undirected/models/simple_gin/results_seed42.json'
]

model_labels = ['DeepSeek V3.2', 'HO-GIN (Tree)', 'HO-GIN (Big Thoughts Graph)', 'HO-GIN (Big Thoughs Causal Graph)']
custom_colors = ['grey', 'yellow', 'green', 'cyan']

# Example with height overrides (hard-coded values)

height_overrides_example = {
    'HO-GIN (cn_k12)': {
        'accuracy': 68
'f1': 0.7303*100
'recall': 0.7065*100
'precision': 0.7558*100


    },
    'HO-GIN (cn_k12)': {
        'accuracy': 76.45,
    }
    
}



results_files = [
    'outputsNEW/encoder64/combined_easy/undirected/models/simple_gin/results_seed42.json',
    'outputsNEW/encoder64/combined_easy/undirected/models/hetero_gin/results_seed42.json',
    'outputsNEW/encoder64/combined_hard/undirected/models/simple_gin/results_seed42.json',
    'outputsNEW/encoder64/combined_hard/undirected/models/hetero_gin/results_seed42.json',
]

model_labels = ['HO-GIN (easy)', 'HE-GIN (easy)', 'HO-GIN (hard)', 'HE-GIN (hard)']
custom_colors = ['lightcoral', 'lightgreen', 'red', 'green']



plot_test_metrics(results_files, labels=model_labels, colors=custom_colors,
                 show_x_labels=False, metrics=['accuracy'])
"""

height_overrides_example = {
    'DeepSeek V3.2': {
        'accuracy': 68.47,
        'f1': 0.7395*100,
        'recall': 0.9251*100,
        'precision': 0.6160*100,
    },
    'HO-GIN (Tree)': {
        'accuracy': 68,
        'f1': 0.7303*100,
        'recall': 0.7065*100,
        'precision': 0.7558*100,
    }
}
                 
results_files = [
    'outputs/models/simple_gin/results_seed42.json',
    'outputs/models/hetero_gin/results_seed42.json',
    'outputs/models/hetero_gat/results_seed42.json',
    'outputsNEW/big_causal_graph/'
]

results_files = [
    'outputsNEW/encoder/cn_k12/directed/models/hetero_gin/results_seed42.json',
    'outputsNEW/graph64/big_causal_graph/undirected/models/simple_gin/results_seed42.json',
    'outputs/models/hetero_gin/results_seed42.json',
    'outputs/models/hetero_gat/results_seed42.json',
    'outputsNEW/graph64/big_graph/undirected/models/simple_gin/results_seed42.json',
    'outputsNEW/text64/combined/undirected/models/simple_gin/results_seed42.json'
]

model_labels = ['DeepSeek V3.2', 'HO-GIN (big causal thoughts)', 'HE-GIN (small)', 'HE-GAT (small)', 'HO-GIN (big thoughts)', 'HO-GIN (text embeddings)']

"""plot_test_metrics(results_files, labels=model_labels, height_overrides=height_overrides_example,
                 show_x_labels=False, metrics=['accuracy'], sort_by_metric='accuracy', baseline_label='DeepSeek V3.2',
                 gradient_base_color='palegreen')  # Use light blue as base for gradient"""