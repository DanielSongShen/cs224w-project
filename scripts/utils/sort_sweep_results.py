import json
import os

def main():
    # Path to the sweep results
    sweep_results_path = 'outputs/sweeps/sweep_20251205_120730/sweep_results.json'
    
    if not os.path.exists(sweep_results_path):
        print(f"Error: {sweep_results_path} not found.")
        return

    try:
        with open(sweep_results_path, 'r') as f:
            results = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {sweep_results_path}")
        return

    # Sort results by best_val_f1 in descending order
    # Handle potential missing keys gracefully, though schema seems consistent
    sorted_results = sorted(
        results, 
        key=lambda x: x.get('best_val_f1', 0.0), 
        reverse=True
    )

    # Format the output as a readable text list
    output_lines = []
    output_lines.append(f"Total Configurations: {len(sorted_results)}")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Rank':<5} | {'Val F1':<8} | {'Val Acc':<8} | {'Test F1':<8} | {'Test Acc':<8} | {'Config'}")
    output_lines.append("-" * 80)

    for rank, res in enumerate(sorted_results, 1):
        config = res.get('config', {})
        val_f1 = res.get('best_val_f1', 0.0)
        val_acc = res.get('best_val_acc', 0.0)
        test_f1 = res.get('test_f1', 0.0)
        test_acc = res.get('test_acc', 0.0)
        
        # Format config string compactly
        config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
        
        output_lines.append(f"{rank:<5} | {val_f1:.4f}   | {val_acc:.4f}   | {test_f1:.4f}   | {test_acc:.4f}   | {config_str}")

    # Write to file
    output_file = 'outputs/sweeps/sorted_configs_by_val_f1.txt'
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))
    
    print(f"Successfully saved sorted configurations to {output_file}")

    # Also print top 5 to console for immediate feedback
    print("\nTop 5 Configurations:")
    print("\n".join(output_lines[:7])) # Header + top 5

if __name__ == "__main__":
    main()

