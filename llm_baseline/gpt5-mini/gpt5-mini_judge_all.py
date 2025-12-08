import json
import requests
import sys
import pathlib
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ===========================
#  CONFIG + API HELPERS
# ===========================

def load_config():
    """Load config.json located in the same directory as this script."""
    config_path = pathlib.Path(__file__).parent / "config.json"
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config.json: {e}")
        sys.exit(1)


def create_gpt5mini_payload(message, model_cfg):
    """Return GPT-5-mini Responses API payload."""
    return {
        "model": model_cfg["model_id"],
        "input": message,
        "temperature": model_cfg.get("temperature", 0.0),
        "max_output_tokens": model_cfg.get("max_tokens", 2048)
    }


def call_gpt5_mini(prompt, model_cfg):
    """
    Send prompt to GPT-5-mini Responses API and return text response.
    On any error or unexpected format, return None (don't crash the whole run).
    """
    url = "https://api.openai.com/v1/responses"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}"
    }

    payload = create_gpt5mini_payload(prompt, model_cfg)

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except Exception:
        # You can uncomment these if you want more debugging noise:
        # print(f"GPT-5-mini request failed: {e}")
        # try:
        #     print("Full error response:", resp.text)
        # except:
        #     pass
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    # Expected structure (based on your working single-example script):
    # data["output"] = [
    #   { ... reasoning object ... },
    #   {
    #     "id": "...",
    #     "type": "message",
    #     "role": "assistant",
    #     "content": [
    #       {"type": "output_text", "text": "...."}
    #     ]
    #   }
    # ]
    try:
        outputs = data.get("output", [])
        if not outputs or len(outputs) < 2:
            return None

        message_obj = outputs[1]
        content_list = message_obj.get("content", [])
        if not content_list:
            return None

        text = content_list[0].get("text")
        return text
    except Exception:
        return None


# ===========================
#  JSON ENFORCER
# ===========================

def parse_json_label(response_text):
    """
    Parse a response like:
        {"label": "CORRECT"}
    or  {"label": "INCORRECT"}

    Returns 1 for CORRECT, 0 for INCORRECT, or None if parsing fails.
    """
    try:
        data = json.loads(response_text)
    except Exception:
        return None

    if "label" not in data:
        return None

    label = str(data["label"]).strip().upper()
    if label not in ["CORRECT", "INCORRECT"]:
        return None

    return 1 if label == "CORRECT" else 0


# ===========================
#  PER-ITEM JUDGE FUNCTION
# ===========================

def judge_item(entry, model_cfg):
    """
    For a single dataset entry:
      - Build judge prompt
      - Call GPT-5-mini
      - Parse JSON label
      - Return (pred_binary, gt) or None on failure
    """
    prediction = entry["prediction"]
    full_prompt = entry["full_prompt"]
    ground_truth = entry["score"]  # "1" or "0"
    gt = 1 if ground_truth == "1" else 0

    judge_prompt = f"""
You are an answer-verification system.

Here is the original problem:

--- PROBLEM ---
{full_prompt}

The model predicted:

--- PREDICTION ---
{prediction}

Your task:
Determine if the prediction is correct or incorrect strictly based on the problem.

You MUST respond in **valid JSON only**:

{{"label": "CORRECT"}}
or
{{"label": "INCORRECT"}}
"""

    response_text = call_gpt5_mini(judge_prompt, model_cfg)
    if not response_text:
        return None

    pred_binary = parse_json_label(response_text)
    if pred_binary is None:
        return None

    return (pred_binary, gt)


# ===========================
#  MAIN PIPELINE
# ===========================

def main():
    # ---------------------------
    # CLI ARGUMENTS
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to dataset JSONL")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of worker threads (default: 10)")
    args = parser.parse_args()

    input_path = args.input
    MAX_WORKERS = args.workers

    dataset_name = pathlib.Path(input_path).parent.name

    # Create results/gpt5-mini/<dataset_name>/ if it doesn't exist
    results_dir = pathlib.Path("results/gpt5-mini") / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    png_path = results_dir / "judge_auc.png"
    stats_path = results_dir / "judge_stats.txt"

    config = load_config()
    model_cfg = config["gpt5-mini"]

    # Load dataset
    with open(input_path, "r") as f:
        lines = f.readlines()

    entries = [json.loads(line) for line in lines]

    total = len(entries)
    print(f"\nLoaded {total} examples from {input_path}.\n")

    y_pred = []
    y_true = []

    print(f"Running with {MAX_WORKERS} worker threads...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(judge_item, entry, model_cfg) for entry in entries]

        for future in tqdm(as_completed(futures), total=total, desc="Judging"):
            pair = future.result()
            if pair is None:
                continue
            pred_binary, gt = pair
            y_pred.append(pred_binary)
            y_true.append(gt)

    if len(y_pred) == 0:
        print("No successful judgments (all failed / malformed). Exiting.")
        return

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    evaluated = len(y_pred)
    correct = np.sum(y_pred == y_true)

    # ===========================
    #   METRICS
    # ===========================

    accuracy = correct / evaluated if evaluated > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred.astype(float))
    except Exception:
        auc = float("nan")

    # ===========================
    #   ROC CURVE (SAVED TO FOLDER)
    # ===========================

    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {dataset_name}")
        plt.legend()
        plt.grid(True)
        plt.savefig(png_path)
        plt.close()
        plot_saved = True
    except Exception:
        plot_saved = False

    # ===========================
    #   FINAL RESULTS STRING
    # ===========================

    results_text = f"""
===== FINAL RESULTS =====
Dataset:   {input_path}
Evaluated: {evaluated}/{total}
Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}
AUC:       {auc:.4f}
Confusion Matrix:
  TP: {np.sum((y_true == 1) & (y_pred == 1))}
  TN: {np.sum((y_true == 0) & (y_pred == 0))}
  FP: {np.sum((y_true == 0) & (y_pred == 1))}
  FN: {np.sum((y_true == 1) & (y_pred == 0))}
"""

    # Print to terminal
    print(results_text)

    # Save stats file in dataset folder
    with open(stats_path, "w") as f:
        f.write(results_text)

    print(f"Stats saved to: {stats_path}")

    if plot_saved:
        print(f"ROC curve saved to: {png_path}")
    else:
        print("Could not generate ROC plot.")

    print("=========================\n")


if __name__ == "__main__":
    main()
