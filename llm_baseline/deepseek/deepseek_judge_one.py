import json
import requests
import sys
import pathlib

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

def create_deepseek_payload(message, model_cfg):
    """Return DeepSeek Chat API request payload."""
    return {
        "model": model_cfg["model_id"],
        "messages": [{"role": "user", "content": message}],
        "temperature": model_cfg.get("temperature", 0.0),
        "max_tokens": model_cfg.get("max_tokens", 2048)
    }

def call_deepseek(prompt, model_cfg):
    """Send prompt to DeepSeek API and return text response."""
    url = f"{model_cfg['url']}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}"
    }

    payload = create_deepseek_payload(prompt, model_cfg)

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"DeepSeek request failed: {e}")
        sys.exit(1)

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        print("Unexpected API response:")
        print(json.dumps(data, indent=2))
        sys.exit(1)


# ===========================
#  MAIN
# ===========================

def main():
    config = load_config()
    model_cfg = config["deepseek-v3.2"]

    # Load ONE entry from process1.json (JSONL)
    path = "data/amc-aime/final_regraded.json"

    with open(path, "r") as f:
        first_line = f.readline().strip()
        entry = json.loads(first_line)

    prediction = entry["prediction"]
    full_prompt = entry["full_prompt"]
    ground_truth = entry["score"]  # "1" or "0"

    # Construct judging prompt
    judge_prompt = f"""
You are an answer-verification model.

Here is the original problem:

--- PROBLEM ---
{full_prompt}

The model predicted:

--- PREDICTION ---
{prediction}

Your task:
Determine if the prediction is **correct or incorrect** strictly based on the problem.

Respond with exactly one token:
CORRECT
or
INCORRECT
"""

    deepseek_answer = call_deepseek(judge_prompt, model_cfg).strip().upper()

    print("\n=== DeepSeek Judgment ===")
    print(deepseek_answer)
    print("=========================\n")

    # Compare with ground truth
    truth = "CORRECT" if ground_truth == "1" else "INCORRECT"

    print(f"Ground Truth: {truth}")
    print(f"DeepSeek Judged: {deepseek_answer}")

    if deepseek_answer == truth:
        print("\n✅ DeepSeek judged correctly!\n")
    else:
        print("\n❌ DeepSeek judged incorrectly.\n")


if __name__ == "__main__":
    main()
