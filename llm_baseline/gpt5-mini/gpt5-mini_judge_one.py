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


def create_gpt5mini_payload(message, model_cfg):
    """Return GPT-5-mini Responses API payload."""
    return {
        "model": model_cfg["model_id"],
        "input": message,
        "temperature": model_cfg.get("temperature", 0.0),
        "max_output_tokens": model_cfg.get("max_tokens", 2048)
    }


def call_gpt5_mini(prompt, model_cfg):
    """Send prompt to GPT-5-mini Responses API and return text response."""
    url = "https://api.openai.com/v1/responses"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}"
    }

    payload = create_gpt5mini_payload(prompt, model_cfg)

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"GPT-5-mini request failed: {e}")
        print("Full error response:")
        try:
            print(response.text)
        except:
            pass
        sys.exit(1)

    data = response.json()

    # ===== Correct extraction based on actual API output ===== #
    try:
        # Output is list: [reasoning_object, message_object]
        message_obj = data["output"][1]
        content_list = message_obj["content"]
        text = content_list[0]["text"]
        return text
    except Exception:
        print("Unexpected API response:")
        print(json.dumps(data, indent=2))
        sys.exit(1)


# ===========================
#  MAIN
# ===========================

def main():
    config = load_config()
    model_cfg = config["gpt5-mini"]

    # Load ONE entry from process1.json (JSONL)
    path = "data/amc-aime/final_regraded.json"

    with open(path, "r") as f:
        first_line = f.readline().strip()
        entry = json.loads(first_line)

    prediction = entry["prediction"]
    full_prompt = entry["full_prompt"]
    ground_truth = entry["score"]  # "1" or "0"

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

    answer = call_gpt5_mini(judge_prompt, model_cfg).strip().upper()

    print("\n=== GPT-5-mini Judgment ===")
    print(answer)
    print("=========================\n")

    truth = "CORRECT" if ground_truth == "1" else "INCORRECT"

    print(f"Ground Truth: {truth}")
    print(f"GPT-5-mini Judged: {answer}")

    if answer == truth:
        print("\n✅ GPT-5-mini judged correctly!\n")
    else:
        print("\n❌ GPT-5-mini judged incorrectly.\n")


if __name__ == "__main__":
    main()
