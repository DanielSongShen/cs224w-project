"""
Merge scores from regraded file into final file by matching "tag" field.

Usage:
    python scripts/merge_scores.py
"""
import json


def main():
    final_path = "data/processed/deepseek/new_graphs_combined/final.json"
    regraded_path = "data/processed/deepseek/combined/final_regraded.json"
    output_path = "data/processed/deepseek/new_graphs_combined/final_with_scores.json"
    
    # Step 1: Build tag -> score mapping from regraded file
    print(f"Loading scores from {regraded_path}...")
    tag_to_score = {}
    with open(regraded_path, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                tag = entry.get("tag")
                score = entry.get("score")
                if tag is not None:
                    tag_to_score[tag] = score
    
    print(f"Loaded {len(tag_to_score)} scores from regraded file")
    
    # Step 2: Read final file, add scores, write output
    print(f"Processing {final_path}...")
    matched = 0
    unmatched = 0
    
    with open(final_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if line.strip():
                entry = json.loads(line)
                tag = "test_" + entry.get("tag")
                if tag in tag_to_score:
                    entry["score"] = tag_to_score[tag]
                    matched += 1
                else:
                    unmatched += 1
                
                f_out.write(json.dumps(entry) + '\n')
    
    print(f"\n=== Done ===")
    print(f"Matched: {matched}")
    print(f"Unmatched: {unmatched}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()

