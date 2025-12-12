#!/usr/bin/env python3
"""View reasoning sketches from LCoT2Tree output"""
import json
import sys

def view_sketches(data_path, max_examples=5):
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            
            item = json.loads(line)
            print(f"\n{'='*80}")
            print(f"Example {i+1}: {item['tag']}")
            print(f"Score: {item['score']}")
            print(f"{'='*80}")
            print("\nReasoning Sketch:")
            print(item['reasoning_sketch'])
            print()

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "./data/processed/lcot2tree_test/final.json"
    view_sketches(data_path)