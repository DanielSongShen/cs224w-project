#!/usr/bin/env python3
"""
Pre-compute sentence embeddings for thought nodes.

This script processes final.json files containing reasoning traces and creates
384-dimensional embeddings using all-MiniLM-L6-v2. The embeddings are saved
as thought_embeddings.pt and an index mapping is saved as embedding_index.json.

Usage:
    poetry run python scripts/03_precompute_embeddings.py \
        --input data/processed/deepseek/cn_k12/final.json \
        --output data/processed/deepseek/cn_k12/ \
        --batch-size 256 \
        --device auto
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from sentence_transformers import SentenceTransformer


def load_jsonl_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file containing reasoning traces with thoughts_list.

    Args:
        filepath: Path to final.json file

    Returns:
        List of reasoning trace dictionaries
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    return data


def extract_thought_texts(data: List[Dict[str, Any]]) -> tuple[List[str], Dict[int, Dict[int, int]]]:
    """
    Extract all thought text strings and create mapping from (graph_idx, node_idx) to global_idx.

    Args:
        data: List of reasoning traces from final.json

    Returns:
        Tuple of (all_texts, index_mapping) where:
        - all_texts: List of thought strings
        - index_mapping: {graph_idx: {node_idx: global_idx}}
    """
    all_texts = []
    index_mapping = {}

    for graph_idx, item in enumerate(data):
        thoughts_list = item.get('thoughts_list', {})

        # Normalize keys to strings and sort by int value
        normalized_thoughts = {}
        for key, value in thoughts_list.items():
            try:
                int_key = int(key)
                normalized_thoughts[int_key] = str(value)
            except (ValueError, TypeError):
                continue

        if not normalized_thoughts:
            print(f"Warning: No valid thoughts found for graph {graph_idx}")
            continue

        # Sort by node index and add to global list
        graph_mapping = {}
        for node_idx in sorted(normalized_thoughts.keys()):
            thought_text = normalized_thoughts[node_idx]
            global_idx = len(all_texts)
            all_texts.append(thought_text)
            graph_mapping[node_idx] = global_idx

        index_mapping[graph_idx] = graph_mapping

    return all_texts, index_mapping


def get_device(device_str: str) -> str:
    """Resolve device string to actual torch device."""
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    return device_str


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute sentence embeddings for thought nodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to final.json file containing reasoning traces"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save embeddings and index mapping"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device for encoding (auto = mps/cuda/cpu)"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length for model"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Pre-computing Sentence Embeddings ===")
    print(f"Input file: {args.input}")
    print(f"Output dir: {args.output}")
    print(f"Model: {args.model}")
    print(f"Device: {get_device(args.device)}")
    print(f"Batch size: {args.batch_size}")

    # Step 1: Load data
    print("\n1. Loading data...")
    data = load_jsonl_data(args.input)
    print(f"   Loaded {len(data)} reasoning traces")

    # Step 2: Extract thought texts
    print("\n2. Extracting thought texts...")
    all_texts, index_mapping = extract_thought_texts(data)

    if not all_texts:
        print("ERROR: No thought texts found! Check that final.json contains 'thoughts_list' fields.")
        return 1

    print(f"   Found {len(all_texts)} total thoughts across {len(index_mapping)} graphs")

    # Step 3: Load model and encode
    print(f"\n3. Loading model and encoding texts...")
    device = get_device(args.device)

    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length

    print(f"   Encoding {len(all_texts)} texts in batches of {args.batch_size}...")

    # Encode in batches
    embeddings = model.encode(
        all_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    # Convert to float32 and move to CPU for storage
    embeddings = embeddings.to(torch.float32).cpu()

    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Expected size: ~{embeddings.numel() * embeddings.element_size() / 1024 / 1024:.1f} MB")

    # Step 4: Save results
    print("\n4. Saving results...")

    # Save embeddings
    emb_path = output_dir / "thought_embeddings.pt"
    torch.save(embeddings, emb_path)
    print(f"   Saved embeddings to {emb_path}")

    # Save index mapping
    idx_path = output_dir / "embedding_index.json"
    with open(idx_path, 'w', encoding='utf-8') as f:
        json.dump(index_mapping, f, indent=2)
    print(f"   Saved index mapping to {idx_path}")

    # Save metadata
    meta_path = output_dir / "embedding_metadata.json"
    metadata = {
        "model": args.model,
        "device_used": device,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "num_graphs": len(index_mapping),
        "total_thoughts": len(all_texts),
        "embedding_dim": embeddings.shape[1],
        "input_file": str(args.input),
        "created_at": str(torch.tensor(0.).new_zeros(1).device),  # timestamp placeholder
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved metadata to {meta_path}")

    print("\n=== SUCCESS ===")
    print(f"Processed {len(all_texts)} thoughts from {len(data)} graphs")
    print(f"Embeddings saved to: {emb_path}")
    print(f"Index mapping saved to: {idx_path}")
    print(f"Memory usage: ~{embeddings.numel() * embeddings.element_size() / 1024 / 1024:.1f} MB")

    return 0


if __name__ == "__main__":
    exit(main())
