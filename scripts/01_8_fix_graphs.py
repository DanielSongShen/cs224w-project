#!/usr/bin/env python3
"""
Fix ALL isolated nodes in completed reasoning graphs (NON-CAUSAL) with DETAILED METRICS.

This script repairs broken graphs in 'final.json' and reports exactly 
how connections were restored (Root vs Previous Step vs Gap Bridging).

Usage:
    python 01_8_fix_graphs_metrics.py --input data/final.json --output data/final_fixed.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict, Counter

# ==========================================
# 1. Data Loading / Saving
# ==========================================
def load_data(path: Path) -> List[Dict[str, Any]]:
    print(f"Loading data from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split('\n') if line.strip()]
    print(f"  Loaded {len(data)} items")
    return data

def save_data(data: List[Dict[str, Any]], path: Path):
    print(f"Saving fixed data to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  Saved {len(data)} items")

# ==========================================
# 2. Helper Logic
# ==========================================
def normalize_key(key: Any) -> int:
    if isinstance(key, str):
        clean = ''.join(filter(str.isdigit, key))
        return int(clean) if clean else 0
    return int(key)

def build_step_maps(item: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    assigned = item.get("assigned_step", {})
    t_to_s = {}
    s_to_t = defaultdict(list)
    
    for k, v in assigned.items():
        tid = normalize_key(k)
        if tid == 0: continue 
        
        steps = v if isinstance(v, list) else [v]
        step_ints = []
        for s in steps:
            try: step_ints.append(normalize_key(s))
            except: continue
            
        if step_ints:
            step_val = min(step_ints)
            t_to_s[tid] = step_val
            s_to_t[step_val].append(tid)
            
    for s in s_to_t:
        s_to_t[s].sort()
        
    return t_to_s, s_to_t

# ==========================================
# 3. Core Repair Logic (With Metrics)
# ==========================================
def fix_isolated_nodes(item: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    stats = {
        'orphans_found': 0, 
        'orphans_fixed': 0,
        'fix_type_step1_root': 0,    # Normal Step 1 -> Root
        'fix_type_deep_root': 0,     # Step 5 -> Root (Emergency)
        'fix_type_immediate': 0,     # Step N -> Step N-1
        'fix_type_gap_bridge': 0     # Step N -> Step N-3
    }
    
    if "reasoning_graph" not in item:
        return stats
        
    edges = item["reasoning_graph"].get("edges", [])
    
    # Handle int list vs dict list for nodes
    all_nodes = set()
    if "nodes" in item["reasoning_graph"]:
        for n in item["reasoning_graph"]["nodes"]:
            if isinstance(n, dict): all_nodes.add(n.get('id'))
            else: all_nodes.add(normalize_key(n))
    else:
        all_nodes = set(normalize_key(k) for k in item.get("assigned_step", {}).keys())

    # 1. Identify Orphans
    has_parent = {edge['target'] for edge in edges}
    orphans = [n for n in all_nodes if n != 0 and n not in has_parent]
    
    if not orphans:
        return stats
        
    stats['orphans_found'] = len(orphans)
    
    # 2. Build Context
    t_to_s, s_to_t = build_step_maps(item)
    
    # 3. Fix Each Orphan
    new_edges = []
    for orphan in orphans:
        orphan_step = t_to_s.get(orphan, 1)
        anchor = 0 
        
        # Track logic for stats
        found_parent_step = -1
        
        # Look backwards
        prev_step = orphan_step - 1
        
        # Gap Bridging Loop
        while prev_step > 0:
            candidates = s_to_t.get(prev_step, [])
            if candidates:
                anchor = max(candidates) # Non-Causal Anchor
                found_parent_step = prev_step
                break
            prev_step -= 1
            
        # --- METRICS CALCULATION ---
        if anchor == 0:
            if orphan_step == 1:
                stats['fix_type_step1_root'] += 1
            else:
                stats['fix_type_deep_root'] += 1
        else:
            if found_parent_step == orphan_step - 1:
                stats['fix_type_immediate'] += 1
            else:
                stats['fix_type_gap_bridge'] += 1
        # ---------------------------

        new_edge = {
            "source": anchor,
            "target": orphan,
            "category": 5 
        }
        new_edges.append(new_edge)
        
        if verbose:
            tag = item.get('tag', 'unknown')
            type_str = "Immediate" if found_parent_step == orphan_step-1 else "Gap/Root"
            print(f"  [{tag}] Fixed T{orphan} (S{orphan_step}) -> T{anchor} ({type_str})")

    if new_edges:
        item["reasoning_graph"]["edges"].extend(new_edges)
        stats['orphans_fixed'] = len(new_edges)
        
    return stats

# ==========================================
# 4. Main Execution & Reporting
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Fix orphans with metrics")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    data = load_data(Path(args.input))
    
    # Aggregation Counters
    total_found = 0
    total_fixed = 0
    fix_types = Counter()
    
    print("\n=== Starting Graph Repair (Non-Causal) ===\n")
    
    for i, item in enumerate(data):
        s = fix_isolated_nodes(item, verbose=args.verbose)
        
        total_found += s['orphans_found']
        total_fixed += s['orphans_fixed']
        
        fix_types['Step 1 -> Root'] += s['fix_type_step1_root']
        fix_types['Step N -> Root (Deep Fallback)'] += s['fix_type_deep_root']
        fix_types['Step N -> Step N-1 (Immediate)'] += s['fix_type_immediate']
        fix_types['Step N -> Step N-k (Gap Bridge)'] += s['fix_type_gap_bridge']
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} items...")

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("REPAIR STATISTICS REPORT")
    print("="*60)
    print(f"Total Graphs Processed: {len(data)}")
    print(f"Total Orphans Found:    {total_found}")
    print(f"Total Orphans Fixed:    {total_fixed}")
    
    if total_fixed > 0:
        print("\nFix Type Distribution:")
        print("-" * 30)
        for ftype, count in fix_types.most_common():
            pct = (count / total_fixed) * 100
            print(f"{ftype:<35}: {count:>6} ({pct:>5.1f}%)")
            
        print("-" * 30)
        
        # Interpretation
        if fix_types['Step N -> Root (Deep Fallback)'] > (total_fixed * 0.1):
            print("\n[!] NOTE: High number of Deep Fallbacks indicates many empty steps or broken assignments.")
        elif fix_types['Step N -> Step N-1 (Immediate)'] > (total_fixed * 0.8):
            print("\n[+] SUCCESS: Most fixes recovered the immediate logical predecessor.")
            
    print("="*60 + "\n")
    
    save_data(data, Path(args.output))

if __name__ == "__main__":
    main()