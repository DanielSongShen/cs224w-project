#!/usr/bin/env python3
"""
Convert a Non-Causal reasoning dataset to a Causal one (With Metrics).

This script performs two major actions:
1. PRUNING: Removes edges where Source >= Target ("Time Travel").
2. REWIRING: Connects resulting orphans to the "Causal Anchor" (Max valid ID of prev step).

It reports detailed statistics on how many edges were pruned and how orphans were fixed.

Usage:
    python 01_9_causal_edges.py --input data/final.json --output data/final_causal.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
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
    print(f"Saving causal data to {path}...")
    with open(path, 'w', encoding='utf-8') as f:
        if path.suffix == '.json':
            json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print("Done.")

# ==========================================
# 2. Helper Logic
# ==========================================
def build_step_lookup(assigned_step: Dict[str, Any]) -> Dict[int, int]:
    t_to_s = {}
    for thought_id, steps in assigned_step.items():
        tid = int(thought_id)
        if not isinstance(steps, list): steps = [steps]
        
        step_ints = []
        for s in steps:
            try: step_ints.append(int(str(s).replace("A", "").replace("B", "")))
            except: continue
                
        if step_ints:
            t_to_s[tid] = min(step_ints)
        else:
            t_to_s[tid] = 0
    return t_to_s

def build_step_inventory(t_to_s: Dict[int, int]) -> Dict[int, List[int]]:
    s_to_t = defaultdict(list)
    for tid, step in t_to_s.items():
        s_to_t[step].append(tid)
    for step in s_to_t:
        s_to_t[step].sort()
    return s_to_t

# ==========================================
# 3. Core Logic (Prune & Rewire)
# ==========================================
def fix_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single item to strictly causal structure and returns local stats.
    """
    stats = {
        'pruned_edges': 0,
        'orphans_fixed': 0,
        'fix_type_step1_root': 0,
        'fix_type_deep_root': 0,
        'fix_type_immediate': 0,
        'fix_type_gap_bridge': 0
    }

    if "assigned_step" not in item:
        return item, stats
        
    t_to_s = build_step_lookup(item["assigned_step"])
    s_to_t = build_step_inventory(t_to_s)
    
    # ---------------------------------------------------------
    # PART A: Fix 'reasoning_graph' (Flat Edges)
    # ---------------------------------------------------------
    if "reasoning_graph" in item:
        edges = item["reasoning_graph"].get("edges", [])
        nodes = item["reasoning_graph"].get("nodes", [])
        valid_edges = []
        
        # 1. Prune Non-Causal Edges
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            
            if src < tgt:
                valid_edges.append(edge)
            else:
                stats['pruned_edges'] += 1

        # 2. Identify Orphans (Nodes that lost all parents)
        has_parent = set()
        for edge in valid_edges:
            has_parent.add(edge['target'])
            
        # --- FIX: Handle both dicts and ints in 'nodes' list ---
        all_node_ids = set()
        if nodes:
            for n in nodes:
                if isinstance(n, dict):
                    all_node_ids.add(n.get('id'))
                else:
                    all_node_ids.add(int(n))
        else:
            all_node_ids = set(t_to_s.keys())
        # -------------------------------------------------------

        orphans = [n for n in all_node_ids if n != 0 and n not in has_parent]
        
        # 3. Rewire Orphans
        for orphan in orphans:
            stats['orphans_fixed'] += 1
            orphan_step = t_to_s.get(orphan, 1)
            anchor = 0 
            found_parent_step = -1
            
            prev_step = orphan_step - 1
            while prev_step > 0:
                prev_thoughts = s_to_t.get(prev_step, [])
                # STRICT CAUSAL FILTER
                valid_ancestors = [t for t in prev_thoughts if t < orphan]
                
                if valid_ancestors:
                    anchor = max(valid_ancestors)
                    found_parent_step = prev_step
                    break
                prev_step -= 1
            
            # Record Stat Type
            if anchor == 0:
                if orphan_step == 1: stats['fix_type_step1_root'] += 1
                else: stats['fix_type_deep_root'] += 1
            else:
                if found_parent_step == orphan_step - 1: stats['fix_type_immediate'] += 1
                else: stats['fix_type_gap_bridge'] += 1

            valid_edges.append({
                "source": anchor,
                "target": orphan,
                "category": 5 
            })
            
        item["reasoning_graph"]["edges"] = valid_edges

    # ---------------------------------------------------------
    # PART B: Fix 'thought_relations' (Adjacency Map)
    # ---------------------------------------------------------
    # Note: Stats are only counted once (from Part A) to avoid double counting
    # if both structures exist. Logic is mirrored here.
    if "thought_relations" in item:
        old_rels = item["thought_relations"]
        new_rels = defaultdict(dict)
        
        # 1. Prune
        for parent, children in old_rels.items():
            pid = int(parent)
            for child, cat in children.items():
                cid = int(child)
                if pid < cid:
                    new_rels[str(pid)][str(cid)] = cat
        
        # 2. Check Orphans
        child_nodes_with_parents = set()
        for p, children in new_rels.items():
            for c in children:
                child_nodes_with_parents.add(int(c))
                
        all_ids = set(t_to_s.keys())
        orphans = [n for n in all_ids if n != 0 and n not in child_nodes_with_parents]
        
        # 3. Rewire
        for orphan in orphans:
            orphan_step = t_to_s.get(orphan, 1)
            prev_step = orphan_step - 1
            anchor = 0
            
            while prev_step > 0:
                prev_thoughts = s_to_t.get(prev_step, [])
                valid_ancestors = [t for t in prev_thoughts if t < orphan]
                if valid_ancestors:
                    anchor = max(valid_ancestors)
                    break
                prev_step -= 1
            
            new_rels[str(anchor)][str(orphan)] = 5
            
        item["thought_relations"] = dict(new_rels)
        
    return item, stats

# ==========================================
# 4. Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input non-causal JSON file")
    parser.add_argument("--output", required=True, help="Output causal JSON file")
    args = parser.parse_args()
    
    data = load_data(Path(args.input))
    
    total_pruned = 0
    total_fixed = 0
    fix_types = Counter()
    
    print("\n=== Starting Causal Conversion ===\n")
    
    fixed_data = []
    for i, item in enumerate(data):
        new_item, s = fix_item(item)
        fixed_data.append(new_item)
        
        total_pruned += s['pruned_edges']
        total_fixed += s['orphans_fixed']
        
        fix_types['Step 1 -> Root'] += s['fix_type_step1_root']
        fix_types['Step N -> Root (Deep Fallback)'] += s['fix_type_deep_root']
        fix_types['Step N -> Step N-1 (Immediate)'] += s['fix_type_immediate']
        fix_types['Step N -> Step N-k (Gap Bridge)'] += s['fix_type_gap_bridge']
        
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} items...")

    # --- REPORT ---
    print("\n" + "="*60)
    print("CAUSAL CONVERSION STATISTICS")
    print("="*60)
    print(f"Total Graphs: {len(data)}")
    print(f"Total Edges Pruned (Time Travel): {total_pruned}")
    print(f"Total Orphans Fixed (Rewired):    {total_fixed}")
    
    if total_fixed > 0:
        print("\nRewiring Strategy Distribution:")
        print("-" * 30)
        for ftype, count in fix_types.most_common():
            pct = (count / total_fixed) * 100
            print(f"{ftype:<35}: {count:>6} ({pct:>5.1f}%)")
    print("="*60 + "\n")
    
    save_data(fixed_data, Path(args.output))

if __name__ == "__main__":
    main()