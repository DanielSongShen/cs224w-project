# LCoT2Tree-GNN

> GNN baselines for analyzing LLM chain-of-thought (CoT) reasoning traces by converting them into graphs and training graph classifiers (e.g., correct vs. incorrect reasoning).

**CS 224W final project**  
*Python 3.12+, PyTorch 2.0+, PyTorch Geometric 2.x*

## Table of Contents

- [What this repo does](#what-this-repo-does)
- [Setup](#setup)
- [Quick Start Pipeline](#quick-start-pipeline)
- [Data Format & Graph Structure](#data-format--graph-structure)
- [Models](#models)
- [Repository Structure](#repository-structure)
- [License](#license)

## What this repo does

- ✅ Parse reasoning traces → structured trees/graphs (optionally using an LLM via LCoT2Tree)
- ✅ Build heterogeneous graphs (multiple edge types)
- ✅ Train/evaluate GNNs on graph-level labels
- ✅ (Optional) Add text embeddings for node features

## Setup

```bash
git clone <YOUR_REPO_URL>
cd <REPO_DIR>

pip install -r requirements.txt
# or
poetry install
```

### LLM Configuration (if using LCoT2Tree)

If you're running the LCoT2Tree/LLM-based parsing steps:

```bash
export OPENAI_API_KEY="..."
# or configure your backend in config.json
```

## Quick Start Pipeline

### 1. Process Data (Build Graphs)

Basic usage:
```bash
python scripts/01_parse_data.py
```

Example with explicit dataset and output directory:
```bash
python scripts/01_parse_data.py \
  --n_samples 100 \
  --dataset "PrimeIntellect/verifiable-math-problems" \
  --output_dir ./data/processed/my_experiment
```

**Fix/regenerate graphs** (if needed):
```bash
python scripts/01_8_fix_graphs.py \
  --input_path ./data/processed/<dataset>/final.json \
  --output_path ./data/processed/<dataset>/final_fixed.json
```

### 2. (Optional) Precompute Text Embeddings

```bash
python scripts/03_precompute_embeddings.py \
  --pt-file ./data/processed/<dataset>/final.pt \
  --model-name all-MiniLM-L6-v2 \
  --output-dir ./data/embeddings
```

### 3. Train Models

Basic training:
```bash
python scripts/02_train_model.py \
  --pt-file ./data/processed/<dataset>/final.pt \
  --model-type hetero_gin \
  --hidden-channels 64 \
  --num-layers 3 \
  --epochs 100
```

**Cross-validation:**
```bash
python scripts/02_train_model.py \
  --pt-file ./data/processed/<dataset>/final.pt \
  --cross-validation \
  --n-folds 5
```

**Train with text embeddings:**
```bash
python scripts/02_train_model.py \
  --pt-file ./data/processed/<dataset>/final.pt \
  --embedding-dir ./data/embeddings \
  --encoder-type text_aware
```

### 4. Evaluate

```bash
python scripts/04_evaluate.py \
  --model-path ./outputs/models/best_model.pth \
  --pt-file ./data/processed/<dataset>/final.pt \
  --model-type hetero_gin
```

## Data Format & Graph Structure

### Input/Output Specification

- **Input**: JSONL reasoning traces (or dataset loaders) with per-example labels
- **Output**: `.pt` file containing PyG `HeteroData` graphs

### Graph Components

- **Nodes**: "thought" steps (optionally with text content)
- **Edges**: Heterogeneous relations including:
  - Continuous flow
  - Exploration/backtracking
  - Validation
  - Root connections

## Models

### Available Architectures

- **HeteroGIN / HeteroGAT**: Per-edge-type message passing on heterogeneous graphs
- **SimpleHeteroGNN**: Projects edge types into simpler structure (baseline)

### Training Features

All models support:
- ✅ Train/validation/test splits
- ✅ Logging and checkpointing
- ✅ Early stopping
- ✅ Optional cross-validation

## Repository Structure

```
├── src/
│   ├── data/          # Parsing + graph construction (+ LLM client hooks)
│   ├── models/        # GNNs + model factory
│   └── training/      # Trainer utilities
├── scripts/           # Entrypoints: parse → train → evaluate
├── LCoT2Tree/         # Submodule (if enabled)
└── data/              # Processed datasets and outputs
```

## License

MIT (see [LICENSE](LICENSE))