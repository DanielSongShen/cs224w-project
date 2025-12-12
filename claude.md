# CS224W Project Overview

## Project Description
Final project for CS 224W focused on using Graph Neural Networks (GNNs) to extract important reasoning steps from LLM reasoning traces. The project builds graph structures from long chain-of-thought (LCoT) reasoning and uses GNNs to analyze and identify critical reasoning steps.

## Technology Stack
- Python 3.12-3.14
- PyTorch 2.9+
- PyTorch Geometric 2.7+
- NetworkX 3.5+
- Sentence Transformers 5.1+
- HuggingFace Datasets 4.3+

## Directory Structure

### Root Configuration Files
- [pyproject.toml](pyproject.toml) - Poetry project configuration with dependencies
- [config.json](config.json) - Model configurations for various LLMs (DeepSeek, GPT-5, Qwen models)
- [requirements.txt](requirements.txt) - Python package dependencies
- [poetry.lock](poetry.lock) - Locked dependency versions
- [README.md](README.md) - Basic project description
- [LICENSE](LICENSE) - Project license
- [commands](commands) - Custom commands or shortcuts

### Source Code ([src/](src/))

#### Data Module ([src/data/](src/data/))
Core data processing and graph construction:
- [dataset.py](src/data/dataset.py) - Dataset loading and management
- [graph_builder.py](src/data/graph_builder.py) - Graph construction from reasoning traces
- [parser.py](src/data/parser.py) - Data parsing utilities
- [lcot2tree_wrapper.py](src/data/lcot2tree_wrapper.py) - Wrapper for LCoT2Tree integration
- [llm_client.py](src/data/llm_client.py) - LLM API client for generating reasoning traces

##### Data Pipeline ([src/data/pipeline/](src/data/pipeline/))
Multi-step processing pipeline for converting chain-of-thought to graphs:
- [config.py](src/data/pipeline/config.py) - Pipeline configuration
- [builder.py](src/data/pipeline/builder.py) - Pipeline orchestration
- [llm_client.py](src/data/pipeline/llm_client.py) - Pipeline-specific LLM client
- [utils.py](src/data/pipeline/utils.py) - Pipeline utility functions

##### Pipeline Steps ([src/data/pipeline/steps/](src/data/pipeline/steps/))
Sequential processing steps:
1. [step1_split.py](src/data/pipeline/steps/step1_split.py) - Split CoT into individual thoughts
2. [step2_sketch.py](src/data/pipeline/steps/step2_sketch.py) - Extract reasoning sketch/outline
3. [step3_assign.py](src/data/pipeline/steps/step3_assign.py) - Assign thoughts to sketch nodes
4. [step4_link.py](src/data/pipeline/steps/step4_link.py) - Link nodes and build connections
5. [step5_graph.py](src/data/pipeline/steps/step5_graph.py) - Generate final graph structure

#### Models ([src/models/](src/models/))
- [gin.py](src/models/gin.py) - Graph Isomorphism Network (GIN) implementation

#### Training ([src/training/](src/training/))
- [trainer.py](src/training/trainer.py) - Model training logic
- [evaluator.py](src/training/evaluator.py) - Model evaluation metrics and procedures

#### Explainer ([src/explainer/](src/explainer/))
- [gnn_explainer.py](src/explainer/gnn_explainer.py) - GNN interpretability and explanation
- [visualizer.py](src/explainer/visualizer.py) - Visualization of graphs and explanations

### Scripts ([scripts/](scripts/))
Executable scripts for various tasks:

#### Data Processing & Visualization
- [01_parse_data.py](scripts/01_parse_data.py) - Parse raw data into structured format
- [01_parse_new_data.py](scripts/01_parse_new_data.py) - Parse new data format
- [01_1_visualize_tree.py](scripts/01_1_visualize_tree.py) - Visualize reasoning trees
- [01_4_visualize_enchanced.py](scripts/01_4_visualize_enchanced.py) - Enhanced tree visualization
- [01_6_regenerate_graphs_new.py](scripts/01_6_regenerate_graphs_new.py) - Regenerate graph structures
- [01_7_regenerate_graphs_new_modified.py](scripts/01_7_regenerate_graphs_new_modified.py) - Modified graph regeneration (loads step 3 from lcot2tree)
- [01_8_fix_graphs.py](scripts/01_8_fix_graphs.py) - Fix graph connectivity issues
- [00_fix_labels.py](scripts/00_fix_labels.py) - Fix dataset labels
- [view_sketches.py](scripts/view_sketches.py) - View reasoning sketches

#### Model Training & Evaluation
- [02_train_model.py](scripts/02_train_model.py) - Train GNN models
- [02_1_hparam_sweeps.py](scripts/02_1_hparam_sweeps.py) - Hyperparameter sweep experiments
- [03_explain.py](scripts/03_explain.py) - Generate model explanations
- [03_precompute_embeddings.py](scripts/03_precompute_embeddings.py) - Precompute node embeddings
- [04_evaluate.py](scripts/04_evaluate.py) - Evaluate trained models

#### Utilities
- [00_sanity_test.py](scripts/00_sanity_test.py) - Sanity checks for data and models
- [test_data_loading.py](scripts/test_data_loading.py) - Test data loading pipeline
- [continue_from_step.py](scripts/continue_from_step.py) - Resume pipeline from specific step
- [analyze_sweep_results.py](scripts/analyze_sweep_results.py) - Analyze hyperparameter sweep results
- [sort_sweep_results.py](scripts/sort_sweep_results.py) - Sort and organize sweep results
- [merge_scores.py](scripts/merge_scores.py) - Merge evaluation scores

### Data Directories

#### [data/](data/)
- [data/raw/](data/raw/) - Raw input data
- [data/processed/](data/processed/) - Processed data and graphs
  - `data/processed/lcot2tree/` - LCoT2Tree pipeline outputs (configured in config.json)

#### [outputs/](outputs/)
- [outputs/models/](outputs/models/) - Trained model checkpoints
- [outputs/results/](outputs/results/) - Evaluation results and metrics
- [outputs/visualizations/](outputs/visualizations/) - Generated visualizations

### External Dependencies

#### [LCoT2Tree/](LCoT2Tree/)
Git submodule containing the LCoT2Tree framework for analyzing long chain-of-thought reasoning. See [LCoT2Tree/README.md](LCoT2Tree/README.md) for details.

Key components:
- LightEval framework for model evaluation
- CoT to tree conversion pipeline
- Mathematical reasoning task evaluation

### [archive/](archive/)
Archived or deprecated scripts:
- [01_3_visual_new_trees.py](archive/01_3_visual_new_trees.py) - Old visualization code
- [01_5_regenerate_graphs.py](archive/01_5_regenerate_graphs.py) - Old graph regeneration

### [notebooks/](notebooks/)
Jupyter notebooks for interactive exploration and analysis

## Typical Workflow

1. **Data Generation**: Use LCoT2Tree to generate reasoning traces with LLMs
2. **Data Processing**: Run pipeline steps (scripts/01_*.py) to convert CoT to graphs
   - Split reasoning into thoughts
   - Extract reasoning sketch
   - Assign thoughts to sketch nodes
   - Link nodes and build graph structure
3. **Embedding**: Precompute node embeddings (03_precompute_embeddings.py)
4. **Training**: Train GNN models (02_train_model.py)
5. **Evaluation**: Evaluate model performance (04_evaluate.py)
6. **Explanation**: Generate and visualize explanations (03_explain.py)

## Key Concepts

- **Chain-of-Thought (CoT)**: Step-by-step reasoning traces from LLMs
- **Reasoning Graphs**: Graph structures where nodes are reasoning steps and edges represent dependencies
- **GNN Analysis**: Using Graph Neural Networks to identify critical reasoning steps
- **Reasoning Sketch**: High-level outline/structure of the reasoning process
- **Node Embeddings**: Vector representations of reasoning steps using sentence transformers

## Recent Updates (from git log)
- Added sentence embeddings and updated node encoders
- Implemented robust node encoder
- Fixed graph connectivity (01_8_fix_graphs.py connects all step 1 nodes to root)
- Updated pipeline configuration to remove `\n\n` splits
- Integrated step4_link.py with graph fixes
