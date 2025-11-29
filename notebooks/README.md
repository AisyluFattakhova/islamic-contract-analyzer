# Notebooks for Islamic Contract Analyzer Dataset

This directory contains Jupyter notebooks for exploring and preparing the Shariaa Standards dataset.

## Notebooks Overview

### 1. `01_document_structure_analysis.ipynb`
**Purpose**: Analyze the structure of the Shariaa Standards document

**What it does**:
- Extracts standard numbers and titles
- Identifies section numbering patterns (e.g., 3//3//2, 5//1)
- Analyzes document hierarchy
- Examines content distribution (paragraph lengths, sentence lengths)
- Identifies major document sections

**Output**: 
- `datasets/analysis/standards_list.csv`
- `datasets/analysis/sections_list.csv`
- `datasets/analysis/standard_content_stats.csv`

### 2. `02_chunking_strategy_exploration.ipynb`
**Purpose**: Explore different chunking strategies to find the optimal approach

**What it does**:
- Tests fixed-size chunking with various sizes and overlaps
- Tests sentence-boundary aware chunking
- Tests structure-aware chunking (by standards)
- Compares strategies and visualizes results

**Output**:
- `datasets/analysis/sample_fixed_chunks.csv`
- `datasets/analysis/sample_structure_chunks.csv`

### 3. `03_metadata_extraction.ipynb`
**Purpose**: Extract structured metadata from the document

**What it does**:
- Extracts standard metadata (numbers, titles, dates)
- Finds cross-references between standards
- Extracts section numbering and hierarchy
- Identifies definitions and key terms

**Output**:
- `datasets/metadata/standards_metadata.csv`
- `datasets/metadata/cross_references.csv`
- `datasets/metadata/sections.csv`
- `datasets/metadata/definitions.csv`

### 4. `04_embedding_quality_analysis.ipynb`
**Purpose**: Analyze embedding quality and test different models

**What it does**:
- Tests different embedding models (BGE-M3, MiniLM, MPNet)
- Analyzes semantic similarity between chunks
- Tests retrieval quality with sample queries
- Compares model performance

**Output**:
- `datasets/analysis/embedding_model_comparison.csv`

## Running the Notebooks

1. **Install dependencies**:
```bash
pip install jupyter pandas numpy matplotlib seaborn sentence-transformers scikit-learn
```

2. **Start Jupyter**:
```bash
jupyter notebook
```

3. **Run notebooks in order**:
   - Start with `01_document_structure_analysis.ipynb` to understand the document
   - Then `02_chunking_strategy_exploration.ipynb` to test chunking
   - Then `03_metadata_extraction.ipynb` to extract metadata
   - Finally `04_embedding_quality_analysis.ipynb` to test embeddings

## Recommended Workflow

1. **Exploration Phase** (Notebooks):
   - Run all notebooks to understand your data
   - Experiment with different parameters
   - Visualize results

2. **Production Phase** (Scripts):
   - Convert successful notebook code to scripts
   - Use scripts for processing the full dataset
   - Integrate into your pipeline

## Notes

- The notebooks use sample data for faster iteration
- Adjust parameters based on your findings
- Save intermediate results for analysis
- Use the insights to build production scripts

