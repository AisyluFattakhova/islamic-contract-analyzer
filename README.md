# Islamic Contract Analyzer

An AI-powered system that allows users to upload contracts and analyze them for Shariah compliance using RAG (Retrieval-Augmented Generation) technology.

## Project Structure

```
islamic-contract-analyzer/
├── raw_pdf/              # Original PDF files
├── clean_text/           # Cleaned text files
├── datasets/             # Processed datasets and embeddings
│   ├── embeddings/       # Generated embeddings
│   └── analysis/         # Analysis results
├── scripts/              # Processing scripts
│   ├── pdf_to_text.py           # Extract text from PDFs
│   ├── construct_dataset.py     # Build dataset from cleaned text
│   ├── generate_embeddings.py    # Generate embeddings for RAG
│   ├── setup_vector_db.py       # Setup vector database
│   └── rag_query.py             # RAG query pipeline
├── notebooks/            # Jupyter notebooks for analysis
└── metadata/            # Document metadata

```

## Quick Start

### 1. Extract Text from PDF
```bash
python scripts/pdf_to_text.py
```

### 2. Construct Dataset
```bash
python scripts/construct_dataset.py
```
This creates `datasets/standards_dataset.csv` with sections following numbering patterns (e.g., "2//1", "2//1//1").

### 3. Generate Embeddings
```bash
# First install requirements
pip install sentence-transformers torch tqdm

# Generate embeddings
python scripts/generate_embeddings.py
```
This uses the BGE-M3 model (same as SmartClause) to generate embeddings for all sections.

### 4. Setup Vector Database
```bash
# Option 1: ChromaDB (easiest)
pip install chromadb
python scripts/setup_vector_db.py
# Choose option 1

# Option 2: FAISS (lightweight)
pip install faiss-cpu
python scripts/setup_vector_db.py
# Choose option 2
```

### 5. Query the RAG System
```bash
python scripts/rag_query.py
```

## Dataset

The dataset contains **2,286 sections** extracted from Shariaa Standards, with only text that appears after section numbers (e.g., "2//1", "2//1//1"). Each section includes:
- `section_number`: The section identifier
- `content`: The text content
- `content_length`: Length in characters
- `line_number`: Original line number

## Next Steps

1. **Generate Embeddings**: Run `generate_embeddings.py` to create vector representations
2. **Setup Vector DB**: Choose ChromaDB (easy) or PostgreSQL+pgvector (production)
3. **Build RAG Pipeline**: Use `rag_query.py` to query the knowledge base
4. **Integrate with LLM**: Connect to OpenAI/Anthropic for contract analysis
5. **Build Web Interface**: Create UI for contract upload and analysis

## Reference

This project follows best practices from [SmartClause](https://github.com/IU-Capstone-Project-2025/SmartClause) for dataset creation and RAG implementation.
