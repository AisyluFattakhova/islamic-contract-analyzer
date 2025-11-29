# Next Steps for Islamic Contract Analyzer

## Current Status ✅

- ✅ PDF extracted to text
- ✅ Text cleaned (removed metadata, TOC, prefaces)
- ✅ Dataset created with **2,286 sections** (only text after section numbers like "2//1", "2//1//1")
- ✅ Dataset saved to `datasets/standards_dataset.csv`

## What to Do Now

### Step 0: Analyze Dataset (Recommended)

Before generating embeddings, analyze your dataset to understand its structure:

```bash
# Open and run the analysis notebook
jupyter notebook notebooks/05_dataset_analysis.ipynb
```

**What this notebook does:**
- Shows dataset statistics (total sections, content lengths)
- Visualizes content length distribution
- Analyzes section numbering patterns
- Checks text quality (empty sections, duplicates, encoding issues)
- Reviews sample content
- Exports analysis results

**Why analyze first:**
- Understand data distribution
- Identify any issues before embedding
- Decide if additional chunking is needed
- Validate data quality

### Step 1: Generate Embeddings (Required)

This converts your text sections into numerical vectors for similarity search.

```bash
# Install required packages
pip install sentence-transformers torch tqdm

# Generate embeddings
python scripts/generate_embeddings.py
```

**What this does:**
- Loads all 2,286 sections from your dataset
- Uses BGE-M3 model (same as SmartClause) to generate embeddings
- Saves embeddings to `datasets/embeddings/embeddings.npy`
- Creates metadata file

**Time:** ~5-10 minutes (first run downloads model ~1.5GB)

### Step 2: Setup Vector Database (Required)

Choose one option:

#### Option A: ChromaDB (Easiest - Recommended for Development)

```bash
pip install chromadb
python scripts/setup_vector_db.py
# Choose option 1
```

#### Option B: FAISS (Lightweight, In-Memory)

```bash
pip install faiss-cpu
python scripts/setup_vector_db.py
# Choose option 2
```

#### Option C: PostgreSQL + pgvector (Production)

Requires PostgreSQL setup. See `scripts/setup_postgresql.sql` (to be created).

### Step 3: Test RAG Query System

```bash
python scripts/rag_query.py
```

This allows you to:
- Query the knowledge base with natural language
- Get relevant Shariaa Standards sections
- Test retrieval quality

**Example queries:**
- "What is Murabahah?"
- "What are the rules for currency trading?"
- "What is Riba and when is it prohibited?"

### Step 4: Build Your Application

After testing the RAG system, you can:

1. **Integrate with LLM** (OpenAI, Anthropic, etc.)
   - Use retrieved sections as context
   - Generate contract analysis

2. **Build Web Interface**
   - Upload contracts
   - Display compliance analysis
   - Show relevant standards

3. **Add Contract Processing**
   - Extract text from uploaded contracts
   - Compare against Shariaa Standards
   - Generate compliance report

## File Structure

```
datasets/
├── standards_dataset.csv              # Your dataset (2,286 sections)
├── embeddings/
│   ├── embeddings.npy                  # Generated after Step 1
│   ├── metadata.json                   # Embedding metadata
│   └── faiss.index                     # FAISS index (if using FAISS)
└── chroma_db/                          # ChromaDB database (if using ChromaDB)
```

## Quick Start Commands

```bash
# 1. Generate embeddings
python scripts/generate_embeddings.py

# 2. Setup vector database (choose ChromaDB)
python scripts/setup_vector_db.py

# 3. Test queries
python scripts/rag_query.py
```

## Troubleshooting

**If embeddings generation fails:**
- Make sure `sentence-transformers` is installed
- Check internet connection (model downloads on first run)
- Ensure you have ~2GB free disk space

**If vector database setup fails:**
- For ChromaDB: `pip install chromadb`
- For FAISS: `pip install faiss-cpu`
- Check that embeddings were generated first

## Next: Integration

Once RAG is working, you can:
1. Connect to an LLM API for contract analysis
2. Build a web interface (Flask/FastAPI + React)
3. Add contract upload and processing
4. Generate compliance reports

