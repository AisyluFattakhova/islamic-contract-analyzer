"""
RAG Query System for Shariaa Standards.

This script allows you to query the vector database using natural language
and retrieve relevant Shariaa Standards sections.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
CHROMA_DB_DIR = DATASET_DIR / "chroma_db"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
FAISS_INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
FAISS_MAPPING_PATH = EMBEDDINGS_DIR / "faiss_mapping.json"
DATASET_PATH = DATASET_DIR / "standards_dataset_with_embeddings.csv"


def load_embedding_model():
    """Load the embedding model used for queries."""
    # Get model name from metadata if available (must match the model used for embeddings)
    model_name = "BAAI/bge-base-en-v1.5"  # Default (must match generate_embeddings.py)
    
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
            model_name = metadata.get('model_name', model_name)
    
    print(f"Loading embedding model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    # Store model name for later use
    model._model_name = model_name
    print(f"✅ Model loaded on {device}")
    
    return model


def load_chromadb():
    """Load ChromaDB collection."""
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError(
            "ChromaDB not installed. Install it with: pip install chromadb"
        )
    
    if not CHROMA_DB_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB database not found at {CHROMA_DB_DIR}\n"
            f"Please run 'python scripts/setup_vector_db.py' first."
        )
    
    print(f"Loading ChromaDB from: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    collection_name = "shariaa_standards"
    try:
        collection = client.get_collection(collection_name)
        print(f"✅ Loaded collection: {collection_name} ({collection.count()} documents)")
    except Exception as e:
        raise FileNotFoundError(
            f"Collection '{collection_name}' not found in ChromaDB.\n"
            f"Please run 'python scripts/setup_vector_db.py' first."
        )
    
    return collection


def load_faiss():
    """Load FAISS index and mapping."""
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "FAISS not installed. Install it with: pip install faiss-cpu"
        )
    
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}\n"
            f"Please run 'python scripts/setup_vector_db.py' first."
        )
    
    if not FAISS_MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"FAISS mapping not found at {FAISS_MAPPING_PATH}\n"
            f"Please run 'python scripts/setup_vector_db.py' first."
        )
    
    print(f"Loading FAISS index from: {FAISS_INDEX_PATH}")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✅ Loaded FAISS index ({index.ntotal} vectors)")
    
    with open(FAISS_MAPPING_PATH, 'r') as f:
        mapping = json.load(f)
    print(f"✅ Loaded FAISS mapping")
    
    return index, mapping


def expand_query(query_text):
    """
    Expand query with variants for better retrieval, especially for definition queries.
    
    Args:
        query_text: Original query
    
    Returns:
        List of query variants
    """
    query_lower = query_text.lower()
    variants = [query_text]  # Always include original
    
    # Detect definition queries
    is_definition_query = any(phrase in query_lower for phrase in [
        "what is", "what are", "define", "definition", "explain", "meaning of"
    ])
    
    if is_definition_query:
        # Extract the term being asked about
        # Simple extraction - look for "what is X" or "define X"
        import re
        match = re.search(r'(?:what is|what are|define|definition of|explain|meaning of)\s+([^?]+)', query_lower)
        if match:
            term = match.group(1).strip()
            # Add variants
            variants.extend([
                f"Definition of {term}",
                f"{term} definition",
                f"{term} is",
                f"{term} means",
                f"Concept of {term}",
            ])
    
    return variants


def query_chromadb(collection, query_text, model, n_results=5, use_query_expansion=True):
    """
    Query ChromaDB collection with improved query handling.
    
    Args:
        collection: ChromaDB collection
        query_text: Query string
        model: SentenceTransformer model for encoding
        n_results: Number of results to return
        use_query_expansion: Whether to use query expansion
    
    Returns:
        Dictionary with documents, metadata, and distances (normalized structure)
    """
    # Expand query if enabled
    if use_query_expansion:
        query_variants = expand_query(query_text)
        # Use the first variant (original) for encoding, but retrieve more results
        query_for_encoding = query_variants[0]
        # Retrieve more results to account for better matching
        n_results_retrieve = min(n_results * 2, 20)
    else:
        query_for_encoding = query_text
        n_results_retrieve = n_results
    
    # Encode query - BGE models work best with instruction prefix
    # Check if model is BGE by checking model name
    model_name = getattr(model, '_model_name', '') or str(model)
    is_bge_model = 'bge' in model_name.lower()
    
    if is_bge_model:
        # BGE models benefit from instruction prefix for queries
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query_for_encoding}"
        query_embedding = model.encode([query_with_instruction], normalize_embeddings=True)[0].tolist()
    else:
        query_embedding = model.encode([query_for_encoding], normalize_embeddings=True)[0].tolist()
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results_retrieve,
        include=['documents', 'metadatas', 'distances']
    )
    
    # ChromaDB returns nested lists (one list per query), extract first query's results
    normalized_results = {
        'ids': results['ids'][0] if results['ids'] else [],
        'documents': results['documents'][0] if results['documents'] else [],
        'metadatas': results['metadatas'][0] if results['metadatas'] else [],
        'distances': results['distances'][0] if results['distances'] else []
    }
    
    # Post-process: boost definition sections for definition queries
    if use_query_expansion and any(phrase in query_text.lower() for phrase in [
        "what is", "what are", "define", "definition", "explain"
    ]):
        normalized_results = boost_definition_sections(normalized_results)
    
    # Return top n_results
    normalized_results = {
        'ids': normalized_results['ids'][:n_results],
        'documents': normalized_results['documents'][:n_results],
        'metadatas': normalized_results['metadatas'][:n_results],
        'distances': normalized_results['distances'][:n_results]
    }
    
    return normalized_results


def boost_definition_sections(results):
    """
    Boost sections that are likely to contain definitions.
    
    Args:
        results: Results dictionary with documents, metadatas, distances
    
    Returns:
        Reordered results with definition sections prioritized
    """
    if not results['documents']:
        return results
    
    # Score each result
    scored_results = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['distances']
    )):
        score = distance  # Lower is better (distance)
        doc_lower = doc.lower()
        
        # Boost definition sections
        if "definition" in doc_lower[:100]:  # Check first 100 chars
            score -= 0.2  # Boost (reduce distance)
        if any(pattern in doc_lower[:100] for pattern in [" is ", " means ", " refers to "]):
            score -= 0.1  # Boost
        
        # Penalize very short sections (likely incomplete)
        if len(doc) < 50:
            score += 0.1  # Penalize (increase distance)
        
        # Penalize wrong topics (simple keyword check)
        # This is a heuristic - could be improved
        if "mudarabah" in doc_lower and "murabahah" not in doc_lower:
            score += 0.15
        
        scored_results.append((score, i, doc, metadata, distance))
    
    # Sort by score (lower is better)
    scored_results.sort(key=lambda x: x[0])
    
    # Rebuild results in new order
    reordered_results = {
        'ids': [],
        'documents': [],
        'metadatas': [],
        'distances': []
    }
    
    for _, orig_idx, doc, metadata, distance in scored_results:
        reordered_results['ids'].append(results['ids'][orig_idx])
        reordered_results['documents'].append(doc)
        reordered_results['metadatas'].append(metadata)
        reordered_results['distances'].append(distance)
    
    return reordered_results


def query_faiss(index, mapping, query_text, model, df, n_results=5, use_query_expansion=True):
    """
    Query FAISS index.
    
    Args:
        index: FAISS index
        query_text: Query string
        model: SentenceTransformer model for encoding
        df: DataFrame with section data
        n_results: Number of results to return
    
    Returns:
        List of results with documents, metadata, and distances
    """
    # Encode query - use expanded query if enabled
    if use_query_expansion:
        query_variants = expand_query(query_text)
        query_for_encoding = query_variants[0]
    else:
        query_for_encoding = query_text
    
    # Check if BGE model and add instruction prefix
    model_name = getattr(model, '_model_name', '') or str(model)
    is_bge_model = 'bge' in model_name.lower()
    
    if is_bge_model:
        query_with_instruction = f"Represent this sentence for searching relevant passages: {query_for_encoding}"
        query_embedding = model.encode([query_with_instruction], normalize_embeddings=True)[0]
    else:
        query_embedding = model.encode([query_for_encoding], normalize_embeddings=True)[0]
    
    query_embedding = query_embedding.astype('float32').reshape(1, -1)
    
    # Retrieve more results for post-processing
    n_results_retrieve = min(n_results * 2, 20) if use_query_expansion else n_results
    
    # Search FAISS
    distances, indices = index.search(query_embedding, n_results_retrieve)
    
    # Get results
    results = {
        'ids': [],
        'documents': [],
        'metadatas': [],
        'distances': []
    }
    
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        # Map FAISS index to dataset row
        embedding_index = mapping['index_to_row'].get(str(idx))
        if embedding_index is None:
            continue
        
        # Get row from dataset
        row = df[df['embedding_index'] == embedding_index].iloc[0]
        
        results['ids'].append(f"section_{embedding_index}")
        results['documents'].append(row['content'])
        results['metadatas'].append({
            'section_number': str(row['section_number']),
            'section_path': str(row['section_path']),
            'content_length': int(row['content_length']),
            'line_number': int(row['line_number']),
            'embedding_index': int(row['embedding_index'])
        })
        results['distances'].append(float(dist))
    
    # Post-process: boost definition sections for definition queries
    if use_query_expansion and any(phrase in query_text.lower() for phrase in [
        "what is", "what are", "define", "definition", "explain"
    ]):
        results = boost_definition_sections(results)
    
    # Return top n_results
    results = {
        'ids': results['ids'][:n_results],
        'documents': results['documents'][:n_results],
        'metadatas': results['metadatas'][:n_results],
        'distances': results['distances'][:n_results]
    }
    
    return results


def display_results(query, results, db_type="ChromaDB"):
    """Display query results in a formatted way."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Database: {db_type}")
    print(f"{'='*80}")
    
    if not results.get('documents') or len(results['documents']) == 0:
        print("\n❌ No results found.")
        return
    
    # Extract lists (handle both ChromaDB nested structure and FAISS flat structure)
    documents = results['documents']
    metadatas = results['metadatas']
    distances = results.get('distances', [])
    
    # Ensure distances is a list of numbers
    if distances and isinstance(distances[0], list):
        distances = distances[0]
    
    for i in range(len(documents)):
        doc = documents[i]
        metadata = metadatas[i] if i < len(metadatas) else {}
        distance = distances[i] if i < len(distances) else None
        
        print(f"\n{'─'*80}")
        if distance is not None:
            print(f"Result {i+1} (Distance: {distance:.4f})")
        else:
            print(f"Result {i+1}")
        
        if metadata:
            section_path = metadata.get('section_path', 'N/A')
            section_number = metadata.get('section_number', 'N/A')
            content_length = metadata.get('content_length', 0)
            print(f"Section: {section_path} ({section_number})")
            print(f"Content Length: {content_length} chars")
        
        print(f"{'─'*80}")
        print(f"{doc[:500]}{'...' if len(doc) > 500 else ''}")
        print()


def interactive_query(collection=None, index=None, mapping=None, model=None, df=None, db_type="ChromaDB"):
    """Interactive query loop."""
    print(f"\n{'='*80}")
    print("RAG Query System - Interactive Mode")
    print(f"{'='*80}")
    print("\nEnter your queries (type 'quit' or 'exit' to stop)")
    print("Example queries:")
    print("  - What is Murabahah?")
    print("  - What are the rules for currency trading?")
    print("  - What is Riba and when is it prohibited?")
    print()
    
    while True:
        try:
            query = input("\n🔍 Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Query the database (ChromaDB only)
            if db_type == "ChromaDB":
                results = query_chromadb(collection, query, model, n_results=5, use_query_expansion=True)
            else:
                raise FileNotFoundError("Only ChromaDB is supported. FAISS has been removed for Streamlit Cloud compatibility.")
            
            # Display results
            display_results(query, results, db_type)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def single_query(query, collection=None, index=None, mapping=None, model=None, df=None, db_type="ChromaDB", n_results=5):
    """Perform a single query and return results (ChromaDB only)."""
    if db_type == "ChromaDB":
        results = query_chromadb(collection, query, model, n_results=n_results, use_query_expansion=True)
    else:
        raise FileNotFoundError("Only ChromaDB is supported. FAISS has been removed for Streamlit Cloud compatibility.")
    
    display_results(query, results, db_type)
    return results


def detect_database_type():
    """Detect which database is available."""
    if CHROMA_DB_DIR.exists():
        try:
            import chromadb
            from chromadb.config import Settings
            client = chromadb.PersistentClient(
                path=str(CHROMA_DB_DIR),
                settings=Settings(anonymized_telemetry=False)
            )
            client.get_collection("shariaa_standards")
            return "ChromaDB"
        except:
            pass
    
    # FAISS removed for Streamlit Cloud compatibility - ChromaDB only
    return None


def main():
    """Main function."""
    print("="*80)
    print("RAG Query System for Shariaa Standards")
    print("="*80)
    
    # Detect database type
    db_type = detect_database_type()
    if db_type is None:
        print("\n❌ No vector database found!")
        print("Please run 'python scripts/setup_vector_db.py' first.")
        sys.exit(1)
    
    print(f"\n✅ Detected database: {db_type}")
    
    # Load embedding model
    try:
        model = load_embedding_model()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Load database
    collection = None
    index = None
    mapping = None
    df = None
    
    try:
        if db_type == "ChromaDB":
            collection = load_chromadb()
        else:
            raise FileNotFoundError("Only ChromaDB is supported. FAISS has been removed for Streamlit Cloud compatibility.")
    except Exception as e:
        print(f"\n❌ Error loading database: {e}")
        sys.exit(1)
    
    # Check if query provided as command line argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}\n")
        single_query(query, collection, index, mapping, model, df, db_type)
    else:
        # Interactive mode
        interactive_query(collection, index, mapping, model, df, db_type)


if __name__ == "__main__":
    main()

