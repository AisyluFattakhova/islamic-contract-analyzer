"""
Script to set up a vector database (ChromaDB or FAISS) for RAG queries.

This script loads embeddings and creates a searchable vector database.
"""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
DATASET_PATH = DATASET_DIR / "standards_dataset_with_embeddings.csv"
CHROMA_DB_DIR = DATASET_DIR / "chroma_db"


def load_embeddings_and_dataset():
    """Load embeddings and dataset with metadata."""
    # Check if embeddings exist
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(
            f"Embeddings not found at {EMBEDDINGS_PATH}\n"
            f"Please run 'python scripts/generate_embeddings.py' first."
        )
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset with embeddings not found at {DATASET_PATH}\n"
            f"Please run 'python scripts/generate_embeddings.py' first."
        )
    
    # Load embeddings
    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH)
    print(f"✅ Loaded embeddings: shape {embeddings.shape}")
    
    # Load metadata
    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✅ Loaded metadata: {metadata['model_name']}, dim={metadata['embedding_dimension']}")
    else:
        metadata = None
        print("⚠️  No metadata file found")
    
    # Load dataset
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded dataset: {len(df)} sections")
    
    # Verify embeddings match dataset
    if len(embeddings) != len(df):
        raise ValueError(
            f"Mismatch: {len(embeddings)} embeddings but {len(df)} dataset rows"
        )
    
    return embeddings, df, metadata


def setup_chromadb(embeddings, df, metadata):
    """
    Set up ChromaDB vector database.
    
    Args:
        embeddings: numpy array of embeddings
        df: DataFrame with section data
        metadata: Embedding metadata dict
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError:
        raise ImportError(
            "ChromaDB not installed. Install it with: pip install chromadb"
        )
    
    print(f"\n{'='*60}")
    print("Setting up ChromaDB Vector Database")
    print(f"{'='*60}")
    
    # Create ChromaDB directory
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client with persistent storage
    print(f"\nInitializing ChromaDB at: {CHROMA_DB_DIR}")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Create or get collection
    collection_name = "shariaa_standards"
    print(f"\nCreating collection: {collection_name}")
    
    # Delete existing collection if it exists (for re-running)
    try:
        client.delete_collection(collection_name)
        print("  (Deleted existing collection)")
    except:
        pass
    
    # Create new collection
    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "Shariaa Standards for Islamic Contract Analysis",
            "model": metadata.get('model_name', 'unknown') if metadata else 'unknown',
            "embedding_dim": metadata.get('embedding_dimension', embeddings.shape[1]) if metadata else embeddings.shape[1]
        }
    )
    
    print(f"✅ Collection created")
    
    # Prepare data for ChromaDB
    # ChromaDB expects: ids, embeddings, documents, metadatas
    print(f"\nPreparing data for ChromaDB...")
    
    ids = []
    documents = []
    metadatas = []
    embeddings_list = []
    
    for idx, row in df.iterrows():
        # Create unique ID
        doc_id = f"section_{row['embedding_index']}"
        ids.append(doc_id)
        
        # Document text (content)
        documents.append(str(row['content']))
        
        # Metadata
        metadata = {
            'section_number': str(row['section_number']),
            'section_path': str(row['section_path']),
            'content_length': int(row['content_length']),
            'line_number': int(row['line_number']),
            'embedding_index': int(row['embedding_index'])
        }
        metadatas.append(metadata)
        
        # Embedding (convert numpy array to list)
        embeddings_list.append(embeddings[idx].tolist())
    
    print(f"  Prepared {len(ids)} documents")
    
    # Add to ChromaDB in batches (ChromaDB can handle large batches, but let's be safe)
    batch_size = 1000
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"\nAdding documents to ChromaDB (in {total_batches} batches)...")
    for i in range(0, len(ids), batch_size):
        batch_end = min(i + batch_size, len(ids))
        batch_ids = ids[i:batch_end]
        batch_docs = documents[i:batch_end]
        batch_embeddings = embeddings_list[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_docs,
            metadatas=batch_metadatas
        )
        
        print(f"  Added batch {i//batch_size + 1}/{total_batches} ({batch_end}/{len(ids)} documents)")
    
    # Verify collection
    count = collection.count()
    print(f"\n✅ ChromaDB setup complete!")
    print(f"   Collection: {collection_name}")
    print(f"   Documents: {count}")
    print(f"   Database location: {CHROMA_DB_DIR}")
    
    # Test query
    print(f"\nTesting query...")
    results = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=3
    )
    print(f"✅ Test query successful! Retrieved {len(results['ids'][0])} results")
    
    return collection, client


def setup_faiss(embeddings, df, metadata):
    """
    Set up FAISS vector database (alternative to ChromaDB).
    
    Args:
        embeddings: numpy array of embeddings
        df: DataFrame with section data
        metadata: Embedding metadata dict
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "FAISS not installed. Install it with: pip install faiss-cpu"
        )
    
    print(f"\n{'='*60}")
    print("Setting up FAISS Vector Database")
    print(f"{'='*60}")
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    print(f"\nCreating FAISS index (dimension={dimension})...")
    
    # Use L2 distance (Euclidean) - ChromaDB uses cosine by default, but we can normalize
    # Since embeddings are already normalized, L2 works well
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings (FAISS expects float32)
    embeddings_float32 = embeddings.astype('float32')
    print(f"Adding {len(embeddings_float32)} embeddings to index...")
    index.add(embeddings_float32)
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    
    # Save FAISS index
    faiss_path = EMBEDDINGS_DIR / "faiss.index"
    print(f"\nSaving FAISS index to: {faiss_path}")
    faiss.write_index(index, str(faiss_path))
    print(f"✅ FAISS index saved")
    
    # Save metadata mapping (FAISS doesn't store metadata, so we need a separate file)
    mapping_path = EMBEDDINGS_DIR / "faiss_mapping.json"
    mapping = {
        'index_to_row': {i: int(df.iloc[i]['embedding_index']) for i in range(len(df))},
        'row_to_index': {int(row['embedding_index']): i for i, row in df.iterrows()}
    }
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ FAISS mapping saved to: {mapping_path}")
    
    return index, faiss_path


def main():
    """Main function to set up vector database."""
    print("="*60)
    print("Vector Database Setup")
    print("="*60)
    
    # Load embeddings and dataset
    try:
        embeddings, df, metadata = load_embeddings_and_dataset()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    # Ask user which database to use
    print(f"\n{'='*60}")
    print("Choose vector database:")
    print("  1. ChromaDB (Recommended - Persistent, easy to use)")
    print("  2. FAISS (Lightweight, in-memory, faster)")
    print(f"{'='*60}")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    if not choice:
        choice = "1"
    
    try:
        if choice == "1":
            # Setup ChromaDB
            collection, client = setup_chromadb(embeddings, df, metadata)
            print(f"\n{'='*60}")
            print("✅ ChromaDB setup complete!")
            print(f"{'='*60}")
            print(f"\nNext steps:")
            print(f"  - Use ChromaDB in your RAG queries")
            print(f"  - Database location: {CHROMA_DB_DIR}")
            print(f"  - Collection name: shariaa_standards")
            
        elif choice == "2":
            # Setup FAISS
            index, faiss_path = setup_faiss(embeddings, df, metadata)
            print(f"\n{'='*60}")
            print("✅ FAISS setup complete!")
            print(f"{'='*60}")
            print(f"\nNext steps:")
            print(f"  - Use FAISS index in your RAG queries")
            print(f"  - Index location: {faiss_path}")
            print(f"  - Note: FAISS is in-memory, load index each time")
            
        else:
            print(f"\n❌ Invalid choice: {choice}")
            sys.exit(1)
            
    except ImportError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error setting up vector database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

