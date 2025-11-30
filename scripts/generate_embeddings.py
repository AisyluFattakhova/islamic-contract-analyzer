"""
Script to generate embeddings for the Shariaa Standards dataset.

This script loads the dataset, generates embeddings using sentence-transformers,
and saves them for use in the RAG system.
"""
import os
import sys
import json
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DATASET_DIR = PROJECT_ROOT / "datasets"
EMBEDDINGS_DIR = DATASET_DIR / "embeddings"
DATASET_PATH = DATASET_DIR / "standards_dataset.csv"

# Create embeddings directory if it doesn't exist
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu_availability():
    """Check if GPU is available and return device."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU Available: {device_name}")
        print(f"   GPU Memory: {gpu_memory:.1f} GB")
        device = "cuda"
    else:
        print("⚠️  No GPU available. Using CPU (will be slower)")
        device = "cpu"
    return device


def load_dataset(dataset_path):
    """Load the dataset from CSV file."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}\n"
            f"Please run 'python scripts/construct_dataset.py' first to create the dataset."
        )
    
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Verify required columns exist
    required_columns = ['section_number', 'section_path', 'content']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")
    
    print(f"✅ Dataset loaded: {len(df):,} sections")
    print(f"   Columns: {list(df.columns)}")
    
    return df


def generate_embeddings(df, model_name="BAAI/bge-base-en-v1.5", 
                        device="cpu", batch_size=None):
    """
    Generate embeddings for all text content in the dataset.
    
    Args:
        df: DataFrame with 'content' column containing text to embed
        model_name: Name of the sentence transformer model to use
        device: Device to use ('cuda' or 'cpu')
        batch_size: Batch size for encoding (None = auto-detect)
    
    Returns:
        numpy array of embeddings with shape (num_texts, embedding_dim)
    """
    texts = df['content'].tolist()
    print(f"\nTotal texts to embed: {len(texts):,}")
    
    # Auto-detect batch size if not provided
    # BGE model is larger, so use smaller batches
    if batch_size is None:
        if device == "cuda":
            batch_size = 64  # Moderate batch for GPU (BGE is larger than MiniLM)
        else:
            batch_size = 16   # Smaller batch for CPU
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    
    # Clean up memory
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Load model
    print(f"\nLoading model...")
    try:
        model = SentenceTransformer(model_name, device=device)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    # Generate embeddings
    print(f"\nGenerating embeddings...")
    embeddings = []
    start_time = time.time()
    
    try:
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with tqdm(total=len(texts), desc="Processing", unit="text") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                
                # Encode batch
                # BGE models work best with instruction prefix for retrieval
                # Check if this is a BGE model
                is_bge_model = 'bge' in model_name.lower()
                
                if is_bge_model:
                    # Prepend instruction to each text in the batch for BGE models
                    instruction = "Represent this sentence for searching relevant passages:"
                    batch_to_encode = [f"{instruction} {text}" for text in batch]
                else:
                    batch_to_encode = batch
                
                batch_embeddings = model.encode(
                    batch_to_encode,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=len(batch)
                )
                
                embeddings.append(batch_embeddings)
                pbar.update(len(batch))
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"✅ Embeddings generated successfully!")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Speed: {len(texts)/total_time:.1f} texts/sec")
        print(f"Shape: {all_embeddings.shape}")
        print(f"Embedding dimension: {all_embeddings.shape[1]}")
        
        return all_embeddings
        
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        raise


def save_embeddings(embeddings, df, embeddings_dir, model_name):
    """
    Save embeddings and metadata.
    
    Args:
        embeddings: numpy array of embeddings
        df: Original dataframe
        embeddings_dir: Directory to save embeddings
        model_name: Name of the model used
    """
    embeddings_path = embeddings_dir / "embeddings.npy"
    metadata_path = embeddings_dir / "metadata.json"
    dataset_output_path = embeddings_dir.parent / "standards_dataset_with_embeddings.csv"
    
    # Save embeddings
    print(f"\nSaving embeddings to: {embeddings_path}")
    np.save(embeddings_path, embeddings)
    print(f"✅ Saved embeddings ({embeddings.nbytes / 1024**2:.1f} MB)")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'embedding_dimension': int(embeddings.shape[1]),
        'num_embeddings': int(embeddings.shape[0]),
        'processing_time': time.time()  # You can calculate actual time if needed
    }
    
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata")
    
    # Update dataset with embedding indices
    df_output = df.copy()
    df_output['embedding_index'] = range(len(df_output))
    
    print(f"Saving dataset with indices to: {dataset_output_path}")
    df_output.to_csv(dataset_output_path, index=False)
    print(f"✅ Saved dataset with embedding indices")
    
    return embeddings_path, metadata_path, dataset_output_path


def main():
    """Main function to generate embeddings."""
    print("="*60)
    print("Shariaa Standards - Embedding Generation")
    print("="*60)
    
    # Check GPU availability
    device = check_gpu_availability()
    
    # Load dataset
    try:
        df = load_dataset(DATASET_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    # Model configuration
    # Using BGE-base-en-v1.5: Better quality for definitional queries and semantic search
    # Alternative: "sentence-transformers/all-mpnet-base-v2" (slower but higher quality)
    model_name = "BAAI/bge-base-en-v1.5"
    
    # Generate embeddings
    try:
        embeddings = generate_embeddings(df, model_name=model_name, device=device)
    except Exception as e:
        print(f"\n❌ Failed to generate embeddings: {e}")
        sys.exit(1)
    
    # Save embeddings and metadata
    try:
        embeddings_path, metadata_path, dataset_path = save_embeddings(
            embeddings, df, EMBEDDINGS_DIR, model_name
        )
        
        print(f"\n{'='*60}")
        print("✅ Embedding generation complete!")
        print(f"{'='*60}")
        print(f"\nGenerated files:")
        print(f"  - {embeddings_path}")
        print(f"  - {metadata_path}")
        print(f"  - {dataset_path}")
        print(f"\nNext steps:")
        print(f"  1. Run 'python scripts/setup_vector_db.py' to create vector database")
        print(f"  2. Run 'python scripts/rag_query.py' to test RAG queries")
        
    except Exception as e:
        print(f"\n❌ Error saving embeddings: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

