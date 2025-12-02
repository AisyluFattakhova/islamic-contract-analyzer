"""
Vector Store Abstraction for RAG System.

Provides unified interface for different vector databases.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from pathlib import Path
import json


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def query(self, query_embedding: List[float], n_results: int = 5, 
              include_metadata: bool = True) -> Dict:
        """
        Query the vector store.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            include_metadata: Whether to include metadata
        
        Returns:
            Dict with 'documents', 'metadatas', 'distances', 'ids'
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total number of documents."""
        pass


class ChromaDBVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore."""
    
    def __init__(self, collection_name: str = "shariaa_standards", 
                 db_path: Optional[Path] = None):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of ChromaDB collection
            db_path: Path to ChromaDB database
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Run: pip install chromadb")
        
        if db_path is None:
            db_path = Path(__file__).parent.parent / "datasets" / "chroma_db"
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception as e:
            raise FileNotFoundError(
                f"Collection '{collection_name}' not found. Run setup first."
            )
    
    def query(self, query_embedding: List[float], n_results: int = 5,
              include_metadata: bool = True) -> Dict:
        """Query ChromaDB."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Normalize nested structure
        return {
            'ids': results['ids'][0] if results['ids'] else [],
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()


class FAISSVectorStore(VectorStore):
    """FAISS implementation of VectorStore."""
    
    def __init__(self, index_path: Optional[Path] = None,
                 mapping_path: Optional[Path] = None,
                 dataset_path: Optional[Path] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            index_path: Path to FAISS index file
            mapping_path: Path to mapping JSON file
            dataset_path: Path to dataset CSV
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")
        
        import pandas as pd
        
        if index_path is None:
            base_path = Path(__file__).parent.parent / "datasets" / "embeddings"
            index_path = base_path / "faiss.index"
            mapping_path = base_path / "faiss_mapping.json"
            dataset_path = Path(__file__).parent.parent / "datasets" / "standards_dataset_with_embeddings.csv"
        
        self.index = faiss.read_index(str(index_path))
        
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
        
        self.df = pd.read_csv(dataset_path)
    
    def query(self, query_embedding: List[float], n_results: int = 5,
              include_metadata: bool = True) -> Dict:
        """Query FAISS."""
        import numpy as np
        
        query_vector = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_vector, n_results)
        
        results = {
            'ids': [],
            'documents': [],
            'metadatas': [],
            'distances': []
        }
        
        for dist, idx in zip(distances[0], indices[0]):
            embedding_index = self.mapping['index_to_row'].get(str(idx))
            if embedding_index is None:
                continue
            
            row = self.df[self.df['embedding_index'] == embedding_index].iloc[0]
            
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
        
        return results
    
    def count(self) -> int:
        """Get document count."""
        return self.index.ntotal

