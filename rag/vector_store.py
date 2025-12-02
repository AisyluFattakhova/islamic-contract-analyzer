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
            # Try multiple path resolution strategies for compatibility
            # Strategy 1: Relative to this file (rag/vector_store.py)
            db_path = Path(__file__).parent.parent / "datasets" / "chroma_db"
            
            # Strategy 2: If that doesn't exist, try relative to current working directory
            if not db_path.exists():
                import os
                cwd_path = Path(os.getcwd()) / "datasets" / "chroma_db"
                if cwd_path.exists():
                    db_path = cwd_path
            
            # Strategy 3: Try absolute path from project root
            if not db_path.exists():
                # Try to find project root by looking for common markers
                current = Path(__file__).resolve()
                while current != current.parent:
                    if (current / "datasets" / "chroma_db").exists():
                        db_path = current / "datasets" / "chroma_db"
                        break
                    current = current.parent
        
        # Convert to absolute path and ensure it exists
        db_path = Path(db_path).resolve()
        
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB database directory not found at: {db_path}\n"
                f"Please run 'python scripts/setup_vector_db.py' first."
            )
        
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # List all collections for debugging
        try:
            all_collections = self.client.list_collections()
            collection_names = [c.name for c in all_collections]
        except Exception:
            collection_names = []
        
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception as e:
            error_msg = (
                f"Collection '{collection_name}' not found.\n"
                f"Database path: {db_path}\n"
                f"Available collections: {collection_names}\n"
                f"Error: {e}\n"
                f"Please run 'python scripts/setup_vector_db.py' first."
            )
            raise FileNotFoundError(error_msg)
    
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

