"""
Embedding Module for RAG System.
"""
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer
from typing import Optional


class Embedder:
    """Handles text embedding operations."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize embedder.
        
        Args:
            model_name: Name of embedding model (default: from metadata)
            device: Device to use ('cuda' or 'cpu', default: auto-detect)
        """
        self.model_name = model_name or self._get_model_name_from_metadata()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
    
    def _get_model_name_from_metadata(self) -> str:
        """Get model name from metadata file."""
        # Default model
        default = "BAAI/bge-base-en-v1.5"
        
        # Try to load from metadata
        metadata_path = Path(__file__).parent.parent / "datasets" / "embeddings" / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('model_name', default)
        
        return default
    
    def _load_model(self) -> SentenceTransformer:
        """Load embedding model."""
        print(f"Loading embedding model: {self.model_name}")
        model = SentenceTransformer(self.model_name, device=self.device)
        model._model_name = self.model_name
        print(f"✅ Model loaded on {self.device}")
        return model
    
    def encode(self, texts: list, normalize: bool = True, instruction: Optional[str] = None) -> list:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to normalize embeddings
            instruction: Instruction prefix for BGE models
        
        Returns:
            List of embedding vectors
        """
        # Check if BGE model
        is_bge = 'bge' in self.model_name.lower()
        
        if is_bge and instruction:
            texts_with_instruction = [f"{instruction} {text}" for text in texts]
            texts = texts_with_instruction
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    
    def encode_query(self, query: str, normalize: bool = True) -> list:
        """
        Encode a single query.
        
        Args:
            query: Query text
            normalize: Whether to normalize embedding
        
        Returns:
            Embedding vector
        """
        is_bge = 'bge' in self.model_name.lower()
        
        if is_bge:
            query_with_instruction = f"Represent this sentence for searching relevant passages: {query}"
            embedding = self.model.encode(
                [query_with_instruction],
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
        else:
            embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
        
        return embedding[0].tolist()

