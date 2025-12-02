"""
RAG Retriever Module.

Main retrieval component that orchestrates embedding, vector store, and routing.
"""
from typing import List, Dict, Optional
from .embedder import Embedder
from .vector_store import VectorStore, ChromaDBVectorStore
from .router import QueryRouter
from pathlib import Path


class RAGRetriever:
    """Main RAG retrieval component."""
    
    def __init__(self, embedder: Optional[Embedder] = None,
                 vector_store: Optional[VectorStore] = None,
                 router: Optional[QueryRouter] = None,
                 use_routing: bool = True):
        """
        Initialize RAG retriever.
        
        Args:
            embedder: Embedder instance (creates default if None)
            vector_store: VectorStore instance (creates default if None)
            router: QueryRouter instance (creates default if None)
            use_routing: Whether to use query routing
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or self._create_default_vector_store()
        self.router = router or QueryRouter()
        self.use_routing = use_routing
    
    def _create_default_vector_store(self) -> VectorStore:
        """Create default vector store (ChromaDB only)."""
        # Use ChromaDB only (FAISS removed for Streamlit Cloud compatibility)
        try:
            return ChromaDBVectorStore()
        except Exception as e:
            raise FileNotFoundError(
                f"ChromaDB vector database not found. Run setup first. Error: {e}"
            )
    
    def retrieve(self, query: str, n_results: int = 5,
                 use_query_expansion: bool = True,
                 boost_definitions: bool = False) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query text
            n_results: Number of results to return
            use_query_expansion: Whether to use query expansion
            boost_definitions: Whether to boost definition sections
        
        Returns:
            List of retrieved documents with metadata
        """
        # Route query if enabled
        if self.use_routing:
            routing_result = self.router.route_query(query)
            strategy = routing_result['strategy']
            
            # Override parameters with strategy
            n_results = strategy.get('n_results', n_results)
            use_query_expansion = strategy.get('use_query_expansion', use_query_expansion)
            boost_definitions = strategy.get('boost_definitions', boost_definitions)
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Query vector store
        results = self.vector_store.query(query_embedding, n_results=n_results)
        
        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            formatted_results.append({
                'section_number': metadata.get('section_number', ''),
                'section_path': metadata.get('section_path', ''),
                'content': doc,
                'distance': distance,
                'relevance_score': 1 - distance,
                'content_length': metadata.get('content_length', len(doc))
            })
        
        # Boost definitions if requested
        if boost_definitions:
            formatted_results = self._boost_definitions(formatted_results)
        
        return formatted_results
    
    def _boost_definitions(self, results: List[Dict]) -> List[Dict]:
        """Boost definition sections in results."""
        scored_results = []
        
        for result in results:
            score = result['distance']
            content_lower = result['content'].lower()
            
            # Boost definition sections
            if "definition" in content_lower[:100]:
                score -= 0.2
            if any(pattern in content_lower[:100] for pattern in [" is ", " means ", " refers to "]):
                score -= 0.1
            
            # Penalize very short sections
            if len(result['content']) < 50:
                score += 0.1
            
            scored_results.append((score, result))
        
        # Sort by score (lower is better)
        scored_results.sort(key=lambda x: x[0])
        
        return [result for _, result in scored_results]

