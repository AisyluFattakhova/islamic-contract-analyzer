"""
Conversation Memory Manager for RAG System.

Manages conversation history and context for multi-turn interactions.
"""
from typing import List, Dict, Optional
from datetime import datetime
import json
from pathlib import Path


class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10, max_context_length: int = 4000):
        """
        Initialize memory manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep
            max_context_length: Maximum context length in characters
        """
        self.max_history = max_history
        self.max_context_length = max_context_length
        self.conversation_history: List[Dict] = []
        self.session_id: Optional[str] = None
    
    def add_turn(self, query: str, response: str, metadata: Optional[Dict] = None):
        """
        Add a conversation turn to history.
        
        Args:
            query: User query
            response: System response
            metadata: Additional metadata (e.g., retrieved docs, timestamps)
        """
        turn = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'metadata': metadata or {}
        }
        
        self.conversation_history.append(turn)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context(self, include_metadata: bool = False) -> str:
        """
        Get conversation context as formatted string.
        
        Args:
            include_metadata: Whether to include metadata in context
        
        Returns:
            Formatted context string
        """
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for turn in self.conversation_history[-5:]:  # Last 5 turns
            context_parts.append(f"User: {turn['query']}")
            context_parts.append(f"Assistant: {turn['response'][:500]}...")  # Truncate long responses
            if include_metadata and turn.get('metadata'):
                context_parts.append(f"Context: {json.dumps(turn['metadata'], indent=2)}")
        
        context = "\n\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        
        return context
    
    def get_recent_queries(self, n: int = 3) -> List[str]:
        """Get recent queries."""
        return [turn['query'] for turn in self.conversation_history[-n:]]
    
    def clear(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def save(self, filepath: str):
        """Save conversation history to file."""
        data = {
            'session_id': self.session_id,
            'conversation_history': self.conversation_history,
            'saved_at': datetime.now().isoformat()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """Load conversation history from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.session_id = data.get('session_id')
        self.conversation_history = data.get('conversation_history', [])


class MemoryManager:
    """Manages multiple conversation sessions."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.sessions: Dict[str, ConversationMemory] = {}
    
    def get_or_create_session(self, session_id: str) -> ConversationMemory:
        """Get existing session or create new one."""
        if session_id not in self.sessions:
            memory = ConversationMemory()
            memory.session_id = session_id
            self.sessions[session_id] = memory
        return self.sessions[session_id]
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            del self.sessions[session_id]

