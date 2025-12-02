"""
Query Router for RAG System.

Routes queries to different retrieval strategies based on query type and complexity.
"""
import re
from typing import Dict
from enum import Enum


class QueryType(Enum):
    """Types of queries."""
    DEFINITION = "definition"
    RULES = "rules"
    PROHIBITION = "prohibition"
    COMPARISON = "comparison"
    COMPLEX = "complex"
    SIMPLE = "simple"


class QueryRouter:
    """Routes queries to appropriate retrieval strategies."""
    
    def __init__(self):
        """Initialize query router."""
        # Keywords for different query types
        self.definition_keywords = [
            "what is", "what are", "define", "definition", "explain", "meaning",
            "concept", "describe"
        ]
        self.rules_keywords = [
            "rules", "requirements", "conditions", "criteria", "guidelines",
            "procedures", "how to", "steps"
        ]
        self.prohibition_keywords = [
            "prohibited", "forbidden", "not allowed", "cannot", "must not",
            "should not", "haram", "impermissible"
        ]
        self.comparison_keywords = [
            "compare", "difference", "versus", "vs", "between", "similar",
            "different"
        ]
    
    def classify_query(self, query: str) -> tuple:
        """
        Classify query type and complexity.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (query_type, complexity_score)
        """
        query_lower = query.lower()
        
        # Check for definition queries
        if any(keyword in query_lower for keyword in self.definition_keywords):
            return (QueryType.DEFINITION, self._calculate_complexity(query))
        
        # Check for rules queries
        if any(keyword in query_lower for keyword in self.rules_keywords):
            return (QueryType.RULES, self._calculate_complexity(query))
        
        # Check for prohibition queries
        if any(keyword in query_lower for keyword in self.prohibition_keywords):
            return (QueryType.PROHIBITION, self._calculate_complexity(query))
        
        # Check for comparison queries
        if any(keyword in query_lower for keyword in self.comparison_keywords):
            return (QueryType.COMPLEX, 0.8)  # Comparisons are usually complex
        
        # Default classification based on complexity
        complexity = self._calculate_complexity(query)
        if complexity > 0.6:
            return (QueryType.COMPLEX, complexity)
        else:
            return (QueryType.SIMPLE, complexity)
    
    def _calculate_complexity(self, query: str) -> float:
        """
        Calculate query complexity score (0-1).
        
        Factors:
        - Length
        - Number of clauses
        - Question words
        - Technical terms
        """
        complexity = 0.0
        
        # Length factor (normalized to 0-0.3)
        length_score = min(len(query) / 200, 1.0) * 0.3
        complexity += length_score
        
        # Number of clauses (sentences, questions)
        clauses = len(re.split(r'[.!?]', query))
        clause_score = min(clauses / 5, 1.0) * 0.2
        complexity += clause_score
        
        # Question words (indicates complexity)
        question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
        question_count = sum(1 for word in question_words if word in query.lower())
        question_score = min(question_count / 3, 1.0) * 0.2
        complexity += question_score
        
        # Technical terms (longer words, specific terminology)
        words = query.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        technical_score = min((avg_word_length - 4) / 5, 1.0) * 0.3
        complexity += max(technical_score, 0)
        
        return min(complexity, 1.0)
    
    def get_retrieval_strategy(self, query_type: QueryType, complexity: float) -> Dict:
        """
        Get retrieval strategy based on query type and complexity.
        
        Args:
            query_type: Type of query
            complexity: Complexity score
        
        Returns:
            Strategy configuration dict
        """
        base_strategy = {
            'use_query_expansion': True,
            'n_results': 5,
            'boost_definitions': False,
            'rerank': False
        }
        
        if query_type == QueryType.DEFINITION:
            base_strategy.update({
                'n_results': 3,  # Fewer results for definitions
                'boost_definitions': True,  # Boost definition sections
                'use_query_expansion': True
            })
        elif query_type == QueryType.RULES:
            base_strategy.update({
                'n_results': 7,  # More results for rules
                'boost_definitions': False,
                'use_query_expansion': True
            })
        elif query_type == QueryType.PROHIBITION:
            base_strategy.update({
                'n_results': 5,
                'boost_definitions': False,
                'use_query_expansion': True
            })
        elif query_type == QueryType.COMPLEX:
            base_strategy.update({
                'n_results': 10,  # More results for complex queries
                'rerank': True,  # Rerank results
                'use_query_expansion': True
            })
        else:  # SIMPLE
            base_strategy.update({
                'n_results': 3,
                'use_query_expansion': False  # No expansion for simple queries
            })
        
        # Adjust based on complexity
        if complexity > 0.7:
            base_strategy['n_results'] = min(base_strategy['n_results'] + 2, 15)
            base_strategy['rerank'] = True
        
        return base_strategy
    
    def route_query(self, query: str) -> Dict:
        """
        Route query and return strategy configuration.
        
        Args:
            query: User query
        
        Returns:
            Dict with query_type, complexity, and strategy
        """
        query_type, complexity = self.classify_query(query)
        strategy = self.get_retrieval_strategy(query_type, complexity)
        
        return {
            'query': query,
            'query_type': query_type.value,
            'complexity': complexity,
            'strategy': strategy
        }

