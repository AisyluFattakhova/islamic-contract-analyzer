"""
Contract Analyzer using LLM + Modular RAG.

This script analyzes contracts for Shariah compliance by:
1. Retrieving relevant Shariaa Standards using Modular RAG (router + retriever)
2. Using LLM to generate compliance analysis
3. Managing conversation memory for multi-turn interactions
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for rag module
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modular RAG components
from rag.retriever import RAGRetriever
from rag.router import QueryRouter
from rag.embedder import Embedder
from rag.vector_store import ChromaDBVectorStore

# Import memory manager
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from memory_manager import ConversationMemory, MemoryManager

# Import API utilities for best practices
try:
    from api_utils import retry_with_backoff, log_api_call, RateLimiter
except ImportError:
    # Fallback if api_utils not available
    def retry_with_backoff(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def log_api_call(func):
        return func
    class RateLimiter:
        def __init__(self, *args, **kwargs):
            pass
        def wait_if_needed(self):
            pass


class ContractAnalyzer:
    """
    Analyze contracts for Shariah compliance using Gemini + Modular RAG.
    
    The analyzer uses a modular RAG architecture:
    1. QueryRouter: Routes queries to appropriate retrieval strategies
    2. RAGRetriever: Retrieves relevant standards from vector database
    3. MemoryManager: Manages conversation history for multi-turn interactions
    4. Gemini LLM: Generates compliance analysis
    
    Best Practices:
    - Retry logic with exponential backoff for API calls
    - Rate limiting to prevent API abuse
    - Logging for debugging and monitoring
    - Memory management for context-aware conversations
    """
    
    def __init__(self, llm_provider: str = "gemini", api_key: Optional[str] = None, 
                 session_id: Optional[str] = None, use_memory: bool = True):
        """
        Initialize the contract analyzer with modular RAG components.
        
        Args:
            llm_provider: Kept for backward compatibility (always uses Gemini)
            api_key: Gemini API key (optional, will use GEMINI_API_KEY env var if not provided)
            session_id: Session ID for conversation memory (auto-generated if None)
            use_memory: Whether to use conversation memory
        """
        # Force Gemini as the only provider
        self.llm_provider = "gemini"
        self.api_key = api_key or self._get_api_key()
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize memory manager
        self.memory_manager = MemoryManager()
        self.session_id = session_id or f"session_{os.getpid()}_{int(os.path.getmtime(__file__))}"
        self.use_memory = use_memory
        if use_memory:
            self.memory = self.memory_manager.get_or_create_session(self.session_id)
        else:
            self.memory = None
        
        # Initialize rate limiter (Gemini free tier: ~15 requests/minute)
        self.rate_limiter = RateLimiter(max_calls=15, time_window=60.0)
        
        # Load modular RAG components
        print("Loading Modular RAG system...")
        try:
            # Use ChromaDB only (FAISS removed for Streamlit Cloud compatibility)
            vector_store = ChromaDBVectorStore()
            print("[OK] Using ChromaDB vector store")
        except Exception as e:
            raise FileNotFoundError(
                f"ChromaDB vector database not found. Run 'python scripts/setup_vector_db.py' first.\n"
                f"Error: {e}"
            )
        
        # Initialize modular components
        self.embedder = Embedder()
        self.router = QueryRouter()
        self.retriever = RAGRetriever(
            embedder=self.embedder,
            vector_store=vector_store,
            router=self.router,
            use_routing=True  # Enable query routing
        )
        
        print("✅ Modular RAG system loaded")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key for Gemini from environment variables."""
        return os.getenv("GEMINI_API_KEY")
    
    def retrieve_relevant_standards(self, contract_text: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve the most relevant Shariaa Standards using Modular RAG.
        
        This uses the modular RAG system (router + retriever) which:
        1. Routes the query based on type (definition, rules, prohibition, etc.)
        2. Adjusts retrieval strategy based on query complexity
        3. Retrieves from the full standards corpus
        4. Returns top N most relevant sections
        
        Args:
            contract_text: Contract text to analyze
            n_results: Number of standards to retrieve (router may adjust this)
        
        Returns:
            List of relevant standards with metadata
        """
        # Use modular RAG retriever (includes routing)
        retrieved = self.retriever.retrieve(
            query=contract_text,
            n_results=n_results,
            use_query_expansion=True,
            boost_definitions=False  # Router will decide
        )
        
        # Format results
        relevant_standards: List[Dict] = []
        for i, result in enumerate(retrieved, 1):
            relevant_standards.append({
                "rank": i,
                "section_number": result.get("section_number", "N/A"),
                "section_path": result.get("section_path", "N/A"),
                "content": result.get("content", ""),
                "relevance_score": result.get("relevance_score", 0.0),
                "content_length": result.get("content_length", 0),
            })
        
        return relevant_standards
    
    def generate_analysis(self, contract_text: str, relevant_standards: List[Dict]) -> Dict:
        """
        Generate compliance analysis using Gemini LLM with best practices.
        
        Args:
            contract_text: Text of the contract
            relevant_standards: List of relevant standards from RAG
        
        Returns:
            Dictionary with analysis results
        """
        # Get conversation context if memory is enabled
        conversation_context = ""
        if self.memory and self.use_memory:
            conversation_context = self.memory.get_context(include_metadata=False)
            if conversation_context:
                conversation_context = f"\n\nPREVIOUS CONVERSATION:\n{conversation_context}\n"
        
        # Prepare context from relevant standards
        standards_context = "\n\n".join([
            f"Standard {std['section_path']} ({std['section_number']}):\n{std['content']}"
            for std in relevant_standards[:5]  # Use top 5
        ])
        
        # Create prompt with conversation context
        prompt = f"""You are an expert in Islamic finance and Shariah compliance. Analyze the following contract against the provided Shariaa Standards.

CONTRACT TEXT:
{contract_text[:3000]}{conversation_context}

RELEVANT SHARIAA STANDARDS:
{standards_context}

Please provide a comprehensive analysis:
1. **Compliance Summary**: Overall compliance status (Compliant/Non-Compliant/Partially Compliant)
2. **Key Findings**: List the main compliance issues or confirmations
3. **Relevant Standards**: Reference specific standards that apply
4. **Recommendations**: Suggestions for ensuring full compliance
5. **Risk Assessment**: Level of risk (Low/Medium/High)

Format your response in clear sections with bullet points where appropriate."""

        # Call LLM with retry and rate limiting (best practices)
        return self._call_gemini(prompt)
    
    def answer_question(self, question: str, n_standards: int = 5) -> Dict:
        """
        Answer a question about Shariaa Standards using RAG + Gemini.
        
        This is Scenario 1: Simple questions like "What is Murabahah?"
        The router will classify this as DEFINITION and optimize retrieval accordingly.
        
        Args:
            question: User's question about Shariaa Standards
            n_standards: Number of relevant standards to retrieve
        
        Returns:
            Dictionary with answer and relevant standards
        """
        print(f"\nAnswering question: {question}")
        
        # Step 1: Retrieve relevant standards (router will classify as DEFINITION, etc.)
        print("Step 1: Retrieving relevant Shariaa Standards using Modular RAG...")
        relevant_standards = self.retrieve_relevant_standards(
            question, n_results=n_standards
        )
        print(f"✅ Found {len(relevant_standards)} relevant standards")
        
        # Step 2: Generate answer using Gemini
        print("Step 2: Generating answer with Gemini...")
        answer = self._generate_answer(question, relevant_standards)
        print("✅ Answer complete")
        
        # Step 3: Update conversation memory if enabled
        if self.memory and self.use_memory:
            self.memory.add_turn(
                query=question,
                response=answer['answer'],
                metadata={
                    'standards_found': len(relevant_standards),
                    'standards': [std['section_path'] for std in relevant_standards],
                    'query_type': 'question'
                }
            )
        
        return {
            'question': question,
            'relevant_standards': relevant_standards,
            'answer': answer,
            'summary': {
                'standards_found': len(relevant_standards),
                'llm_provider': answer['provider'],
                'llm_model': answer['model'],
                'session_id': self.session_id if self.use_memory else None,
            },
        }
    
    def _generate_answer(self, question: str, relevant_standards: List[Dict]) -> Dict:
        """
        Generate answer to question using Gemini LLM.
        
        Args:
            question: User's question
            relevant_standards: List of relevant standards from RAG
        
        Returns:
            Dictionary with answer results
        """
        # Get conversation context if memory is enabled
        conversation_context = ""
        if self.memory and self.use_memory:
            conversation_context = self.memory.get_context(include_metadata=False)
            if conversation_context:
                conversation_context = f"\n\nPREVIOUS CONVERSATION:\n{conversation_context}\n"
        
        # Prepare context from relevant standards
        standards_context = "\n\n".join([
            f"Standard {std['section_path']} ({std['section_number']}):\n{std['content']}"
            for std in relevant_standards
        ])
        
        # Create prompt for answering questions
        prompt = f"""You are an expert in Islamic finance and Shariah compliance. Answer the following question based on the provided Shariaa Standards.

QUESTION:
{question}{conversation_context}

RELEVANT SHARIAA STANDARDS:
{standards_context}

Please provide a clear, comprehensive answer:
1. **Direct Answer**: Answer the question directly based on the standards
2. **Key Points**: Highlight the most important aspects
3. **Standard References**: Reference the specific standards that support your answer
4. **Additional Context**: Provide any relevant additional information if helpful

Format your response in clear sections with bullet points where appropriate."""

        # Call LLM with retry and rate limiting
        result = self._call_gemini(prompt)
        
        # Rename 'analysis' to 'answer' for consistency
        return {
            'answer': result['analysis'],
            'model': result['model'],
            'provider': result['provider'],
            'tokens_used': result.get('tokens_used')
        }
    
    @log_api_call
    def _call_gemini(self, prompt: str) -> Dict:
        """
        Call Google Gemini API with best practices:
        - Retry with exponential backoff
        - Rate limiting
        - Logging
        - Error handling
        - Quota/rate limit error handling
        """
        try:
            import google.generativeai as genai
            from google.api_core import exceptions as google_exceptions
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
        
        # Rate limiting (best practice)
        self.rate_limiter.wait_if_needed()
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use stable model with better rate limits (avoid experimental models on free tier)
        # gemini-1.5-flash has better rate limits than experimental models
        model_name = 'gemini-2.5-flash'
        model = genai.GenerativeModel(model_name)
        
        # Generate response with retry logic for quota errors
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                response = model.generate_content(prompt)
                analysis_text = response.text
                
                return {
                    'analysis': analysis_text,
                    'model': model_name,
                    'provider': 'gemini',
                    'tokens_used': None  # Gemini API doesn't always provide token usage in free tier
                }
            
            except Exception as e:
                error_str = str(e).lower()
                error_full = str(e)
                
                # Check for quota/rate limit errors
                is_quota_error = (
                    '429' in error_str or 
                    'quota' in error_str or 
                    'rate limit' in error_str or
                    'exceeded' in error_str
                )
                
                if is_quota_error:
                    # Extract retry delay from error if available
                    import re
                    import time
                    
                    # Try to extract retry delay from error message
                    retry_match = re.search(r'retry.*?(\d+(?:\.\d+)?)\s*s', error_full, re.IGNORECASE)
                    if retry_match:
                        retry_delay = float(retry_match.group(1)) + 2.0  # Add 2 second buffer
                    else:
                        # Exponential backoff
                        retry_delay = min(retry_delay * 2, 60.0)  # Max 60s
                    
                    if attempt < max_retries:
                        wait_msg = (
                            f"⚠️ Rate limit/quota exceeded. "
                            f"Retrying in {retry_delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})..."
                        )
                        print(wait_msg)
                        time.sleep(retry_delay)
                        continue
                    else:
                        # All retries exhausted - provide helpful error message
                        error_summary = error_full[:300] if len(error_full) > 300 else error_full
                        raise ValueError(
                            f"❌ Gemini API quota/rate limit exceeded after {max_retries + 1} attempts.\n\n"
                            f"**What happened:**\n"
                            f"The free tier has rate limits on requests per minute. You've hit the limit.\n\n"
                            f"**Solutions:**\n"
                            f"1. Wait a few minutes and try again\n"
                            f"2. Check your quota usage: https://ai.dev/usage?tab=rate-limit\n"
                            f"3. Consider upgrading your API plan for higher limits\n"
                            f"4. Use a stable model (gemini-1.5-flash) instead of experimental models\n\n"
                            f"**Error details:** {error_summary}"
                        )
                else:
                    # Other errors - raise immediately
                    raise
    
    def analyze_contract(self, contract_text: str, n_standards: int = 5) -> Dict:
        """
        Complete contract analysis pipeline with modular RAG and memory.
        
        Args:
            contract_text: Text of the contract to analyze
            n_standards: Number of relevant standards to retrieve (router may adjust)
        
        Returns:
            Dictionary with full analysis results
        """
        print(f"\nAnalyzing contract ({len(contract_text)} characters)...")
        
        # Step 1: Retrieve relevant standards using modular RAG (with routing)
        print("Step 1: Retrieving relevant Shariaa Standards using Modular RAG...")
        relevant_standards = self.retrieve_relevant_standards(
            contract_text, n_results=n_standards
        )
        print(f"✅ Found {len(relevant_standards)} relevant standards")
        
        # Step 2: Generate Gemini analysis using those standards as context
        print("Step 2: Generating compliance analysis with Gemini...")
        analysis = self.generate_analysis(contract_text, relevant_standards)
        print("✅ Analysis complete")
        
        # Step 3: Update conversation memory if enabled
        if self.memory and self.use_memory:
            self.memory.add_turn(
                query=f"Analyze contract ({len(contract_text)} chars)",
                response=analysis['analysis'],
                metadata={
                    'standards_found': len(relevant_standards),
                    'standards': [std['section_path'] for std in relevant_standards]
                }
            )
        
        return {
            'contract_length': len(contract_text),
            "relevant_standards": relevant_standards,
            "analysis": analysis,
            "summary": {
                "standards_found": len(relevant_standards),
                "llm_provider": analysis["provider"],
                "llm_model": analysis["model"],
                "session_id": self.session_id if self.use_memory else None,
            },
        }


def main():
    """CLI interface for contract analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze contract for Shariah compliance")
    parser.add_argument("contract_file", help="Path to contract text file")
    # Provider is fixed to Gemini but argument kept for backward compatibility
    parser.add_argument(
        "--provider",
        choices=["gemini"],
        default="gemini",
        help="LLM provider (fixed to gemini)",
    )
    parser.add_argument("--api-key", help="API key (or set env var)")
    parser.add_argument("--standards", type=int, default=5,
                       help="Number of relevant standards to retrieve (default: 5)")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Read contract
    contract_path = Path(args.contract_file)
    if not contract_path.exists():
        print(f"❌ Contract file not found: {contract_path}")
        sys.exit(1)
    
    contract_text = contract_path.read_text(encoding="utf-8")
    
    # Initialize analyzer
    try:
        analyzer = ContractAnalyzer(llm_provider=args.provider, api_key=args.api_key)
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        sys.exit(1)
    
    # Analyze
    try:
        results = analyzer.analyze_contract(contract_text, n_standards=args.standards)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(f"\n{results['analysis']['analysis']}")
    print("\n" + "="*80)
    print("RELEVANT STANDARDS")
    print("="*80)
    for std in results['relevant_standards']:
        print(f"\n[{std['rank']}] Section {std['section_path']} (Relevance: {std['relevance_score']:.2%})")
        print(f"{std['content'][:200]}...")
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

