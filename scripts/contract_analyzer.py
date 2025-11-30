"""
Contract Analyzer using LLM + RAG.

This script analyzes contracts for Shariah compliance by:
1. Retrieving relevant Shariaa Standards using RAG
2. Using LLM to generate compliance analysis
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# Get project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Import RAG query functions
# Add scripts directory to path for imports
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from rag_query import (
        load_embedding_model, load_chromadb, load_faiss,
        query_chromadb, query_faiss, detect_database_type,
        DATASET_PATH
    )
except ImportError as e:
    raise ImportError(
        f"Could not import from rag_query.py: {e}\n"
        f"Make sure rag_query.py exists in {SCRIPT_DIR}"
    )

import pandas as pd


class ContractAnalyzer:
    """Analyze contracts for Shariah compliance using RAG + LLM."""
    
    def __init__(self, llm_provider: str = "gemini", api_key: Optional[str] = None):
        """
        Initialize the contract analyzer.
        
        Args:
            llm_provider: "gemini", "openai", or "anthropic"
            api_key: API key for LLM (or set GEMINI_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY env var)
        """
        self.llm_provider = llm_provider.lower()
        self.api_key = api_key or self._get_api_key()
        
        # Load RAG components
        print("Loading RAG system...")
        self.db_type = detect_database_type()
        if self.db_type is None:
            raise FileNotFoundError(
                "No vector database found. Run 'python scripts/setup_vector_db.py' first."
            )
        
        self.model = load_embedding_model()
        
        if self.db_type == "ChromaDB":
            self.collection = load_chromadb()
            self.index = None
            self.mapping = None
            self.df = None
        else:  # FAISS
            self.collection = None
            self.index, self.mapping = load_faiss()
            self.df = pd.read_csv(DATASET_PATH)
        
        print("✅ RAG system loaded")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.llm_provider == "gemini":
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "GEMINI_API_KEY not found. Set it as environment variable or pass api_key parameter."
                )
            return key
        elif self.llm_provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OPENAI_API_KEY not found. Set it as environment variable or pass api_key parameter."
                )
            return key
        elif self.llm_provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found. Set it as environment variable or pass api_key parameter."
                )
            return key
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def retrieve_relevant_standards(self, contract_text: str, n_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant Shariaa Standards for the contract.
        
        Args:
            contract_text: Text of the contract to analyze
            n_results: Number of relevant standards to retrieve
        
        Returns:
            List of dictionaries with section info and content
        """
        # Query RAG system
        if self.db_type == "ChromaDB":
            results = query_chromadb(
                self.collection, contract_text, self.model, 
                n_results=n_results, use_query_expansion=True
            )
        else:  # FAISS
            results = query_faiss(
                self.index, self.mapping, contract_text, self.model,
                self.df, n_results=n_results, use_query_expansion=True
            )
        
        # Format results
        relevant_standards = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ), 1):
            relevant_standards.append({
                'rank': i,
                'section_number': metadata.get('section_number', 'N/A'),
                'section_path': metadata.get('section_path', 'N/A'),
                'content': doc,
                'relevance_score': 1 - distance,  # Convert distance to similarity
                'content_length': metadata.get('content_length', len(doc))
            })
        
        return relevant_standards
    
    def generate_analysis(self, contract_text: str, relevant_standards: List[Dict]) -> Dict:
        """
        Generate compliance analysis using LLM.
        
        Args:
            contract_text: Text of the contract
            relevant_standards: List of relevant standards from RAG
        
        Returns:
            Dictionary with analysis results
        """
        # Prepare context from relevant standards
        standards_context = "\n\n".join([
            f"Standard {std['section_path']} ({std['section_number']}):\n{std['content']}"
            for std in relevant_standards[:5]  # Use top 5
        ])
        
        # Create prompt
        prompt = f"""You are an expert in Islamic finance and Shariah compliance. Analyze the following contract against the provided Shariaa Standards.

CONTRACT TEXT:
{contract_text[:3000]}  # Limit contract text to avoid token limits

RELEVANT SHARIAA STANDARDS:
{standards_context}

Please provide a comprehensive analysis:
1. **Compliance Summary**: Overall compliance status (Compliant/Non-Compliant/Partially Compliant)
2. **Key Findings**: List the main compliance issues or confirmations
3. **Relevant Standards**: Reference specific standards that apply
4. **Recommendations**: Suggestions for ensuring full compliance
5. **Risk Assessment**: Level of risk (Low/Medium/High)

Format your response in clear sections with bullet points where appropriate."""

        # Call LLM
        if self.llm_provider == "gemini":
            return self._call_gemini(prompt)
        elif self.llm_provider == "openai":
            return self._call_openai(prompt)
        elif self.llm_provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _call_gemini(self, prompt: str) -> Dict:
        """Call Google Gemini API."""
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
        
        # Configure Gemini
        Client=genai.Client(api_key="AIzaSyDMZllwjXTgnhu8aJv1yacULF5NRu-_Yvo")
        
        try:
            # Use Gemini Pro model (free tier available)
            model = 'gemini-2.5-flash'
            
            # Generate response
            response = Client.models.generate_content(
                model=model,
                contents=prompt
            )
            
            analysis_text = response.text
            
            return {
                'analysis': analysis_text,
                'model': 'gemini-2.5-flash',
                'provider': 'gemini',
                'tokens_used': None  # Gemini API doesn't always provide token usage in free tier
            }
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")
    
    def _call_openai(self, prompt: str) -> Dict:
        """Call OpenAI API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        client = OpenAI(api_key=self.api_key)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cost-effective, can use "gpt-4" for better quality
                messages=[
                    {"role": "system", "content": "You are an expert in Islamic finance and Shariah compliance analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                'analysis': analysis_text,
                'model': 'gpt-4o-mini',
                'provider': 'openai',
                'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def _call_anthropic(self, prompt: str) -> Dict:
        """Call Anthropic API."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        client = Anthropic(api_key=self.api_key)
        
        try:
            response = client.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            analysis_text = response.content[0].text
            
            return {
                'analysis': analysis_text,
                'model': 'claude-3-haiku-20240307',
                'provider': 'anthropic',
                'tokens_used': response.usage.input_tokens + response.usage.output_tokens if hasattr(response, 'usage') else None
            }
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def analyze_contract(self, contract_text: str, n_standards: int = 5) -> Dict:
        """
        Complete contract analysis pipeline.
        
        Args:
            contract_text: Text of the contract to analyze
            n_standards: Number of relevant standards to retrieve
        
        Returns:
            Dictionary with full analysis results
        """
        print(f"\nAnalyzing contract ({len(contract_text)} characters)...")
        
        # Step 1: Retrieve relevant standards
        print("Step 1: Retrieving relevant Shariaa Standards...")
        relevant_standards = self.retrieve_relevant_standards(contract_text, n_results=n_standards)
        print(f"✅ Found {len(relevant_standards)} relevant standards")
        
        # Step 2: Generate LLM analysis
        print("Step 2: Generating compliance analysis with LLM...")
        analysis = self.generate_analysis(contract_text, relevant_standards)
        print("✅ Analysis complete")
        
        return {
            'contract_length': len(contract_text),
            'relevant_standards': relevant_standards,
            'analysis': analysis,
            'summary': {
                'standards_found': len(relevant_standards),
                'llm_provider': analysis['provider'],
                'llm_model': analysis['model']
            }
        }


def main():
    """CLI interface for contract analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze contract for Shariah compliance")
    parser.add_argument("contract_file", help="Path to contract text file")
    parser.add_argument("--provider", choices=["gemini", "openai", "anthropic"], default="gemini",
                       help="LLM provider (default: gemini)")
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

