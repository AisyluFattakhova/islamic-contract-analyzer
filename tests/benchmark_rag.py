"""
Automated Benchmark Test Suite for RAG System.

Evaluates RAG performance using standard metrics:
- Mean Reciprocal Rank (MRR)
- Precision@K
- Recall@K
- NDCG@K
"""
import sys
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from rag_query import (
    load_embedding_model, load_chromadb, load_faiss,
    query_chromadb, query_faiss, detect_database_type,
    DATASET_PATH
)
import pandas as pd


class RAGBenchmark:
    """Benchmark RAG system performance."""
    
    def __init__(self, test_dataset_path: str = None):
        """
        Initialize benchmark.
        
        Args:
            test_dataset_path: Path to test dataset JSON file
        """
        self.test_dataset_path = test_dataset_path or Path(__file__).parent / "test_dataset.json"
        self.results = []
        
        # Load RAG system
        print("Loading RAG system for benchmarking...")
        self.db_type = detect_database_type()
        if self.db_type is None:
            raise FileNotFoundError("No vector database found. Run setup first.")
        
        self.model = load_embedding_model()
        
        if self.db_type == "ChromaDB":
            self.collection = load_chromadb()
            self.index = None
            self.mapping = None
            self.df = None
        else:
            raise FileNotFoundError("Only ChromaDB is supported. FAISS has been removed for Streamlit Cloud compatibility.")
        
        print("✅ RAG system loaded")
    
    def load_test_dataset(self) -> List[Dict]:
        """Load test dataset with queries and expected results."""
        if not Path(self.test_dataset_path).exists():
            # Create default test dataset if doesn't exist
            self._create_default_test_dataset()
        
        with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _create_default_test_dataset(self):
        """Create a default test dataset for Shariaa Standards."""
        default_dataset = [
            {
                "query": "What is Murabahah?",
                "expected_sections": ["3//1"],  # Section numbers that should be in top results
                "category": "definition"
            },
            {
                "query": "currency trading",
                "expected_sections": ["2//1", "2//1//1", "2//1//2"],
                "category": "rules"
            },
            {
                "query": "What is Riba and when is it prohibited?",
                "expected_sections": ["2//1", "2//2"],
                "category": "prohibition"
            },
            {
                "query": "Mudarabah",
                "expected_sections": ["5//1", "5//2"],
                "category": "definition"
            },
            {
                "query": "Ijarah contract",
                "expected_sections": ["4//1", "4//1//1"],
                "category": "conditions"
            },
            {
                "query": "Salam contract",
                "expected_sections": ["6//1", "6//1//1"],
                "category": "definition"
            },
            {
                "query": "Musharakah",
                "expected_sections": ["5//1", "5//1//1"],
                "category": "rules"
            },
            {
                "query": "currency transactions",
                "expected_sections": ["2//2", "2//3"],
                "category": "prohibition"
            }
        ]
        
        # Save default dataset
        Path(self.test_dataset_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.test_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(default_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created default test dataset at {self.test_dataset_path}")
    
    def query_rag(self, query: str, k: int = 5) -> List[Dict]:
        """Query RAG system and return results."""
        if self.db_type == "ChromaDB":
            results = query_chromadb(
                self.collection, query, self.model, 
                n_results=k, use_query_expansion=True
            )
        else:
            raise FileNotFoundError("Only ChromaDB is supported. FAISS has been removed for Streamlit Cloud compatibility.")
        
        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
            results['documents'],
            results['metadatas'],
            results['distances']
        ):
            formatted_results.append({
                'section_path': metadata.get('section_path', ''),
                'section_number': metadata.get('section_number', ''),
                'content': doc,
                'distance': distance,
                'relevance_score': 1 - distance
            })
        
        return formatted_results
    
    def calculate_precision_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len([r for r in retrieved_k if any(exp in r for exp in expected)])
        return relevant_retrieved / k
    
    def calculate_recall_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if len(expected) == 0:
            return 0.0
        
        retrieved_k = retrieved[:k]
        relevant_retrieved = len([r for r in retrieved_k if any(exp in r for exp in expected)])
        return relevant_retrieved / len(expected)
    
    def calculate_mrr(self, retrieved: List[str], expected: List[str]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for rank, item in enumerate(retrieved, 1):
            if any(exp in item for exp in expected):
                return 1.0 / rank
        return 0.0
    
    def calculate_ndcg_at_k(self, retrieved: List[str], expected: List[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        retrieved_k = retrieved[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(retrieved_k, 1):
            if any(exp in item for exp in expected):
                dcg += 1.0 / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(expected), k) + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, test_case: Dict, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Evaluate a single query."""
        query = test_case['query']
        expected_sections = test_case.get('expected_sections', [])
        
        # Query RAG
        results = self.query_rag(query, k=max(k_values))
        retrieved_sections = [r['section_path'] for r in results]
        
        # Calculate metrics
        metrics = {}
        for k in k_values:
            metrics[f'precision@{k}'] = self.calculate_precision_at_k(
                retrieved_sections, expected_sections, k
            )
            metrics[f'recall@{k}'] = self.calculate_recall_at_k(
                retrieved_sections, expected_sections, k
            )
            metrics[f'ndcg@{k}'] = self.calculate_ndcg_at_k(
                retrieved_sections, expected_sections, k
            )
        
        metrics['mrr'] = self.calculate_mrr(retrieved_sections, expected_sections)
        
        return {
            'query': query,
            'category': test_case.get('category', 'unknown'),
            'expected_sections': expected_sections,
            'retrieved_sections': retrieved_sections[:max(k_values)],
            'metrics': metrics,
            'top_result': results[0] if results else None
        }
    
    def run_benchmark(self, k_values: List[int] = [1, 3, 5]) -> Dict:
        """Run full benchmark suite."""
        print("\n" + "="*80)
        print("RAG Benchmark Test Suite")
        print("="*80)
        
        test_dataset = self.load_test_dataset()
        print(f"\nLoaded {len(test_dataset)} test queries")
        
        results = []
        for i, test_case in enumerate(test_dataset, 1):
            print(f"\n[{i}/{len(test_dataset)}] Evaluating: {test_case['query']}")
            result = self.evaluate_query(test_case, k_values)
            results.append(result)
            
            # Print quick summary
            print(f"  MRR: {result['metrics']['mrr']:.3f}")
            print(f"  Precision@5: {result['metrics']['precision@5']:.3f}")
            print(f"  Recall@5: {result['metrics']['recall@5']:.3f}")
        
        # Calculate aggregate metrics
        aggregate = self._calculate_aggregate_metrics(results, k_values)
        
        benchmark_results = {
            'timestamp': datetime.now().isoformat(),
            'total_queries': len(test_dataset),
            'k_values': k_values,
            'individual_results': results,
            'aggregate_metrics': aggregate,
            'summary': self._generate_summary(aggregate)
        }
        
        return benchmark_results
    
    def _calculate_aggregate_metrics(self, results: List[Dict], k_values: List[int]) -> Dict:
        """Calculate aggregate metrics across all queries."""
        aggregate = {}
        
        for k in k_values:
            aggregate[f'mean_precision@{k}'] = np.mean([
                r['metrics'][f'precision@{k}'] for r in results
            ])
            aggregate[f'mean_recall@{k}'] = np.mean([
                r['metrics'][f'recall@{k}'] for r in results
            ])
            aggregate[f'mean_ndcg@{k}'] = np.mean([
                r['metrics'][f'ndcg@{k}'] for r in results
            ])
        
        aggregate['mean_mrr'] = np.mean([r['metrics']['mrr'] for r in results])
        
        return aggregate
    
    def _generate_summary(self, aggregate: Dict) -> str:
        """Generate human-readable summary."""
        summary = f"""
Benchmark Summary:
==================
Mean Reciprocal Rank (MRR): {aggregate['mean_mrr']:.3f}

Precision@K:
  - Precision@1: {aggregate['mean_precision@1']:.3f}
  - Precision@3: {aggregate['mean_precision@3']:.3f}
  - Precision@5: {aggregate['mean_precision@5']:.3f}

Recall@K:
  - Recall@1: {aggregate['mean_recall@1']:.3f}
  - Recall@3: {aggregate['mean_recall@3']:.3f}
  - Recall@5: {aggregate['mean_recall@5']:.3f}

NDCG@K:
  - NDCG@1: {aggregate['mean_ndcg@1']:.3f}
  - NDCG@3: {aggregate['mean_ndcg@3']:.3f}
  - NDCG@5: {aggregate['mean_ndcg@5']:.3f}
"""
        return summary
    
    def save_results(self, results: Dict, output_path: str = None):
        """Save benchmark results to JSON file."""
        if output_path is None:
            output_path = Path(__file__).parent / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Results saved to: {output_path}")
        return output_path


def main():
    """Run benchmark tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG benchmark tests")
    parser.add_argument("--test-dataset", help="Path to test dataset JSON file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5],
                       help="K values for evaluation (default: 1 3 5)")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = RAGBenchmark(test_dataset_path=args.test_dataset)
    results = benchmark.run_benchmark(k_values=args.k)
    
    # Print summary
    print("\n" + "="*80)
    print(results['summary'])
    print("="*80)
    
    # Save results
    output_path = benchmark.save_results(results, args.output)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()

