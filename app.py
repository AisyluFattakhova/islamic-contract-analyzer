"""
Streamlit Web Application for Islamic Contract Analyzer.

Simple frontend for uploading and analyzing contracts for Shariah compliance.
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

# Add scripts directory to path
SCRIPT_DIR = Path(__file__).parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contract_analyzer import ContractAnalyzer

# Page configuration
st.set_page_config(
    page_title="Islamic Contract Analyzer",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set dark mode theme
st.markdown("""
<style>
    /* Force dark mode */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Main content background */
    .main .block-container {
        background-color: #1e1e1e;
        padding: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Text */
    p, li, div {
        color: #e0e0e0 !important;
    }
    
    /* Cards and containers */
    .analysis-section {
        background-color: #2d2d2d !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #e0e0e0 !important;
        border: 1px solid #404040;
    }
    
    .standard-card {
        background-color: #2d2d2d !important;
        padding: 1rem;
        border-left: 4px solid #4a9eff;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        color: #e0e0e0 !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e1e1e;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4a9eff;
        color: white;
        border: none;
    }
    
    .stButton > button:hover {
        background-color: #3a8eef;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #4a9eff !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #1e3a1e;
        color: #90ee90;
    }
    
    .stError {
        background-color: #3a1e1e;
        color: #ff6b6b;
    }
    
    .stWarning {
        background-color: #3a3a1e;
        color: #ffd700;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #2d2d2d;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > select {
        background-color: #2d2d2d;
        color: #e0e0e0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e1e1e;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #b0b0b0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #4a9eff;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4a9eff;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_analyzer(llm_provider: str, api_key: str = None):
    """Load and cache the contract analyzer."""
    try:
        return ContractAnalyzer(llm_provider=llm_provider, api_key=api_key)
    except Exception as e:
        st.error(f"Error loading analyzer: {e}")
        return None


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📜 Islamic Contract Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze contracts for Shariah compliance using AI-powered RAG technology")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # LLM Provider selection
        llm_provider = st.selectbox(
            "LLM Provider",
            ["gemini", "openai", "anthropic"],
            index=0,  # Default to Gemini
            help="Choose your LLM provider"
        )
        
        # API Key input
        env_var_name = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }.get(llm_provider, "GEMINI_API_KEY")
        
        api_key_env = os.getenv(env_var_name)
        if api_key_env:
            st.success("✅ API key found in environment")
            api_key = None  # Will use env var
        else:
            api_key = st.text_input(
                f"{llm_provider.upper()} API Key",
                type="password",
                help=f"Enter your {llm_provider.upper()} API key or set {env_var_name} as environment variable"
            )
            if not api_key:
                st.warning("⚠️ API key required")
        
        # Number of standards to retrieve
        n_standards = st.slider(
            "Number of Standards to Retrieve",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of relevant Shariaa Standards to use in analysis"
        )
        
        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This tool analyzes contracts for Shariah compliance by:
        1. **RAG Retrieval**: Finding relevant Shariaa Standards
        2. **LLM Analysis**: Generating compliance assessment
        3. **Report**: Providing actionable recommendations
        """)
    
    # Main content area
    tab1, tab2 = st.tabs(["📄 Contract Analysis", "🔍 Standards Query"])
    
    # Tab 1: Contract Analysis
    with tab1:
        st.header("Upload Contract for Analysis")
        
        # Contract input methods
        input_method = st.radio(
            "Input Method",
            ["📝 Paste Text", "📁 Upload File"],
            horizontal=True
        )
        
        contract_text = ""
        
        if input_method == "📝 Paste Text":
            contract_text = st.text_area(
                "Contract Text",
                height=300,
                placeholder="Paste your contract text here...",
                help="Paste the full text of the contract you want to analyze"
            )
        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload Contract",
                type=["txt", "pdf"],
                help="Upload a text or PDF file containing the contract"
            )
            
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    contract_text = uploaded_file.read().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    # For PDF, we'd need pdfplumber - for now, show message
                    st.warning("PDF upload requires pdfplumber. Please extract text first or use text upload.")
                    # You can add PDF processing here if needed
                    # import pdfplumber
                    # with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    #     tmp.write(uploaded_file.read())
                    #     # Process PDF...
                else:
                    st.error("Unsupported file type")
        
        # Analyze button
        if st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
            if not contract_text.strip():
                st.error("❌ Please provide contract text")
            elif not api_key and not api_key_env:
                st.error("❌ Please provide API key in sidebar")
            else:
                with st.spinner("Analyzing contract... This may take a minute."):
                    try:
                        # Load analyzer
                        analyzer = load_analyzer(llm_provider, api_key)
                        if analyzer is None:
                            st.stop()
                        
                        # Analyze
                        results = analyzer.analyze_contract(contract_text, n_standards=n_standards)
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Analysis section
                        st.markdown("---")
                        st.header("📊 Compliance Analysis")
                        st.markdown(f'<div class="analysis-section">{results["analysis"]["analysis"]}</div>', 
                                   unsafe_allow_html=True)
                        
                        # Relevant Standards
                        st.markdown("---")
                        st.header("📚 Relevant Shariaa Standards")
                        
                        for std in results['relevant_standards']:
                            with st.expander(
                                f"Section {std['section_path']} ({std['section_number']}) - "
                                f"Relevance: {std['relevance_score']:.1%}"
                            ):
                                st.write(std['content'])
                        
                        # Summary
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Standards Found", results['summary']['standards_found'])
                        with col2:
                            st.metric("LLM Provider", results['summary']['llm_provider'].upper())
                        with col3:
                            st.metric("LLM Model", results['summary']['llm_model'])
                        
                        # Download results
                        import json
                        results_json = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Analysis (JSON)",
                            data=results_json,
                            file_name="contract_analysis.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    
    # Tab 2: Standards Query
    with tab2:
        st.header("Query Shariaa Standards")
        st.markdown("Search the Shariaa Standards knowledge base directly")
        
        query = st.text_input(
            "Enter your query",
            placeholder="e.g., What is Murabahah?",
            help="Ask questions about Shariaa Standards"
        )
        
        if st.button("🔍 Search Standards", type="primary"):
            if not query.strip():
                st.error("❌ Please enter a query")
            else:
                with st.spinner("Searching standards..."):
                    try:
                        from rag_query import (
                            load_embedding_model, load_chromadb, load_faiss,
                            query_chromadb, query_faiss, detect_database_type,
                            DATASET_PATH
                        )
                        import pandas as pd
                        
                        # Load RAG system
                        db_type = detect_database_type()
                        if db_type is None:
                            st.error("No vector database found. Run setup first.")
                            st.stop()
                        
                        model = load_embedding_model()
                        
                        if db_type == "ChromaDB":
                            collection = load_chromadb()
                            results = query_chromadb(collection, query, model, n_results=5, use_query_expansion=True)
                        else:
                            index, mapping = load_faiss()
                            df = pd.read_csv(DATASET_PATH)
                            results = query_faiss(index, mapping, query, model, df, n_results=5, use_query_expansion=True)
                        
                        # Display results
                        st.success(f"✅ Found {len(results['documents'])} relevant sections")
                        
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results['documents'],
                            results['metadatas'],
                            results['distances']
                        ), 1):
                            with st.expander(
                                f"Result {i}: Section {metadata.get('section_path', 'N/A')} "
                                f"(Distance: {distance:.4f})"
                            ):
                                st.write(f"**Section Number:** {metadata.get('section_number', 'N/A')}")
                                st.write(f"**Content Length:** {metadata.get('content_length', len(doc))} characters")
                                st.write("**Content:**")
                                st.write(doc)
                        
                    except Exception as e:
                        st.error(f"❌ Error during search: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

