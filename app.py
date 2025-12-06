"""
Streamlit Web Application for Islamic Contract Analyzer.

Simple frontend for uploading and analyzing contracts for Shariah compliance.
"""
import streamlit as st
import sys
from pathlib import Path
import tempfile
import os
import pdfplumber

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

# Set Islamic-inspired theme with greens, soft yellows, and golden accents
st.markdown("""
<style>
    /* Main app background - soft yellow/green sky */
    .stApp {
        background: linear-gradient(135deg, #f5f8e8 0%, #f0f8e8 50%, #e8f5e8 100%);
    }
    
    /* Main content background - light mint green */
    .main .block-container {
        background-color: #f0f8f0;
        padding: 2rem;
        border-radius: 0.5rem;
    }
    
    /* Headers - dark forest green */
    h1, h2, h3 {
        color: #2d5016 !important;
    }
    
    /* Text - dark green for readability */
    p, li, div {
        color: #1a3d0a !important;
    }
    
    /* Cards and containers - mint green with golden border */
    .analysis-section {
        background-color: #e8f5e8 !important;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #1a3d0a !important;
        border: 2px solid #d4af37;
        box-shadow: 0 2px 8px rgba(45, 80, 22, 0.1);
    }
    
    .standard-card {
        background-color: #f0f8f0 !important;
        padding: 1rem;
        border-left: 4px solid #d4af37;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        color: #1a3d0a !important;
        box-shadow: 0 1px 4px rgba(45, 80, 22, 0.1);
    }
    
    /* Sidebar - light green */
    .css-1d391kg {
        background-color: #e8f5e8;
    }
    
    /* Input fields - light mint with green text */
    .stTextInput > div > div > input {
        background-color: #f5f8f0;
        color: #1a3d0a;
        border: 1px solid #a8d5a8;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #f5f8f0;
        color: #1a3d0a;
        border: 1px solid #a8d5a8;
    }
    
    /* Buttons - golden with dark green text */
    .stButton > button {
        background-color: #d4af37;
        color: #1a3d0a;
        border: none;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(212, 175, 55, 0.3);
    }
    
    .stButton > button:hover {
        background-color: #c9a028;
        box-shadow: 0 4px 8px rgba(212, 175, 55, 0.4);
    }
    
    /* Expanders - light green */
    .streamlit-expanderHeader {
        background-color: #e8f5e8;
        color: #2d5016;
        border: 1px solid #a8d5a8;
    }
    
    /* Metrics - golden values, dark green labels */
    [data-testid="stMetricValue"] {
        color: #d4af37 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2d5016 !important;
    }
    
    /* Success/Error messages - green themed */
    .stSuccess {
        background-color: #e8f5e8;
        color: #2d5016;
        border-left: 4px solid #4a9e4a;
    }
    
    .stError {
        background-color: #ffe8e8;
        color: #8b1a1a;
        border-left: 4px solid #cc4444;
    }
    
    .stWarning {
        background-color: #fff8e8;
        color: #8b6f1a;
        border-left: 4px solid #d4af37;
    }
    
    .stInfo {
        background-color: #e8f0f8;
        color: #1a3d0a;
        border-left: 4px solid #4a9e4a;
    }
    
    /* Radio buttons - light green */
    .stRadio > div {
        background-color: #f0f8f0;
    }
    
    /* Selectbox - light green */
    .stSelectbox > div > div > select {
        background-color: #f5f8f0;
        color: #1a3d0a;
        border: 1px solid #a8d5a8;
    }
    
    /* Tabs - green themed */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e8f5e8;
        border-bottom: 2px solid #a8d5a8;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #2d5016;
    }
    
    .stTabs [aria-selected="true"] {
        color: #d4af37;
        border-bottom: 3px solid #d4af37;
    }
    
    /* Main header - golden with dark green shadow */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d4af37;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(45, 80, 22, 0.3);
    }
    
    /* Sidebar elements */
    [data-testid="stSidebar"] {
        background-color: #e8f5e8;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1a3d0a;
    }
    
    /* Slider */
    .stSlider > div > div {
        background-color: #a8d5a8;
    }
    
    .stSlider > div > div > div {
        background-color: #d4af37;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #f0f8f0;
        border: 2px dashed #a8d5a8;
    }
</style>
""", unsafe_allow_html=True)


def get_session_id():
    """Get or create session ID for this Streamlit session."""
    if 'session_id' not in st.session_state:
        import uuid
        st.session_state.session_id = f"streamlit_{uuid.uuid4().hex[:8]}"
    return st.session_state.session_id


@st.cache_resource
def load_analyzer_base():
    """Load base analyzer components (without memory)."""
    # This is cached separately to avoid recreating RAG components
    pass


def setup_chromadb_if_needed():
    """Automatically set up ChromaDB database if collection doesn't exist."""
    # Check if setup has already been attempted in this session
    if 'chromadb_setup_attempted' in st.session_state:
        return
    
    try:
        import chromadb
        from chromadb.config import Settings
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import json
        
        # Determine database path (try multiple strategies for Streamlit Cloud compatibility)
        db_path = None
        possible_paths = [
            Path(__file__).parent / "datasets" / "chroma_db",
            Path(os.getcwd()) / "datasets" / "chroma_db",
            Path("/mount/src/islamic-contract-analyzer/datasets/chroma_db"),  # Streamlit Cloud default
        ]
        
        for path in possible_paths:
            if path.exists():
                db_path = path
                break
        
        if db_path is None:
            # Create directory in the most likely location
            db_path = Path(__file__).parent / "datasets" / "chroma_db"
            db_path.mkdir(parents=True, exist_ok=True)
        
        # Check if collection exists
        client = chromadb.PersistentClient(
            path=str(db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = "shariaa_standards"
        collections = client.list_collections()
        collection_exists = any(c.name == collection_name for c in collections)
        
        if collection_exists:
            st.session_state.chromadb_setup_attempted = True
            return
        
        # Collection doesn't exist - set it up
        st.info("🔧 Setting up ChromaDB database... This may take a minute.")
        
        # Load embeddings and dataset (try multiple path strategies)
        possible_base_paths = [
            Path(__file__).parent,
            Path(os.getcwd()),
            Path("/mount/src/islamic-contract-analyzer"),  # Streamlit Cloud default
        ]
        
        embeddings_path = None
        metadata_path = None
        dataset_path = None
        
        for base_path in possible_base_paths:
            test_embeddings = base_path / "datasets" / "embeddings" / "embeddings.npy"
            test_dataset = base_path / "datasets" / "standards_dataset_with_embeddings.csv"
            if test_embeddings.exists() and test_dataset.exists():
                embeddings_path = test_embeddings
                metadata_path = base_path / "datasets" / "embeddings" / "metadata.json"
                dataset_path = test_dataset
                break
        
        if not embeddings_path.exists() or not dataset_path.exists():
            st.error(
                f"❌ Required files not found for database setup:\n"
                f"- Embeddings: {embeddings_path}\n"
                f"- Dataset: {dataset_path}\n\n"
                f"**For Streamlit Cloud deployment:**\n"
                f"1. Ensure `datasets/embeddings/embeddings.npy` and `datasets/standards_dataset_with_embeddings.csv` are committed to your repository\n"
                f"2. Or remove `datasets/embeddings/` from `.gitignore` if files are small enough\n"
                f"3. The database will be automatically set up on first run"
            )
            st.session_state.chromadb_setup_attempted = True
            return
        
        # Load data
        embeddings = np.load(embeddings_path)
        df = pd.read_csv(dataset_path)
        
        if len(embeddings) != len(df):
            st.error(f"❌ Mismatch: {len(embeddings)} embeddings but {len(df)} dataset rows")
            st.session_state.chromadb_setup_attempted = True
            return
        
        # Create collection
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "description": "Shariaa Standards for Islamic Contract Analysis",
                "model": "BAAI/bge-base-en-v1.5",
                "embedding_dim": embeddings.shape[1]
            }
        )
        
        # Prepare and add data in batches
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[i:batch_end]
            
            ids = [f"section_{row['embedding_index']}" for _, row in batch_df.iterrows()]
            documents = [str(row['content']) for _, row in batch_df.iterrows()]
            embeddings_list = [embeddings[idx].tolist() for idx in range(i, batch_end)]
            metadatas = [{
                'section_number': str(row['section_number']),
                'section_path': str(row['section_path']),
                'content_length': int(row['content_length']),
                'line_number': int(row['line_number']),
                'embedding_index': int(row['embedding_index'])
            } for _, row in batch_df.iterrows()]
            
            collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            progress = (i + batch_size) / len(df)
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Added batch {i//batch_size + 1}/{total_batches} ({batch_end}/{len(df)} documents)")
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ ChromaDB database set up successfully! ({collection.count()} documents)")
        st.session_state.chromadb_setup_attempted = True
        
    except Exception as e:
        st.error(f"❌ Error setting up ChromaDB: {e}")
        st.session_state.chromadb_setup_attempted = True
        import traceback
        st.code(traceback.format_exc())


def get_analyzer():
    """Get analyzer instance with memory for current session."""
    session_id = get_session_id()
    
    # Automatically set up ChromaDB if needed
    setup_chromadb_if_needed()
    
    try:
        # Get API keys from Streamlit secrets (for deployment) or environment variables
        api_keys = []
        try:
            # Try Streamlit secrets first (for Streamlit Cloud deployment)
            key1 = st.secrets.get("GEMINI_API_KEY", None)
            key2 = st.secrets.get("GEMINI_API_KEY_2", None)
            if key1:
                api_keys.append(key1)
            if key2:
                api_keys.append(key2)
        except (AttributeError, KeyError, FileNotFoundError):
            # Fall back to environment variables
            key1 = os.getenv("GEMINI_API_KEY")
            key2 = os.getenv("GEMINI_API_KEY_2")
            if key1:
                api_keys.append(key1)
            if key2:
                api_keys.append(key2)
        
        if not api_keys:
            st.error(
                "⚠️ Gemini API key not found. Please set at least GEMINI_API_KEY in:\n"
                "- Environment variables (for local development)\n"
                "- Streamlit secrets (for Streamlit Cloud deployment)\n\n"
                "Optional: Set GEMINI_API_KEY_2 for automatic fallback on rate limit errors."
            )
            return None
        
        # Create analyzer with memory enabled for this session
        # Pass api_keys list for automatic fallback (works with single or multiple keys)
        analyzer = ContractAnalyzer(
            api_keys=api_keys,
            session_id=session_id,
            use_memory=True
        )
        return analyzer
    except Exception as e:
        st.error(f"Error loading analyzer: {e}")
        return None


def main():
    """Main Streamlit application."""
    # Header
    st.markdown('<h1 class="main-header">📜 Islamic Contract Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze contracts for Shariah compliance using AI-powered RAG technology")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("📖 About")
        st.markdown(
            "This tool analyzes contracts for Shariah compliance using the **Gemini** model with **conversation memory**."
        )
        st.markdown("---")
        
        # Memory management section
        st.header("💬 Conversation Memory")
        st.markdown(
            "The system remembers previous analyses in this session for context-aware responses."
        )
        
        # Show conversation history count
        history_count = len(st.session_state.conversation_history)
        st.metric("Conversation Turns", history_count)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.conversation_history = []
            # Also clear the analyzer's memory if it exists
            try:
                analyzer = get_analyzer()
                if analyzer and analyzer.memory:
                    analyzer.memory.clear()
            except:
                pass
            st.success("Conversation history cleared!")
            st.rerun()
        
        # Show recent queries if any
        if history_count > 0:
            with st.expander("📜 Recent Analyses", expanded=False):
                for i, turn in enumerate(st.session_state.conversation_history[-5:], 1):
                    st.markdown(f"**Turn {i}:**")
                    st.caption(f"Contract: {turn.get('contract_preview', 'N/A')[:50]}...")
                    st.caption(f"Standards: {turn.get('standards_found', 0)}")
        
        st.markdown("---")
        
        # Number of standards to retrieve
        n_standards = st.slider(
            "Number of Standards to Consider",
            min_value=3,
            max_value=10,
            value=5,
            help="Number of relevant Shariaa Standards to retrieve and use in analysis"
        )
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📄 Contract Analysis", "❓ Ask Questions", "📜 Previous Analyses"])
    
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
                    st.success(f"✅ Loaded text file: {uploaded_file.name} ({len(contract_text)} characters)")
                elif uploaded_file.type == "application/pdf":
                    # Process PDF file
                    with st.spinner(f"Extracting text from PDF: {uploaded_file.name}..."):
                        try:
                            # Save uploaded file to temporary location
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            # Extract text from PDF
                            text_pages = []
                            with pdfplumber.open(tmp_path) as pdf:
                                total_pages = len(pdf.pages)
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, page in enumerate(pdf.pages, 1):
                                    try:
                                        page_text = page.extract_text() or ""
                                        text_pages.append(page_text)
                                        progress_bar.progress(i / total_pages)
                                        status_text.text(f"Processing page {i}/{total_pages}...")
                                    except Exception as e:
                                        st.warning(f"⚠️ Error extracting page {i}: {e}")
                                        text_pages.append("")
                                
                                progress_bar.empty()
                                status_text.empty()
                            
                            # Combine all pages
                            contract_text = "\n\n".join(text_pages)
                            
                            # Clean up temporary file
                            os.unlink(tmp_path)
                            
                            if contract_text.strip():
                                st.success(f"✅ Extracted text from PDF: {uploaded_file.name} ({total_pages} pages, {len(contract_text)} characters)")
                            else:
                                st.warning("⚠️ No text could be extracted from the PDF. The file might be image-based or corrupted.")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {e}")
                            import traceback
                            with st.expander("Error Details"):
                                st.code(traceback.format_exc())
                            # Clean up temp file if it exists
                            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                else:
                    st.error("Unsupported file type")
        
        # Analyze button
        if st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
            if not contract_text.strip():
                st.error("❌ Please provide contract text")
            else:
                with st.spinner("Analyzing contract with Gemini (using conversation memory)... This may take a minute."):
                    try:
                        # Get analyzer with memory for this session
                        analyzer = get_analyzer()
                        if analyzer is None:
                            st.stop()
                        
                        # Analyze (memory is automatically used)
                        results = analyzer.analyze_contract(contract_text, n_standards=n_standards)
                        
                        # Store in conversation history
                        st.session_state.conversation_history.append({
                            'timestamp': results.get('summary', {}).get('session_id', 'unknown'),
                            'contract_preview': contract_text[:100],
                            'contract_length': results['contract_length'],
                            'standards_found': results['summary']['standards_found'],
                            'analysis_preview': results['analysis']['analysis'][:200],
                            'full_results': results  # Store full results for reference
                        })
                        
                        # Keep only last 10 turns in session state (matches memory limit)
                        if len(st.session_state.conversation_history) > 10:
                            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                        
                        # Display results
                        st.success("✅ Analysis Complete!")
                        
                        # Analysis section
                        st.markdown("---")
                        st.header("📊 Compliance Analysis")
                        st.markdown(
                            f'<div class="analysis-section">{results["analysis"]["analysis"]}</div>',
                            unsafe_allow_html=True,
                        )
                        
                        # Relevant Standards section
                        if results.get("relevant_standards"):
                            st.markdown("---")
                            st.header("📚 Relevant Shariaa Standards Retrieved")
                            st.markdown(
                                f"*Retrieved {len(results['relevant_standards'])} most relevant standards using Modular RAG*"
                            )
                            
                            for std in results['relevant_standards']:
                                with st.expander(
                                    f"**Rank {std['rank']}**: Section {std['section_path']} "
                                    f"({std['section_number']}) - "
                                    f"Relevance: {std['relevance_score']:.1%}"
                                ):
                                    st.markdown(f"**Section Path:** `{std['section_path']}`")
                                    st.markdown(f"**Section Number:** `{std['section_number']}`")
                                    st.markdown(f"**Content Length:** {std['content_length']} characters")
                                    st.markdown("**Content:**")
                                    st.markdown(f'<div class="standard-card">{std["content"]}</div>', 
                                               unsafe_allow_html=True)
                        
                        # Memory indicator and context display
                        if results.get('summary', {}).get('session_id'):
                            session_id = results['summary']['session_id']
                            st.info(f"💬 **Conversation Memory Active** - Session: `{session_id}`")
                            
                            # Show conversation context if available
                            if analyzer and analyzer.memory and len(analyzer.memory.conversation_history) > 1:
                                with st.expander("📜 View Conversation Context Used", expanded=False):
                                    st.markdown("**Previous conversation turns that were included in this analysis:**")
                                    context = analyzer.memory.get_context(include_metadata=True)
                                    if context:
                                        st.code(context, language=None)
                                    else:
                                        st.caption("No previous context (this is the first analysis)")
                        
                        # Summary
                        st.markdown("---")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Standards Found", results['summary']['standards_found'])
                        with col2:
                            st.metric("Contract Length", f"{results['contract_length']} chars")
                        with col3:
                            st.metric("LLM Provider", results['summary']['llm_provider'].upper())
                        with col4:
                            st.metric("LLM Model", results['summary']['llm_model'])
                        with col5:
                            st.metric("Conversation Turns", len(st.session_state.conversation_history))
                        
                        # Download results
                        import json
                        results_json = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Analysis (JSON)",
                            data=results_json,
                            file_name="contract_analysis.json",
                            mime="application/json",
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    
    # Tab 2: Ask Questions (Scenario 1)
    with tab2:
        st.header("❓ Ask Questions About Shariaa Standards")
        st.markdown(
            "Ask questions about Islamic finance concepts, rules, and standards. "
            "The system will retrieve relevant standards and provide answers using Gemini."
        )
        st.markdown("---")
        
        # Example questions
        st.markdown("**Example Questions:**")
        example_questions = [
            "What is Murabahah?",
            "What are the rules for currency trading?",
            "What is prohibited in Islamic finance?",
            "What is Mudarabah?",
            "What are the conditions for Ijarah contract?"
        ]
        
        # Create columns for example buttons
        cols = st.columns(len(example_questions))
        for i, example in enumerate(example_questions):
            with cols[i]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.selected_question = example
        
        st.markdown("---")
        
        # Question input - use key to properly manage state
        question_key = 'question_input'
        if 'selected_question' in st.session_state:
            # Set the question from example button click
            st.session_state[question_key] = st.session_state.selected_question
            # Clear selected_question after setting it
            del st.session_state.selected_question
        
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get(question_key, ''),
            key=question_key,
            placeholder="e.g., What is Murabahah?",
            help="Ask any question about Shariaa Standards"
        )
        
        # Answer button
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if not question.strip():
                st.error("❌ Please enter a question")
            else:
                with st.spinner("Retrieving relevant standards and generating answer... This may take a moment."):
                    try:
                        # Get analyzer with memory for this session
                        analyzer = get_analyzer()
                        if analyzer is None:
                            st.stop()
                        
                        # Answer question (Scenario 1: Simple question)
                        results = analyzer.answer_question(question, n_standards=n_standards)
                        
                        # Store in conversation history
                        st.session_state.conversation_history.append({
                            'timestamp': results.get('summary', {}).get('session_id', 'unknown'),
                            'question': question,
                            'type': 'question',
                            'standards_found': results['summary']['standards_found'],
                            'answer_preview': results['answer']['answer'][:200],
                            'full_results': results
                        })
                        
                        # Keep only last 10 turns
                        if len(st.session_state.conversation_history) > 10:
                            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
                        
                        # Display results
                        st.success("✅ Answer Generated!")
                        
                        # Answer section
                        st.markdown("---")
                        st.header("💡 Answer")
                        st.markdown(
                            f'<div class="analysis-section">{results["answer"]["answer"]}</div>',
                            unsafe_allow_html=True,
                        )
                        
                        # Relevant Standards section
                        if results.get("relevant_standards"):
                            st.markdown("---")
                            st.header("📚 Relevant Shariaa Standards Retrieved")
                            st.markdown(
                                f"*Retrieved {len(results['relevant_standards'])} most relevant standards using Modular RAG*"
                            )
                            
                            # Show query type info if available
                            with st.expander("🔍 Query Routing Information", expanded=False):
                                st.markdown("**How the system processed your question:**")
                                st.info(
                                    "The router classified your question and optimized retrieval. "
                                    "For definition questions like 'What is X?', the system: "
                                    "- Classifies as DEFINITION query type\n"
                                    "- Boosts definition sections in results\n"
                                    "- Uses query expansion for better matching\n"
                                    "- Retrieves the most relevant standards"
                                )
                            
                            for std in results['relevant_standards']:
                                with st.expander(
                                    f"**Rank {std['rank']}**: Section {std['section_path']} "
                                    f"({std['section_number']}) - "
                                    f"Relevance: {std['relevance_score']:.1%}"
                                ):
                                    st.markdown(f"**Section Path:** `{std['section_path']}`")
                                    st.markdown(f"**Section Number:** `{std['section_number']}`")
                                    st.markdown(f"**Content Length:** {std['content_length']} characters")
                                    st.markdown("**Content:**")
                                    st.markdown(f'<div class="standard-card">{std["content"]}</div>', 
                                               unsafe_allow_html=True)
                        
                        # Summary
                        st.markdown("---")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Standards Found", results['summary']['standards_found'])
                        with col2:
                            st.metric("LLM Provider", results['summary']['llm_provider'].upper())
                        with col3:
                            st.metric("LLM Model", results['summary']['llm_model'])
                        with col4:
                            st.metric("Query Type", "Question")
                        
                        # Download results
                        import json
                        results_json = json.dumps(results, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="📥 Download Answer (JSON)",
                            data=results_json,
                            file_name="question_answer.json",
                            mime="application/json",
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during question answering: {e}")
                        import traceback
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
    
    # Tab 3: Previous Analyses
    with tab3:
        st.header("📜 Previous Analyses & Questions")
        
        history_count = len(st.session_state.conversation_history)
        
        if history_count == 0:
            st.info("📭 No previous analyses or questions yet. Use the 'Contract Analysis' or 'Ask Questions' tabs to see results here.")
        else:
            st.markdown(f"**Total Analyses:** {history_count}")
            st.markdown("---")
            
            # Create a selectbox to choose which analysis to view
            # Build options with analysis numbers (most recent first)
            analysis_options = []
            total_analyses = len(st.session_state.conversation_history)
            
            for i in range(total_analyses):
                # Reverse index: most recent is #total_analyses, oldest is #1
                analysis_num = total_analyses - i
                turn = st.session_state.conversation_history[total_analyses - 1 - i]
                
                # Handle both contract analysis and question types
                if turn.get('type') == 'question':
                    preview = turn.get('question', 'Unknown question')[:60]
                    item_type = "Question"
                    size_info = ""
                else:
                    preview = turn.get('contract_preview', 'Unknown contract')[:60]
                    item_type = "Analysis"
                    size_info = f", {turn.get('contract_length', 0)} chars"
                
                standards = turn.get('standards_found', 0)
                analysis_options.append(
                    f"{item_type} #{analysis_num} - {preview}... ({standards} standards{size_info})"
                )
            
            selected_index = st.selectbox(
                "Select Analysis to View:",
                options=range(len(analysis_options)),
                format_func=lambda x: analysis_options[x],
                help="Choose an analysis to view its full details"
            )
            
            # Get the selected analysis
            # selected_index 0 = most recent (index total_analyses-1), selected_index 1 = second most recent, etc.
            actual_index = total_analyses - 1 - selected_index
            analysis_number = total_analyses - selected_index  # Analysis # for display
            selected_analysis = st.session_state.conversation_history[actual_index]
            results = selected_analysis.get('full_results')
            
            if results:
                st.markdown("---")
                
                # Determine if this is a question or contract analysis
                is_question = selected_analysis.get('type') == 'question'
                
                # Analysis metadata
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    item_type = "Question" if is_question else "Analysis"
                    st.metric(f"{item_type} #", f"#{analysis_number}")
                with col2:
                    st.metric("Standards Found", results['summary']['standards_found'])
                with col3:
                    if is_question:
                        st.metric("Type", "Question")
                    else:
                        st.metric("Contract Length", f"{results.get('contract_length', 0)} chars")
                with col4:
                    st.metric("LLM Model", results['summary']['llm_model'])
                
                # Show question or contract preview
                if is_question:
                    st.markdown("---")
                    st.header("❓ Question")
                    st.info(f"**{selected_analysis.get('question', 'Unknown question')}**")
                else:
                    # Contract preview
                    with st.expander("📄 Contract Preview", expanded=False):
                        contract_preview = selected_analysis.get('contract_preview', '')
                        st.text_area(
                            "Contract Text (Preview)",
                            value=contract_preview + ("..." if len(contract_preview) == 100 else ""),
                            height=200,
                            disabled=True,
                            label_visibility="collapsed"
                        )
                        st.caption(f"Full contract length: {results.get('contract_length', 0)} characters")
                
                # Answer or Analysis section
                st.markdown("---")
                if is_question:
                    st.header("💡 Answer")
                    st.markdown(
                        f'<div class="analysis-section">{results["answer"]["answer"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.header("📊 Compliance Analysis")
                    st.markdown(
                        f'<div class="analysis-section">{results["analysis"]["analysis"]}</div>',
                        unsafe_allow_html=True,
                    )
                
                # Relevant Standards section
                if results.get("relevant_standards"):
                    st.markdown("---")
                    st.header("📚 Relevant Shariaa Standards Retrieved")
                    st.markdown(
                        f"*Retrieved {len(results['relevant_standards'])} most relevant standards using Modular RAG*"
                    )
                    
                    for std in results['relevant_standards']:
                        with st.expander(
                            f"**Rank {std['rank']}**: Section {std['section_path']} "
                            f"({std['section_number']}) - "
                            f"Relevance: {std['relevance_score']:.1%}"
                        ):
                            st.markdown(f"**Section Path:** `{std['section_path']}`")
                            st.markdown(f"**Section Number:** `{std['section_number']}`")
                            st.markdown(f"**Content Length:** {std['content_length']} characters")
                            st.markdown("**Content:**")
                            st.markdown(f'<div class="standard-card">{std["content"]}</div>', 
                                       unsafe_allow_html=True)
                
                # Download button for this specific analysis/question
                st.markdown("---")
                import json
                results_json = json.dumps(results, indent=2, ensure_ascii=False)
                file_type = "question" if is_question else "analysis"
                label_text = f"📥 Download {item_type} #{analysis_number} (JSON)"
                file_name = f"{file_type}_{analysis_number}.json"
                st.download_button(
                    label=label_text,
                    data=results_json,
                    file_name=file_name,
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Full results not available for this analysis.")


if __name__ == "__main__":
    main()

