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


def get_analyzer():
    """Get analyzer instance with memory for current session."""
    session_id = get_session_id()
    try:
        # Create analyzer with memory enabled for this session
        analyzer = ContractAnalyzer(
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
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            value=st.session_state.get('selected_question', ''),
            placeholder="e.g., What is Murabahah?",
            help="Ask any question about Shariaa Standards"
        )
        
        # Clear selected question after use
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question
        
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

