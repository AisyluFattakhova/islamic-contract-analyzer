# 📜 Islamic Contract Analyzer

An AI-powered web application for analyzing contracts for Shariah compliance using Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs).

## 🌟 Features

- **Contract Analysis**: Upload or paste contract text to analyze for Shariah compliance
- **Question Answering**: Ask questions about Islamic finance concepts and Shariaa Standards
- **Modular RAG Architecture**: Intelligent retrieval of relevant standards using query routing
- **Conversation Memory**: Context-aware multi-turn conversations with conversation history
- **Multiple API Key Support**: Automatic fallback to backup API key on rate limit errors (429)
- **PDF Support**: Extract and analyze text from PDF documents
- **Dark Mode UI**: Modern, user-friendly Streamlit interface

## 🏗️ Architecture

### Core Components

1. **Modular RAG System**
   - **Query Router**: Classifies queries (definitions, rules, prohibitions) and optimizes retrieval
   - **RAG Retriever**: Retrieves relevant Shariaa Standards from vector database
   - **Embedder**: Uses `sentence-transformers` (BAAI/bge-base-en-v1.5) for embeddings
   - **Vector Store**: ChromaDB for persistent vector storage

2. **LLM Integration**
   - **Google Gemini 2.5 Flash**: Primary LLM for analysis and question answering
   - **Automatic Key Fallback**: Switches to backup API key on 429 rate limit errors
   - **Rate Limiting & Retry Logic**: Exponential backoff for robust API calls

3. **Memory Management**
   - **ConversationMemory**: Maintains session-based conversation history
   - **MemoryManager**: Manages multiple conversation sessions
   - **Context Injection**: Previous turns included in prompts for context-aware responses

4. **Web Interface**
   - **Streamlit Frontend**: Three-tab interface (Contract Analysis, Ask Questions, Previous Analyses)
   - **Dark Mode Theme**: Custom CSS styling
   - **Real-time Progress**: Progress bars and status updates
   - **Results Export**: Download analysis results as JSON

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key (get one at [Google AI Studio](https://makersuite.google.com/app/apikey))
- Optional: Second API key for fallback (`GEMINI_API_KEY_2`)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd islamic-contract-analyzer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file or set environment variables:
   ```bash
   # Required
   export GEMINI_API_KEY="your-api-key-here"
   
   # Optional (for automatic fallback on rate limits)
   export GEMINI_API_KEY_2="your-backup-api-key-here"
   ```

   Or for Streamlit Cloud deployment, add to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-api-key-here"
   GEMINI_API_KEY_2 = "your-backup-api-key-here"  # Optional
   ```

5. **Set up vector database** (if not already done)
   
   The application will automatically set up ChromaDB on first run if the collection doesn't exist. Ensure you have:
   - `datasets/embeddings/embeddings.npy`
   - `datasets/embeddings/metadata.json`
   - `datasets/standards_dataset_with_embeddings.csv`

   Or run the setup script manually:
   ```bash
   python scripts/setup_vector_db.py
   ```

## 💻 Usage

### Running the Web Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the CLI

```bash
# Analyze a contract file
python scripts/contract_analyzer.py contract.txt --standards 5 --output results.json

# Answer a question
python scripts/rag_query.py "What is Murabahah?"
```

### Web Interface Tabs

1. **📄 Contract Analysis**
   - Upload a PDF or paste contract text
   - Get comprehensive Shariah compliance analysis
   - View relevant standards retrieved
   - Download results as JSON

2. **❓ Ask Questions**
   - Ask questions about Islamic finance concepts
   - Click example questions or type your own
   - Get answers based on Shariaa Standards
   - View query routing information

3. **📜 Previous Analyses**
   - View history of all analyses and questions
   - Download previous results as JSON
   - Review conversation context

## 📁 Project Structure

```
islamic-contract-analyzer/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
│
├── scripts/                        # Core analysis scripts
│   ├── contract_analyzer.py       # Main analyzer class
│   ├── memory_manager.py           # Conversation memory management
│   ├── api_utils.py                # API utilities (retry, rate limiting)
│   ├── setup_vector_db.py         # Vector database setup
│   ├── generate_embeddings.py      # Embedding generation
│   ├── construct_dataset.py        # Dataset construction
│   └── ...
│
├── rag/                            # Modular RAG components
│   ├── router.py                   # Query routing logic
│   ├── retriever.py                # RAG retriever
│   ├── embedder.py                 # Embedding generation
│   └── vector_store.py             # ChromaDB integration
│
├── datasets/                       # Data and vector database
│   ├── chroma_db/                  # ChromaDB database
│   ├── embeddings/                 # Precomputed embeddings
│   └── standards_dataset*.csv     # Standards dataset
│
├── tests/                          # Test files
│   └── benchmark_rag.py            # RAG benchmarks
│
├── clean_text/                     # Cleaned text files
├── raw_pdf/                        # Original PDF files
└── .streamlit/                      # Streamlit configuration
```

## 🔧 Configuration

### API Keys

- **Primary Key**: `GEMINI_API_KEY` (required)
- **Backup Key**: `GEMINI_API_KEY_2` (optional, for automatic fallback)

The system automatically switches to the backup key when the primary key hits a rate limit (429 error).

### Number of Standards

Adjust the number of relevant standards to retrieve in the sidebar slider (default: 5, range: 3-10).

### Conversation Memory

- Memory is enabled by default for context-aware responses
- Clear conversation history using the sidebar button
- Memory persists for the duration of the Streamlit session
- Last 10 conversation turns are maintained

## 🛠️ Technical Highlights

- **Modern LLM Stack**: Google Gemini 2.5 Flash with automatic failover
- **Modular RAG**: Query routing + vector retrieval for optimal results
- **Conversation Memory**: Session-based context management
- **Robust Error Handling**: Retry logic, rate limiting, and key switching
- **Streamlit UI**: Dark mode, responsive design, real-time feedback
- **PDF Processing**: Automatic text extraction with progress tracking
- **Auto-Setup**: ChromaDB initialization on first run

## 📊 How It Works

1. **Contract Analysis Flow**:
   - User uploads/pastes contract text
   - System retrieves relevant Shariaa Standards using RAG
   - Query router classifies and optimizes retrieval strategy
   - Gemini LLM generates compliance analysis
   - Results displayed with relevant standards and recommendations

2. **Question Answering Flow**:
   - User asks a question about Islamic finance
   - Router classifies query type (definition, rules, etc.)
   - Retriever fetches most relevant standards
   - Gemini generates comprehensive answer
   - Conversation memory updated for context

3. **Error Handling**:
   - On 429 rate limit error, automatically switches to backup API key
   - Exponential backoff retry logic
   - Clear error messages with solutions

## 🧪 Testing

```bash
# Run RAG benchmarks
python tests/benchmark_rag.py

# Test dataset construction
python scripts/construct_dataset.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Aisylu Fattakhova**

## 📞 Support

For issues, questions, or contributions text to @Aisylu_Fattakhova on Telegram.

---

**Note**: This tool is designed to assist with Shariah compliance analysis but should not replace professional legal or Shariah advisory services.

