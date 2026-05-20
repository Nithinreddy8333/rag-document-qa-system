# 📚 RAG-Powered Document Q&A System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-purple.svg)](https://www.trychroma.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B.svg)](https://rag-docs-chat.streamlit.app)

A powerful Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using AI. Upload PDFs, DOCX, or text files and get intelligent answers with source citations.

## 🔗 Live Demo

**👉 [https://rag-docs-chat.streamlit.app](https://rag-docs-chat.streamlit.app)**

## ✨ Features

- **📁 Multi-document upload** - Upload and query multiple documents simultaneously
- **🔍 Semantic search** - ChromaDB vector database for intelligent document retrieval
- **📎 Source citations** - Every answer includes references to the source documents
- **💬 Conversation memory** - Maintains context across multiple questions
- **📊 Cost tracking** - Real-time token usage and cost estimation per query
- **📄 Multi-format support** - PDF, DOCX, TXT, and Markdown files
- **🤖 Multiple LLM models** - GPT-4o-mini, GPT-4o, GPT-3.5-turbo
- **⚙️ Configurable chunking** - Adjustable chunk size and overlap

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **UI Framework** | Streamlit |
| **LLM** | OpenAI GPT-4o-mini / GPT-4o |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Database** | ChromaDB |
| **Document Processing** | PyPDF2, python-docx |
| **Orchestration** | LangChain |

## 🚀 Quick Start

### Option 1: Use the Live App

Visit the live deployment - no installation needed!

**👉 [https://rag-docs-chat.streamlit.app](https://rag-docs-chat.streamlit.app)**

### Option 2: Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/Nithinreddy8333/rag-document-qa-system.git
cd rag-document-qa-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run app.py
```

4. **Open your browser** at `http://localhost:8501`

## 📖 How to Use

1. **Enter your OpenAI API key** in the sidebar
2. **Select your preferred model** (gpt-4o-mini recommended for cost efficiency)
3. **Adjust chunking settings** if needed
4. **Upload documents** (PDF, DOCX, TXT, or MD files)
5. **Click "Process Documents"** to index them in ChromaDB
6. **Start chatting!** Ask questions about your documents

## 💡 Example Questions

- *"What is the main topic of this document?"*
- *"Summarize the key findings"*
- *"What does it say about [specific topic]?"*
- *"Compare the information from document A and B"*
- *"What are the conclusions or recommendations?"*

## 🏗️ Architecture

```
User Query
    |
    ▼
Embedding (OpenAI text-embedding-3-small)
    |
    ▼
Semantic Search (ChromaDB)
    |
    ▼
Top-K Relevant Chunks + Context Building
    |
    ▼
LLM Generation (GPT-4o-mini/GPT-4o)
    |
    ▼
Answer + Source Citations
```

## 📁 Project Structure

```
rag-document-qa-system/
├── app.py                  # Main Streamlit application
├── document_processor.py   # Document loading & chunking
├── rag_engine.py           # RAG engine (ChromaDB + OpenAI)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔒 Privacy

Your documents and API key are processed in memory only. No data is stored permanently or sent to third parties (except OpenAI for embeddings and responses).

## 📄 License

MIT License - feel free to use and modify!
