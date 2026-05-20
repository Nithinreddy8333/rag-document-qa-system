import streamlit as st
import os
from document_processor import DocumentProcessor
from rag_engine import RAGEngine

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.citation-box {background:#e8f4fd;border-left:4px solid #2196F3;padding:0.8rem;margin:0.5rem 0;border-radius:4px;}
.chat-message {padding:1rem;border-radius:8px;margin:0.5rem 0;}
.user-message {background:#e3f2fd;border-left:4px solid #1976D2;}
.assistant-message {background:#f3e5f5;border-left:4px solid #7B1FA2;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state: st.session_state.messages = []
if "rag_engine" not in st.session_state: st.session_state.rag_engine = None
if "doc_processor" not in st.session_state: st.session_state.doc_processor = DocumentProcessor()
if "total_tokens" not in st.session_state: st.session_state.total_tokens = 0
if "total_cost" not in st.session_state: st.session_state.total_cost = 0.0
if "processed_files" not in st.session_state: st.session_state.processed_files = []

st.title("📚 RAG Document Q&A System")
st.markdown("Upload your documents and chat with them using AI-powered retrieval")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    st.divider()
    model = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    st.subheader("📄 Document Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
    top_k = st.slider("Top K Results", 1, 10, 4)
    st.divider()
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    if uploaded_files and api_key:
        if st.button("🔄 Process Documents", use_container_width=True):
            with st.spinner("Processing documents..."):
                try:
                    all_chunks = []
                    processed = []
                    progress_bar = st.progress(0)
                    for i, file in enumerate(uploaded_files):
                        st.write(f"Processing: {file.name}")
                        chunks = st.session_state.doc_processor.process_file(file, chunk_size, chunk_overlap)
                        all_chunks.extend(chunks)
                        processed.append(file.name)
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    st.session_state.rag_engine = RAGEngine(api_key=api_key, model=model, top_k=top_k)
                    st.session_state.rag_engine.build_index(all_chunks)
                    st.session_state.processed_files = processed
                    st.session_state.messages = []
                    st.success(f"Processed {len(processed)} doc(s) with {len(all_chunks)} chunks!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    elif uploaded_files and not api_key:
        st.warning("Please enter your OpenAI API key first")
    st.divider()
    st.subheader("📊 Usage Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Tokens Used", f"{st.session_state.total_tokens:,}")
    with col2:
        st.metric("Est. Cost", f"${st.session_state.total_cost:.4f}")
    if st.session_state.processed_files:
        st.subheader("📋 Loaded Documents")
        for fname in st.session_state.processed_files:
            st.markdown(f"✅ {fname}")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not api_key:
    st.info("Please enter your OpenAI API key in the sidebar to get started.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### How to use")
        st.markdown("1. Enter your OpenAI API key")
        st.markdown("2. Upload documents (PDF, DOCX, TXT, MD)")
        st.markdown("3. Click Process Documents")
        st.markdown("4. Start chatting!")
    with col2:
        st.markdown("### Features")
        st.markdown("✅ Multi-document upload")
        st.markdown("✅ Semantic search with citations")
        st.markdown("✅ Conversation memory")
        st.markdown("✅ Token & cost tracking")
        st.markdown("✅ ChromaDB vector store")
elif not st.session_state.rag_engine:
    st.info("Upload and process your documents to start chatting!")
    with st.expander("What can you ask?"):
        st.markdown("- What is the main topic?")
        st.markdown("- Summarize the key findings")
        st.markdown("- What does it say about [topic]?")
        st.markdown("- What are the conclusions?")
else:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander(f"Sources ({len(message['citations'])} references)"):
                        for i, citation in enumerate(message["citations"], 1):
                            st.markdown(f"**Source {i}:** {citation.get('source', 'Unknown')}")
                            st.markdown(f"*{citation.get('content', '')[:200]}...*")
                            st.divider()

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = st.session_state.rag_engine.query(prompt, st.session_state.messages[:-1])
                st.session_state.total_tokens += result.get("tokens_used", 0)
                st.session_state.total_cost += result.get("cost", 0.0)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("sources", [])
                })
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
        st.rerun()
