from typing import List, Dict, Any, Optional
import json

class RAGEngine:
    """RAG engine using ChromaDB + OpenAI."""

    # Pricing per 1K tokens (USD)
    MODEL_PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, api_key: str, model: str = "gpt-4o-mini", top_k: int = 4):
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.collection = None
        self._setup_clients()

    def _setup_clients(self):
        """Initialize ChromaDB and OpenAI clients."""
        import chromadb
        from openai import OpenAI
        self.chroma_client = chromadb.Client()
        self.openai_client = OpenAI(api_key=self.api_key)

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build ChromaDB vector index from document chunks."""
        if not chunks:
            raise ValueError("No chunks provided")

        # Delete existing collection if any
        try:
            self.chroma_client.delete_collection("documents")
        except Exception:
            pass

        self.collection = self.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Get embeddings in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk["content"] for chunk in batch]
            embeddings = self._get_embeddings(texts)
            ids = [f"chunk_{i + j}" for j in range(len(batch))]
            metadatas = [{"source": chunk["source"], "chunk_id": chunk["chunk_id"]}
                         for chunk in batch]
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get OpenAI embeddings for texts."""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def query(self, question: str, history: List[Dict] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.collection:
            raise ValueError("Index not built. Call build_index first.")

        # Get query embedding
        query_embedding = self._get_embeddings([question])[0]

        # Search similar chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(self.top_k, self.collection.count())
        )

        # Format context
        sources = []
        context_parts = []
        if results["documents"] and results["documents"][0]:
            for j, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                sources.append({"source": meta.get("source", "Unknown"), "content": doc})
                context_parts.append(f"[Source {j+1}: {meta.get('source', 'Unknown')}]\n{doc}")

        context = "\n\n---\n\n".join(context_parts)

        # Build messages
        system_msg = """You are a helpful assistant that answers questions based on provided documents.
Use the context below to answer. Always cite sources. If the answer is not in the context, say so.

Context:
""" + context

        messages = [{"role": "system", "content": system_msg}]
        if history:
            for msg in history[-6:]:  # Keep last 6 messages for context
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": question})

        # Call LLM
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )

        answer = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        # Calculate cost
        pricing = self.MODEL_PRICING.get(self.model, {"input": 0.001, "output": 0.002})
        cost = (prompt_tokens / 1000 * pricing["input"]) + (completion_tokens / 1000 * pricing["output"])

        return {
            "answer": answer,
            "sources": sources,
            "tokens_used": total_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "model": self.model
        }
