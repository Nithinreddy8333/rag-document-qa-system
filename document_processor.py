undefinedimport io
from typing import List, Dict, Any

class DocumentProcessor:
    """Handles document loading and chunking."""

    def process_file(self, uploaded_file, chunk_size=500, chunk_overlap=50):
        """Process an uploaded file and return chunks with metadata."""
        file_name = uploaded_file.name
        ext = file_name.rsplit(".", 1)[-1].lower()

        if ext == "pdf":
            text = self._extract_pdf(uploaded_file)
        elif ext == "docx":
            text = self._extract_docx(uploaded_file)
        elif ext in ["txt", "md"]:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
        else:
            raise ValueError("Unsupported file type: " + ext)

        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        return [{"content": chunk, "source": file_name, "chunk_id": i}
                for i, chunk in enumerate(chunks)]

    def _extract_pdf(self, uploaded_file):
        """Extract text from PDF."""
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            return "".join(p.extract_text() + "\n" for p in reader.pages)
        except ImportError:
            raise ImportError("Install PyPDF2: pip install PyPDF2")

    def _extract_docx(self, uploaded_file):
        """Extract text from DOCX."""
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

    def _chunk_text(self, text, chunk_size, chunk_overlap):
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        text = " ".join(text.split())
        chunks = []
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            if end >= len(words):
                break
            start += chunk_size - chunk_overlap
        return chunks
