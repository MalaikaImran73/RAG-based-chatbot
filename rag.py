"""Core RAG logic: load -> chunk -> embed -> retrieve -> augment -> generate (Ollama)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf"}


# ---------- 1. Knowledge base loading ----------
def read_bytes(name: str, data: bytes) -> str:
    """Extract text from an uploaded/stored file."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    return data.decode("utf-8", errors="ignore")


def load_folder(folder: str) -> dict[str, str]:
    docs = {}
    p = Path(folder)
    if p.exists():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in SUPPORTED:
                docs[f.name] = read_bytes(f.name, f.read_bytes())
    return docs


# ---------- Chunking (paragraph aware, with overlap) ----------
def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    paras = [p for p in paras if p]
    chunks, cur = [], ""
    for para in paras:
        if len(cur) + len(para) + 1 <= size:
            cur = f"{cur} {para}".strip()
            continue
        tail = cur[-overlap:] if cur else ""
        if cur:
            chunks.append(cur)
        # very long paragraph -> hard split
        while len(para) > size:
            chunks.append(para[:size])
            para = para[size - overlap:]
        cur = (tail + " " + para).strip() if tail else para
    if cur:
        chunks.append(cur)
    return chunks


# ---------- Embeddings + vector store ----------
@dataclass
class Hit:
    text: str
    source: str
    score: float


class VectorStore:
    def __init__(self, client: ollama.Client, embed_model: str):
        self.client = client
        self.embed_model = embed_model
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.matrix: np.ndarray | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):  # batch
            resp = self.client.embed(model=self.embed_model, input=texts[i:i + 32])
            out.extend(resp["embeddings"])
        m = np.array(out, dtype=np.float32)
        return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-10)

    def build(self, docs: dict[str, str], size: int = 800, overlap: int = 150) -> int:
        self.texts, self.sources = [], []
        for name, text in docs.items():
            for ch in chunk_text(text, size, overlap):
                self.texts.append(ch)
                self.sources.append(name)
        self.matrix = self._embed(self.texts) if self.texts else None
        return len(self.texts)

    # ---------- 2. Retrieval ----------
    def search(self, question: str, k: int = 4, min_score: float = 0.25) -> list[Hit]:
        if self.matrix is None:
            return []
        q = self._embed([question])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [Hit(self.texts[i], self.sources[i], float(scores[i]))
                for i in top if scores[i] >= min_score]


# ---------- 3. Augmentation ----------
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided. If the answer is not in the context, say you could not find it in "
    "the knowledge base. Be concise and mention the source file names you used."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    if hits:
        context = "\n\n".join(
            f"[Source {i}: {h.source}]\n{h.text}" for i, h in enumerate(hits, 1)
        )
    else:
        context = "(no relevant context found)"
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


# ---------- 4. Generation ----------
def generate_stream(client: ollama.Client, model: str, prompt: str, history=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": prompt})
    for part in client.chat(model=model, messages=messages, stream=True):
        yield part["message"]["content"]
