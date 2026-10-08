import os

import ollama
import streamlit as st

from rag import SUPPORTED, VectorStore, build_prompt, generate_stream, load_folder, read_bytes

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 RAG Chatbot (Ollama)")


def _secret(key, default):
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    host = st.text_input("Ollama host", _secret("OLLAMA_HOST", "http://localhost:11434"))
    chat_model = st.text_input("Chat model", _secret("CHAT_MODEL", "llama3.2"))
    embed_model = st.text_input("Embedding model", _secret("EMBED_MODEL", "nomic-embed-text"))
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, 4)
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, 100)

    st.header("📄 Knowledge base")
    uploads = st.file_uploader("Upload files", type=[e[1:] for e in SUPPORTED],
                               accept_multiple_files=True)
    use_folder = st.checkbox("Include files from knowledge_base/ folder", True)
    build = st.button("🔄 Build / rebuild index", type="primary")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []

client = ollama.Client(host=host)

# ---------------- Build index ----------------
if build:
    docs = load_folder("knowledge_base") if use_folder else {}
    for f in uploads or []:
        docs[f.name] = read_bytes(f.name, f.getvalue())
    if not docs:
        st.sidebar.warning("No documents found.")
    else:
        try:
            with st.spinner("Embedding documents..."):
                store = VectorStore(client, embed_model)
                n = store.build(docs, size=chunk_size, overlap=chunk_size // 5)
            st.session_state.store = store
            st.sidebar.success(f"Indexed {len(docs)} file(s) → {n} chunks")
        except Exception as e:
            st.sidebar.error(f"Could not reach Ollama / embed model:\n{e}")

store = st.session_state.get("store")
st.session_state.setdefault("messages", [])

if store is None:
    st.info("👈 Add documents and click **Build / rebuild index** to begin.")

# ---------------- Chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Retrieved context"):
                for h in m["sources"]:
                    st.caption(f"**{h.source}** · similarity {h.score:.2f}")
                    st.write(h.text)

if question := st.chat_input("Ask a question about your documents...", disabled=store is None):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            hits = store.search(question, k=top_k)                # retrieval
            prompt = build_prompt(question, hits)                  # augmentation
            history = [{"role": m["role"], "content": m["content"]}
                       for m in st.session_state.messages[-7:-1]]
            answer = st.write_stream(generate_stream(client, chat_model, prompt, history))  # generation
            if hits:
                with st.expander("Retrieved context"):
                    for h in hits:
                        st.caption(f"**{h.source}** · similarity {h.score:.2f}")
                        st.write(h.text)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits})
        except Exception as e:
            st.error(f"Ollama error: {e}")
