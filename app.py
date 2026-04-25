import os
import io
import sys
import logging
import warnings
import tempfile

# Suppress ModuleNotFoundError for optional vision deps (e.g. torchvision)
# sentence-transformers tries to load CLIP/vision modules at import time;
# they are not needed for text-only embeddings.
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

_stderr = sys.stderr
sys.stderr = io.StringIO()
try:
    from ingest import ingest
    from rag_pipeline import build_rag_chain
    from embeddings import AVAILABLE_MODELS, DEFAULT_MODEL
finally:
    sys.stderr = _stderr

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)

# --- Sidebar ---
with st.sidebar:
    st.header("Upload a PDF")

    selected_model = st.selectbox(
        "Embedding model",
        options=list(AVAILABLE_MODELS.keys()),
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.embedding_model),
        format_func=lambda key: f"{key} — {AVAILABLE_MODELS[key]}",
    )

    if selected_model != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_model
        st.session_state.rag_chain = None
        st.session_state.pdf_name = None

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Indexing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                ingest(tmp_path, embedding_model=st.session_state.embedding_model)
                st.session_state.rag_chain = build_rag_chain(
                    embedding_model=st.session_state.embedding_model
                )
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                os.unlink(tmp_path)

            st.success(f"Ready! Chatting about: {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.info(f"Active: {st.session_state.pdf_name}")

    st.divider()
    st.caption(f"LangChain · ChromaDB · {st.session_state.embedding_model}")

# --- Main area ---
st.title("PDF Chatbot")
st.caption("Upload a PDF in the sidebar, then ask questions about its contents.")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.rag_chain is None:
    st.info("Upload a PDF using the sidebar to get started.")

user_input = st.chat_input(
    "Ask a question about your PDF...", disabled=(st.session_state.rag_chain is None)
)

if user_input and st.session_state.rag_chain:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            response = st.session_state.rag_chain.invoke(user_input)
        st.markdown(response)

    st.session_state.chat_history.append({"role": "assistant", "content": response})
