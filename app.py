import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest
from rag_pipeline import build_rag_chain

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# --- Sidebar ---
with st.sidebar:
    st.header("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Indexing your PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                ingest(tmp_path)
                st.session_state.rag_chain = build_rag_chain()
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
                os.unlink(tmp_path)

            st.success(f"Ready! Chatting about: {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.info(f"Active: {st.session_state.pdf_name}")

    st.divider()
    st.caption("LangChain · ChromaDB · all-mpnet-base-v2")

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
