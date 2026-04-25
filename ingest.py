import os
import sys

# Validate args before loading heavy dependencies
if __name__ == "__main__":
    _pdf_path = sys.argv[1] if len(sys.argv) > 1 else "sample_pdfs/test.pdf"
    if not os.path.exists(_pdf_path):
        print(f"Error: PDF not found: {_pdf_path}", file=sys.stderr)
        sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from embeddings import AVAILABLE_MODELS, DEFAULT_MODEL, get_embedding_model

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"


def load_pdf(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    doc = fitz.open(file_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": page_num + 1, "source": file_path})
    doc.close()
    return pages


def chunk_pages(pages: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    documents = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page["page"], "source": page["source"]},
                )
            )
    return documents


def build_vector_store(documents: list[Document], embedding_model) -> Chroma:
    if os.path.exists(CHROMA_DIR):
        import chromadb as _chromadb

        _client = _chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        del _client

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def ingest(pdf_path: str, embedding_model: str = DEFAULT_MODEL):
    print(f"Loading PDF: {pdf_path}")
    pages = load_pdf(pdf_path)
    print(f"  Extracted {len(pages)} pages")

    print("Chunking text...")
    documents = chunk_pages(pages)
    print(f"  Created {len(documents)} chunks")

    size_hint = AVAILABLE_MODELS.get(embedding_model, "")
    print(
        f"Loading embedding model '{embedding_model}' ({size_hint}, first run downloads model)..."
    )
    embeddings = get_embedding_model(embedding_model)

    print("Embedding and storing in ChromaDB...")
    vectorstore = build_vector_store(documents, embeddings)
    print(f"  Done. Vector store saved to {CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest(_pdf_path)
