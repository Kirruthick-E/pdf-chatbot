import os

from dotenv import load_dotenv

load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from embeddings import DEFAULT_MODEL, get_embedding_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about PDF documents.
Use ONLY the context provided below to answer the question.
If the answer cannot be found in the context, say "I couldn't find information about that in the document."
Always mention which page(s) the information came from when possible.

Context:
{context}

Question: {question}

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def format_docs(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        parts.append(f"[Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def load_retriever(embedding_model: str = DEFAULT_MODEL):
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Vector store not found at '{CHROMA_DIR}'. Run ingest.py on a PDF first."
        )
    embedding_fn = get_embedding_model(embedding_model)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


def load_llm():
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.0-flash",
            temperature=0.1,
            max_output_tokens=1024,
        )
    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or "gemma3",
            temperature=0.1,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. Use 'gemini' or 'ollama'."
        )


def build_rag_chain(embedding_model: str = DEFAULT_MODEL):
    retriever = load_retriever(embedding_model)
    llm = load_llm()
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    question = input("Ask a question about your PDF: ")
    answer = chain.invoke(question)
    print("\nAnswer:", answer)
