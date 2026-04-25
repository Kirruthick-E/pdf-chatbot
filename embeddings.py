from langchain_huggingface import HuggingFaceEmbeddings

AVAILABLE_MODELS: dict[str, str] = {
    "all-MiniLM-L6-v2": "Fast (~90 MB)",
    "all-MiniLM-L12-v2": "Balanced (~120 MB)",
    "all-mpnet-base-v2": "High quality (~420 MB)",
}

DEFAULT_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> HuggingFaceEmbeddings:
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown embedding model: {model_name!r}. "
            f"Choose one of: {list(AVAILABLE_MODELS)}"
        )
    return HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{model_name}",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
