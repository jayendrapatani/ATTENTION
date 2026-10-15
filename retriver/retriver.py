from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize the same embedding model used during indexing
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    num_ctx=8192,
    dimensions=1024
)

# Load the existing Chroma database
vectorstore = Chroma(
    collection_name="attention_posts",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create a retriever
def retrieve_posts(query: str, k: int = 10):

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    documents = retriever.invoke(query)

    return documents
