from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize the same embedding model used during indexing
embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b"
)

# Load the existing Chroma database
vectorstore = Chroma(
    collection_name="attention_posts",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Create a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# Search for relevant posts
query = "What is happening in the india?"

documents = retriever.invoke(query)

for i, doc in enumerate(documents, start=1):
    print(f"\n--- Post {i} ---")
    print(doc.page_content)
    print(doc.metadata)