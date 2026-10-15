from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader

from pathlib import Path
import boto3


s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.in-west2.purestore.io/",
    aws_access_key_id="cfdfca6d721b4930LG8S",
    aws_secret_access_key="cyaApPOO02EycUAim0JFDUlTtwv2OoXyy8L1KGnW"
)

# 1. Check that the input file exists
file_path = Path("raw.jsonl")
file_path = Path("raw.jsonl")

s3.download_file(
    "attention-db",
    "x/raw.jsonl",
    str(file_path)
)
if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path.resolve()}")

def metadata_func(record, metadata):
    user = record.get("user") or {}

    metadata["id"] = str(record.get("id", ""))
    metadata["url"] = record.get("url") or ""
    metadata["date"] = str(record.get("date") or "")

    metadata["username"] = user.get("username") or ""
    metadata["followersCount"] = user.get("followersCount") or 0

    metadata["lang"] = record.get("lang") or ""

    metadata["likeCount"] = record.get("likeCount") or 0
    metadata["retweetCount"] = record.get("retweetCount") or 0
    metadata["replyCount"] = record.get("replyCount") or 0
    metadata["quoteCount"] = record.get("quoteCount") or 0
    metadata["viewCount"] = record.get("viewCount") or 0

    return metadata


loader = JSONLoader(
    file_path=str(file_path),
    jq_schema=".",
    content_key="rawContent",
    json_lines=True,
    metadata_func=metadata_func
)

doc = loader.load()

print("Documents loaded:", len(doc))

# 2. Load JSONL documents
model= OllamaEmbeddings(
    model="qwen3-embedding:8b",
    num_ctx=8192,
    dimensions=1024
)


# vector_store 
vector_store = Chroma(
    collection_name="attention_posts",
    embedding_function=model,
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}
)
total = len(doc)

for i, document in enumerate(doc, start=1):
    vector_store.add_documents(
        documents=[document],
        ids=[f"attention_post_{i-1}"]
    )

    print(f"\rStored: {i}/{total} documents", end="", flush=True)


print("\nAll documents stored successfully!")