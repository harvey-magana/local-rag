# load the persisted ChromaDB collection, retrieve relevant chunks, and pass them to Ollama for generation
import chromadb

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from ollama import Client


# 1. Configure the same embedding model used during ingestion
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest"
)

# 2. Connect to the existing persistent ChromaDB
db = chromadb.PersistentClient(
    path="./chroma_db"
)

chroma_collection = db.get_or_create_collection(
    name="local_rag"
)

# 3. Connect ChromaDB to LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# 4. Reconstruct the index from the persisted vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)

print("Loaded persisted vector index.")

# 5. Create retriever
retriever = index.as_retriever(
    similarity_top_k=2
)

# 6. Create Ollama client
client = Client(
    host="http://localhost:11434",
    timeout=600
)

# 7. Interactive query loop
while True:
    question = input("\nAsk a question ('quit' to exit): ").strip()

    if question.lower() in {"quit", "exit"}:
        print("Exiting.")
        break

    if not question:
        continue

    # Retrieve relevant chunks
    nodes = retriever.retrieve(question)

    print("\nQuestion:")
    print(question)

    print("\nRetrieved chunks:")

    for i, node in enumerate(nodes, start=1):
        print(f"\n--- Source {i} ---")
        print(f"Similarity score: {node.score}")

        metadata = node.metadata or {}
        print(f"File name: {metadata.get('file_name', 'Unknown')}")

        print("\nSupporting text:")
        print(node.text[:1000])

    # Build context from retrieved chunks
    context = "\n\n".join(
        node.text for node in nodes
    )

    # Build grounded prompt
    prompt = f"""
Answer the question using only the context provided below.

If the answer cannot be determined from the context, say:
"I do not have enough information in the provided context."

Context:
{context}

Question:
{question}

Give a concise answer.
"""

    print("\nGenerating answer with llama3.1...")

    response = client.chat(
        model="llama3.1:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "num_predict": 150
        }
    )

    print("\nGenerated answer:")
    print(response["message"]["content"])

    source_names = sorted(
    {
        (node.metadata or {}).get("file_name", "Unknown")
        for node in nodes
    }
)

print("\nSources:")

for source_name in source_names:
    print(f"- {source_name}")





