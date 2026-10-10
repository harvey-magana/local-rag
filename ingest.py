# load documents, chunk them, embed them, and persist them to ChromaDB
import chromadb

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Configure embeddings
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text:latest"
)

# 2. Configure chunking
Settings.text_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

# 3. Load documents
documents = SimpleDirectoryReader("data").load_data()

print(f"Loaded {len(documents)} document(s).")

# 4. Show how many chunks will be created
nodes = Settings.text_splitter.get_nodes_from_documents(documents)

print(f"Created {len(nodes)} chunk(s).")

# 5. Create persistent ChromaDB client
db = chromadb.PersistentClient(
    path="./chroma_db"
)

collection_name = "local_rag"

try:
    db.delete_collection(collection_name)
    print("Existing ChromaDB collection removed.")
except Exception:
    pass

chroma_collection = db.create_collection(
    name=collection_name
)

# 6. Get or create collection
chroma_collection = db.get_or_create_collection(
    name="local_rag"
)

# 7. Connect ChromaDB to LlamaIndex
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# 8. Embed documents and persist them
VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

print("Documents embedded and stored in ChromaDB.")