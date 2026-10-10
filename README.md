# Local RAG with Ollama, LlamaIndex, and ChromaDB

A local Retrieval-Augmented Generation (RAG) learning project built with Python, Ollama, LlamaIndex, and ChromaDB.

The project demonstrates how to ingest local documents, split them into chunks, generate local embeddings, persist those embeddings in a vector database, retrieve semantically relevant context, and use a locally running large language model to generate grounded answers.

No cloud-hosted LLM is required.

## Architecture

```text
                 INGESTION

Local Documents
      |
      v
LlamaIndex
      |
      v
SentenceSplitter
      |
      v
nomic-embed-text
      |
      v
ChromaDB
(persistent vectors)


                  QUERY

User Question
      |
      v
nomic-embed-text
      |
      v
ChromaDB Similarity Search
      |
      v
Relevant Document Chunks
      |
      v
Prompt Augmentation
      |
      v
Ollama
      |
      v
llama3.1
      |
      v
Grounded Answer + Sources
```

## Technologies

- Python 3.13
- Ollama
- Llama 3.1
- nomic-embed-text
- LlamaIndex
- ChromaDB

## Project Structure

```text
local-rag/
├── data/
│   ├── client_authentication_details_OAuth2.0.txt
│   ├── mtls_basics.txt
│   └── short_description_of_rest_api.txt
├── ingest.py
├── query.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `chroma_db/` directory is generated locally during ingestion and is intentionally excluded from Git.

## How It Works

### 1. Document ingestion

`ingest.py`:

1. Reads documents from `data/`.
2. Splits documents into chunks.
3. Generates embeddings with `nomic-embed-text`.
4. Stores the vectors and document metadata in ChromaDB.

### 2. Retrieval

When a user asks a question, `query.py` converts the question into an embedding using the same embedding model.

ChromaDB performs semantic similarity search against the stored document vectors and returns the most relevant chunks.

### 3. Augmentation

The retrieved chunks are combined into context and inserted into a prompt.

The prompt instructs the language model to answer only from the supplied context.

If the context does not contain enough information, the model is instructed to respond:

> I do not have enough information in the provided context.

### 4. Generation

The augmented prompt is sent to a locally running `llama3.1` model through the Ollama Python client.

The resulting answer and supporting source filenames are displayed in the terminal.

## Prerequisites

Install Ollama before running the project.

Verify the installation:

```bash
ollama --version
```

The project uses these models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Verify them with:

```bash
ollama list
```

## Installation

Clone the repository:

```bash
git clone https://github.com/harvey-magana/local-rag.git
cd local-rag
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Add Documents

Place supported documents in:

```text
data/
```

The initial learning project uses text documents covering REST APIs, OAuth 2.0 client authentication, and mutual TLS.

## Build the Knowledge Base

Run:

```bash
python ingest.py
```

Example:

```text
Loaded 3 document(s).
Created 5 chunk(s).
Documents embedded and stored in ChromaDB.
```

This creates the local persistent vector database.

## Query the Knowledge Base

Run:

```bash
python query.py
```

Example:

```text
Loaded persisted vector index.

Ask a question ('quit' to exit):
How does private_key_jwt authenticate a client?
```

The application displays:

- retrieved document chunks
- similarity scores
- source filenames
- supporting context
- the generated answer

After answering, the application prompts for another question.

Enter:

```text
quit
```

or:

```text
exit
```

to end the session.

## Example RAG Behavior

A question about REST API methods retrieves the REST document.

A question about OAuth 2.0 client authentication retrieves the OAuth document.

A question about mutual TLS retrieves the mTLS document.

If a retrieved document does not contain sufficient information to answer the question, the language model is instructed not to rely on its pretrained knowledge and instead reports that the supplied context is insufficient.

This provides basic protection against ungrounded responses.

## Key Concepts Demonstrated

This project demonstrates:

- Retrieval-Augmented Generation
- semantic search
- local embeddings
- document chunking
- vector databases
- persistent vector storage
- metadata-aware retrieval
- grounded prompting
- multi-document retrieval
- local LLM inference
- source traceability

## Models

### `nomic-embed-text`

Used for embeddings and semantic similarity search.

It converts both document chunks and questions into vectors that can be compared mathematically.

### `llama3.1`

Used for response generation.

The retrieved document context is supplied to the model before it generates an answer.

## Design Decision: Direct Ollama Generation

LlamaIndex is used for document processing, embedding integration, vector indexing, and retrieval.

Generation is performed using the native Ollama Python client.

During development, direct Ollama generation proved more reliable for local inference and provided explicit control over HTTP timeouts and generation parameters.

## Possible Future Enhancements

The current project intentionally remains a small educational RAG implementation.

Potential extensions include:

- incremental document ingestion
- document change detection
- similarity-score thresholds
- reranking
- metadata filtering
- streaming generation
- retrieval evaluation
- additional document formats
- a web interface
- conversational history
- hybrid keyword/vector search

## Purpose

This repository was created as a hands-on learning project for understanding how a complete RAG pipeline operates locally.

The goal was not simply to produce answers from documents, but to expose each major RAG stage independently:

```text
Ingestion
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Storage
   ↓
Retrieval
   ↓
Augmentation
   ↓
Generation
```

This makes the system easier to understand, troubleshoot, and extend.