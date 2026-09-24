# 🔎 Hybrid RAG with LlamaIndex

A **Hybrid Retrieval-Augmented Generation (RAG)** system built with **LlamaIndex** that combines **semantic vector search** and **keyword-based BM25 search** to retrieve relevant information from PDF documents.

The retrieved information is then passed to **Google Gemini** to generate a grounded response.

---

## 🚀 Overview

Traditional RAG systems generally rely on vector similarity search. While semantic search is powerful, it can sometimes miss exact keywords, technical terms, names, or specific phrases.

This project solves that problem by combining:

* 🧠 **Dense Retrieval** → Semantic similarity using embeddings
* 🔤 **Sparse Retrieval** → Keyword matching using BM25
* 🔀 **Hybrid Retrieval** → Reciprocal Rank Fusion (RRF)
* 📚 **PDF Document Loading** → LlamaIndex
* 🗄️ **Vector Storage** → Qdrant
* 🤗 **Embeddings** → BAAI BGE-small-en-v1.5
* ✨ **LLM** → Google Gemini
* ⚡ **Framework** → LlamaIndex

### Architecture

```text
                 PDF Documents
                       │
                       ▼
             SimpleDirectoryReader
                       │
                       ▼
                Document Parsing
                       │
                       ▼
                SentenceSplitter
                       │
                       ▼
                  Text Chunks
                   /       \
                  /         \
                 ▼           ▼
        Hugging Face       BM25
         Embeddings       Retriever
              │               │
              ▼               ▼
           Qdrant       Keyword Search
          Vector DB           │
              │               │
              └───────┬───────┘
                      ▼
             Hybrid Retrieval
             QueryFusionRetriever
                      │
                      ▼
              Reciprocal Rank
                  Fusion
                      │
                      ▼
                Gemini LLM
                      │
                      ▼
                 Final Answer
```

---

## ✨ Features

* 📄 Load PDF documents from the `data/` directory
* ✂️ Split documents into manageable chunks
* 🧠 Generate semantic embeddings using Hugging Face
* 🔍 Perform dense vector similarity search
* 🔤 Perform BM25 keyword-based retrieval
* 🔀 Combine both retrieval strategies using Reciprocal Rank Fusion
* 🗄️ Store vectors using Qdrant
* 🤖 Generate answers using Google Gemini
* 📚 Retrieve information from your own documents instead of relying only on the LLM's knowledge

---

## 🛠️ Tech Stack

| Technology             | Purpose              |
| ---------------------- | -------------------- |
| Python                 | Programming language |
| LlamaIndex             | RAG framework        |
| Qdrant                 | Vector database      |
| Hugging Face           | Embedding model      |
| BAAI BGE-small-en-v1.5 | Text embeddings      |
| BM25                   | Keyword retrieval    |
| Google Gemini          | Large Language Model |
| nest_asyncio           | Async compatibility  |

---

## 📁 Project Structure

```text
Hybrid_RAG/
│
├── data/
│   ├── compiler.pdf
│   └── insurance.csv
│
├── src/
│   └── hybrid_rag/
│       ├── __init__.py
│       └── main.py
│
├── tests/
│   └── test_gemini_config.py
│
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

> The current implementation loads `.pdf` files from the `data/` directory.

---

## ⚙️ How It Works

### 1. Load PDF Documents

The project uses LlamaIndex's `SimpleDirectoryReader` to load PDFs:

```python
reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf"],
    recursive=True
)

documents = reader.load_data()
```

---

### 2. Split Documents into Chunks

Documents are divided into smaller chunks using `SentenceSplitter`.

```python
parser = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)
```

The overlap helps preserve context between neighboring chunks.

---

### 3. Generate Embeddings

The project uses:

```text
BAAI/bge-small-en-v1.5
```

to convert text into numerical vectors.

```python
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

These vectors represent the semantic meaning of the text.

---

### 4. Store Embeddings in Qdrant

Qdrant is used as the vector store.

```python
qdrant_client = QdrantClient(location=":memory:")
```

The current configuration uses **in-memory Qdrant**, meaning the vector database exists only while the application is running.

---

### 5. Dense Retrieval

The vector retriever searches for documents based on semantic similarity.

```python
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)
```

This is useful when the query uses different words but has similar meaning to the document.

---

### 6. Sparse Retrieval with BM25

BM25 performs keyword-based retrieval.

```python
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10
)
```

This can be useful for:

* Exact terms
* Names
* Technical terminology
* Keywords
* Specific phrases

---

### 7. Hybrid Retrieval

The project combines both retrievers:

```python
hybrid_retriever = QueryFusionRetriever(
    retrievers=[
        vector_retriever,
        bm25_retriever
    ],
    similarity_top_k=10,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False
)
```

The system uses **Reciprocal Rank Fusion (RRF)** to combine the rankings produced by the semantic and keyword retrievers.

This gives the system access to both:

```text
Semantic Understanding
        +
Keyword Matching
        ↓
Hybrid Retrieval
```

---

## 🤖 Gemini Integration

Google Gemini is used as the generation model:

```python
llm = GoogleGenAI(
    model="gemini-3.5-flash"
)
```

The retrieved document context is used to help Gemini generate the final response.

---

## 🔑 API Key Setup

Set your Gemini API key before running the application.

### Windows PowerShell

```powershell
$env:GOOGLE_API_KEY="YOUR_API_KEY"
```

Or configure it in your Python environment.

**Never commit your real API key to GitHub.**

Do not write:

```python
os.environ["GOOGLE_API_KEY"] = "your-real-api-key"
```

Instead, use an environment variable:

```python
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
```

---

## 📦 Installation

### Using pip

Create and activate a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Install the required dependencies:

```powershell
pip install llama-index
pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-google-genai
pip install llama-index-retrievers-bm25
pip install qdrant-client
pip install nest-asyncio
```

---

## ▶️ Run the Project

After installing the dependencies and configuring your Gemini API key:

```powershell
python src/hybrid_rag/main.py
```

The application loads the PDFs from:

```text
./data
```

and creates the hybrid retrieval pipeline.

---

## 💬 Example Query

The current example asks:

```python
ask(
    "What is the main topic of these documents, describe in detail?"
)
```

The system performs:

```text
User Query
     ↓
Dense Vector Search
     +
BM25 Keyword Search
     ↓
Reciprocal Rank Fusion
     ↓
Relevant Document Chunks
     ↓
Gemini
     ↓
Generated Answer
```

---

## 🧠 Why Hybrid RAG?

A vector-only RAG system primarily relies on semantic similarity.

For example:

```text
Query:
"How does memory management work?"
```

It can retrieve content discussing:

```text
memory allocation
memory organization
RAM management
```

However, exact keyword matching can be important for queries containing specific technical terms.

BM25 helps with this by looking at the actual words appearing in the documents.

Hybrid RAG combines both approaches:

```text
Dense Retrieval
→ Understands meaning

BM25
→ Finds important keywords

Hybrid Retrieval
→ Combines both
```

---

## 🔄 Retrieval Pipeline

```text
PDF
 │
 ▼
Document Loader
 │
 ▼
Sentence Splitter
 │
 ▼
 ┌───────────────────────┐
 │                       │
 ▼                       ▼
Embeddings              BM25
 │                       │
 ▼                       ▼
Qdrant              Keyword Search
 │                       │
 └───────────┬───────────┘
             ▼
       Query Fusion
             │
             ▼
          RRF
             │
             ▼
        Gemini LLM
             │
             ▼
       Final Response
```

---

## 📌 Current Limitations

The current implementation uses:

* Qdrant in-memory storage
* A single query without query expansion
* PDF documents as the primary input
* Gemini API for generation
* Local Hugging Face embeddings

Because Qdrant is configured with:

```python
QdrantClient(location=":memory:")
```

the vector collection is recreated when the application starts.

For production usage, Qdrant can instead be deployed as a persistent/local or remote vector database.

---

## 🔮 Future Improvements

Possible improvements include:

* [ ] Persistent Qdrant storage
* [ ] Support for DOCX, TXT, and CSV files
* [ ] Metadata filtering
* [ ] Source/page citations
* [ ] Reranking models
* [ ] Query expansion
* [ ] Multi-query retrieval
* [ ] Streaming responses
* [ ] FastAPI backend
* [ ] React frontend
* [ ] Chat history
* [ ] Conversation memory
* [ ] Docker deployment
* [ ] Evaluation using RAG metrics
* [ ] Authentication and user-specific document collections

---

## 🎯 Project Goal

The goal of this project is to demonstrate how **hybrid retrieval** can improve a RAG pipeline by combining:

```text
Semantic Search
      +
Keyword Search
      +
LLM Generation
      ↓
Hybrid RAG
```

This project is
