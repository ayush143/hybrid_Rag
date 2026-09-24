import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine

import nest_asyncio

nest_asyncio.apply()


# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = ""

# Embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Gemini LLM
llm = GoogleGenAI(model="gemini-3.5-flash")

# Set global settings
Settings.embed_model = embed_model
Settings.llm = llm
Settings.chunk_size = 512
Settings.chunk_overlap = 50


qdrant_client = QdrantClient(location=":memory:")

# Get embedding dimension
test_embedding = embed_model.get_text_embedding("test")
embedding_dim = len(test_embedding)

# Create collection
collection_name = "hybrid_rag"
qdrant_client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
)

# Vector store
vector_store = QdrantVectorStore(client=qdrant_client, collection_name=collection_name)


reader = SimpleDirectoryReader(input_dir="./data", required_exts=[".pdf"], recursive=True)
documents = reader.load_data()
print(f"Loaded {len(documents)} documents")


from llama_index.core.node_parser import SentenceSplitter

# Parse documents into nodes
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents, show_progress=True)

# Build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)


from llama_index.core.query_engine import RetrieverQueryEngine

# Dense retriever (vector embeddings - semantic search)
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# Sparse retriever (BM25 - keyword/exact match search)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
)

# Hybrid retriever (combines both with fusion)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=10,
    num_queries=1,  # No query generation, just fusion
    mode="reciprocal_rerank",  # Reciprocal Rank Fusion
    use_async=False,
)

# Create query engine with hybrid retriever
query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)


def ask(q:str):
  """Query with grounding information (page numbers and sources)"""
  response = query_engine.query(q)

  # Print the answer
  print("=" * 80)
  print("ANSWER:")
  print("=" * 80)
  print(response.response)
  print()

ask("What is the main topic of these documents, describe in detail?")