import os
import requests
from typing import List
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_core.embeddings import Embeddings

class TEIEmbeddings(Embeddings):
    def __init__(self, api_url: str):
        self.api_url = api_url
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"inputs": texts},
                timeout=30
            )
            response.raise_for_status()
            embeddings = response.json()
            return embeddings
        except Exception as e:
            print(f"✗ Error embedding documents: {e}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        try:
            response = requests.post(
                f"{self.api_url}/embed",
                json={"inputs": [text]},  #  wrap in list, because the req expect list
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            # Handle both single embedding and list of embeddings
            embedding = result[0] if isinstance(result, list) else result
            return embedding
        except Exception as e:
            print(f"✗ Error embedding query: {e}")
            raise

def get_vector_store():
    qdrant_host = os.getenv('QDRANT_HOST', 'axora-qdrant')
    qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
    collection_name = os.getenv('QDRANT_COLLECTION', 'crawl_collection')
    embedding_url = os.getenv('EMBEDDING_API_URL', 'http://axora-mpnetbasev2:8000')
    
    print(f"Connecting to Qdrant: {qdrant_host}:{qdrant_port}")
    print(f"Collection: {collection_name}")
    print(f"Embedding API: {embedding_url}")
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    try:
        collection_info = client.get_collection(collection_name)
        print(f"✓ Collection exists with {collection_info.points_count} points")
    except Exception as e:
        print(f"✗ Error accessing collection: {e}")
        raise
    
    embeddings = TEIEmbeddings(api_url=embedding_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings
    )
    
    return vector_store


def get_retriever(search_type="mmr", search_kwargs=None):
    """
    Get a retriever with specified search strategy
    
    Args:
        search_type: Type of search to perform
            - "similarity": Standard similarity search
            - "mmr": Maximal Marginal Relevance (for diversity) - DEFAULT for Phase 1
            - "similarity_score_threshold": Filter by similarity score
        search_kwargs: Additional search parameters
            For MMR (Phase 1 optimized):
            - k: Final number of documents (default: 30 for reranking)
            - fetch_k: Number of docs to fetch before MMR (default: 50)
            - lambda_mult: Diversity factor 0-1 (default: 0.5)
                * 0 = max relevance, might get redundant chunks
                * 1 = max diversity, might miss details
                * 0.5 = balanced (good for articles)
    """
    
    # Phase 1 optimized defaults for MMR + Reranking
    if search_kwargs is None:
        search_kwargs = {
            "k": 30,           # Retrieve 30 for reranking
            "fetch_k": 50,     # Fetch 50 candidates before MMR
            "lambda_mult": 0.5 # Balanced relevance/diversity
        }
    
    print(f"\nInitializing retriever:")
    print(f"  - Search type: '{search_type}'")
    print(f"  - Parameters: {search_kwargs}")
    
    vector_store = get_vector_store()
    
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    return retriever