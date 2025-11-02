import os
import requests
from typing import List, Tuple
from langchain_core.documents import Document

class BGEReranker:
    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.getenv('RERANKER_API_URL', 'http://localhost:8002')
        self._check_health()
    
    def _check_health(self):
        """Check if reranker service is healthy"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"✓ Reranker service healthy - Device: {health.get('device')}")
                return True
        except Exception as e:
            print(f"⚠ Warning: Reranker service not available: {e}")
            return False
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Document]:
        """
        Rerank documents using BGE reranker
        
        Args:
            query: The search query
            documents: List of LangChain Document objects
            top_k: Number of top documents to return
            include_metadata: Whether to include metadata in reranking
        
        Returns:
            List of reranked Document objects
        """
        if not documents:
            return []
        
        try:
            # Prepare documents for reranking
            doc_texts = []
            for doc in documents:
                if include_metadata:
                    # Enrich with metadata for better reranking
                    text = self._format_document_with_metadata(doc)
                else:
                    text = doc.page_content
                doc_texts.append(text)
            
            # Call reranker API
            response = requests.post(
                f"{self.api_url}/rerank",
                json={
                    "query": query,
                    "documents": doc_texts,
                    "top_k": top_k
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Reorder documents based on reranker scores
            reranked_docs = []
            for idx, score in zip(result['indices'], result['scores']):
                doc = documents[idx]
                # Add reranker score to metadata
                doc.metadata['rerank_score'] = score
                reranked_docs.append(doc)
            
            print(f"✓ Reranked {len(documents)} → {len(reranked_docs)} documents")
            self._log_scores(reranked_docs)
            
            return reranked_docs
            
        except Exception as e:
            print(f"✗ Reranking failed: {e}")
            print(f"  Falling back to original order (top {top_k})")
            return documents[:top_k]
    
    def _format_document_with_metadata(self, doc: Document) -> str:
        """Format document with metadata for reranking"""
        metadata = doc.metadata
        parts = []
        
        # Add available metadata fields
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        if metadata.get('author'):
            parts.append(f"Author: {metadata['author']}")
        
        if metadata.get('published_date'):
            parts.append(f"Published: {metadata['published_date']}")
        
        if metadata.get('excerpt'):
            parts.append(f"Excerpt: {metadata['excerpt']}")
        
        if metadata.get('categories'):
            cats = metadata['categories']
            if isinstance(cats, list):
                cats = ', '.join(cats)
            parts.append(f"Categories: {cats}")
        
        if metadata.get('tags'):
            tags = metadata['tags']
            if isinstance(tags, list):
                tags = ', '.join(tags)
            parts.append(f"Tags: {tags}")
        
        # Add content
        parts.append(f"\nContent:\n{doc.page_content}")
        
        return '\n'.join(parts)
    
    def _log_scores(self, documents: List[Document]):
        """Log reranking scores for debugging"""
        print("\nReranking Scores:")
        for i, doc in enumerate(documents, 1):
            score = doc.metadata.get('rerank_score', 'N/A')
            title = doc.metadata.get('title', 'Untitled')
            # Truncate title if too long
            title = title[:50] + '...' if len(title) > 50 else title
            print(f"  {i}. Score: {score:.4f} | {title}")


def get_reranker(api_url: str = None) -> BGEReranker:
    """Factory function to get reranker instance"""
    return BGEReranker(api_url=api_url)