import os
from typing import List
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class BGEReranker:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-large')
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load the reranker model"""
        print(f"Loading reranker model: {self.model_name}")
        
        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Reranker model loaded successfully on {self.device}")
        except Exception as e:
            print(f"✗ Error loading reranker model: {e}")
            raise
    
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
            
            # Create query-document pairs
            pairs = [[query, doc] for doc in doc_texts]
            
            # Tokenize and get scores
            with torch.no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                ).to(self.device)
                
                # Get scores
                scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores = scores.cpu().numpy().tolist()
            
            # Sort by score (descending)
            scored_docs = [(i, score) for i, score in enumerate(scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k
            reranked_docs = []
            for idx, score in scored_docs[:top_k]:
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


# Global reranker instance (lazy loaded)
_reranker_instance = None

def get_reranker(model_name: str = None) -> BGEReranker:
    """Factory function to get reranker instance (singleton pattern)"""
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = BGEReranker(model_name=model_name)
    
    return _reranker_instance