import os
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from embed import get_retriever
from reranker import get_reranker

def get_llm():
    ollama_host = os.getenv('OLLAMA_HOST', 'http://axora-ollama:11434')
    ollama_model = os.getenv('OLLAMA_MODEL', 'phi3:mini')
    
    print(f"Initializing LLM:")
    print(f"  - Model: {ollama_model}")
    print(f"  - Host: {ollama_host}\n")
    
    llm = OllamaLLM(
        base_url=ollama_host,
        model=ollama_model,
        temperature=0.7,
    )
    
    return llm


def format_docs_with_metadata(documents):
    """Format documents with metadata for LLM context"""
    formatted_parts = []
    
    for i, doc in enumerate(documents, 1):
        parts = [f"[Source {i}]"]
        
        # Add metadata if available
        metadata = doc.metadata
        
        if metadata.get('title'):
            parts.append(f"Title: {metadata['title']}")
        
        if metadata.get('author'):
            parts.append(f"Author: {metadata['author']}")
        
        if metadata.get('published_date'):
            parts.append(f"Published: {metadata['published_date']}")
        
        if metadata.get('rerank_score'):
            parts.append(f"Relevance Score: {metadata['rerank_score']:.4f}")
        
        # Add content
        parts.append(f"\nContent:\n{doc.page_content}")
        
        formatted_parts.append('\n'.join(parts))
    
    return '\n\n' + ('\n' + '='*80 + '\n\n').join(formatted_parts)


def retrieve_and_rerank(query):
    """
    Phase 1 Retrieval Pipeline:
    1. MMR retrieval (k=30, fetch_k=50, lambda=0.5)
    2. Rerank with BGE-reranker-large
    3. Return top 5
    """
    print("\n" + "="*80)
    print("PHASE 1 RETRIEVAL PIPELINE")
    print("="*80)
    
    # Step 1: MMR Retrieval
    print("\n[Step 1/2] MMR Retrieval...")
    retriever = get_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 30,           # Retrieve 30 candidates
            "fetch_k": 50,     # From 50 initial docs
            "lambda_mult": 0.5 # Balanced diversity
        }
    )
    
    documents = retriever.invoke(query)
    print(f"✓ Retrieved {len(documents)} documents via MMR")
    
    # Step 2: Rerank
    print("\n[Step 2/2] Reranking with BGE-reranker-large...")
    reranker = get_reranker()
    reranked_docs = reranker.rerank_documents(
        query=query,
        documents=documents,
        top_k=5,  # Final top-5 after reranking
        include_metadata=True  # Use metadata for better reranking
    )
    
    print(f"✓ Final context: {len(reranked_docs)} documents")
    print("="*80 + "\n")
    
    return reranked_docs


def get_rag_chain():    
    llm = get_llm()

    prompt = PromptTemplate.from_template(
        """You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{input}

Answer clearly and concisely based only on the given context. If the context doesn't contain relevant information, say so. When citing information, you can reference sources by their numbers (e.g., "According to Source 1...").
"""
    )
    

    def log_prompt(inputs):
        print("\n" + "=" * 80)
        print("PROMPT SENT TO LLM")
        print("=" * 80)
        # inputs here has 'context' and 'input' keys already populated
        rendered = prompt.format(**inputs)
        print(rendered)
        print("=" * 80 + "\n")
        return rendered

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_and_rerank),
            "input": RunnablePassthrough(),
        }
        | RunnableLambda(log_prompt)
        | llm
        | StrOutputParser()
    )

    return rag_chain