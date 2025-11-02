import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI(title="BGE Reranker Service")

# Global model variables
model = None
tokenizer = None
device = None

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5

class RerankResponse(BaseModel):
    scores: List[float]
    indices: List[int]

@app.on_event("startup")
async def load_model():
    global model, tokenizer, device
    
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-reranker-large")
    
    print(f"Loading reranker model: {model_name}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        print(f"✓ Model loaded successfully on {device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        query = request.query
        documents = request.documents
        top_k = min(request.top_k, len(documents))
        
        if not documents:
            return RerankResponse(scores=[], indices=[])
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(device)
            
            # Get scores
            scores = model(**inputs, return_dict=True).logits.view(-1).float()
            scores = scores.cpu().numpy().tolist()
        
        # Sort by score (descending)
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k
        top_indices = [idx for idx, _ in scored_docs[:top_k]]
        top_scores = [score for _, score in scored_docs[:top_k]]
        
        return RerankResponse(
            scores=top_scores,
            indices=top_indices
        )
        
    except Exception as e:
        print(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)