from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from openai import OpenAI
import os

app = FastAPI(
    title="InfoCore Semantic Search API",
    description="Semantic search API using text embeddings",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST"],  # Allow OPTIONS and POST
    allow_headers=["*"],  # Allow all headers
)

# OpenAI client will be initialized lazily
_client = None

def get_openai_client():
    """Get or create OpenAI client instance."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _client = OpenAI(api_key=api_key)
    return _client

class SimilarityRequest(BaseModel):
    docs: List[str] = Field(..., description="Array of document texts")
    query: str = Field(..., description="Search query string")

class SimilarityResponse(BaseModel):
    matches: List[str] = Field(..., description="Top 3 most similar documents")

def get_embedding(text: str) -> List[float]:
    """
    Generate embedding for a given text using OpenAI's text-embedding-3-small model.
    """
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

@app.get("/")
async def root():
    """
    Root endpoint - health check.
    """
    return {
        "message": "InfoCore Semantic Search API is running",
        "status": "healthy",
        "endpoints": {
            "similarity": "/similarity (POST)"
        }
    }

@app.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    Calculate semantic similarity between query and documents.
    Returns the top 3 most similar documents.
    """
    if not request.docs:
        raise HTTPException(status_code=400, detail="The 'docs' array cannot be empty")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="The 'query' cannot be empty")
    
    try:
        # Generate embedding for the query
        query_embedding = np.array(get_embedding(request.query))
        
        # Generate embeddings for all documents
        doc_embeddings = []
        for doc in request.docs:
            if not doc.strip():
                # Skip empty documents but keep track of position
                doc_embeddings.append(None)
            else:
                embedding = np.array(get_embedding(doc))
                doc_embeddings.append(embedding)
        
        # Calculate cosine similarity for each document
        similarities = []
        for idx, doc_embedding in enumerate(doc_embeddings):
            if doc_embedding is not None:
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append((idx, similarity, request.docs[idx]))
            else:
                # Assign very low similarity to empty documents
                similarities.append((idx, -1.0, request.docs[idx]))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 matches
        top_matches = similarities[:3]
        
        # Extract the document contents
        matches = [doc_content for _, _, doc_content in top_matches]
        
        return SimilarityResponse(matches=matches)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

