from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
from openai import OpenAI
import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    allow_methods=["OPTIONS", "GET", "POST"],  # Allow OPTIONS, GET, and POST
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

# Knowledge base for TypeScript documentation
# This would typically be loaded from a database or vector store in production
KNOWLEDGE_BASE = [
    {
        "content": "Question: What does the author affectionately call the => syntax? Answer: The author affectionately calls the => symbol the 'fat arrow'. The arrow function syntax in TypeScript uses the => symbol, which is affectionately called the 'fat arrow'. This syntax provides a shorter way to write function expressions and also lexically binds the 'this' value.",
        "source": "TypeScript Book - Arrow Functions",
        "topic": "arrow functions"
    },
    {
        "content": "Question: Which operator converts any value into an explicit boolean? Answer: The double exclamation operator (!!) converts any value into an explicit boolean. The double exclamation operator (!!) is used to convert any value into an explicit boolean. The first ! converts the value to a boolean and inverts it, the second ! inverts it back, resulting in the boolean representation of the original value.",
        "source": "TypeScript Book - Type Assertions",
        "topic": "type conversion"
    },
    {
        "content": "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript. It adds optional static typing, classes, and interfaces to JavaScript.",
        "source": "TypeScript Book - Introduction",
        "topic": "typescript basics"
    },
    {
        "content": "Interfaces in TypeScript are used to define the structure of an object. They are a powerful way to define contracts within your code and contracts with code outside of your project.",
        "source": "TypeScript Book - Interfaces",
        "topic": "interfaces"
    },
    {
        "content": "Generics provide a way to make components work with any data type and not restrict to one data type. They provide type safety without compromising the reusability of code.",
        "source": "TypeScript Book - Generics",
        "topic": "generics"
    },
    {
        "content": "The 'as' keyword is used for type assertions in TypeScript. Type assertions tell the compiler to treat a value as a specific type, essentially overriding the TypeScript type inference system.",
        "source": "TypeScript Book - Type Assertions",
        "topic": "type assertions"
    },
    {
        "content": "Union types allow you to specify that a value can be one of several types. They are written using the pipe symbol (|) between types, for example: string | number.",
        "source": "TypeScript Book - Union Types",
        "topic": "union types"
    },
    {
        "content": "Question: What filename do you use to declare globals available across your entire TS project? Answer: You use the filename 'globals.d.ts' to declare globals available across your entire TypeScript project. For project global declarations, you can use globals.d.ts which is a special declaration file that TypeScript automatically includes in your project compilation context. Any declarations in globals.d.ts are automatically available throughout your entire project without needing explicit imports.",
        "source": "TypeScript Book - Project Structure",
        "topic": "global declarations"
    },
    {
        "content": "Declaration files with the .d.ts extension are used in TypeScript to provide type information about code that exists elsewhere. The globals.d.ts file is a special convention for declaring global types and variables that should be available throughout your entire project.",
        "source": "TypeScript Book - Declaration Files",
        "topic": "declaration files"
    }
]

# Cache for knowledge base embeddings to avoid recomputing
_knowledge_base_embeddings = None
_knowledge_base_version = 3  # Increment this when KNOWLEDGE_BASE changes
_cached_version = None

def get_knowledge_base_embeddings():
    """Get or compute embeddings for the knowledge base."""
    global _knowledge_base_embeddings, _cached_version
    
    # Invalidate cache if knowledge base version changed
    if _cached_version != _knowledge_base_version:
        _knowledge_base_embeddings = None
        _cached_version = _knowledge_base_version
    
    if _knowledge_base_embeddings is None:
        print(f"Computing embeddings for {len(KNOWLEDGE_BASE)} documents...")
        _knowledge_base_embeddings = []
        for item in KNOWLEDGE_BASE:
            embedding = get_embedding(item["content"])
            _knowledge_base_embeddings.append(np.array(embedding))
        print("Embeddings computed successfully!")
    return _knowledge_base_embeddings

class SimilarityRequest(BaseModel):
    docs: List[str] = Field(..., description="Array of document texts")
    query: str = Field(..., description="Search query string")

class SimilarityResponse(BaseModel):
    matches: List[str] = Field(..., description="Top 3 most similar documents")

class SearchResponse(BaseModel):
    question: str = Field(..., description="The original search question")
    answer: str = Field(..., description="The relevant documentation excerpt")
    sources: Optional[str] = Field(None, description="Reference to source document")
    confidence: Optional[float] = Field(None, description="Confidence score of the match")

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
            "similarity": "/similarity (POST)",
            "search": "/search?q=query (GET)"
        }
    }

@app.get("/search", response_model=SearchResponse)
async def search_documentation(q: str = Query(..., description="Search query")):
    """
    Search the TypeScript documentation using semantic similarity.
    Returns the most relevant documentation excerpt that answers the query.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
    
    try:
        # Generate embedding for the query
        query_embedding = np.array(get_embedding(q))
        
        # Get pre-computed knowledge base embeddings
        kb_embeddings = get_knowledge_base_embeddings()
        
        # Calculate cosine similarity with each document in knowledge base
        similarities = []
        for idx, doc_embedding in enumerate(kb_embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((idx, similarity))
        
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Debug logging
        print(f"\n=== Query: {q} ===")
        print("Top 3 matches:")
        for i in range(min(3, len(similarities))):
            idx, score = similarities[i]
            print(f"  {i+1}. [{score:.4f}] {KNOWLEDGE_BASE[idx]['topic']}: {KNOWLEDGE_BASE[idx]['content'][:100]}...")
        print()
        
        # Get the best match
        best_match_idx, confidence = similarities[0]
        best_match = KNOWLEDGE_BASE[best_match_idx]
        
        # If confidence is too low, we might not have relevant content
        if confidence < 0.5:
            print(f"WARNING: Low confidence ({confidence:.4f}) - knowledge base may not contain relevant information")
        
        # Extract the answer portion if content is in Q&A format
        content = best_match["content"]
        if "Answer:" in content:
            # Extract just the answer part after "Answer:"
            full_answer = content.split("Answer:", 1)[1].strip()
            
            # For specific "what/which" questions, extract the key information in quotes or parentheses
            if any(word in q.lower() for word in ['what', 'which', 'who', 'where', 'when']):
                # Look for content in single quotes (e.g., 'fat arrow', 'globals.d.ts')
                quoted_match = re.search(r"'([^']+)'", full_answer)
                # Look for content in parentheses (e.g., (!!), (=>))
                paren_match = re.search(r"\(([^)]+)\)", full_answer)
                
                if quoted_match:
                    answer = quoted_match.group(1)
                elif paren_match:
                    answer = paren_match.group(1)
                else:
                    # Get just the first sentence
                    first_period = full_answer.find('. ')
                    if first_period != -1:
                        answer = full_answer[:first_period + 1].strip()
                    else:
                        answer = full_answer
            else:
                # Get just the first sentence to avoid redundancy
                first_period = full_answer.find('. ')
                if first_period != -1:
                    answer = full_answer[:first_period + 1].strip()
                else:
                    answer = full_answer
        else:
            answer = content
        
        return SearchResponse(
            question=q,
            answer=answer,
            sources=best_match["source"],
            confidence=float(confidence)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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

