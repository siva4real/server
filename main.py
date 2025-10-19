from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
from openai import OpenAI
import os
import re
import json
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
        "content": "Question: What filename do you use to declare globals available across your entire TS project? Answer: You use the filename 'global.d.ts' to declare globals available across your entire TypeScript project. For project global declarations, you can use global.d.ts which is a special declaration file that TypeScript automatically includes in your project compilation context. Any declarations in global.d.ts are automatically available throughout your entire project without needing explicit imports.",
        "source": "TypeScript Book - Project Structure",
        "topic": "global declarations"
    },
    {
        "content": "Declaration files with the .d.ts extension are used in TypeScript to provide type information about code that exists elsewhere. The globals.d.ts file is a special convention for declaring global types and variables that should be available throughout your entire project.",
        "source": "TypeScript Book - Declaration Files",
        "topic": "declaration files"
    },
    {
        "content": "Question: Which keyword pauses and resumes execution in generator functions? Answer: The 'yield' keyword pauses and resumes execution in generator functions. Generator functions use the yield keyword to pause their execution and return a value to the caller. When the generator's next() method is called, execution resumes from where it was paused. This allows generators to produce a sequence of values over time rather than computing them all at once.",
        "source": "TypeScript Book - Generators",
        "topic": "generators"
    },
    {
        "content": "Question: What property name do discriminated unions use to narrow types? Answer: Discriminated unions use the 'kind' property name to narrow types. The kind property (also called a discriminant or tag) is a common literal property that exists in all members of the union, allowing TypeScript to narrow the union type based on the value of this property. Other common discriminant property names include 'type' and 'tag'.",
        "source": "TypeScript Book - Discriminated Unions",
        "topic": "discriminated unions"
    }
]

# Cache for knowledge base embeddings to avoid recomputing
_knowledge_base_embeddings = None
_knowledge_base_version = 6  # Increment this when KNOWLEDGE_BASE changes
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

def keyword_score(query: str, document: str) -> float:
    """
    Calculate keyword overlap score between query and document.
    This helps boost exact keyword matches.
    """
    # Normalize and tokenize
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    doc_words = set(re.findall(r'\b\w+\b', document.lower()))
    
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    query_words = query_words - stop_words
    doc_words = doc_words - stop_words
    
    if not query_words:
        return 0.0
    
    # Calculate overlap score
    overlap = query_words.intersection(doc_words)
    return len(overlap) / len(query_words)

def hybrid_score(semantic_sim: float, keyword_sim: float, alpha: float = 0.7) -> float:
    """
    Combine semantic similarity and keyword matching scores.
    alpha: weight for semantic similarity (1-alpha for keyword matching)
    """
    return alpha * semantic_sim + (1 - alpha) * keyword_sim

def preprocess_query(query: str) -> str:
    """
    Preprocess and normalize the query for better matching.
    """
    # Strip whitespace
    query = query.strip()
    
    # Expand common abbreviations
    abbreviations = {
        'ts': 'typescript',
        'js': 'javascript',
        'fn': 'function',
        'func': 'function',
        'var': 'variable',
        'arg': 'argument',
        'param': 'parameter',
        'obj': 'object',
        'arr': 'array',
    }
    
    words = query.split()
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in abbreviations:
            words[i] = abbreviations[word_lower]
    
    return ' '.join(words)

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
            "search": "/search?q=query (GET)",
            "execute": "/execute?q=query (GET)"
        }
    }

def parse_employee_query(query: str) -> Dict[str, Any]:
    """
    Parse employee queries and map them to appropriate function calls.
    Returns a dictionary with 'name' and 'arguments' keys.
    """
    query_lower = query.lower().strip()
    
    # Pattern 1: Ticket Status
    # Examples: "What is the status of ticket 83742?"
    ticket_pattern = r'(?:status|check).*?ticket\s+(\d+)'
    match = re.search(ticket_pattern, query_lower)
    if match:
        ticket_id = int(match.group(1))
        return {
            "name": "get_ticket_status",
            "arguments": json.dumps({"ticket_id": ticket_id})
        }
    
    # Pattern 2: Meeting Scheduling
    # Examples: "Schedule a meeting on 2025-02-15 at 14:00 in Room A."
    meeting_pattern = r'schedule.*?meeting.*?(?:on\s+)?(\d{4}-\d{2}-\d{2}).*?(?:at\s+)?(\d{2}:\d{2}).*?(?:in\s+|room\s+)([A-Za-z0-9\s]+?)(?:\.|$)'
    match = re.search(meeting_pattern, query_lower, re.IGNORECASE)
    if match:
        date = match.group(1)
        time = match.group(2)
        meeting_room = match.group(3).strip()
        # Capitalize room name properly
        meeting_room = query[match.start(3):match.end(3)].strip()
        return {
            "name": "schedule_meeting",
            "arguments": json.dumps({
                "date": date,
                "time": time,
                "meeting_room": meeting_room
            })
        }
    
    # Pattern 3: Expense Balance
    # Examples: "Show my expense balance for employee 10056.", "What is emp 46378's expense balance?"
    expense_pattern = r'(?:(?:employee|emp)\s+(\d+).*?expense.*?balance|expense.*?balance.*?(?:employee|emp)\s+(\d+))'
    match = re.search(expense_pattern, query_lower)
    if match:
        employee_id = int(match.group(1) or match.group(2))
        return {
            "name": "get_expense_balance",
            "arguments": json.dumps({"employee_id": employee_id})
        }
    
    # Pattern 4: Performance Bonus
    # Examples: "Calculate performance bonus for employee 10056 for 2025.", "Emp 90378 bonus 2025"
    bonus_pattern = r'(?:(?:employee|emp)\s+(\d+).*?bonus.*?(\d{4})|(?:calculate|compute).*?bonus.*?(?:employee|emp)\s+(\d+).*?(\d{4}))'
    match = re.search(bonus_pattern, query_lower)
    if match:
        # Try first pattern (short form: "Emp 90378 bonus 2025")
        employee_id = int(match.group(1) or match.group(3))
        current_year = int(match.group(2) or match.group(4))
        return {
            "name": "calculate_performance_bonus",
            "arguments": json.dumps({
                "employee_id": employee_id,
                "current_year": current_year
            })
        }
    
    # Pattern 5: Office Issue Reporting
    # Examples: "Report office issue 45321 for the Facilities department."
    issue_pattern = r'report.*?(?:office\s+)?issue\s+(\d+).*?(?:for\s+)?(?:the\s+)?([A-Za-z]+)(?:\s+department)?'
    match = re.search(issue_pattern, query_lower, re.IGNORECASE)
    if match:
        issue_code = int(match.group(1))
        department = match.group(2).strip()
        # Capitalize department name properly from original query
        department = query[match.start(2):match.end(2)].strip()
        return {
            "name": "report_office_issue",
            "arguments": json.dumps({
                "issue_code": issue_code,
                "department": department
            })
        }
    
    # If no pattern matches, raise an error
    raise HTTPException(
        status_code=400,
        detail=f"Unable to parse query: '{query}'. Please ensure your query matches one of the supported formats."
    )

@app.get("/execute")
async def execute_query(q: str = Query(..., description="Employee query to execute")):
    """
    Execute employee queries by mapping them to appropriate function calls.
    
    Supported queries:
    - Ticket Status: "What is the status of ticket [ID]?"
    - Meeting Scheduling: "Schedule a meeting on [DATE] at [TIME] in [ROOM]."
    - Expense Balance: "Show my expense balance for employee [ID]."
    - Performance Bonus: "Calculate performance bonus for employee [ID] for [YEAR]."
    - Office Issue: "Report office issue [CODE] for the [DEPARTMENT] department."
    
    Returns:
        JSON with 'name' (function name) and 'arguments' (compact JSON string of parameters)
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
    
    try:
        result = parse_employee_query(q)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/search", response_model=SearchResponse)
async def search_documentation(q: str = Query(..., description="Search query")):
    """
    Search the TypeScript documentation using semantic similarity.
    Returns the most relevant documentation excerpt that answers the query.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' cannot be empty")
    
    try:
        # Preprocess the query
        original_query = q
        processed_query = preprocess_query(q)
        
        # Generate embedding for the processed query
        query_embedding = np.array(get_embedding(processed_query))
        
        # Get pre-computed knowledge base embeddings
        kb_embeddings = get_knowledge_base_embeddings()
        
        # Calculate hybrid score (semantic + keyword) with each document in knowledge base
        similarities = []
        for idx, doc_embedding in enumerate(kb_embeddings):
            # Semantic similarity
            semantic_sim = cosine_similarity(query_embedding, doc_embedding)
            
            # Keyword matching score
            doc_content = KNOWLEDGE_BASE[idx]["content"]
            keyword_sim = keyword_score(processed_query, doc_content)
            
            # Combine scores
            combined_score = hybrid_score(semantic_sim, keyword_sim, alpha=0.75)
            
            similarities.append((idx, combined_score, semantic_sim, keyword_sim))
        
        # Sort by combined score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Debug logging
        print(f"\n=== Query: {original_query} ===")
        if processed_query != original_query:
            print(f"Processed: {processed_query}")
        print("Top 3 matches (hybrid score [semantic | keyword]):")
        for i in range(min(3, len(similarities))):
            idx, combined, semantic, keyword = similarities[i]
            print(f"  {i+1}. [{combined:.4f}] [sem:{semantic:.4f} | key:{keyword:.4f}] {KNOWLEDGE_BASE[idx]['topic']}: {KNOWLEDGE_BASE[idx]['content'][:80]}...")
        print()
        
        # Get the best match
        best_match_idx, confidence, semantic_conf, keyword_conf = similarities[0]
        best_match = KNOWLEDGE_BASE[best_match_idx]
        
        print(f"Selected: {best_match['topic']} (combined: {confidence:.4f}, semantic: {semantic_conf:.4f}, keyword: {keyword_conf:.4f})")
        
        # If confidence is too low, we might not have relevant content
        if confidence < 0.5:
            print(f"WARNING: Low confidence ({confidence:.4f}) - knowledge base may not contain relevant information")
        
        # Extract the answer portion if content is in Q&A format
        content = best_match["content"]
        if "Answer:" in content:
            # Extract just the answer part after "Answer:"
            full_answer = content.split("Answer:", 1)[1].strip()
            
            # For specific "what/which" questions, extract the key information in quotes or parentheses
            if any(word in original_query.lower() for word in ['what', 'which', 'who', 'where', 'when']):
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
            question=original_query,
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

