# InfoCore Semantic Search API

A FastAPI-based semantic search service that uses OpenAI's text-embedding-3-small model to find the most relevant documents based on cosine similarity.

## Features

- **Semantic Search**: Uses text embeddings to understand contextual meaning
- **Cosine Similarity**: Ranks documents by similarity to the query
- **CORS Enabled**: Allows cross-origin requests from any domain
- **Production Ready**: Designed for deployment on Render or similar platforms

## API Endpoints

### POST /similarity

Calculate semantic similarity between a query and multiple documents.

**Request Body:**
```json
{
  "docs": [
    "Contents of document 1",
    "Contents of document 2",
    "Contents of document 3"
  ],
  "query": "Your search query"
}
```

**Response:**
```json
{
  "matches": [
    "Contents of document 3",
    "Contents of document 1",
    "Contents of document 2"
  ]
}
```

The response returns up to 3 documents ranked by similarity (highest first).

### GET /

Health check endpoint.

## Local Development

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd server
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

6. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Deployment to Render

### Steps:

1. **Create a new Web Service** on [Render](https://render.com)

2. **Connect your GitHub repository**

3. **Configure the service:**
   - **Name**: `infocore-semantic-search` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add environment variables:**
   - `OPENAI_API_KEY`: Your OpenAI API key

5. **Deploy** the service

Your API will be available at: `https://your-service-name.onrender.com`

## Testing the API

### Using cURL:

```bash
curl -X POST https://your-service-name.onrender.com/similarity \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      "Machine learning is a subset of artificial intelligence",
      "FastAPI is a modern web framework for building APIs",
      "Neural networks are inspired by biological neurons"
    ],
    "query": "What is AI?"
  }'
```

### Using Python:

```python
import requests

url = "https://your-service-name.onrender.com/similarity"

payload = {
    "docs": [
        "Machine learning is a subset of artificial intelligence",
        "FastAPI is a modern web framework for building APIs",
        "Neural networks are inspired by biological neurons"
    ],
    "query": "What is AI?"
}

response = requests.post(url, json=payload)
print(response.json())
```

### Using JavaScript (fetch):

```javascript
fetch('https://your-service-name.onrender.com/similarity', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    docs: [
      "Machine learning is a subset of artificial intelligence",
      "FastAPI is a modern web framework for building APIs",
      "Neural networks are inspired by biological neurons"
    ],
    query: "What is AI?"
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## API Documentation

Once the server is running, you can access:
- **Interactive API docs (Swagger UI)**: `http://localhost:8000/docs`
- **Alternative API docs (ReDoc)**: `http://localhost:8000/redoc`

## Technical Details

- **Embedding Model**: text-embedding-3-small (OpenAI)
- **Similarity Metric**: Cosine Similarity
- **Framework**: FastAPI
- **CORS**: Enabled for all origins with OPTIONS and POST methods

## Error Handling

The API includes comprehensive error handling:
- Empty docs array: 400 Bad Request
- Empty query string: 400 Bad Request
- OpenAI API errors: 500 Internal Server Error with detailed message
- Invalid JSON: 422 Unprocessable Entity (handled by FastAPI)

## License

MIT

