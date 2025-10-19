"""
Test script to debug the search endpoint locally
"""
import requests
import json

BASE_URL = "http://localhost:8000"

test_queries = [
    "What does the author affectionately call the => syntax?",
    "Which operator converts any value into an explicit boolean?",
    "What filename do you use to declare globals available across your entire TS project?"
]

def test_search(query):
    """Test a single search query"""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print('='*80)
    
    try:
        response = requests.get(f"{BASE_URL}/search", params={"q": query})
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✅ Status: {response.status_code}")
        print(f"Confidence: {result.get('confidence', 'N/A'):.4f}")
        print(f"Sources: {result.get('sources', 'N/A')}")
        print(f"\nAnswer:\n{result.get('answer', 'N/A')[:300]}...")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")

def main():
    print("\n🔍 Testing Search Endpoint")
    print(f"Target: {BASE_URL}")
    
    # Test health check first
    try:
        response = requests.get(BASE_URL)
        print(f"\n✅ Server is running: {response.json()}")
    except:
        print(f"\n❌ Server is not running at {BASE_URL}")
        print("Start the server with: python main.py")
        return
    
    # Test each query
    for query in test_queries:
        test_search(query)
    
    print(f"\n{'='*80}")
    print("Testing complete!")
    print('='*80)

if __name__ == "__main__":
    main()

