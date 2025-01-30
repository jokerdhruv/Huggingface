from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_cpp import Llama
import json
import atexit
import numpy as np

# Initialize Qdrant
client = QdrantClient(
    url="https://e3a7688b-e866-4993-9a7e-ee68084c0a56.us-east4-0.gcp.cloud.qdrant.io:6333",  
    api_key="InsertKEYHERE"
)

# Initialize LLaMA model
llama = Llama(
    model_path="llama-2-7b.Q2_K.gguf",
    n_ctx=2048,
    embedding=True,
    n_gpu_layers=0
)

# Get embedding dimension
test_embedding = llama.embed("test")
EMBEDDING_DIM = len(np.array(test_embedding).flatten())
print(f"Detected embedding dimension: {EMBEDDING_DIM}")

# Create collection if it doesn't exist
collection_name = "company_data"
try:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE
        )
    )
except Exception as e:
    print(f"Collection might already exist: {e}")

def cleanup():
    try:
        llama.reset()
        del llama
    except:
        pass

atexit.register(cleanup)

def normalize_embedding(embedding, target_dim=None):
    """Normalize embedding to target dimension using padding or truncation"""
    if target_dim is None:
        target_dim = EMBEDDING_DIM
    
    # Ensure embedding is a flat list of floats
    if isinstance(embedding, (list, np.ndarray)):
        embedding = np.array(embedding).flatten()
        embedding = embedding.astype(float).tolist()
    
    if len(embedding) > target_dim:
        return embedding[:target_dim]
    elif len(embedding) < target_dim:
        padding = [0.0] * (target_dim - len(embedding))
        return embedding + padding
    return embedding

def generate_embedding(text, llama_model):
    """Generate and normalize embedding for given text"""
    try:
        raw_embedding = llama_model.embed(text)
        normalized = normalize_embedding(raw_embedding)
        return normalized
    except Exception as e:
        raise ValueError(f"Error generating embedding: {str(e)}")

def hash_id(id_string):
    """Convert string ID to a positive integer hash"""
    # Using a simple hash function that maintains consistency
    hash_val = hash(id_string) & 0xffffffff  # Ensure positive 32-bit integer
    return hash_val

def load_and_index_data(file_path):
    """Load and index company data into Qdrant"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        points = []
        id_mapping = {}  # Store mapping between original IDs and hash IDs
        
        for company in data:
            try:
                # Generate normalized embeddings
                embedding = generate_embedding(company["text"], llama)
                
                # Create a consistent hash ID for the string ID
                hashed_id = hash_id(company["id"])
                id_mapping[hashed_id] = company["id"]
                
                points.append(models.PointStruct(
                    id=hashed_id,  # Use hashed ID for Qdrant
                    vector=embedding,
                    payload={
                        "original_id": company["id"],  # Store original ID in payload
                        "text": company["text"][:1000]  # Limit text size in payload
                    }
                ))
                print(f"Successfully processed company {company['id']} (hash: {hashed_id})")
            except Exception as e:
                print(f"Error processing company {company['id']}: {e}")
                continue

        # Upload in batches
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                print(f"Successfully uploaded batch {i//BATCH_SIZE + 1} of {len(points)//BATCH_SIZE + 1}")
            except Exception as e:
                print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")
                # Try uploading one by one if batch fails
                for point in batch:
                    try:
                        client.upsert(
                            collection_name=collection_name,
                            points=[point]
                        )
                        print(f"Successfully uploaded single point {point.id}")
                    except Exception as e:
                        print(f"Error uploading point {point.id}: {e}")

        return data, id_mapping
    except Exception as e:
        print(f"Error loading or indexing data: {e}")
        return None, None

def semantic_search(query_text, top_k=3):
    """Search for companies based on semantic similarity to query text"""
    try:
        # Generate normalized embedding for the query
        query_vector = generate_embedding(query_text, llama)

        # Search Qdrant
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )

        # Format and return results
        formatted_results = []
        for match in results:
            formatted_results.append({
                "score": round(float(match.score), 3),
                "id": match.payload["original_id"],  # Use original ID from payload
                "text": match.payload["text"]
            })
        return formatted_results
    except Exception as e:
        print(f"Error during semantic search: {str(e)}")
        return []

def search_by_id(company_id, data):
    """Search for a company by its ID"""
    try:
        company_text = next((company["text"] for company in data if company["id"] == company_id), None)
        if company_text is None:
            return "Company not found."

        # Generate and normalize embedding
        query_vector = generate_embedding(company_text, llama)

        # Search Qdrant
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=1
        )

        if results:
            return results[0].payload["text"]
        return "Company not found."
    except Exception as e:
        return f"Error during search: {str(e)}"

if __name__ == "__main__":
    try:
        file_path = 'C:\\Users\\DhruvSingh\\Downloads\\Work\\Huggingface\\company_data.json'
        
        print("Loading and indexing data...")
        data, id_mapping = load_and_index_data(file_path)
        
        if data:
            print("\nSearching for 'AI companies'...")
            results = semantic_search("AI companies")
            
            if not results:
                print("No results found.")
            else:
                print("\nTop matches:")
                print("=" * 80)
                
                for i, result in enumerate(results, 1):
                    print(f"\nMatch #{i}")
                    print("-" * 40)
                    print(f"Similarity Score: {result['score']}")
                    print(f"Company ID: {result['id']}")
                    print("\nCompany Description:")
                    print(result['text'])
                    print("=" * 80)
    finally:
        cleanup()