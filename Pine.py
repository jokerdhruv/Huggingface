from pinecone import Pinecone
from pinecone import ServerlessSpec
import json
from llama_cpp import Llama
import atexit
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key="your api key")

# Initialize LLaMA model first to get embedding dimension
llama = Llama(
    model_path="llama-2-7b.Q2_K.gguf",
    n_ctx=2048,
    embedding=True,
    n_gpu_layers=0
)
def process_embedding(embedding):
    """Process embedding to ensure it's a flat list of floats"""
    if isinstance(embedding, (list, np.ndarray)):
        if isinstance(embedding[0], (list, np.ndarray)):
            embedding = [item for sublist in embedding for item in sublist]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        embedding = [float(val) for val in embedding]
    return embedding

# Get the actual embedding dimension by running a test embedding
test_embedding = llama.embed("test")
EMBEDDING_DIM = len(process_embedding(test_embedding))
print(f"Detected embedding dimension: {EMBEDDING_DIM}")

# Create or connect to the Pinecone index with correct dimension
try:
    pc.create_index(
        name="company-data-index2",
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
except Exception as e:
    print(f"Index might already exist: {e}")

# Get the index reference
index = pc.Index("company-data-index2")

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
        
    processed = process_embedding(embedding)
    
    if len(processed) > target_dim:
        # Truncate to target dimension
        return processed[:target_dim]
    elif len(processed) < target_dim:
        # Pad with zeros to target dimension
        padding = [0.0] * (target_dim - len(processed))
        return processed + padding
    return processed

def process_embedding(embedding):
    """Process embedding to ensure it's a flat list of floats"""
    if isinstance(embedding, (list, np.ndarray)):
        # Flatten nested arrays
        flat_embedding = np.array(embedding).flatten()
        # Convert to float list
        return flat_embedding.astype(float).tolist()
    return embedding

def generate_embedding(text, llama_model):
    """Generate and normalize embedding for given text"""
    try:
        raw_embedding = llama_model.embed(text)
        normalized = normalize_embedding(raw_embedding)
        return normalized
    except Exception as e:
        raise ValueError(f"Error generating embedding: {str(e)}")

def verify_embedding_dimension(embedding):
    """Verify that embedding has the correct dimension"""
    if len(embedding) != EMBEDDING_DIM:
        # Instead of raising an error, normalize the embedding
        return normalize_embedding(embedding)
    return embedding

def load_and_index_data(file_path, namespace="company-namespace"):
    """Load and index company data into a specified namespace"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        vectors = []
        for company in data:
            try:
                # Generate normalized embeddings
                embedding = generate_embedding(company["text"], llama)
                
                vectors.append({
                    'id': str(company["id"]),
                    'values': embedding,
                    'metadata': {
                        "id": str(company["id"]),
                        "text": company["text"][:1000]  # Limit text size in metadata
                    }
                })
                print(f"Successfully processed company {company['id']}")
            except Exception as e:
                print(f"Error processing company {company['id']}: {e}")
                continue

        # Upload in small batches
        BATCH_SIZE = 10
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i + BATCH_SIZE]
            try:
                index.upsert(vectors=batch, namespace=namespace)
                print(f"Successfully uploaded batch {i//BATCH_SIZE + 1} of {len(vectors)//BATCH_SIZE + 1}")
            except Exception as e:
                print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")
                # Try uploading one by one if batch fails
                for vector in batch:
                    try:
                        index.upsert(vectors=[vector], namespace=namespace)
                        print(f"Successfully uploaded single vector {vector['id']}")
                    except Exception as e:
                        print(f"Error uploading vector {vector['id']}: {e}")

        return data
    except Exception as e:
        print(f"Error loading or indexing data: {e}")
        return None

def semantic_search(query_text, top_k=3, namespace="company-namespace"):
    """Search for companies based on semantic similarity to query text"""
    try:
        # Generate normalized embedding for the query
        query_vector = generate_embedding(query_text, llama)

        # Query Pinecone with namespace
        results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        # Format and return results
        formatted_results = []
        for match in results["matches"]:
            formatted_results.append({
                "score": round(float(match["score"]), 3),
                "id": match["metadata"]["id"],
                "text": match["metadata"]["text"]
            })
        return formatted_results
    except Exception as e:
        print(f"Error during semantic search: {str(e)}")
        return []

def search_by_id(company_id, data, namespace="company-namespace"):
    """Search for a company by its ID"""
    try:
        company_text = next((company["text"] for company in data if company["id"] == company_id), None)
        if company_text is None:
            return "Company not found."

        # Generate and normalize embedding
        query_vector = generate_embedding(company_text, llama)

        # Query Pinecone
        query_result = index.query(
            vector=query_vector,
            top_k=1,
            include_metadata=True,
            namespace=namespace
        )

        if query_result["matches"]:
            return query_result["matches"][0]["metadata"]["text"]
        return "Company not found."
    except Exception as e:
        return f"Error during search: {str(e)}"

# Example usage
# if __name__ == "__main__":
#     try:
#         file_path = 'C:\\Users\\DhruvSingh\\Downloads\\Work\\Huggingface\\company_data.json'
#         namespace = "company-namespace"
        
#         print("Loading and indexing data...")
#         data = load_and_index_data(file_path, namespace=namespace)
        
#         if data:
#             # Try some searches
#             print("\nSearching for 'AI companies'...")
#             results = semantic_search("AI companies", namespace=namespace)
#             for i, result in enumerate(results, 1):
#                 print(f"\n{i}. Score: {result['score']}")
#                 print(f"   ID: {result['id']}")
#                 print(f"   Text: {result['text'][:200]}...")
#     finally:
#         cleanup()
if __name__ == "__main__":
    try:
        file_path = 'C:\\Users\\DhruvSingh\\Downloads\\Work\\Huggingface\\company_data.json'
        namespace = "company-namespace"
        
        print("Loading and indexing data...")
        data = load_and_index_data(file_path, namespace=namespace)
        
        if data:
            print("\nSearching for 'AI companies'...")
            results = semantic_search("AI companies", namespace=namespace)
            
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
                    print(result['text'])  # Print full text
                    print("=" * 80)
    finally:
        cleanup()