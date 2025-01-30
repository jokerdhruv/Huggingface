# # from pinecone import Pinecone
# # from pinecone import ServerlessSpec
# # import json
# # from llama_cpp import Llama
# # import atexit
# # import numpy as np

# # # Initialize Pinecone
# pc = Pinecone("YOUR API KEY")

# # # Create or connect to the Pinecone index
# # try:
# #     pc.create_index(
# #         name="company-data-index2",
# #         dimension=758,
# #         metric="cosine",
# #         spec=ServerlessSpec(
# #             cloud="aws",
# #             region="us-east-1"
# #         )
# #     )
# # except Exception as e:
# #     print(f"Index might already exist: {e}")

# # # Get the index reference
# # index = pc.Index("company-data-index2")

# # # Initialize LLaMA model
# # llama = Llama(
# #     model_path="llama-2-7b.Q2_K.gguf",
# #     n_ctx=2048,
# #     embedding=True,
# #     n_gpu_layers=0
# # )

# # def cleanup():
# #     try:
# #         llama.reset()
# #         del llama
# #     except:
# #         pass

# # atexit.register(cleanup)




# # def load_and_index_data(file_path, namespace="company-namespace"):
# #     """Load and index company data into a specified namespace"""
# #     try:
# #         with open(file_path, 'r') as file:
# #             data = json.load(file)

# #         vectors = []
# #         for company in data:
# #             try:
# #                 # Generate embeddings
# #                 embedding = llama.embed(company["text"])
                
# #                 # Process the embedding to ensure correct format
# #                 processed_embedding = process_embedding(embedding)
                
# #                 # Use company ID as name if actual name is not available
# #                 company_name = str(company["id"])
                
# #                 vectors.append({
# #                     'id': str(company["id"]),
# #                     'values': processed_embedding,
# #                     'metadata': {
# #                         "id": str(company["id"]),
# #                         "text": company["text"]
# #                     }
# #                 })
# #                 print(f"Successfully processed company {company_name}")
# #             except Exception as e:
# #                 print(f"Error processing company {company['id']}: {e}")
# #                 continue

# #         # Upload in batches to a specific namespace
# #         BATCH_SIZE = 100
# #         for i in range(0, len(vectors), BATCH_SIZE):
# #             batch = vectors[i:i + BATCH_SIZE]
# #             try:
# #                 index.upsert(vectors=batch, namespace=namespace)
# #                 print(f"Uploaded batch {i//BATCH_SIZE + 1} of {len(vectors)//BATCH_SIZE + 1}")
# #             except Exception as e:
# #                 print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")

# #         return data
# #     except Exception as e:
# #         print(f"Error loading or indexing data: {e}")
# #         return None

# # def semantic_search(query_text, top_k=3, namespace="company-namespace"):
# #     """Search for companies based on semantic similarity to query text"""
# #     try:
# #         # Generate embedding for the query
# #         query_vector = llama.embed(query_text)
# #         processed_vector = process_embedding(query_vector)

# #         # Query Pinecone with namespace
# #         results = index.query(
# #             vector=processed_vector,
# #             top_k=top_k,
# #             include_metadata=True,
# #             namespace=namespace
# #         )

# #         # Format and return results
# #         formatted_results = []
# #         for match in results["matches"]:
# #             formatted_results.append({
# #                 "score": round(float(match["score"]), 3),
# #                 "id": match["metadata"]["id"],
# #                 "text": match["metadata"]["text"]
# #             })
# #         return formatted_results
# #     except Exception as e:
# #         print(f"Error during semantic search: {str(e)}")
# #         return []

# # def search_by_id(company_id, data):
# #     """Search for a company by its ID"""
# #     try:
# #         company_text = next((company["text"] for company in data if company["id"] == company_id), None)
# #         if company_text is None:
# #             return "Company not found."

# #         # Generate and process embedding
# #         query_vector = llama.embed(company_text)
# #         processed_vector = process_embedding(query_vector)

# #         # Query Pinecone
# #         query_result = index.query(
# #             vector=processed_vector,
# #             top_k=1,
# #             include_metadata=True
# #         )

# #         if query_result["matches"]:
# #             return query_result["matches"][0]["metadata"]["text"]
# #         return "Company not found."
# #     except Exception as e:
# #         return f"Error during search: {str(e)}"

# # # Example usage
# # if __name__ == "__main__":
# #     try:
# #         file_path = 'C:\\Users\\DhruvSingh\\Downloads\\Work\\Huggingface\\company_data.json'
# #         print("Loading and indexing data...")
# #         data = load_and_index_data(file_path, namespace="company-namespace")
        
# #         if data:
# #             # Example ID-based search
# #             print("\nID-based search for vec3:")
# #             result = search_by_id('vec3', data)
# #             print(result)

# #             # Example semantic searches
# #             print("\nSemantic search for 'social media companies':")
# #             results = semantic_search("social media companies", namespace="company-namespace")
# #             if results:
# #                 for i, result in enumerate(results, 1):
# #                     print(f"\n{i}. Score: {result['score']}")
# #                     print(f"   ID: {result['id']}")
# #                     print(f"   {result['text']}")

# #             print("\nSemantic search for 'AI and machine learning companies':")
# #             results = semantic_search("AI and machine learning companies", namespace="company-namespace")
# #             if results:
# #                 for i, result in enumerate(results, 1):
# #                     print(f"\n{i}. Score: {result['score']}")
# #                     print(f"   ID: {result['id']}")
# #                     print(f"   {result['text']}")

# #     finally:
# #         cleanup()



# from pinecone import Pinecone
# from pinecone import ServerlessSpec
# import json
# from llama_cpp import Llama
# import atexit
# import numpy as np

# # Initialize Pinecone
# pc = Pinecone(api_key="Your api key")

# # Create or connect to the Pinecone index
# try:
#     pc.create_index(
#         name="company-data-index2",
#         dimension=758,
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
# except Exception as e:
#     print(f"Index might already exist: {e}")

# # Get the index reference
# index = pc.Index("company-data-index2")

# # Initialize LLaMA model
# llama = Llama(
#     model_path="llama-2-7b.Q2_K.gguf",
#     n_ctx=2048,
#     embedding=True,
#     n_gpu_layers=0
# )

# def cleanup():
#     try:
#         llama.reset()
#         del llama
#     except:
#         pass

# atexit.register(cleanup)

def process_embedding(embedding):
    """Process embedding to ensure it's a flat list of floats"""
    if isinstance(embedding, (list, np.ndarray)):
        if isinstance(embedding[0], (list, np.ndarray)):
            embedding = [item for sublist in embedding for item in sublist]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        embedding = [float(val) for val in embedding]
    return embedding

# def load_and_index_data(file_path, namespace="company-namespace"):
#     """Load and index company data into a specified namespace"""
#     try:
#         with open(file_path, 'r') as file:
#             data = json.load(file)

#         vectors = []
#         for company in data:
#             try:
#                 # Generate embeddings
#                 embedding = llama.embed(company["text"])
                
#                 # Process the embedding to ensure correct format
#                 processed_embedding = process_embedding(embedding)
                
#                 vectors.append({
#                     'id': str(company["id"]),
#                     'values': processed_embedding,
#                     'metadata': {
#                         "id": str(company["id"]),
#                         "text": company["text"][:1000]  # Limit text size in metadata
#                     }
#                 })
#                 print(f"Successfully processed company {company['id']}")
#             except Exception as e:
#                 print(f"Error processing company {company['id']}: {e}")
#                 continue

#         # Upload in much smaller batches
#         BATCH_SIZE = 10  # Reduced from 100 to 10
#         for i in range(0, len(vectors), BATCH_SIZE):
#             batch = vectors[i:i + BATCH_SIZE]
#             try:
#                 index.upsert(vectors=batch, namespace=namespace)
#                 print(f"Successfully uploaded batch {i//BATCH_SIZE + 1} of {len(vectors)//BATCH_SIZE + 1}")
#             except Exception as e:
#                 print(f"Error uploading batch {i//BATCH_SIZE + 1}: {e}")
#                 # Try uploading one by one if batch fails
#                 for vector in batch:
#                     try:
#                         index.upsert(vectors=[vector], namespace=namespace)
#                         print(f"Successfully uploaded single vector {vector['id']}")
#                     except Exception as e:
#                         print(f"Error uploading vector {vector['id']}: {e}")

#         return data
#     except Exception as e:
#         print(f"Error loading or indexing data: {e}")
#         return None

# def semantic_search(query_text, top_k=3, namespace="company-namespace"):
#     """Search for companies based on semantic similarity to query text"""
#     try:
#         # Generate embedding for the query
#         query_vector = llama.embed(query_text)
#         processed_vector = process_embedding(query_vector)

#         # Query Pinecone with namespace
#         results = index.query(
#             vector=processed_vector,
#             top_k=top_k,
#             include_metadata=True,
#             namespace=namespace
#         )

#         # Format and return results
#         formatted_results = []
#         for match in results["matches"]:
#             formatted_results.append({
#                 "score": round(float(match["score"]), 3),
#                 "id": match["metadata"]["id"],
#                 "text": match["metadata"]["text"]
#             })
#         return formatted_results
#     except Exception as e:
#         print(f"Error during semantic search: {str(e)}")
#         return []

# def search_by_id(company_id, data, namespace="company-namespace"):
#     """Search for a company by its ID"""
#     try:
#         company_text = next((company["text"] for company in data if company["id"] == company_id), None)
#         if company_text is None:
#             return "Company not found."

#         # Generate and process embedding
#         query_vector = llama.embed(company_text)
#         processed_vector = process_embedding(query_vector)

#         # Query Pinecone
#         query_result = index.query(
#             vector=processed_vector,
#             top_k=1,
#             include_metadata=True,
#             namespace=namespace
#         )

#         if query_result["matches"]:
#             return query_result["matches"][0]["metadata"]["text"]
#         return "Company not found."
#     except Exception as e:
#         return f"Error during search: {str(e)}"
    
# # Example usage
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
#                 print(f"   Text: {result['text'][:200]}...")  # Show first 200 chars
#     finally:
#         cleanup()




from pinecone import Pinecone
from pinecone import ServerlessSpec
import json
from llama_cpp import Llama
import atexit
import numpy as np

# Initialize Pinecone
pc = Pinecone(api_key="yourAPIKEY")

# Initialize LLaMA model first to get embedding dimension
llama = Llama(
    model_path="llama-2-7b.Q2_K.gguf",
    n_ctx=2048,
    embedding=True,
    n_gpu_layers=0
)

# Get the actual embedding dimension by running a test embedding
test_embedding = llama.embed("test")
EMBEDDING_DIM = len(process_embedding(test_embedding))
print(f"Detected embedding dimension: {EMBEDDING_DIM}")

# Create or connect to the Pinecone index with correct dimension
try:
    pc.create_index(
        name="company-data-index2",
        dimension=EMBEDDING_DIM,  # Use actual embedding dimension
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

def process_embedding(embedding):
    """Process embedding to ensure it's a flat list of floats"""
    if isinstance(embedding, (list, np.ndarray)):
        if isinstance(embedding[0], (list, np.ndarray)):
            embedding = [item for sublist in embedding for item in sublist]
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        embedding = [float(val) for val in embedding]
    return embedding

def verify_embedding_dimension(embedding):
    """Verify that embedding has the correct dimension"""
    if len(embedding) != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {len(embedding)}")
    return embedding

def load_and_index_data(file_path, namespace="company-namespace"):
    """Load and index company data into a specified namespace"""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

        vectors = []
        for company in data:
            try:
                # Generate embeddings
                embedding = llama.embed(company["text"])
                processed_embedding = process_embedding(embedding)
                
                # Verify dimension
                try:
                    verified_embedding = verify_embedding_dimension(processed_embedding)
                except ValueError as e:
                    print(f"Skipping company {company['id']}: {e}")
                    continue
                
                vectors.append({
                    'id': str(company["id"]),
                    'values': verified_embedding,
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
        # Generate embedding for the query
        query_vector = llama.embed(query_text)
        processed_vector = process_embedding(query_vector)
        
        # Verify dimension
        verified_vector = verify_embedding_dimension(processed_vector)

        # Query Pinecone with namespace
        results = index.query(
            vector=verified_vector,
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

        # Generate and process embedding
        query_vector = llama.embed(company_text)
        processed_vector = process_embedding(query_vector)
        
        # Verify dimension
        verified_vector = verify_embedding_dimension(processed_vector)

        # Query Pinecone
        query_result = index.query(
            vector=verified_vector,
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
if __name__ == "__main__":
    try:
        file_path = 'C:\\Users\\DhruvSingh\\Downloads\\Work\\Huggingface\\company_data.json'
        namespace = "company-namespace"
        
        print("Loading and indexing data...")
        data = load_and_index_data(file_path, namespace=namespace)
        
        if data:
            # Try some searches
            print("\nSearching for 'AI companies'...")
            results = semantic_search("AI companies", namespace=namespace)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result['score']}")
                print(f"   ID: {result['id']}")
                print(f"   Text: {result['text'][:200]}...")
    finally:
        cleanup()