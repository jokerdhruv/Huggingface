# from langchain_community.embeddings import LlamaCppEmbeddings
# from llama_cpp import Llama
# # Load the Llama model for embeddings
# llama_embeddings = LlamaCppEmbeddings(model_path="llama-2-7b.Q2_K.gguf")
# llama = Llama(
#     model_path="llama-2-7b.Q2_K.gguf",  # Replace with the actual model path
#     n_ctx=2048,
#     n_gpu_layers=0  # For CPU usage
# )
# # Text to embed
# text = "This is a test document."

# Generate embeddings
# query_result = llama_embeddings.embed_query(text)  # Embedding for a single query
# doc_result = llama_embeddings.embed_documents([text])  

from llama_cpp import Llama


# Load the Llama model
llama = Llama(
    model_path="llama-2-7b.Q2_K.gguf",  # Replace with the actual model path
    n_ctx=2048,
    embedding=True,
    
    n_gpu_layers=0  # For CPU usage
)

# Text to embed
text = "This is a test document."

# Generate embedding for the text
embedding = llama.embed(text)

# Output the embedding
print("Embedding:", embedding)




help(Llama)

# Output the results
# print("Query Embedding:", query_result)
# print("Document Embedding:", doc_result)
