from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from llama_cpp import Llama
import faiss
import numpy as np

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

def transcribe_audio(audio_path):
    waveform, original_sample_rate = torchaudio.load(audio_path)
    target_sample_rate = 16000
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return ' '.join(transcription)

# Example usage
audio_path = "audio.mp3"
transcribed_text = transcribe_audio(audio_path)
print("Transcribed Text:", transcribed_text)


def generate_embeddings(text, llama_model_path):
    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        embedding=True
    )
    embeddings = []
    tokens = text.split()  # Split text into tokens (or sentences if preferred)
    for token in tokens:
        embedding = llama.embed(token)  # Generate embedding for each token
        embeddings.append((token, embedding))
    del llama  # Cleanup LLaMA model
    return embeddings

# Example usage
llama_model_path = "llama-2-7b.Q2_K.gguf"
embeddings = generate_embeddings(transcribed_text, llama_model_path)



# def create_faiss_index(embeddings, expected_dim=768):
#     valid_embeddings = []
#     id_to_text = {}

#     for i, (token, embedding) in enumerate(embeddings):
#         # Ensure embedding is a numpy array
#         if isinstance(embedding, list):
#             embedding = np.array(embedding, dtype=np.float32)

#         # Filter embeddings with incorrect dimensions
#         if len(embedding) == expected_dim:
#             valid_embeddings.append(embedding)
#             id_to_text[len(valid_embeddings) - 1] = token
#         else:
#             print(f"Skipping invalid embedding at index {i}: size {len(embedding)}")

#     # Stack embeddings into a 2D array
#     vectors = np.stack(valid_embeddings, axis=0)  # Convert list of arrays into a 2D array

#     # Check the final shape
#     print(f"Final shape of vectors: {vectors.shape}")

#     # Initialize and populate the FAISS index
#     dim = expected_dim  # Expected dimensionality of embeddings
#     index = faiss.IndexFlatL2(dim)  # L2 (Euclidean) distance metric
#     index.add(vectors)

#     return index, id_to_text



# index, id_to_text = create_faiss_index(embeddings)
# print(f"FAISS index created with {index.ntotal} vectors.")
# print(f"Mapping of IDs to text: {id_to_text}")


# def create_faiss_index(embeddings, target_dim=768):
#     """
#     Create a FAISS index from embeddings with enforced dimensionality.
#     """
#     valid_embeddings = []
#     id_to_text = {}

#     for i, (token, embedding) in enumerate(embeddings):
#         # Ensure the embedding has the correct dimension
#         embedding = force_dimension(np.array(embedding, dtype=np.float32), target_dim)
#         valid_embeddings.append(embedding)
#         id_to_text[len(valid_embeddings) - 1] = token

#     if len(valid_embeddings) == 0:
#         raise ValueError("No valid embeddings found to create the FAISS index.")

#     # Convert to a 2D numpy array
#     vectors = np.stack(valid_embeddings, axis=0)

#     # Initialize and populate the FAISS index
#     index = faiss.IndexFlatL2(target_dim)
#     index.add(vectors)

#     return index, id_to_text


# def create_faiss_index(embeddings, target_dim=768):
#     """
#     Create a FAISS index from embeddings with enforced dimensionality.
#     """
#     valid_embeddings = []
#     id_to_text = {}

#     for i, (token, embedding) in enumerate(embeddings):
#         # Ensure the embedding has the correct dimension
#         embedding = force_dimension(np.array(embedding, dtype=np.float32), target_dim)

#         # Check shape of embedding before adding to valid_embeddings
#         print(f"Embedding {i} shape: {embedding.shape}")

#         valid_embeddings.append(embedding)
#         id_to_text[len(valid_embeddings) - 1] = token

#     if len(valid_embeddings) == 0:
#         raise ValueError("No valid embeddings found to create the FAISS index.")

#     # Convert to a 2D numpy array
#     vectors = np.stack(valid_embeddings, axis=0)

#     # Initialize and populate the FAISS index
#     index = faiss.IndexFlatL2(target_dim)
#     index.add(vectors)

#     return index, id_to_text



def force_dimension(embedding, target_dim=768):
    """
    Adjust the embedding to the target dimension by padding or truncating.
    """
    if len(embedding) < target_dim:
        # Pad with zeros if the embedding is smaller
        print(f"Padding embedding from size {len(embedding)} to {target_dim}")
        return np.pad(embedding, (0, target_dim - len(embedding)), mode='constant')
    elif len(embedding) > target_dim:
        # Truncate if the embedding is larger
        print(f"Truncating embedding from size {len(embedding)} to {target_dim}")
        return embedding[:target_dim]
    return embedding

def create_faiss_index(embeddings, target_dim=768):
    """
    Create a FAISS index from embeddings with enforced dimensionality.
    """
    valid_embeddings = []
    id_to_text = {}

    for i, (token, embedding) in enumerate(embeddings):
        # Convert to numpy array if it's not already
        embedding = np.array(embedding, dtype=np.float32)
        
        # Ensure the embedding has the correct dimension
        original_size = len(embedding)
        embedding = force_dimension(embedding, target_dim)

        # Check if the dimension is valid
        if len(embedding) != target_dim:
            print(f"Skipping embedding at index {i} due to size mismatch. Expected {target_dim}, got {len(embedding)}")
            continue

        # Check shape before adding to valid_embeddings
        print(f"Embedding {i} shape after adjustment: {embedding.shape}")

        valid_embeddings.append(embedding)
        id_to_text[len(valid_embeddings) - 1] = token

    if len(valid_embeddings) == 0:
        raise ValueError("No valid embeddings found to create the FAISS index.")

    # Convert to a 2D numpy array
    vectors = np.stack(valid_embeddings, axis=0)

    # Initialize and populate the FAISS index
    index = faiss.IndexFlatL2(target_dim)
    index.add(vectors)

    return index, id_to_text





def query_faiss_index(index, query_text, llama_model_path, id_to_text, top_k=3):
    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        embedding=True
    )
    query_embedding = llama.embed(query_text)  # Generate embedding for query
    del llama
    distances, indices = index.search(np.array([query_embedding], dtype=np.float32), top_k)
    results = [(id_to_text[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return results

# Generate embeddings from the transcribed text
embeddings = generate_embeddings(transcribed_text, llama_model_path)

# Create the FAISS index
index, id_to_text = create_faiss_index(embeddings)
print(f"FAISS index created with {index.ntotal} vectors.")
print(f"Mapping of IDs to text: {id_to_text}")

# Example usage
query_text = "What is the main topic of the audio?"
results = query_faiss_index(index, query_text, llama_model_path, id_to_text)
for text, distance in results:
    print(f"Match: {text} (Distance: {distance})")



def generate_response(retrieved_text, query, llama_model_path):
    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0
    )
    input_prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\nAnswer:"
    response = llama(input_prompt)
    del llama
    return response

# Example usage
retrieved_text = ' '.join([text for text, _ in results])
final_response = generate_response(retrieved_text, query_text, llama_model_path)
print("Generated Response:", final_response)
