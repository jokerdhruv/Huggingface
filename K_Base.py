from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from llama_cpp import Llama
import faiss
import numpy as np

def transcribe_audio(audio_path):
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None
    
    waveform, original_sample_rate = torchaudio.load(audio_path)
    target_sample_rate = 16000
    if original_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    input_features = processor(waveform.squeeze().numpy(), sampling_rate=target_sample_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return ' '.join(transcription)

def generate_embeddings(text, llama_model_path):
    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        embedding=True
    )
    
    # Split text into chunks if it's too long
    max_chunk_size = 512  # Adjust based on your needs
    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    embeddings = []
    for chunk in chunks:
        try:
            embedding = llama.embed(chunk)
            embeddings.append((chunk, np.array(embedding, dtype=np.float32)))
        except Exception as e:
            print(f"Error generating embedding for chunk: {e}")
            continue
    
    del llama
    return embeddings

def force_dimension(embedding, target_dim=768):
    """
    Adjust the embedding to the target dimension by padding or truncating.
    Ensures the output is a 1D array.
    """
    embedding = np.array(embedding, dtype=np.float32)
    
    # If embedding is 2D, flatten it first
    if len(embedding.shape) > 1:
        embedding = embedding.flatten()
    
    current_size = len(embedding)
    if current_size < target_dim:
        return np.pad(embedding, (0, target_dim - current_size), mode='constant')
    elif current_size > target_dim:
        return embedding[:target_dim]
    return embedding

def create_faiss_index(embeddings, target_dim=768):
    """
    Create a FAISS index from embeddings with consistent dimensionality.
    """
    valid_embeddings = []
    id_to_text = {}

    for i, (text, embedding) in enumerate(embeddings):
        try:
            adjusted_embedding = force_dimension(embedding, target_dim)
            
            if len(adjusted_embedding.shape) == 1 and adjusted_embedding.shape[0] == target_dim:
                valid_embeddings.append(adjusted_embedding)
                id_to_text[len(valid_embeddings) - 1] = text
                print(f"Successfully added embedding {i} to index")
            else:
                print(f"Skipping embedding {i} due to incorrect shape after adjustment")
        except Exception as e:
            print(f"Error processing embedding {i}: {e}")
            continue

    if not valid_embeddings:
        raise ValueError("No valid embeddings found to create the FAISS index.")

    vectors = np.stack(valid_embeddings, axis=0)
    print(f"Final vectors shape: {vectors.shape}")

    index = faiss.IndexFlatL2(target_dim)
    index.add(vectors.astype(np.float32))
    
    print(f"Created FAISS index with {index.ntotal} vectors")
    return index, id_to_text

def query_faiss_index(index, query_text, llama_model_path, id_to_text, top_k=3):
    """
    Query the FAISS index with improved error handling and fallback options.
    """
    if index.ntotal == 0:
        print("Warning: FAISS index is empty")
        return []

    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        embedding=True
    )
    
    try:
        # Generate and process query embedding
        query_embedding = llama.embed(query_text)
        query_embedding = force_dimension(query_embedding, 768)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Adjust top_k if it's larger than the number of vectors in the index
        actual_top_k = min(top_k, index.ntotal)
        
        # Search the index
        distances, indices = index.search(query_vector, actual_top_k)
        
        # Filter out invalid results (-1 indices) and create results list
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Only include valid results
                try:
                    text = id_to_text[int(idx)]
                    distance = distances[0][i]
                    results.append((text, distance))
                except KeyError:
                    print(f"Warning: Index {idx} not found in id_to_text mapping")
                    continue
        
        if not results:
            print("No valid results found in search")
            
        return results
    
    except Exception as e:
        print(f"Error during search: {e}")
        return []
    
    finally:
        del llama

def generate_response(retrieved_text, query, llama_model_path):
    if not retrieved_text:
        return "I couldn't find any relevant information to answer your question."
        
    llama = Llama(
        model_path=llama_model_path,
        n_ctx=4096,
        n_gpu_layers=0
    )
    
    input_prompt = f"Context: {retrieved_text}\n\nQuestion: {query}\nAnswer:"
    response = llama(input_prompt)
    del llama
    return response['choices'][0]['text'] if isinstance(response, dict) else response

def main():
    audio_path = "audio.mp3"
    llama_model_path = "llama-2-7b.Q2_K.gguf"
    
    print("Starting transcription...")
    transcribed_text = transcribe_audio(audio_path)
    print("Transcribed Text:", transcribed_text)
    
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(transcribed_text, llama_model_path)
    print(f"Generated {len(embeddings)} embeddings")
    
    print("\nCreating FAISS index...")
    index, id_to_text = create_faiss_index(embeddings)
    print(f"FAISS index created with {index.ntotal} vectors")
    
    query_text = "What is the main topic of the audio?"
    print(f"\nQuerying index with: {query_text}")
    results = query_faiss_index(index, query_text, llama_model_path, id_to_text)
    
    if results:
        print("\nGenerating response...")
        retrieved_text = ' '.join([text for text, _ in results])
        final_response = generate_response(retrieved_text, query_text, llama_model_path)
        print("Generated Response:", final_response)
    else:
        print("No matches found in the index")

if __name__ == "__main__":
    main()  