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





def create_faiss_index(embeddings):
    dim = len(embeddings[0][1])  # Dimension of embeddings
    index = faiss.IndexFlatL2(dim)  # L2 distance
    vectors = np.array([embedding[1] for embedding in embeddings], dtype=np.float32)
    index.add(vectors)
    return index, {i: token for i, (token, _) in enumerate(embeddings)}

# Example usage
index, id_to_text = create_faiss_index(embeddings)



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
